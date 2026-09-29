# Complete Luna v2 retrieval execution, validation, and routing signals.

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from adaptive_rag.retrieval import (BM25Retriever, DenseRetriever,
    cross_encoder_rerank_with_model, evaluate_rankings, run_hybrid, run_retriever)
from .common import atomic_json, digest, file_hash, load_table, read_json, save_table, versions


# Complete every retrieval method for every query before caching a domain.
def run_retrieval(cfg, qa, docs, *, dense_factory=None, reranker_factory=None):
    settings = cfg['retrieval']
    signature = digest({'settings': settings, 'qa': digest(qa.to_dict('records')),
                        'docs': digest(docs.to_dict('records')), 'version': cfg['version']})
    manifest_path = cfg['paths']['artifacts'] / 'retrieval_manifest.json'
    if manifest_path.exists() and read_json(manifest_path)['signature'] != signature:
        raise ValueError('Retrieval inputs changed. Use a fresh luna_v2 run directory.')
    atomic_json(manifest_path, {'signature': signature, 'versions': versions()})
    # Prevent a library compatibility fallback from downloading model weights.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if reranker_factory is None:
        from sentence_transformers import CrossEncoder
        reranker_factory = lambda: CrossEncoder(settings['reranker_model'], max_length=512, local_files_only=True)
    reranker = None
    parts = []
    for domain, questions in qa.groupby('domain', sort=True):
        name = 'rankings_' + domain
        cache = cfg['paths']['artifacts'] / (name + '.parquet')
        if cache.exists():
            parts.append(load_table(cfg, name))
            continue
        documents = docs[docs.domain.eq(domain)].reset_index(drop=True)
        if documents.empty:
            raise ValueError(f'No documents for {domain}')
        bm25 = BM25Retriever().fit(documents)
        dense = (dense_factory() if dense_factory else DenseRetriever(settings['dense_model'],
                 local_files_only=True, batch_size=settings['batch_size'])).fit(documents)
        # Model loading and warm-up are outside measured request time.
        dense.search(questions.iloc[0].question, top_k=1)
        if reranker is None:
            reranker = reranker_factory()
        reranker.predict([(questions.iloc[0].question, documents.iloc[0].text[:1000])],
                         batch_size=1, show_progress_bar=False)
        sparse = run_retriever(questions, bm25, method='bm25', top_k=settings['top_k'])
        dense_results = run_retriever(questions, dense, method='dense', top_k=settings['top_k'])
        hybrid = run_hybrid(questions, bm25, dense, top_k=max(settings['top_k'], settings['rerank_depth']),
                            candidate_depth=settings['candidate_depth'], rrf_k=settings['rrf_k'])
        lookup = documents.set_index('document_id')
        reranked = []
        for query in questions.itertuples():
            h = hybrid[hybrid.example_id.eq(query.example_id)].iloc[0]
            candidates = lookup.loc[list(h.document_ids)[:settings['rerank_depth']]].reset_index()
            start = time.perf_counter()
            ids, scores, meta = cross_encoder_rerank_with_model(query.question, candidates,
                model=reranker, model_id=settings['reranker_model'], top_k=settings['top_k'],
                batch_size=settings['batch_size'])
            rerank_ms = (time.perf_counter() - start) * 1000
            reranked.append({'example_id': query.example_id, 'domain': domain,
                'retrieval_method': 'hybrid_rrf_reranked', 'document_ids': ids, 'scores': scores,
                'latency_ms': float(h.latency_ms) + rerank_ms, 'rerank_only_ms': rerank_ms,
                'metadata': meta})
        frame = pd.concat([sparse, dense_results, hybrid, pd.DataFrame(reranked)], ignore_index=True)
        frame['metadata'] = frame.metadata.map(lambda x: json.dumps(x, sort_keys=True))
        save_table(cfg, name, frame)
        parts.append(frame)
    rankings = pd.concat(parts, ignore_index=True)
    validate_rankings(qa, docs, rankings)
    save_table(cfg, 'rankings', rankings)
    metrics = evaluate_rankings(qa, rankings, cutoffs=[1, 5, 10])
    save_table(cfg, 'retrieval_metrics', metrics)
    atomic_json(manifest_path, {'signature': signature, 'versions': versions(),
        'rankings_sha256': file_hash(cfg['paths']['artifacts'] / 'rankings.parquet'),
        'latency_note': 'Reranked latency includes hybrid + rerank; warm-up excluded. Local hardware only.'})
    return rankings, metrics


# Verify method coverage, row counts, identifiers, and ranked documents.
def validate_rankings(qa, docs, rankings):
    methods = {'bm25', 'dense', 'hybrid_rrf', 'hybrid_rrf_reranked'}
    if rankings.duplicated(['example_id', 'retrieval_method']).any():
        raise ValueError('Duplicate retrieval results.')
    if set(rankings.retrieval_method) != methods:
        raise ValueError('Missing retrieval methods.')
    for method, group in rankings.groupby('retrieval_method'):
        if set(group.example_id) != set(qa.example_id):
            raise ValueError(f'{method} does not cover the complete QA cohort.')
    doc_domain = dict(zip(docs.document_id, docs.domain))
    for r in rankings.itertuples():
        if not len(r.document_ids) or len(r.document_ids) != len(r.scores):
            raise ValueError('Empty or malformed ranking.')
        if len(set(r.document_ids)) != len(r.document_ids) or not np.isfinite(r.scores).all():
            raise ValueError('Duplicate documents or nonfinite scores.')
        if any(doc_domain.get(d) != r.domain for d in r.document_ids):
            raise ValueError('Cross-domain or missing retrieval document.')
        if not np.isfinite(r.latency_ms) or r.latency_ms < 0:
            raise ValueError('Invalid retrieval latency.')
    hybrid = rankings[rankings.retrieval_method.eq('hybrid_rrf')].set_index('example_id')
    for r in rankings[rankings.retrieval_method.eq('hybrid_rrf_reranked')].itertuples():
        if r.latency_ms < hybrid.loc[r.example_id, 'latency_ms']:
            raise ValueError('Reranked latency must include hybrid retrieval.')


# Expose only signals available after hybrid retrieval and before escalation.
def hybrid_features(rankings):
    rows = []
    for r in rankings[rankings.retrieval_method.eq('hybrid_rrf')].itertuples():
        scores = np.asarray(r.scores, dtype=float)
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores)) / len(scores)
        rows.append({'example_id': r.example_id, 'hybrid_top1': scores[0],
            'hybrid_margin': scores[0] - scores[1] if len(scores) > 1 else 0.,
            'hybrid_mean': scores.mean(), 'hybrid_std': scores.std(),
            'hybrid_entropy': float(-np.sum(probs * np.log(probs + 1e-12)))})
    return pd.DataFrame(rows)

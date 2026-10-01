# Annotation-based labels, bounded corpora, and frozen Luna v2 splits.

from __future__ import annotations

import hashlib
import json
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from adaptive_rag.data import (deterministic_subset, load_domain_from_dir,
                               normalize_natural_questions)
from adaptive_rag.schemas import normalize_list
from .common import atomic_json, digest, file_hash, load_table, read_json, save_table, versions


# Build one reference per NQ annotation, joining multi-span answers.
def annotation_labels(record):
    tokens = record.get('document_tokens') or record.get('tokens') or []
    tokens = [('' if x.get('html_token') else x.get('token', x.get('text', '')))
              if isinstance(x, dict) else str(x)
              for x in tokens] or str(record.get('document_text', '')).split()
    refs, has_long = [], False
    for ann in record.get('annotations') or []:
        long = ann.get('long_answer') or {}
        has_long |= (long.get('start_token', -1) >= 0 and
                     long.get('end_token', -1) > long.get('start_token', -1))
        yes_no = str(ann.get('yes_no_answer', 'NONE')).upper()
        if yes_no in ('YES', 'NO'):
            refs.append(yes_no.lower())
            continue
        spans = []
        for span in ann.get('short_answers') or []:
            start, end = span.get('start_token', -1), span.get('end_token', -1)
            text = span.get('text', '')
            if not text and 0 <= start < end <= len(tokens):
                text = ' '.join(token for token in tokens[start:end] if token)
            if text:
                spans.append(str(text).strip())
        if spans:
            refs.append(' '.join(spans))
    refs = sorted(set(refs))
    category = 'short_or_yes_no' if refs else 'long_only' if has_long else 'no_positive_annotation'
    return refs, has_long or bool(refs), category


# Create NQ labels and supervision categories from raw annotations.
def corrected_nq(record, source):
    qa, docs = normalize_natural_questions([record], source_name=source, split='train')
    if qa.empty:
        return qa, docs
    refs, positive, category = annotation_labels(record)
    qa.at[0, 'answer_aliases'] = refs
    qa.at[0, 'answer'] = refs[0] if refs else ''
    qa.at[0, 'answerable'] = positive
    if not positive:
        qa.at[0, 'gold_document_ids'] = []
    qa['annotation_category'] = category
    qa['generation_eligible'] = bool(refs)
    return qa, docs


# Keep all positive documents and sample deterministic corpus negatives.
def bounded_corpus(docs, qa, maximum, seed):
    gold = {str(x) for values in qa.gold_document_ids for x in normalize_list(values)}
    docs = docs.drop_duplicates('document_id').copy()
    missing = gold - set(docs.document_id)
    if missing:
        raise ValueError(f'{len(missing)} gold documents missing from raw corpus.')
    required = docs[docs.document_id.isin(gold)]
    if len(required) > maximum:
        raise ValueError('Corpus cap is smaller than the required gold-document count.')
    others = docs[~docs.document_id.isin(gold)]
    count = min(maximum - len(required), len(others))
    sampled = deterministic_subset(others, count, seed=seed, id_col='document_id') if count else others.iloc[:0]
    return pd.concat([required, sampled], ignore_index=True).sort_values('document_id').reset_index(drop=True)


# Group duplicate questions and shared pages before assigning frozen splits.
def split_cohort(qa, seed):
    eligible = qa[qa.generation_eligible].copy().reset_index(drop=True)
    eligible['split'] = 'techqa_test'
    source = eligible[eligible.domain.eq('natural_questions')]
    parents = {i: i for i in source.index}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, row in source.iterrows():
        keys = ['q:' + ' '.join(row.question.lower().split())]
        keys += ['d:' + str(x) for x in normalize_list(row.gold_document_ids)]
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            owners[key] = i
    groups = np.array([find(i) for i in source.index])
    if len(set(groups)) < 5:
        raise ValueError('Need at least five independent NQ groups to freeze train/validation/test.')
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(source, groups=groups))
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
                      .split(source.iloc[trainval], groups=groups[trainval]))
    eligible.loc[source.index[trainval[train]], 'split'] = 'train'
    eligible.loc[source.index[trainval[val]], 'split'] = 'validation'
    eligible.loc[source.index[test], 'split'] = 'nq_test'
    eligible['split_group'] = eligible.example_id
    eligible.loc[source.index, 'split_group'] = ['nq_group_' + str(x) for x in groups]
    # Avoid duplicate question text across the source and transfer test.
    source_text = set(source.question.str.lower().str.split().str.join(' '))
    eligible['cross_domain_duplicate'] = (eligible.domain.eq('techqa') &
        eligible.question.str.lower().str.split().str.join(' ').isin(source_text))
    eligible.loc[eligible.cross_domain_duplicate, 'split'] = 'excluded_duplicate'
    return eligible


# Build or load the official run's dataset artifacts.
def prepare_data(cfg):
    manifest_path = cfg['paths']['processed'] / 'dataset_manifest.json'
    raw_paths = [cfg['paths']['nq']] + sorted(cfg['paths']['techqa'].rglob('*.json'))

    # A release copy can reuse the frozen, checksum-verified tables without
    # redistributing the much larger source corpora.
    if manifest_path.exists():
        old = read_json(manifest_path)
        for name, checksum in old['table_sha256'].items():
            if file_hash(cfg['paths']['processed'] / f'{name}.parquet') != checksum:
                raise ValueError(f'Cached {name} has changed; use a fresh run directory.')
        if not raw_paths[0].exists() or len(raw_paths) < 2:
            return (load_table(cfg, 'qa', processed=True),
                    load_table(cfg, 'docs', processed=True),
                    load_table(cfg, 'cohort', processed=True))

    if not raw_paths[0].exists() or len(raw_paths) < 2:
        raise FileNotFoundError(
            'Raw NQ/TechQA files or a complete checksum-verified Luna dataset cache are required.'
        )
    signature = digest({'data': cfg['data'], 'seed': cfg['seed'], 'version': cfg['version'],
                        'raw': [(str(p.relative_to(cfg['_root'])), p.stat().st_size,
                                 p.stat().st_mtime_ns) for p in raw_paths]})
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['signature'] != signature:
            raise ValueError('Data settings/raw files changed. Use fresh luna_v2 output directories.')
        return (load_table(cfg, 'qa', processed=True),
                load_table(cfg, 'docs', processed=True),
                load_table(cfg, 'cohort', processed=True))
    nq_rows, nq_docs = [], {}
    nq_prefix_hash = hashlib.sha256()
    with cfg['paths']['nq'].open(encoding='utf-8') as stream:
        for line in islice(stream, cfg['data']['nq_record_limit']):
            nq_prefix_hash.update(line.encode('utf-8'))
            q, d = corrected_nq(json.loads(line), cfg['paths']['nq'].name)
            nq_rows.extend(q.to_dict('records'))
            nq_docs.update({r['document_id']: r for r in d.to_dict('records')})
    nq, nd = pd.DataFrame(nq_rows), pd.DataFrame(nq_docs.values())
    tq, td, sources = load_domain_from_dir('techqa', cfg['paths']['techqa'])
    tq['annotation_category'] = np.where(tq.answerable, 'answerable', 'benchmark_unanswerable')
    tq['generation_eligible'] = [not bool(row.answerable) or bool(normalize_list(row.answer_aliases))
                                 for row in tq.itertuples()]
    qa_parts, doc_parts = [], []
    for q, d in ((nq, nd), (tq, td)):
        q = q[q.question.str.len().ge(4)].drop_duplicates('example_id')
        q = deterministic_subset(q, cfg['data']['sample_per_domain'], seed=cfg['seed'], id_col='example_id')
        d = d[d.text.str.len().ge(20)]
        d = bounded_corpus(d, q, cfg['data']['max_documents_per_domain'], cfg['seed'])
        qa_parts.append(q)
        doc_parts.append(d)
    qa, docs = pd.concat(qa_parts, ignore_index=True), pd.concat(doc_parts, ignore_index=True)
    if qa.example_id.duplicated().any() or docs.document_id.duplicated().any():
        raise ValueError('Identifiers must be globally unique.')
    cohort = split_cohort(qa, cfg['seed'])
    for frame in (qa, docs, cohort):
        frame['metadata'] = frame.metadata.map(lambda x: json.dumps(x, sort_keys=True))
    tables = {'qa': qa, 'docs': docs, 'cohort': cohort}
    checksums = {name: file_hash(save_table(cfg, name, frame, processed=True)) for name, frame in tables.items()}
    atomic_json(manifest_path, {'signature': signature, 'table_sha256': checksums,
        'versions': versions(), 'source_files': [str(p.relative_to(cfg['_root'])) for p in raw_paths],
        'source_sha256': {str(cfg['paths']['nq'].relative_to(cfg['_root'])) +
            f'#first-{cfg["data"]["nq_record_limit"]}-records': nq_prefix_hash.hexdigest(),
            **{str(p.relative_to(cfg['_root'])): file_hash(p) for p in raw_paths[1:]}},
        'nq_records_examined': len(nq_rows), 'split_counts': cohort.split.value_counts().to_dict(),
        'note': 'NQ negatives are not globally unanswerable. Only short/yes-no references enter generation.'})
    return qa, docs, cohort

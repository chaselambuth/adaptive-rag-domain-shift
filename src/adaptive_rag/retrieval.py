# Sparse, dense, hybrid, and reranked retrieval implementations.

from __future__ import annotations

import math
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import normalize_list

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# Apply the shared lowercase word tokenizer used by sparse retrieval.
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


# Store ranked identifiers, scores, timing, and retrieval metadata.
@dataclass
class RankingResult:
    example_id: str
    domain: str
    retrieval_method: str
    document_ids: list[str]
    scores: list[float]
    latency_ms: float
    metadata: dict


# Fit, search, and persist a local BM25 document index.
class BM25Retriever:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_titles: dict[str, str] = {}
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.doc_term_counts: list[Counter[str]] = []

    # Build document frequencies and BM25 state from the fixed corpus.
    def fit(self, docs: pd.DataFrame) -> "BM25Retriever":
        self.doc_ids = docs["document_id"].astype(str).tolist()
        self.doc_titles = dict(zip(docs["document_id"].astype(str), docs.get("title", pd.Series([""] * len(docs))).fillna("").astype(str)))
        self.doc_tokens = [tokenize(text) for text in docs["text"].fillna("").astype(str)]
        self.doc_term_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = float(np.mean(lengths)) if lengths else 0.0
        n_docs = max(len(self.doc_ids), 1)
        self.idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }
        return self

    # Score every indexed document for one tokenized query.
    def score(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        if not q_terms or not self.doc_ids:
            return scores
        avgdl = self.avgdl or 1.0
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, counts in enumerate(self.doc_term_counts):
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                doc_len = max(len(self.doc_tokens[idx]), 1)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                scores[idx] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    # Return the highest-scoring BM25 document identifiers.
    def search(self, query: str, *, top_k: int = 10) -> tuple[list[str], list[float]]:
        scores = self.score(query)
        if len(scores) == 0:
            return [], []
        order = np.argsort(-scores)[:top_k]
        return [self.doc_ids[i] for i in order], [float(scores[i]) for i in order]

    # Persist the fitted sparse index.
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    # Load a previously fitted sparse index.
    @staticmethod
    def load(path: Path) -> "BM25Retriever":
        with path.open("rb") as f:
            return pickle.load(f)


# Encode and search a locally cached sentence-transformer index.
class DenseRetriever:
    def __init__(self, model_id: str, *, normalize_embeddings: bool = True, local_files_only: bool = True, batch_size: int = 32):
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.model = None
        self.doc_ids: list[str] = []
        self.doc_titles: dict[str, str] = {}
        self.embeddings: np.ndarray | None = None

    # Load the sentence transformer only when first needed.
    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_id, local_files_only=self.local_files_only)
        return self.model

    # Encode text in deterministic batches with optional normalization.
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        model = self.load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # Encode and retain the fixed document corpus.
    def fit(self, docs: pd.DataFrame) -> "DenseRetriever":
        self.doc_ids = docs["document_id"].astype(str).tolist()
        self.doc_titles = dict(zip(docs["document_id"].astype(str), docs.get("title", pd.Series([""] * len(docs))).fillna("").astype(str)))
        corpus_text = (docs.get("title", "").fillna("").astype(str) + "\n" + docs["text"].fillna("").astype(str)).tolist()
        self.embeddings = self.encode(corpus_text)
        return self

    # Return documents with the largest embedding similarity.
    def search(self, query: str, *, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.embeddings is None:
            raise ValueError("DenseRetriever.fit must be called before search.")
        q = self.encode([query])
        scores = np.matmul(self.embeddings, q[0])
        order = np.argsort(-scores)[:top_k]
        return [self.doc_ids[i] for i in order], [float(scores[i]) for i in order]

    # Persist dense vectors and their document mapping.
    def save_embeddings(self, embeddings_path: Path, mapping_path: Path) -> tuple[Path, Path]:
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        pd.DataFrame({"embedding_row": range(len(self.doc_ids)), "document_id": self.doc_ids, "model_id": self.model_id}).to_csv(mapping_path, index=False)
        return embeddings_path, mapping_path


# Fuse multiple rankings with reciprocal-rank scores.
def reciprocal_rank_fusion(rankings: list[tuple[list[str], list[float]]], *, top_k: int, rrf_k: int = 60) -> tuple[list[str], list[float]]:
    fused: dict[str, float] = defaultdict(float)
    for doc_ids, _scores in rankings:
        for rank, doc_id in enumerate(doc_ids, start=1):
            fused[str(doc_id)] += 1.0 / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [doc_id for doc_id, _ in ordered], [float(score) for _, score in ordered]


# Load a cached cross-encoder without allowing network fallback.
def _load_cross_encoder(model_id: str, *, local_files_only: bool = True):
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(model_id, max_length=512, automodel_args={"local_files_only": local_files_only}, tokenizer_args={"local_files_only": local_files_only})
    except TypeError:
        return CrossEncoder(model_id, max_length=512)


# Load a cross-encoder and rerank a candidate document table.
def cross_encoder_rerank(query: str, candidates: pd.DataFrame, *, model_id: str, top_k: int, batch_size: int = 16, local_files_only: bool = True) -> tuple[list[str], list[float], dict]:
    model = _load_cross_encoder(model_id, local_files_only=local_files_only)
    return cross_encoder_rerank_with_model(query, candidates, model=model, model_id=model_id, top_k=top_k, batch_size=batch_size)


# Rerank candidates with a preloaded cross-encoder instance.
def cross_encoder_rerank_with_model(query: str, candidates: pd.DataFrame, *, model, model_id: str, top_k: int, batch_size: int = 16) -> tuple[list[str], list[float], dict]:
    pairs = [(query, f"{row.title}\n{row.text}") for row in candidates.itertuples()]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))[:top_k]
    ids = candidates.iloc[order]["document_id"].astype(str).tolist()
    return ids, [float(np.asarray(scores)[i]) for i in order], {"reranker_model_id": model_id, "candidate_count": len(candidates)}


# Execute one retriever over every question and record per-query latency.
def run_retriever(qa: pd.DataFrame, retriever, *, method: str, top_k: int) -> pd.DataFrame:
    rows: list[dict] = []
    for row in qa.itertuples():
        start = time.perf_counter()
        doc_ids, scores = retriever.search(row.question, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            RankingResult(
                example_id=str(row.example_id),
                domain=str(row.domain),
                retrieval_method=method,
                document_ids=[str(x) for x in doc_ids],
                scores=[float(x) for x in scores],
                latency_ms=float(latency_ms),
                metadata={},
            ).__dict__
        )
    return pd.DataFrame(rows)


# Run sparse and dense retrieval before reciprocal-rank fusion.
def run_hybrid(qa: pd.DataFrame, bm25: BM25Retriever, dense: DenseRetriever, *, top_k: int, candidate_depth: int, rrf_k: int) -> pd.DataFrame:
    rows: list[dict] = []
    for row in qa.itertuples():
        start = time.perf_counter()
        bm25_rank = bm25.search(row.question, top_k=candidate_depth)
        dense_rank = dense.search(row.question, top_k=candidate_depth)
        doc_ids, scores = reciprocal_rank_fusion([bm25_rank, dense_rank], top_k=top_k, rrf_k=rrf_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            RankingResult(
                example_id=str(row.example_id),
                domain=str(row.domain),
                retrieval_method="hybrid_rrf",
                document_ids=doc_ids,
                scores=scores,
                latency_ms=float(latency_ms),
                metadata={"candidate_depth": candidate_depth, "rrf_k": rrf_k},
            ).__dict__
        )
    return pd.DataFrame(rows)


# Rerank hybrid candidates while preserving cumulative retrieval latency.
def run_reranker(qa: pd.DataFrame, docs: pd.DataFrame, hybrid_rankings: pd.DataFrame, *, model_id: str, top_k: int, candidate_depth: int, batch_size: int, local_files_only: bool) -> pd.DataFrame:
    rows = []
    doc_lookup = docs.set_index("document_id")
    model = _load_cross_encoder(model_id, local_files_only=local_files_only)
    for row in qa.itertuples():
        candidates = hybrid_rankings.loc[hybrid_rankings["example_id"].eq(row.example_id)]
        if candidates.empty:
            continue
        candidate_ids = normalize_list(candidates.iloc[0]["document_ids"])[:candidate_depth]
        candidate_docs = doc_lookup.loc[[doc_id for doc_id in candidate_ids if doc_id in doc_lookup.index]].reset_index()
        start = time.perf_counter()
        doc_ids, scores, metadata = cross_encoder_rerank_with_model(
            row.question,
            candidate_docs,
            model=model,
            model_id=model_id,
            top_k=top_k,
            batch_size=batch_size,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            RankingResult(
                example_id=str(row.example_id),
                domain=str(row.domain),
                retrieval_method="hybrid_rrf_reranked",
                document_ids=doc_ids,
                scores=scores,
                latency_ms=float(latency_ms),
                metadata=metadata,
            ).__dict__
        )
    return pd.DataFrame(rows)


# Compute MRR and recall at configured cutoffs on supported examples.
def evaluate_rankings(qa: pd.DataFrame, rankings: pd.DataFrame, *, cutoffs: list[int]) -> pd.DataFrame:
    qa_gold = qa.set_index("example_id")["gold_document_ids"].to_dict()
    rows = []
    for method, group in rankings.groupby("retrieval_method"):
        for domain, domain_group in group.groupby("domain"):
            supported = 0
            reciprocal_ranks = []
            hit_counts = {k: 0 for k in cutoffs}
            for result in domain_group.itertuples():
                gold = {str(x) for x in normalize_list(qa_gold.get(result.example_id, []))}
                if not gold:
                    continue
                supported += 1
                docs = [str(x) for x in normalize_list(result.document_ids)]
                first_hit = next((idx for idx, doc_id in enumerate(docs, start=1) if doc_id in gold), None)
                reciprocal_ranks.append(0.0 if first_hit is None else 1.0 / first_hit)
                for k in cutoffs:
                    hit_counts[k] += int(any(doc_id in gold for doc_id in docs[:k]))
            if supported == 0:
                rows.append({"retrieval_method": method, "domain": domain, "supported_examples": 0, "metric": "unavailable", "value": np.nan})
                continue
            rows.append({"retrieval_method": method, "domain": domain, "supported_examples": supported, "metric": "MRR", "value": float(np.mean(reciprocal_ranks))})
            for k in cutoffs:
                rows.append({"retrieval_method": method, "domain": domain, "supported_examples": supported, "metric": f"Recall@{k}", "value": float(hit_counts[k] / supported)})
    return pd.DataFrame(rows)


# Summarize retrieval latency by domain and method.
def latency_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    return rankings.groupby(["domain", "retrieval_method"]).agg(
        queries=("example_id", "count"),
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda x: float(np.percentile(x, 95)) if len(x) else np.nan),
    ).reset_index()

# Query-only linguistic and source-domain similarity features.

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CODE_SYMBOL_RE = re.compile(r"[{}\[\]();=<>/\\._:-]")
TECH_TERMS = {
    "api",
    "array",
    "boot",
    "cache",
    "class",
    "client",
    "code",
    "command",
    "compile",
    "database",
    "driver",
    "error",
    "exception",
    "function",
    "http",
    "install",
    "kernel",
    "linux",
    "memory",
    "network",
    "node",
    "python",
    "query",
    "server",
    "service",
    "sql",
    "stack",
    "thread",
    "version",
    "windows",
}


# Extract lowercase word tokens from question text.
def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


# Return the leading interrogative token when present.
def _question_word(text: str) -> str:
    tokens = _tokens(text)
    if not tokens:
        return "none"
    first = tokens[0]
    return first if first in {"what", "who", "when", "where", "why", "how", "which", "is", "can", "does", "do"} else "other"


# Compute routing features available before retrieval begins.
def extract_query_features(qa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in qa.itertuples():
        question = str(row.question or "")
        tokens = _tokens(question)
        chars = max(len(question), 1)
        token_count = len(tokens)
        numeric_tokens = sum(tok.isdigit() for tok in tokens)
        tech_terms = sum(tok in TECH_TERMS for tok in tokens)
        rows.append(
            {
                "example_id": str(row.example_id),
                "domain": str(row.domain),
                "source_split": str(getattr(row, "source_split", "")),
                "question_char_length": len(question),
                "question_token_length": token_count,
                "query_lexical_diversity": float(len(set(tokens)) / token_count) if token_count else np.nan,
                "question_word": _question_word(question),
                "punctuation_density": float(sum(ch in "?!.,;:" for ch in question) / chars),
                "code_symbol_density": float(len(CODE_SYMBOL_RE.findall(question)) / chars),
                "numeric_token_density": float(numeric_tokens / token_count) if token_count else 0.0,
                "technical_term_density": float(tech_terms / token_count) if token_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


# Add source-centroid and nearest-source TF-IDF similarities.
def add_tfidf_domain_features(qa: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if qa.empty:
        return features.copy()
    texts = qa["question"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    domains = qa["domain"].astype(str).to_numpy()
    example_ids = qa["example_id"].astype(str).to_numpy()
    centroids = {
        domain: np.asarray(matrix[domains == domain].mean(axis=0)).reshape(1, -1)
        for domain in sorted(set(domains))
    }
    rows = []
    for idx, example_id in enumerate(example_ids):
        record = {"example_id": example_id}
        vector = matrix[idx]
        for domain, centroid in centroids.items():
            record[f"query_tfidf_similarity_to_{domain}_centroid"] = float(cosine_similarity(vector, centroid)[0, 0])
        nq_mask = domains == "natural_questions"
        if nq_mask.any():
            sims = cosine_similarity(vector, matrix[nq_mask]).ravel()
            same_self = example_id in set(example_ids[nq_mask])
            if same_self and len(sims) > 1:
                sims = np.sort(sims)[:-1]
            record["nearest_natural_questions_query_similarity"] = float(np.max(sims)) if len(sims) else np.nan
        rows.append(record)
    return features.merge(pd.DataFrame(rows), on="example_id", how="left")

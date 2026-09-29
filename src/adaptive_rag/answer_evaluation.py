# Reference-based answer scoring, abstention checks, and oracle labels.

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import normalize_list


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
ABSTENTION_RE = re.compile(
    r"\b(insufficient evidence|not enough (?:information|context)|cannot determine|can't determine|"
    r"cannot answer|can't answer|unknown|not provided|not specified)\b",
    flags=re.IGNORECASE,
)


# Normalize casing, punctuation, articles, and whitespace before comparison.
def normalize_answer(text: str) -> str:
    lowered = str(text or "").lower()
    without_punct = "".join(ch if ch not in string.punctuation else " " for ch in lowered)
    without_articles = ARTICLES_RE.sub(" ", without_punct)
    return " ".join(without_articles.split())


# Tokenize an answer after applying the benchmark normalization rules.
def answer_tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


# Check whether a prediction exactly matches any normalized reference.
def exact_match(prediction: str, references: Iterable[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(ref) for ref in references if str(ref).strip())


# Check whether a prediction contains a complete normalized reference span.
def normalized_contains(prediction: str, references: Iterable[str]) -> bool:
    pred = normalize_answer(prediction)
    if not pred:
        return False
    for ref in references:
        normalized_ref = normalize_answer(ref)
        if normalized_ref and re.search(rf"(^|\s){re.escape(normalized_ref)}($|\s)", pred):
            return True
    return False


# Return the best token-overlap F1 score across all references.
def token_f1(prediction: str, references: Iterable[str]) -> float:
    pred_tokens = answer_tokens(prediction)
    best = 0.0
    for ref in references:
        ref_tokens = answer_tokens(ref)
        if not pred_tokens and not ref_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return float(best)


# Detect the experiment's accepted natural-language abstention phrases.
def is_abstention(answer: str) -> bool:
    return bool(ABSTENTION_RE.search(str(answer or "")))


# Collect and deduplicate the canonical answer and its aliases.
def collect_references(row) -> list[str]:
    refs = []
    answer = getattr(row, "answer", "")
    if str(answer or "").strip():
        refs.append(str(answer))
    refs.extend(str(x) for x in normalize_list(getattr(row, "answer_aliases", [])) if str(x).strip())
    return sorted(set(refs))


# Score one generated answer while keeping answerability and abstention distinct.
def evaluate_answer(
    generated_answer: str,
    references: Iterable[str],
    *,
    answerable: bool,
    min_f1_for_correct: float = 0.5,
) -> dict:
    refs = [str(ref) for ref in references if str(ref).strip()]
    abstained = is_abstention(generated_answer)
    em = exact_match(generated_answer, refs) if refs else False
    contains = normalized_contains(generated_answer, refs) if refs else False
    f1 = token_f1(generated_answer, refs) if refs else np.nan
    has_reference = bool(refs)

    if bool(answerable) and has_reference:
        correct = bool(em or contains or (np.isfinite(f1) and f1 >= min_f1_for_correct))
        source = "benchmark-derived"
    elif not bool(answerable):
        correct = bool(abstained)
        source = "answerability-derived"
    else:
        correct = np.nan
        source = "unavailable"

    return {
        "exact_match": bool(em),
        "normalized_contains": bool(contains),
        "token_f1": float(f1) if np.isfinite(f1) else np.nan,
        "answer_correct": correct,
        "abstained": bool(abstained),
        "successful_abstention": bool((not bool(answerable)) and abstained),
        "failed_abstention": bool(bool(answerable) and abstained),
        "answered_unanswerable": bool((not bool(answerable)) and not abstained),
        "evaluation_source": source,
        "reference_count": len(refs),
    }


# Join generated rows to references and append answer-level metrics.
def evaluate_generation_results(results: pd.DataFrame, qa: pd.DataFrame, *, min_f1_for_correct: float = 0.5) -> pd.DataFrame:
    qa_lookup = qa.set_index("example_id")
    rows = []
    for row in results.itertuples():
        if row.example_id not in qa_lookup.index:
            raise ValueError(f"Generated row has unknown example_id: {row.example_id}")
        qrow = qa_lookup.loc[row.example_id]
        references = collect_references(qrow)
        record = row._asdict()
        record.pop("Index", None)
        record.update(
            evaluate_answer(
                getattr(row, "generated_answer", ""),
                references,
                answerable=bool(qrow["answerable"]),
                min_f1_for_correct=min_f1_for_correct,
            )
        )
        rows.append(record)
    return pd.DataFrame(rows)


# Choose the cheapest successful action available for each example.
def derive_oracle_actions(evaluated: pd.DataFrame) -> pd.DataFrame:
    required = {"example_id", "condition", "answer_correct", "evaluation_source"}
    missing = sorted(required - set(evaluated.columns))
    if missing:
        raise ValueError(f"Missing evaluated generation columns: {missing}")

    action_order = [
        ("direct", "DIRECT"),
        ("hybrid_rrf", "RETRIEVE"),
        ("hybrid_rrf_reranked", "ESCALATE"),
    ]
    rows = []
    for example_id, group in evaluated.groupby("example_id", sort=True):
        by_condition = {str(row.condition): row for row in group.itertuples()}
        oracle_action = None
        sources = []
        for condition, action in action_order:
            row = by_condition.get(condition)
            if row is None:
                continue
            sources.append(str(row.evaluation_source))
            if row.answer_correct is True or str(row.answer_correct).lower() == "true":
                oracle_action = action
                break
        if oracle_action is None:
            if sources and all(source != "unavailable" for source in sources):
                oracle_action = "ABSTAIN"
                confidence = "medium"
                label_source = "benchmark-derived"
            else:
                oracle_action = "UNAVAILABLE"
                confidence = "unavailable"
                label_source = "unavailable"
        else:
            confidence = "high" if "benchmark-derived" in sources or "answerability-derived" in sources else "low"
            label_source = "benchmark-derived" if "benchmark-derived" in sources else "heuristic"
        rows.append(
            {
                "example_id": example_id,
                "oracle_action": oracle_action,
                "oracle_action_source": label_source,
                "oracle_label_confidence": confidence,
                "available_conditions": sorted(by_condition),
            }
        )
    return pd.DataFrame(rows)

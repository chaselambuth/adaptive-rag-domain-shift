# Shared table-column contracts and scalar normalization helpers.

from __future__ import annotations

QA_COLUMNS = [
    "example_id",
    "domain",
    "question",
    "answer",
    "answer_aliases",
    "answerable",
    "gold_document_ids",
    "source_split",
    "metadata",
]

DOCUMENT_COLUMNS = [
    "document_id",
    "domain",
    "title",
    "text",
    "source",
    "metadata",
]

REQUIRED_QA_COLUMNS = ["example_id", "domain", "question", "answerable", "gold_document_ids"]
REQUIRED_DOCUMENT_COLUMNS = ["document_id", "domain", "text"]


# Convert scalar and array-like values into a predictable Python list.
def normalize_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [value]
    return [value]


# Normalize common boolean encodings while preserving a caller default.
def normalize_bool(value, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "answerable"}:
            return True
        if text in {"false", "no", "n", "0", "unanswerable"}:
            return False
    return default

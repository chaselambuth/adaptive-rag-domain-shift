# Raw dataset loading, schema normalization, sampling, and corpus construction.

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schemas import DOCUMENT_COLUMNS, QA_COLUMNS, normalize_bool, normalize_list


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".gz", ".csv", ".parquet"}


# Build a reproducible identifier from normalized record fields.
def stable_id(*parts: Any, prefix: str) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


# List supported raw-data files in deterministic order.
def discover_raw_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    files = [p for p in path.rglob("*") if p.is_file()]
    return sorted(p for p in files if p.suffix.lower() in SUPPORTED_SUFFIXES or p.name.endswith(".jsonl.gz"))


# Read records from a supported JSON, JSONL, CSV, or Parquet source.
def read_records(path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = []
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if max_records is not None and len(rows) >= max_records:
                        break
            return rows
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = []
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if max_records is not None and len(rows) >= max_records:
                        break
            return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[:max_records] if max_records is not None else data
        if isinstance(data, dict):
            for key in ["data", "examples", "questions", "records", "instances"]:
                if isinstance(data.get(key), list):
                    rows = data[key]
                    return rows[:max_records] if max_records is not None else rows
            return [data]
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported raw file format: {path}")


# Return the first populated value among candidate record keys.
def _first(record: dict[str, Any], keys: Iterable[str], default=None):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        upper = key.upper()
        if upper in record and record[upper] not in (None, ""):
            return record[upper]
        lower = key.lower()
        if lower in record and record[lower] not in (None, ""):
            return record[lower]
    return default


# Extract annotated Natural Questions short and yes/no answers.
def _extract_nq_short_answers(record: dict[str, Any]) -> list[str]:
    annotations = record.get("annotations") or []
    document_tokens = record.get("document_tokens") or record.get("tokens") or []
    answers: list[str] = []
    if isinstance(document_tokens, list):
        token_text = [
            token.get("token", token.get("text", "")) if isinstance(token, dict) else str(token)
            for token in document_tokens
        ]
    else:
        token_text = []
    if not token_text and record.get("document_text"):
        token_text = str(record["document_text"]).split()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        yes_no = annotation.get("yes_no_answer")
        if yes_no and str(yes_no).upper() not in {"NONE", "NULL"}:
            answers.append(str(yes_no).lower())
        for short in annotation.get("short_answers") or []:
            if isinstance(short, dict):
                text = short.get("text")
                if not text and token_text:
                    start = short.get("start_token")
                    end = short.get("end_token")
                    if isinstance(start, int) and isinstance(end, int) and start < end:
                        text = " ".join(token_text[start:end])
                if text:
                    answers.append(str(text))
    return [a for a in answers if a]


# Recover document text from the supported raw-record layouts.
def _document_text_from_record(record: dict[str, Any]) -> str:
    text = _first(record, ["document_text", "doc_text", "text", "body", "contents", "passage", "context"], "")
    if isinstance(text, list):
        return " ".join(str(part) for part in text)
    return str(text or "")


# Convert Natural Questions records into shared QA and document schemas.
def normalize_natural_questions(records: list[dict[str, Any]], *, source_name: str, split: str = "unknown") -> tuple[pd.DataFrame, pd.DataFrame]:
    qa_rows: list[dict[str, Any]] = []
    doc_rows: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(records):
        question = str(_first(record, ["question", "question_text", "query"], "") or "").strip()
        if not question:
            continue
        answer_aliases = normalize_list(_first(record, ["answer_aliases", "answers", "short_answers"], []))
        extracted = _extract_nq_short_answers(record)
        if extracted:
            answer_aliases.extend(extracted)
        answer = _first(record, ["answer", "long_answer"], None)
        if answer is None and answer_aliases:
            answer = answer_aliases[0]
        text = _document_text_from_record(record).strip()
        title = str(_first(record, ["title", "document_title", "wikipedia_title"], "") or "")
        source_doc_id = _first(record, ["document_id", "doc_id", "page_id", "document_url"], None)
        doc_id = str(source_doc_id) if source_doc_id else stable_id("nq", title, text[:500], prefix="doc")
        example_id = str(_first(record, ["example_id", "id", "qid"], stable_id("nq", source_name, i, question, prefix="ex")))
        gold_document_ids = normalize_list(_first(record, ["gold_document_ids", "gold_document_id"], []))
        if not gold_document_ids and text:
            gold_document_ids = [doc_id]
        qa_rows.append(
            {
                "example_id": f"nq_{example_id}" if not example_id.startswith("nq_") else example_id,
                "domain": "natural_questions",
                "question": question,
                "answer": "" if answer is None else str(answer),
                "answer_aliases": sorted(set(str(a) for a in answer_aliases if str(a).strip())),
                "answerable": bool(answer or answer_aliases or gold_document_ids),
                "gold_document_ids": [str(x) for x in gold_document_ids],
                "source_split": split,
                "metadata": {"source_file": source_name, "native_id": example_id},
            }
        )
        if text:
            doc_rows[doc_id] = {
                "document_id": doc_id,
                "domain": "natural_questions",
                "title": title,
                "text": text,
                "source": source_name,
                "metadata": {"source_split": split},
            }
    return pd.DataFrame(qa_rows, columns=QA_COLUMNS), pd.DataFrame(doc_rows.values(), columns=DOCUMENT_COLUMNS)


# Collect embedded TechQA document candidates from a raw record.
def _techqa_candidate_documents(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = _first(record, ["candidate_documents", "documents", "docs", "doc", "paragraphs"], [])
    if isinstance(candidates, dict):
        candidates = [candidates]
    if not isinstance(candidates, list):
        return []
    docs = []
    for item in candidates:
        if isinstance(item, dict):
            docs.append(item)
        elif isinstance(item, str):
            docs.append({"text": item})
    return docs


# Convert TechQA question records into shared QA and document schemas.
def normalize_techqa(records: list[dict[str, Any]], *, source_name: str, split: str = "unknown") -> tuple[pd.DataFrame, pd.DataFrame]:
    qa_rows: list[dict[str, Any]] = []
    doc_rows: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(records):
        question_title = str(_first(record, ["question_title", "question", "query", "utterance", "title"], "") or "").strip()
        question_text = str(_first(record, ["question_text"], "") or "").strip()
        question = question_title if not question_text else f"{question_title}\n{question_text}".strip()
        if not question:
            continue
        example_id = str(_first(record, ["qid", "question_id", "id", "example_id"], stable_id("techqa", source_name, i, question, prefix="ex")))
        answer = _first(record, ["answer", "answer_text", "accepted_answer", "response"], "")
        aliases = normalize_list(_first(record, ["answer_aliases", "answers", "answer_texts"], []))
        if answer and str(answer) not in aliases:
            aliases.append(str(answer))
        candidate_docs = _techqa_candidate_documents(record)
        gold_ids = [
            str(doc_id)
            for doc_id in normalize_list(_first(record, ["gold_document_ids", "gold_doc_ids", "document"], []))
            if str(doc_id).strip() not in {"", "-", "None", "none", "NULL", "null"}
        ]
        for j, doc in enumerate(candidate_docs):
            text = _document_text_from_record(doc).strip()
            if not text:
                continue
            doc_id = str(_first(doc, ["document_id", "doc_id", "id", "uri"], stable_id("techqa", example_id, j, text[:500], prefix="doc")))
            title = str(_first(doc, ["title", "document_title", "name"], "") or "")
            doc_rows[doc_id] = {
                "document_id": doc_id,
                "domain": "techqa",
                "title": title,
                "text": text,
                "source": source_name,
                "metadata": {"source_split": split, "candidate_for": example_id},
            }
            if not gold_ids and normalize_bool(_first(doc, ["is_gold", "gold", "relevant"], None), default=False):
                gold_ids.append(doc_id)
        qa_rows.append(
            {
                "example_id": f"techqa_{example_id}" if not example_id.startswith("techqa_") else example_id,
                "domain": "techqa",
                "question": question,
                "answer": "" if answer is None else str(answer),
                "answer_aliases": sorted(set(str(a) for a in aliases if str(a).strip())),
                "answerable": normalize_bool(_first(record, ["answerable", "is_answerable", "has_answer"], bool(answer or aliases)), default=bool(answer or aliases)),
                "gold_document_ids": [str(x) for x in gold_ids],
                "source_split": split,
                "metadata": {
                    "source_file": source_name,
                    "native_id": example_id,
                    "candidate_document_ids": normalize_list(_first(record, ["doc_ids"], [])),
                    "retrieval_setting": "candidate-set" if normalize_list(_first(record, ["doc_ids"], [])) else "broader-corpus",
                },
            }
        )
    return pd.DataFrame(qa_rows, columns=QA_COLUMNS), pd.DataFrame(doc_rows.values(), columns=DOCUMENT_COLUMNS)


# Normalize standalone TechQA corpus records into the document schema.
def normalize_techqa_documents(data: dict[str, Any] | list[dict[str, Any]], *, source_name: str, split: str = "unknown") -> pd.DataFrame:
    rows = []
    if isinstance(data, dict):
        iterable = data.items()
    else:
        iterable = [(None, record) for record in data]
    for key, record in iterable:
        if not isinstance(record, dict):
            continue
        doc_id = str(_first(record, ["id", "document_id", "doc_id"], key))
        text = _document_text_from_record(record).strip()
        if not doc_id or not text:
            continue
        rows.append(
            {
                "document_id": doc_id,
                "domain": "techqa",
                "title": str(_first(record, ["title", "document_title"], "") or ""),
                "text": text,
                "source": source_name,
                "metadata": {"source_split": split},
            }
        )
    return pd.DataFrame(rows, columns=DOCUMENT_COLUMNS)


# Infer a source split from a raw file name.
def infer_split(path: Path) -> str:
    name = path.name.lower()
    for split in ["train", "dev", "validation", "test"]:
        if split in name:
            return "dev" if split == "validation" else split
    return "unknown"


# Load and combine every supported Natural Questions source file.
def _load_natural_questions_from_dir(path: Path, *, max_records: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    files = discover_raw_files(path)
    messages: list[str] = []
    if not files:
        return pd.DataFrame(columns=QA_COLUMNS), pd.DataFrame(columns=DOCUMENT_COLUMNS), [f"No raw files found in {path}."]
    qa_parts = []
    doc_parts = []
    for file_path in files:
        if file_path.suffix.lower() not in {".jsonl", ".gz"} and not file_path.name.endswith(".jsonl.gz"):
            continue
        try:
            records = read_records(file_path, max_records=max_records)
            qa, docs = normalize_natural_questions(records, source_name=str(file_path), split=infer_split(file_path))
            qa_parts.append(qa)
            doc_parts.append(docs)
            messages.append(f"Loaded {len(qa)} QA rows and {len(docs)} documents from {file_path.name}.")
        except Exception as exc:  # Keep notebook diagnostics actionable across heterogeneous raw files.
            messages.append(f"Skipped {file_path}: {type(exc).__name__}: {exc}")
    qa_all = pd.concat(qa_parts, ignore_index=True) if qa_parts else pd.DataFrame(columns=QA_COLUMNS)
    docs_all = pd.concat(doc_parts, ignore_index=True) if doc_parts else pd.DataFrame(columns=DOCUMENT_COLUMNS)
    return qa_all, docs_all, messages


# Load and combine TechQA questions and standalone corpus files.
def _load_techqa_from_dir(path: Path, *, max_records: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    messages: list[str] = []
    qa_files = [
        path / "training_and_dev" / "training_Q_A.json",
        path / "training_and_dev" / "dev_Q_A.json",
    ]
    doc_files = [
        path / "training_and_dev" / "training_dev_technotes.json",
        path / "validation" / "validation_technotes.json",
    ]
    qa_parts = []
    doc_parts = []
    for file_path in qa_files:
        if not file_path.exists():
            messages.append(f"Missing expected TechQA QA file: {file_path}.")
            continue
        records = read_records(file_path, max_records=max_records)
        qa, embedded_docs = normalize_techqa(records, source_name=str(file_path), split=infer_split(file_path))
        qa_parts.append(qa)
        if not embedded_docs.empty:
            doc_parts.append(embedded_docs)
        messages.append(f"Loaded {len(qa)} TechQA QA rows from {file_path.name}.")
    for file_path in doc_files:
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        docs = normalize_techqa_documents(data, source_name=str(file_path), split=infer_split(file_path))
        doc_parts.append(docs)
        messages.append(f"Loaded {len(docs)} TechQA documents from {file_path.name}.")
    qa_all = pd.concat(qa_parts, ignore_index=True) if qa_parts else pd.DataFrame(columns=QA_COLUMNS)
    docs_all = pd.concat(doc_parts, ignore_index=True) if doc_parts else pd.DataFrame(columns=DOCUMENT_COLUMNS)
    docs_all = docs_all.drop_duplicates("document_id").reset_index(drop=True) if not docs_all.empty else docs_all
    return qa_all, docs_all, messages


# Dispatch domain-specific loading through a shared directory interface.
def load_domain_from_dir(domain: str, path: Path, *, max_records: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if domain == "natural_questions":
        return _load_natural_questions_from_dir(path, max_records=max_records)
    if domain == "techqa":
        return _load_techqa_from_dir(path, max_records=max_records)
    raise ValueError(f"Unsupported domain: {domain}")


# Select a reproducible shuffled subset keyed by a stable identifier.
def deterministic_subset(df: pd.DataFrame, n: int, *, seed: int, id_col: str) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.sort_values(id_col).reset_index(drop=True)
    sampled = df.sample(n=n, random_state=seed)
    return sampled.sort_values(id_col).reset_index(drop=True)


# Build a fixed corpus without query-specific negative-document selection.
def construct_corpus(docs: pd.DataFrame, qa: pd.DataFrame, *, max_docs: int, seed: int) -> pd.DataFrame:
    if docs.empty:
        return docs.copy()
    gold_ids = set()
    for ids in qa.get("gold_document_ids", []):
        gold_ids.update(str(x) for x in normalize_list(ids))
    gold_docs = docs[docs["document_id"].astype(str).isin(gold_ids)]
    remaining = docs[~docs["document_id"].astype(str).isin(gold_ids)]
    remaining_budget = max(max_docs - len(gold_docs), 0)
    distractors = deterministic_subset(remaining, remaining_budget, seed=seed, id_col="document_id")
    corpus = pd.concat([gold_docs, distractors], ignore_index=True)
    return corpus.drop_duplicates("document_id").sort_values("document_id").reset_index(drop=True)

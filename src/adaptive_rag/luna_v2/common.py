# Shared hashing, configuration, version, and artifact helpers for Luna v2.

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import pandas as pd
import yaml

CONDITIONS = ('direct', 'hybrid_rrf', 'hybrid_rrf_reranked')
ABSTAIN = 'INSUFFICIENT_EVIDENCE'


# Hash structured values with stable JSON serialization.
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode('utf-8')).hexdigest()


# Replace a JSON artifact atomically after writing a temporary file.
def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=str, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


# Load a JSON artifact from disk.
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# Capture the runtime and library versions used by an experiment stage.
def versions():
    result = {'python': platform.python_version()}
    for package in ('openai', 'pandas', 'numpy', 'scikit-learn', 'sentence-transformers', 'torch'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = 'not installed'
    return result


# Load Luna v2 settings and resolve all configured paths under the project root.
def load_config(root, path=None):
    root = Path(root).resolve()
    path = Path(path or os.environ.get('ADAPTIVE_RAG_LUNA_CONFIG',
                                      root / 'config/experiment_config_luna_v2.yaml'))
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['_root'] = root
    for key in ('processed', 'artifacts', 'nq', 'techqa'):
        cfg['paths'][key] = (root / cfg['paths'][key]).resolve()
    for key in ('processed', 'artifacts'):
        target = cfg['paths'][key]
        if root not in target.parents or 'luna_v2' not in target.parts:
            raise ValueError('Outputs must stay in a luna_v2 subdirectory of the workspace.')
        target.mkdir(parents=True, exist_ok=True)
    return cfg


# Persist a dataframe atomically in the Luna processed or artifact tree.
def save_table(cfg, name, frame, *, processed=False):
    folder = cfg['paths']['processed' if processed else 'artifacts']
    path = folder / (name + '.parquet')
    temporary = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return path


# Load a required Luna table with a preceding-notebook error message.
def load_table(cfg, name, *, processed=False):
    folder = cfg['paths']['processed' if processed else 'artifacts']
    path = folder / (name + '.parquet')
    if not path.exists():
        raise FileNotFoundError(f'Run the preceding notebook first: {path}')
    return pd.read_parquet(path)


# Compute a SHA-256 digest for a persisted artifact.
def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()

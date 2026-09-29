# Setup

## Flask with Docker

Run from this folder with Docker Desktop running and the private Luna v2 outputs available in the sibling project folder:

```powershell
docker compose up --build
```

Open http://localhost:5000. Check http://localhost:5000/health and enter a question or try an example. This runs Flask; the notebook server on port 8889 is separate. To stop the app, run `docker compose down`.

To use an existing private `config.json` for live paid answers, run `./scripts/run_with_local_key.ps1` instead. It looks in this folder and its parent, reads `OPENAI_API_KEY` or `API_KEY`, and passes only the key to Docker Compose. Use `-ConfigPath` if your configuration lives elsewhere. Keep `config.json` out of Git.

The build installs CPU PyTorch and pinned dependencies. Docker mounts the sibling project's `data/processed/luna_v2` and `artifacts/luna_v2` folders read-only by default. For a standalone clone, create a private `.env` from `.env.example` and set `ADAPTIVE_RAG_DATA_DIR` and `ADAPTIVE_RAG_ARTIFACT_DIR` to the absolute paths of your private copies. Both directories must exist before startup. A named volume retains downloaded model weights.

Optional local configuration:

```powershell
Copy-Item .env.example .env
```

Edit `.env` to set the two external data paths for a standalone clone. Set `OPENAI_API_KEY` only if you want paid answers for new questions, and keep `.env` private. Enable Advanced Settings → Replay saved experiment question to use an exact saved example without a key. Set `ADAPTIVE_RAG_LOCAL_FILES_ONLY=true` to require cached weights; the Docker default is `false`, permitting model downloads. Missing dense/reranker weights produce a labeled retrieval fallback. BM25 only works without weights. Initial live indexing can be slow; later queries reuse the in-memory index.

`/ask` accepts a JSON object with `question`, `domain`, `mode`, `top_k` (1–10), `generate` (boolean), and `use_saved` (boolean, default false). Supported strategies are `direct`, `hybrid_rrf_static`, `hybrid_rrf_reranked`, `adaptive`, and `bm25`. `/compare` compares the four primary strategies. New-question comparison with generation enabled can make up to three distinct paid calls; saved answers do not make new calls.

## Environment

```powershell
conda create -n adaptive-rag-luna python=3.11
conda activate adaptive-rag-luna
python -m pip install -r requirements.txt
```

Start Flask locally with `python run.py`. Local Python reads configuration from environment variables, so set `$env:OPENAI_API_KEY` and `$env:ADAPTIVE_RAG_LOCAL_FILES_ONLY` in PowerShell when needed; `.env` is consumed by Docker Compose.

Run commands from the repository root. The notebooks locate `src/adaptive_rag` and the Luna configuration relative to that root.

## Run Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

The test suite uses synthetic fixtures and checks notebook structure, frozen-cache loading, Batch resume behavior, paired evaluation, policy fitting, HTTP input validation, and Flask response behavior without private data or paid calls.

## Run the Notebooks

```powershell
python -m jupyter notebook
```

Open `experiment_notebooks/luna_v2/` and run notebooks 01 through 05 in order.

The notebooks expect local `data/processed/luna_v2` and `artifacts/luna_v2` folders. If you have private copies of the completed outputs, place them there before running the notebooks; Notebook 01 validates the processed-table checksums, Notebook 02 reuses completed rankings, and Notebook 03 reuses completed generation results. Otherwise, provide the original raw sources and rebuild in order. New API submission requires `OPENAI_API_KEY` and an explicit incomplete or fresh run directory.

## Fresh Data Rebuild

A complete rebuild of Notebook 01 requires the original Natural Questions and TechQA source data at the paths configured in `config/experiment_config_luna_v2.yaml`. Those raw corpora are intentionally absent from this repository. The published numerical results remain in the notebooks and [case analysis](case_analysis.md); a fresh clone needs private data and outputs before the interactive app can run.

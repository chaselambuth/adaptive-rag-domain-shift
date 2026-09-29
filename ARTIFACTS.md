# Private Data and Experiment Artifacts

This repository contains code, notebooks, figures, and the written results. It does not contain raw corpora, processed tables, completed generation records, or other generated experiment artifacts. The original project keeps its private Luna v2 outputs outside this GitHub-ready folder.

## Required Files

The notebooks write processed data to `data/processed/luna_v2/` and outputs to `artifacts/luna_v2/` inside a local working copy. Notebook 01 needs the original Natural Questions and TechQA source data at the paths in `config/experiment_config_luna_v2.yaml` unless a private copy of the checksum-verified processed tables is supplied. Running Notebook 03 from scratch makes paid OpenAI Batch calls.

The Flask app requires the complete frozen processed tables and these artifacts: `rankings.parquet`, `generation_results.parquet`, `final_metrics.parquet`, `frozen_policy.joblib`, `evaluated_generations.parquet`, and `generation/manifest.json`. Keep `dataset_manifest.json` alongside the processed tables. Startup verifies dataset checksums and the frozen policy input signature. Only use trusted policy bundles: Joblib loads serialized Python objects.

Docker Compose mounts these directories read-only and does not copy them into the image. By default it uses `../data/processed/luna_v2` and `../artifacts/luna_v2`, matching this copy's location inside the original project. For a standalone clone, set `ADAPTIVE_RAG_DATA_DIR` and `ADAPTIVE_RAG_ARTIFACT_DIR` in a private `.env` to absolute paths containing your copies. A clone without these files can run the synthetic test suite and read the published results, but the app cannot start or replay the experiment until the files are provided.

## Publication

`data/` and `artifacts/` are ignored by Git. `config.json`, `.env`, API keys, and provider credentials are also ignored. The largest processed table is approximately 117 MB, above GitHub's normal 100 MB per-file limit. The original raw corpora are approximately 30.6 GB. Do not force-add either to a normal commit.

If distributing a separate data bundle later, version it, include checksums and setup instructions, and confirm the redistribution terms for each dataset and derived corpus. No artifact download is configured automatically.

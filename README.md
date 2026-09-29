# Adaptive RAG Under Technical Domain Shift

This repository evaluates whether a retrieval-augmented generation system can choose the least expensive reliable action for each question: answer directly, retrieve evidence, retrieve and rerank, or abstain.

## Why These Experiments Matter

Direct generation is inexpensive but can fail when the model lacks reliable knowledge. Retrieval can improve grounded accuracy, while reranking adds more computation and latency. An adaptive system is useful only if it can choose among those routes more effectively than a fixed strategy.

Domain shift makes that decision harder. A policy trained on short, general-domain Natural Questions may not transfer to long technical-support questions with different vocabulary, answer length, and retrieval behavior. These experiments measure that transfer with frozen data splits and matched comparisons across direct generation, Hybrid RRF, and reranked Hybrid RRF.

## Main Result

The corrected Luna v2 experiment completed 3,813 of 3,831 planned generation calls across 369 Natural Questions and 908 TechQA examples. Hybrid RRF improved paired TechQA task success from **2.9% for direct generation to 14.3%**. Reranking reached similar answer quality at higher retrieval cost.

The learned policy routed every validation and held-out example to `RETRIEVE`, making it operationally identical to static Hybrid RRF. This is a useful negative result: good source-domain validation performance did not produce meaningful per-query adaptation after technical-domain shift.

The project remains research-only. TechQA accuracy is based on automatic lexical evaluation, the blind human answer audit is incomplete, and Batch generation does not provide synchronous end-to-end latency.

See [case_analysis.md](case_analysis.md) for the complete interpretation.

## Why GPT-5.6 Luna

The experiment uses `gpt-5.6-luna` as a budget-conscious generator for 3,831 planned calls, with three matched generation conditions per question. OpenAI describes Luna as intended for cost-sensitive, high-volume workloads; that fits an experiment measuring whether retrieval can improve an economical model's answers. [Official Luna model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

The frozen September 18, 2026 configuration records Batch rates of **$0.10 input / $0.60 output per million tokens**, a **$5 local spending guard**, and a conservative first-pass reservation of **$3.34**. Batch offers a 50% discount over synchronous processing and suits offline evaluation; the interactive Flask app uses synchronous generation, so these Batch estimates are not its live prices. [OpenAI Batch documentation](https://developers.openai.com/api/docs/guides/batch).

Every condition uses the same model, reasoning setting (`none`), and 256-token output cap, keeping generator settings constant while comparing retrieval strategies. This is a controlled, affordable baseline, not evidence that Luna is the best model for technical support. Results are specific to this model and setup; stronger-model comparisons and human evaluation remain necessary.

## Repository Contents

- `experiment_notebooks/luna_v2/` — the five corrected experiment notebooks.
- `src/adaptive_rag/luna_v2/` — Luna data, retrieval, generation, evaluation, policy, and reporting code.
- `src/adaptive_rag/` — shared schemas, loaders, retrieval, feature, and evaluation utilities.
- `config/experiment_config_luna_v2.yaml` — frozen experiment settings.
- `app/` — Flask explorer connected to corrected Luna data, saved answers, and the frozen policy.
- `Dockerfile` and `compose.yaml` — CPU container serving Flask on port 5000.
- `tests/` — experiment integrity and Flask behavior tests.
- `data/processed/luna_v2/` and `artifacts/luna_v2/` — expected output paths for local notebook runs; the data and outputs are excluded from this repository.
- [LUNA_V2_EXPERIMENT.md](LUNA_V2_EXPERIMENT.md) — execution order and Batch controls.
- [ARTIFACTS.md](ARTIFACTS.md) — external data, artifact, and publication guidance.

The superseded original notebooks are excluded. The Flask explorer uses the Luna artifacts rather than the original experiment outputs.

## Run the Adaptive RAG Explorer

From this repository folder, with Docker Desktop running and the private Luna v2 processed data and artifacts available in the sibling project folder:

```powershell
docker compose up --build
```

Open **http://localhost:5000**. This starts the Adaptive RAG Explorer Flask app. Enter a question or use an example button to run a live query. Enable Advanced Settings → Replay saved experiment question to test saved examples without an API key or model downloads. Health status is available at `/health`. Stop it with `docker compose down`.

For live answers using an existing private `config.json` in this folder or its parent project folder, run `./scripts/run_with_local_key.ps1` from PowerShell. The script reads `OPENAI_API_KEY` or `API_KEY`, starts the container with that key in its environment, and leaves the file outside the image and Git. You can pass `-ConfigPath` for a different location. Live API requests are paid; the UI submits them only when Generate answer is selected.

For new questions, the app retrieves from the local frozen corpus. Hybrid and reranked retrieval require cached model weights; unavailable weights produce a clearly labeled fallback. Docker permits model downloads by default; set ADAPTIVE_RAG_LOCAL_FILES_ONLY=true to require cached weights. BM25-only retrieval needs no model downloads. A local `.env` can override that behavior. The first live hybrid query may take several minutes to initialize indexes.

Saved replay is optional and uses historical Luna answers and retrieval timings. Live questions need `OPENAI_API_KEY` for optional paid generation; with no key the app still shows evidence. Strategy comparison can make up to three distinct paid calls for a new question. Successful live answers are cached in memory for the running process. Saved failed calls remain unavailable and are not retried. The UI's Top K controls displayed evidence; answer context remains five documents as in the experiment.

Historical Batch costs and unavailable Batch generation latency are labeled explicitly. Live requests are separate demonstrations, not additional evaluated experiment results. The frozen policy's stage scores are not calibrated answer-confidence estimates.

The app requires the processed data and artifacts described in [ARTIFACTS.md](ARTIFACTS.md). Docker mounts the sibling project's `data/processed/luna_v2` and `artifacts/luna_v2` folders read-only by default. For a standalone clone, set `ADAPTIVE_RAG_DATA_DIR` and `ADAPTIVE_RAG_ARTIFACT_DIR` in a private `.env` to the full paths of your local copies. The app cannot start without those files; they are not included in Git or the image.

## Local Python and Notebooks

Create an environment and install the tested dependencies:

```powershell
conda create -n adaptive-rag-luna python=3.11
conda activate adaptive-rag-luna
python -m pip install -r requirements.txt
```

Start the Flask app locally with `python run.py`, then open http://localhost:5000. Docker uses Gunicorn; the local command uses Flask’s development server with debug disabled.

Run the test suite:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Start Jupyter from the repository root and run the notebooks in order:

```powershell
python -m jupyter notebook
```

1. `01_data_and_domain_setup.ipynb`

2. `02_retrieval_systems.ipynb`

3. `03_rag_generation_and_signal_extraction.ipynb`

4. `04_hierarchical_adaptive_retrieval_policy.ipynb`

5. `05_final_domain_shift_evaluation.ipynb`

The GitHub repository does not include the approximately 30.6 GB raw corpora, processed tables, or completed artifacts. To rerun the notebooks, provide the original source datasets and execute notebooks 01–05 in order. If you already have a private copy of the frozen processed tables and outputs, place them at the configured local paths before running the notebooks; completed saved Batch results can then be inspected without resubmission.

## Data and Artifact Status

No raw or processed datasets and no generated experiment artifacts are included in this GitHub-ready folder. The original project's copies remain outside it. `.gitignore` protects local output paths from accidental commits. The largest processed table exceeds GitHub's normal per-file limit; consult [ARTIFACTS.md](ARTIFACTS.md) before distributing any dataset-derived files.

Do not add `config.json`, `.env`, API keys, provider credentials, or the original raw datasets to the repository.

## License

The project code, notebooks, and documentation are available under the [MIT License](LICENSE). Third-party datasets and pretrained models retain their original licenses.

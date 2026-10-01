# Official Experiment Run

Run these five notebooks in order with the project's Python environment:

1. [Data and domain setup](experiment_notebooks/luna_v2/01_data_and_domain_setup.ipynb)
2. [Retrieval systems](experiment_notebooks/luna_v2/02_retrieval_systems.ipynb)
3. [Generation and signals](experiment_notebooks/luna_v2/03_rag_generation_and_signal_extraction.ipynb)
4. [Adaptive policy](experiment_notebooks/luna_v2/04_hierarchical_adaptive_retrieval_policy.ipynb)
5. [Final domain-shift evaluation](experiment_notebooks/luna_v2/05_final_domain_shift_evaluation.ipynb)

Settings: [`config/experiment_config_luna_v2.yaml`](config/experiment_config_luna_v2.yaml).
Run from this project so the setup cell can find `src` and `config`. Dependencies are in the root
`requirements.txt`; the checked local environment uses OpenAI SDK 3.3.1. Private checksum-verified tables and completed rankings can be reused if supplied at the configured paths. This repository does not include them. A fresh data/retrieval rebuild requires the raw sources and cached weights; its initial scan and indexing may take a while.

## Paid generation

The model is `gpt-5.6-luna`, reasoning `none`, maximum 256 output tokens, using `/v1/responses` through Batch.
The notebook is saved with `ACTION = 'run'`. If a private completed ledger is present, it reuses the finished results without contacting OpenAI. Without that ledger, prepare and review a fresh experiment before running: set `ACTION = 'prepare'` and inspect the request count and conservative cost reservation. Authentication uses `OPENAI_API_KEY`, or `OPENAI_API_KEY`/`API_KEY`
in the existing local `config.json`. Never paste a key into a notebook or commit that file.

For a small compatibility check, set `ACTION = 'submit'`, keep `REQUEST_LIMIT = 30`, and run the action cell.
Later set `ACTION = 'collect'` to import the batch. Inspect errors and usage. Set `ACTION = 'run'` to finish
the remaining batches automatically; this may wait many hours. `run` uses the configured batch chunk limits,
not `REQUEST_LIMIT`. Stop/restart the kernel whenever needed; successful results are reused.
The `run` cell remains busy while OpenAI processes batches and shows a progress bar with saved responses,
live provider request counts, and active Batch status. It keeps up to four conservative chunks active concurrently;
with the configured 1M-per-chunk bound, this remains below Luna's documented Tier-1 5M queued-token limit.
Interrupting that waiting cell does not cancel the provider batch; restart the kernel and rerun with `ACTION = 'run'`
to resume collection and subsequent submissions from the saved ledger.

The local spending guard uses a conservative UTF-8 input bound, capped outputs,
and 25% headroom. Batch prices are configured as **$0.10 input / $0.60 output per million tokens**
(checked September 18, 2026). This is an estimate and local guard, not a provider-enforced billing limit.
Failed/uncertain attempts retain their reservation; retries consume an additional reservation. If the guard
stops the run, inspect the plan and ledger before explicitly changing `budget_usd` in YAML. Increasing that
budget does not invalidate successful cached outputs. Changing prompts/model/context/output limits does:
use new directories such as `data/processed/luna_v2/run_02` and `artifacts/luna_v2/run_02` for a different experiment.

Sources: [Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[Batch](https://developers.openai.com/api/docs/guides/batch),
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
Explicit cache mode without breakpoints avoids cache writes. Actual provider usage, resolved model,
response/request IDs, errors, incomplete details, and raw Batch records are saved locally.
Cost estimates assume no cache discount and include all output tokens, including reasoning usage if any.

## Resume and error handling

- Keep `RETRY_FAILED = False` to record failed rows as unavailable and continue the remaining requests;
  completed answers are never resubmitted. Inspect quota, authentication, and model-access failures because
  they may affect many later calls. If 256 tokens regularly truncate answers, use a separately identified run.
- A failed condition excludes that entire example from all three conditions in downstream paired analysis.
  The exclusion is displayed in notebooks 04 and 05 and is never converted into an incorrect-answer label.
- In-flight batches must be collected before another chunk is submitted. Out-of-order output is matched by
  request hash. Missing responses stay unavailable; notebooks 04–05 retain only examples with all three conditions.
- A network error during submission can leave `state: submitting` in `artifacts/luna_v2/generation/ledger.json`.
  Do **not** delete the ledger or resubmit blindly. Find the provider batch with the same `submission_nonce`
  and `input_file_id` in the OpenAI Batch dashboard or `client.batches.list()`, then attach its verified ID:

  ```python
  from adaptive_rag.luna_v2.generation import reconcile_submission
  reconcile_submission(cfg, nonce='nonce-from-ledger', batch_id='verified-batch-id')
  ```

  If the provider definitively rejected creation and no batch exists, preserve the ledger and error evidence;
  ask for help reconciling that rejection before retrying. A submission lock blocks simultaneous kernels.
  After a hard kernel/process crash, remove `submission.lock` only once that process is confirmed stopped;
  the ledger still protects an uncertain paid submission.

## Experimental Method

- Derive NQ gold labels from annotations and join multi-span references. Keep long-only and negative-page
  examples visible in the audit, but exclude them from primary short-answer generation metrics.
- Freeze NQ train/validation/test groups before fitting text features or observing generation outcomes.
  Keep duplicate questions and shared gold pages together. Exclude exact source/target question duplicates.
- Fit TF-IDF vocabulary, source centroid, and nearest-neighbour reference vectors on NQ training only.
  TechQA is never used for model/threshold fitting. Decision features never include future reranker scores.
- Compute all retrieval methods fully. Include hybrid time in reranking and in late ABSTAIN decisions.
- Compare all three generation conditions on identical complete cohorts. Separate availability, answer
  correctness, task success, answer coverage, and abstention. An answerable all-fail oracle is not a claim
  that the question is unanswerable; abstaining does not earn correct-answer credit.
- Use conservative rare-class fallbacks and explicit validation-only utility/cost threshold selection.
  Export paired uncertainty, feature shift, failures, and blind human-review sheets.
- Batch generation latency stays unavailable. Validate synchronous latency and human-rated answer quality
  before making deployment claims.

Outputs live in `data/processed/luna_v2` and `artifacts/luna_v2`, including figures. The source helpers are
in `src/adaptive_rag/luna_v2`. These outputs are excluded from Git. A private copy of the completed outputs
can be reused without the raw corpora or new API calls.

## Flask Explorer

The Flask app serves outputs from the official run at http://localhost:5000 when the private processed data and artifacts are available. Run `docker compose up --build` from the repository root; see [SETUP.md](SETUP.md). Saved answer replay is separate from optional synchronous API generation for new questions. App interactions do not add evaluated rows to the frozen experiment.

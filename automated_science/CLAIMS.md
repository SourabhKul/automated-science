# Claims Audit

This file tracks what the current repository actually supports. It is intentionally stricter than the manuscript: a claim is marked **supported** only when there is code plus reproducible local evidence in this checkout.

## Status Key

- **Supported:** implemented and backed by files/artifacts in this checkout.
- **Partially supported:** there is some implementation or artifact evidence, but the claim needs narrower wording or more verification.
- **Unsupported:** manuscript or roadmap claim without corresponding implemented evidence in this checkout.

## Claim Matrix

| Claim | Status | Evidence | Action |
| --- | --- | --- | --- |
| LLM-proposed JAX dynamics are evaluated by ABC-SMC. | Supported | `scripts/run_domain.py` and `core/sandbox_eval.py` scaffold generated `dynamics`/`metadata`; `core/sbi_engine.py` runs ABC-SMC. | Keep as primary contribution. |
| The repo contains preserved 25-domain, 10-run Qwen result artifacts. | Supported | `artifacts/runs/qwen36_27b/` contains the primary run tree; `scheduler_logs` symlink preserves existing script paths. | Keep, but document artifact location. |
| Results table is generated from run-history CSV files. | Supported | `compile_paper_results.py` reads `scheduler_logs/run_*/qwen36_27b_*_run_history.csv` and writes `paper/results_table.tex`. | Keep. |
| Current local environment is reproducibly captured. | Partially supported | `requirements-freeze.txt` captures current `.venv`; no container or cross-platform lockfile yet. | Add setup docs and eventually a pinned environment file. |
| Runs are independent replications. | Unsupported as written | `master_scheduler.py` feeds previous-run `best_logic.py` into later runs as meta-hints. | Reword as sequential/multi-run curriculum, or rerun with independent seeds and no cross-run hints. |
| Random seeds are strictly standardized. | Partially supported for future runs | `SBIEngine` accepts seeds and all top-level orchestrators accept optional `E3_SEED` plus write run configs; historical artifacts do not include seeds. | Do not claim for old artifacts; use seeded reruns going forward. |
| All continuous ODE solves used explicit `rtol=1e-5` and `atol=1e-5`. | Unsupported | Orchestrators call `diffeqsolve(..., Tsit5(), ..., max_steps=10000)` without `rtol`/`atol`. | Reword or patch new runner templates. |
| `E^3` currently implements PPP/MMD stochastic scoring. | Unsupported | `core/sbi_engine.py` has Euclidean/summary-stat distances only; no MMD implementation or stochastic ensemble rollouts. | Move PPP/MMD to future work until implemented. |
| `E^3` currently implements SDE discovery/inference. | Unsupported | No integrated `VirtualBrownianTree` or SDE scoring path in the active engine/orchestrators. | Move SDE support to future work until implemented. |
| GRPO alignment was trained and evaluated. | Unsupported | No training scripts, datasets, model checkpoints, or result artifacts for GRPO are present. | Reframe as proposed future work or add artifacts. |
| The paper's baseline comparison is fully apples-to-apples. | Partially supported | `run_baselines.py` now saves integrated trajectory metrics for PySINDy/PySR, and new `scripts/run_domain.py --held-out` runs select by held-out test MSE. Historical $E^3$ artifacts were not generated under that protocol. | Rerun selected $E^3$ domains with matched train/test splits and compute budgets before making final comparison claims. |
| The generated equations are mechanistic discoveries, not just curve fits. | Partially supported | Some best models are compact and interpretable; others are time-forced or show minimal improvement. | Add train/test/forecast and parsimony reporting. |
| Generated-code evaluation is fully sandboxed. | Partially supported | Centralized domain runs evaluate candidates in a subprocess via `core/sandbox_eval.py` after AST validation, but this is not an OS security sandbox and legacy special runners still execute locally. | Treat distributed execution as future work; add real resource/container sandboxing before deployment. |

## Immediate Paper Edits Required

1. Replace "10 independent runs" with "10 sequential runs" or "10-run curriculum" unless rerun independently.
2. Replace PPP/SDE/MMD/GRPO demonstrated claims with roadmap/future-work language.
3. Replace Docker/Python/JAX version claims with the actual environment snapshot or add the missing container/lockfile.
4. Mark baseline comparison as preliminary until selected $E^3$ domains are rerun with the same held-out trajectory protocol used by `run_baselines.py`.
5. Add caveats around domains with negligible mean improvement.

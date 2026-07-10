# Automated Science

Autonomous white-box scientific model discovery.

The canonical architecture is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md). In one line:

```text
data -> model -> parameter fit -> diagnostics -> structural hypothesis -> validation -> model replacement -> memory -> meta-learning
```

The core idea is to hold observed scientific data fixed, fit a candidate model's parameters, diagnose how the model succeeds and fails, then let LLM agents propose in-place white-box model replacements. Parameter inference strategies such as Gaussian ABC-SMC, BDSS, adaptive kernels, or future methods are swappable inner-loop components; they are not the whole system.

The May-June qwen36 hero run is summarized in [`artifacts/evaluations/hero_run_deep_dive_20260703.md`](artifacts/evaluations/hero_run_deep_dive_20260703.md). It is the strongest proof-of-life artifact, but it used in-sample ABC distance rather than the newer held-out protocol.

## Current Layout

- `ARCHITECTURE.md`: source of truth for the canonical discovery loop and improvement rules.
- `core/`: model interfaces, parameter inference, candidate evaluation, validation, and metrics.
- `scripts/run_domain.py`: canonical single-domain model-replacement loop.
- `scripts/run_gauntlet.py`: one model across many domains.
- `scripts/run_model_matrix.py`: many LM Studio models across many domains.
- `scripts/repo_inventory.py`: visibility report for canonical files, legacy entrypoints, generated outputs, and hero-run traces.
- `core/domain_configs.py`: active 25-domain registry with seed models, solver settings, and prompt hints.
- `run_baselines.py`: PySINDy/PySR baseline refresh and trajectory-level metrics.
- `qwen36_*_orchestrator.py`, `gemma4_*_orchestrator.py`: backwards-compatible stubs, not canonical architecture surfaces.
- `master_scheduler.py`, `setup_run_circuit.py`, and special old orchestrators: legacy/provenance paths.
- `artifacts/`: preserved generated outputs, logs, archives, model snapshots, reports, and legacy traces.
- `scheduler_logs`: compatibility symlink to `artifacts/runs/qwen36_27b`.

## Environment

The current local virtualenv has been moved outside the project to keep the repo small:

```bash
.venv -> ../.venvs/automated_science_py314
```

`requirements-freeze.txt` records the currently installed package set. The historical paper text mentions a different Python/JAX stack, so treat this file as the actual local environment snapshot for this checkout.

## Common Commands

Read the canonical architecture:

```bash
make architecture
```

Show the current canonical/legacy inventory:

```bash
make inventory
```

Regenerate synthetic datasets:

```bash
./bootstrap_data.sh
```

Regenerate public real-data arrays:

```bash
.venv/bin/python build_real_data.py
```

Run the lightweight engine stress test:

```bash
.venv/bin/python test_stress.py
```

Or use the Makefile wrapper:

```bash
make test
```

Rebuild the manuscript results table from preserved run histories:

```bash
.venv/bin/python compile_paper_results.py
```

Audit preserved run histories and flag weak domains:

```bash
.venv/bin/python scripts/audit_results.py
```

Run a small integrated-baseline smoke test:

```bash
make baseline-smoke
```

Run the full preliminary PySINDy/PySR baseline script:

```bash
make baselines
```

Run one domain orchestrator:

```bash
make run-domain DOMAIN=ecology FAMILY=qwen36
```

Use held-out trajectory selection for new scientific runs. This is the preferred mode for new model-replacement experiments:

```bash
.venv/bin/python scripts/run_domain.py ecology --family qwen36 --held-out --seed 123
```

Inspect the resolved model family/folder without launching an LLM run:

```bash
make run-domain-dry DOMAIN=ecology FAMILY=qwen3_coder_next
```

Run tiny end-to-end LM Studio smoke tests:

```bash
make smoke-qwen-coder
make smoke-qwen36
```

Hardened runners accept `E3_SEED`, validate generated `dynamics`/`metadata` blocks before scaffolding, and write run provenance configs for new runs.

Model-family segregation:

- `qwen36`: `qwen3.6-27b-nvfp4`, output under `models/qwen36_27b_{domain}/`.
- `qwen3_coder_next`: `qwen/qwen3-coder-next`, output under `models/qwen3_coder_next_{domain}/`.
- `gemma4`: `gemma-4-26b-it`, output under `models/gemma4_26b_{domain}/`.

Reusable train/test metric helpers live in `core/evaluation.py`; baseline trajectory helpers live in `run_baselines.py`. New `scripts/run_domain.py --held-out` experiments fit parameters on the first 80% of each trajectory and accept/reject candidate hypotheses by held-out test MSE instead of in-sample ABC-SMC distance.

## Canonical Improvement Targets

Every new change should name the loop it improves:

- parameter inference,
- model replacement,
- evaluation and limitation reporting,
- meta/self-improvement,
- domain/model-class adaptation.

BDSS belongs in the parameter-inference loop as a first-class strategy to restore and compare. The next major system work should make fit diagnostics, model limitations, and held-out performance available to the model-generating agent before each replacement proposal.

## Cleanup Notes

The repository was spring-cleaned to separate source from generated research artifacts. Nothing material was intentionally deleted except disposable caches (`__pycache__`, `.DS_Store`, `*.pyc`). Large generated outputs were moved under `artifacts/`, and the primary Qwen run logs remain available through the `scheduler_logs` symlink.

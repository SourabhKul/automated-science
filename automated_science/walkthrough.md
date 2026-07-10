# Automated Science Walkthrough

This is a practical routing note. The canonical architecture is `ARCHITECTURE.md`.

Canonical loop:

```text
data -> model -> parameter fit -> diagnostics -> structural hypothesis -> validation -> model replacement -> memory -> meta-learning
```

Use this file for commands and current implementation surfaces, not for architectural decisions.

## Model Families

- `qwen36`: legacy Qwen 3.6 family using `qwen3.6-27b-nvfp4`; outputs go to `models/qwen36_27b_{domain}/`.
- `qwen3_coder_next`: new family using `qwen/qwen3-coder-next`; outputs go to `models/qwen3_coder_next_{domain}/`.
- `gemma4`: Gemma family using `gemma-4-26b-it`; outputs go to `models/gemma4_26b_{domain}/`.

Check routing without launching a model:

```bash
make run-domain-dry DOMAIN=ecology FAMILY=qwen36
make run-domain-dry DOMAIN=ecology FAMILY=qwen3_coder_next
```

## Central Runner

The 25 active scientific domains are registered in `core/domain_configs.py`. The registry stores each domain's initial conditions, seed logic, solver settings, state description, and prompt hints.

`scripts/run_domain.py` is the main config-driven runner. It:

- resolves model family and output folder,
- writes `run_config.json`,
- calls the LM Studio/OpenAI-compatible endpoint,
- validates generated code with `core/generated_code.py`,
- evaluates candidates through `core/sandbox_eval.py`,
- optionally uses held-out train/test evaluation with `--held-out`.

With `--held-out`, the runner fits ABC-SMC parameters on the first 80% of the trajectory, computes full-trajectory train/test metrics, and accepts new hypotheses by held-out test MSE. The ABC-SMC median distance is still logged as a fitting diagnostic.

Most `qwen36_*_orchestrator.py` and `gemma4_*_orchestrator.py` files are now thin compatibility stubs into this runner.

## Visibility

```bash
make architecture
make inventory
```

`make inventory` reports canonical files, legacy entrypoints, generated model directories, and the preserved qwen36 hero-run trace summary.

## Candidate Evaluation

`core/sandbox_eval.py` runs candidate evaluation in a separate Python process. It is a robustness boundary, not a complete security sandbox. Generated code is AST-validated first; runtime-broken dynamics now fail evaluation instead of being converted into silent all-zero trajectories.

## Baselines

`run_baselines.py` runs preliminary PySINDy/PySR comparisons. PySR is still fit to numerical derivatives, but the learned RHS is now integrated with `solve_ivp` so saved metrics are trajectory-level train/test errors. Derivative errors are retained only as diagnostics.

## Useful Commands

```bash
make test
make compile-results
make audit-results
make baseline-smoke
make smoke-qwen-coder
make smoke-qwen36
```

## Remaining Gaps

- Historical Qwen artifacts were not produced with held-out evaluation or persisted seeds.
- The subprocess evaluator is not a full OS/container sandbox.
- Nine special legacy runners still contain independent loop logic.
- Several domains remain weak in the preserved Qwen run audit: oncology, econ, pkpd, cardio, epidemiology, neuro, astro, pop genetics, and agriculture.
- BDSS/beta-distributed step-size parameter search is not active in the current `core/sbi_engine.py`; it should be restored as a swappable inference strategy after the end-to-end system is stable.
- The qwen36 hero-run scheduler invoked the meta-agent, but completed-run `meta_hints.txt` files are empty. Meta-steering needs a structured, testable rebuild.

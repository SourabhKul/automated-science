# Code Correctness Audit

Scope: current source tree after artifact cleanup. This audit prioritizes correctness, reproducibility, and scientific validity over style.

## High Priority Findings

1. **Generated code is executed with raw `exec()`.**
   - Evidence: generated modules are still dynamically executed after AST validation. The centralized runner evaluates candidates through `core/sandbox_eval.py` in a subprocess, but legacy special runners still execute locally.
   - Risk: arbitrary code execution, accidental file/network access, and unbounded side effects.
   - Fix path: `core/generated_code.py` rejects obvious non-math side effects, and `core/sandbox_eval.py` now fails runtime-broken generated math instead of silently returning zeros. This is still not a true OS security sandbox; generated code should eventually execute in a constrained subprocess/container with resource limits.

2. **Historical runs are not reproducibly seeded or configured.**
   - Evidence: preserved historical run artifacts do not include persisted seeds/configs.
   - Risk: future reruns cannot be tied to an exact random state or hyperparameter set.
   - Fix path: seed support now exists in `SBIEngine`, and `core/run_config.py` writes provenance records. New centralized runs accept optional `E3_SEED`/`--seed` and write a run config.

3. **Manuscript claims exceed implementation.**
   - Evidence: `CLAIMS.md` still tracks PPP/MMD/SDE/GRPO as unsupported roadmap items; the active engine computes Euclidean raw-trajectory or summary-feature distances only.
   - Risk: future edits could accidentally turn roadmap items back into demonstrated contributions.
   - Fix path: keep `CLAIMS.md` updated and require source/artifact evidence before promoting roadmap items to claims.

4. **Baseline script now supports integrated trajectory diagnostics, but historical $E^3$ runs are not yet matched to that protocol.**
   - Evidence: `run_baselines.py` rolls PySINDy and PySR-discovered dynamics forward to saved train/test trajectory metrics, and keeps PySR derivative errors as diagnostics. New `scripts/run_domain.py --held-out` runs fit on the first 80% of the trajectory and select accepted hypotheses by held-out test MSE.
   - Risk: comparing those held-out trajectory metrics to historical in-sample ABC-SMC distances is still not apples-to-apples.
   - Fix path: rerun selected $E^3$ domains with `--held-out`, fixed seeds, saved prompts, and compute-normalized budgets.

5. **Domain orchestrators are heavily duplicated.**
   - Evidence: 50 qwen36/gemma4 domain orchestrators are now thin stubs into the config-driven runner, but 9 special/legacy runners still contain independent loop logic.
   - Risk: fixes can still drift in the legacy special runners.
   - Fix path: either migrate the remaining special runners into `core/domain_configs.py` or explicitly mark them archival.

## Medium Priority Findings

1. **Generated data loaders mostly omit random seeds.**
   - Evidence: most `*_data_loader.py` files call `np.random.normal` without setting or accepting a seed.
   - Fix path: add optional seed arguments and write dataset provenance.

2. **Distance metric is labeled MSE in logs but is often Euclidean summary distance.**
   - Evidence: `compute_distance()` returns sqrt Euclidean distance over summary stats by default.
   - Fix path: rename log labels to `distance` or add explicit metric metadata.

3. **`real_*_data_loader.py` files can regenerate synthetic stand-ins for real-data domains.**
   - Evidence: `build_real_data.py` creates public real datasets; `real_nile_data_loader.py` creates synthetic mean reversion.
   - Fix path: rename synthetic loaders or prevent fallback from silently replacing real data.

4. **The ABC-SMC implementation still has limited numerical diagnostics.**
   - Evidence: sampler arguments and priors are now validated, but non-finite particle weights and repeated simulation failures are not persisted as structured diagnostics.
   - Fix path: return structured failure reasons and weight diagnostics from `run_abc_smc`.

## Current Green Checks

- `python -m py_compile core/*.py *.py`
- `python test_generated_code.py`
- `python test_sbi_engine.py`
- `python test_sandbox_eval.py`
- `python test_run_domain_config.py`
- `python test_baselines.py`
- `python test_stress.py`
- `python compile_paper_results.py`
- `python scripts/audit_results.py`
- `python run_baselines.py --domains ecology --pysr-iterations 1 --output artifacts/baselines/baseline_smoke.json`
- `python scripts/run_domain.py ecology --family qwen36 --dry-run`
- `python scripts/run_domain.py ecology --family qwen3_coder_next --dry-run`
- `python scripts/smoke_qwen_lmstudio.py` against LM Studio `qwen/qwen3-coder-next` or `qwen3.6-27b-nvfp4`

## Migration Order

1. Patch paper claims and create claim-audit map.
2. Add seed/config support in core for future runs.
3. Add generated-code validator and use it in every top-level orchestrator.
4. Rerun selected $E^3$ domains with the same held-out protocol as the saved baselines.
5. Add train/test/forecast evaluation utilities to selected $E^3$ reruns.
6. Finish migrating or archiving the remaining special legacy runners.
7. Re-run selected domains with new config, safety, and evaluation metadata.

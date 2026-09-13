# First-principles code audit

Date: 2026-09-07  
Revision: `6ff92f2` (`Clean up repository and improve documentation`)  
Checkout: `/Users/sourabh/Documents/Codex/automated_science`

This is a bounded source review. It does not launch a scientific experiment,
load or unload a model, commit, or push. The local oMLX endpoint was checked
separately and is reachable; see `reports/2026-09-07/runtime-audit.md` for the
endpoint and targeted-runtime evidence.

## Assessment

The maintained core is an LLM-guided white-box ODE mutation loop:

```text
domain seed + observed trajectory
  -> LLM Python/JAX proposal
  -> AST filter + subprocess evaluation
  -> ABC-SMC-style parameter search
  -> incumbent replacement
  -> diagnostic context for the next proposal
```

This is a useful engineering proof of life for automated model refinement. It
is not yet a trustworthy automated-science benchmark because the outer loop
optimizes the reported held-out test segment, then sends test-derived residuals
back to the proposal model. The inner loop is also not Bayesian-correct ABC-SMC
as currently implemented: generation 0 uses uniform priors, whose constant
density would cancel after weight normalization; later proposals are clipped
into the prior box, which creates boundary point mass rather than a merely
truncated continuous kernel, and no corresponding boundary-mass density is
used in the weights. The BDSS transition is weighted with a Gaussian density
calculation even though its proposal uses beta-distributed directional steps.
Top-k selection by itself is not proof of invalidity; the concern is its
combination with an unrecorded quantile/epsilon schedule and proposal weights
that do not match the implemented transition.

The repository currently has two loosely coupled projects. The canonical
discovery engine consists of 27 domain configurations plus the ODE, Diffrax,
JAX, LLM, and ABC-SMC code. The second project contains 38 real-data adapters
and 124 `run_*.py` scripts for source gates, synthetic controls, and classical
classification/regression baselines. Most of those application scripts report
`abc_smc_calls: 0` and `llm_calls: 0`; they establish data and isolation
contracts but do not exercise automated scientific model discovery. They should
be treated as an application-evaluation surface until a common adapter contract
connects them to the discovery loop.

As of 2026-09-12, the research runtime is fixed to
`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` on local oMLX at
`http://127.0.0.1:8000/v1`. No LLM-matrix or cross-model comparison
development is planned. Same-Qwen ablations of prompts, inference strategies,
priors, proposal kernels, data handling, diagnostics, seeds, and budgets remain
within scope while the model and endpoint stay fixed; near-term work is data
exploration and ABC-SMC-style scientific discovery.

## What the code actually does

The domain registry supplies an initial hand-written dynamics function,
parameter ranges, initial state, solver settings, and prose hints
(`core/domain_configs.py:4-22`, `:599-669`). `scripts/run_domain.py` calls an
OpenAI-compatible chat endpoint (`:91-107`), extracts one `dynamics(t, y,
args)` function plus `metadata`, and sends each distinct proposal through
`core/sandbox_eval.py`.

The evaluator uses the first 80% of a trajectory to construct `SBIEngine`
when `--held-out` is enabled (`core/sandbox_eval.py:649-657`). The posterior
median is then rolled over the full trajectory and scored on train and suffix
segments (`:729-761`). For ordinary domains, the default distance is Euclidean
over four summary features per state: mean, standard deviation, lag-1
autocorrelation, and maximum non-DC FFT amplitude
(`core/sbi_engine.py:167-205`). Masked observations bypass summaries and use a
raw trajectory distance (`:193-200`).

At the outer-loop level, a candidate is accepted when its score is lower than
the incumbent (`scripts/run_domain.py:845-916`). In held-out mode that score is
explicitly the test MSE (`:421-430`). The current diagnostic packet contains
posterior summaries, residuals, and warnings (`core/sandbox_eval.py:165-290`).
The next prompt includes that packet (`scripts/run_domain.py:657-675`), so the
loop is adaptive but its current information boundary is invalid for final
generalization claims.

## Findings

### P0: the held-out suffix is used as an optimization and prompt signal

`selection_score` returns `eval_data["test_metrics"]["mse"]` in held-out mode
(`scripts/run_domain.py:421-425`). Every generated candidate and repair is
therefore ranked against the same suffix, and the accepted candidate is the
one that wins repeated searches on that suffix (`:845-916`). This is a
validation set in the statistical sense, despite being labelled test.

The leak is stronger than score selection. After fitting on the prefix, the
evaluator simulates the full trajectory and builds diagnostics using the full
observed array (`core/sandbox_eval.py:729-761`). The next proposal prompt
includes `residual_summary`, warnings, and failure modes
(`scripts/run_domain.py:77-88`, `:657-675`). Test residuals therefore guide the
next LLM mutation. `held_out_metrics.json` also records each accepted test
score (`scripts/run_domain.py:627-637`, `:899-906`). A final external test or
native repeat using that same suffix is not an untouched final test; a genuinely
separate external set would remain independent if it is never consulted.

Required protocol: split data into train, proposal-selection validation, and a
sealed test/external repeat. Fit parameters on train only; expose train and
validation diagnostics to the LLM; freeze the search; evaluate the selected
model once on test. For grouped applications, make the same distinction at the
subject/cell/file level. Current tests explicitly codify test-MSE selection
(`tests/test_run_domain_config.py:47-52`) and do not test that test data is
absent from prompt context.

### P1: ABC-SMC weights are not the stated Bayesian weights

The engine selects the top `target_samples` particles each generation
(`core/sbi_engine.py:270-310`), which is an empirical quantile/epsilon choice;
top-k selection alone does not establish invalidity, but the epsilon schedule
and acceptance trace are not recorded. For generations greater than zero, the
weight is computed as `1 / sum(previous_weight * Gaussian_kernel)`
(`:285-303`). Because each declared prior is uniform, its density is constant
on the support and would cancel when the new weights are normalized; the
omitted prior numerator is therefore not the defect. The transition strategies
clip proposals independently into each prior range (`:53-56`, `:105-114`).
That operation creates point mass at the bounds, so treating the proposal as a
smooth uncorrected density does not represent the implemented kernel. The
resulting weights do not correspond to the clipped proposal distribution.

The `bdss` strategy proposes beta-distributed, normalized directional steps
(`core/sbi_engine.py:89-114`), but `run_abc_smc` still evaluates every strategy
with `jax.scipy.stats.multivariate_normal.pdf` (`:297-303`) using the returned
`kernel_cov`. Its reported posterior is therefore inconsistent with the BDSS
transition kernel. Accepted
weights are not returned in the result, so downstream diagnostics use an
unweighted median and percentiles (`core/sandbox_eval.py:191-230`). Effective
sample size, kernel density diagnostics, weight degeneracy, and posterior
predictive uncertainty are unavailable.

Required protocol: either implement a mathematically explicit ABC-SMC variant
with prior support, normalized transition densities, boundary handling, ESS,
and a recorded epsilon schedule, or describe the current method as a
population-based quantile search rather than Bayesian inference. Return all
weights and posterior-predictive draws in the artifact schema.

### P1: generated-code execution is a filter plus a subprocess, not a secure sandbox

The validator correctly rejects obvious imports and names, but documents itself
as a lightweight guard (`core/generated_code.py:212-220`). `ALLOWED_CALL_ROOTS`
is declared but never enforced (`:21-22`). The normal candidate path builds a
module with imported JAX, Diffrax, and base-model objects and executes it with
raw `exec` (`core/sandbox_eval.py:293-327`, `:691-697`). The subprocess has a
wall-clock timeout in the parent runner (`scripts/run_domain.py:340-401`) but
no memory, CPU, filesystem, or network isolation. The AST filter is therefore
not a security boundary for public or distributed candidate code.

Required protocol: compile a restricted expression/function representation, or
run generated candidates in a disposable process/container with no network,
read-only inputs, bounded CPU/memory, and a separate output directory. Add
tests for resource exhaustion, attribute escape paths, and attempts to access
the imported module namespace.

### P1: the Warfarin adapter configuration is disconnected from dispatch

`real_warfarin_pkpd` declares `evaluation_mode="dense_masked"` and points to
normalized CSV and subject-holdout files (`core/domain_configs.py:599-611`).
The evaluator dispatches only `subject_holdout` and `cell_holdout`
(`core/sandbox_eval.py:626-639`); every other mode falls through to
`data/{domain}_ground_truth.npy` and `data/{domain}_time_points.npy`
(`:641-657`). Those arrays are absent in this clean checkout, while the
configured normalized CSV and holdout JSON are present. Thus the intended
Warfarin adapter cannot be reached through the canonical runner without a mode
or dispatch correction. The current Warfarin helper is a separate bounded
prior-rejection smoke path (`core/sandbox_eval.py:342-445`), not an integrated
LLM-discovered grouped ABC-SMC evaluation.

### P1: provenance is insufficient for exact reruns

`RunConfig` records experiment tag, domain, model ID, endpoint, one seed, a few
ABC-SMC sizes, Python version, and NumPy version
(`core/run_config.py:13-40`). It does not record the git commit, package
versions for JAX/Diffrax/SciPy, solver tolerances, data file hashes, split
hashes, prompt text, LLM request/response, response fingerprint, chat-template
settings, candidate code hash, or per-candidate failure category. The LLM
payload sets temperature but no generation seed and silently converts all
request errors to `None` (`scripts/run_domain.py:91-107`). Every candidate in a
run receives the same inference seed (`:733-743`); there are no independent
replicates or confidence intervals around the acceptance decision.

The local oMLX endpoint is reachable, but the canonical defaults still target
LM Studio at `localhost:1234` (`scripts/run_domain.py:21`,
`scripts/lmstudio_models.py:10`). The matrix runner also calls LM Studio native
`/api/v1/models/load` and `/api/v1/models/unload`; those endpoints are not part
of the observed oMLX surface. The runtime audit found that oMLX generation
works more reliably when `chat_template_kwargs.enable_thinking` is set to
`false`, but the current request builders do not expose that field.

### P1: claimed hero evidence is absent from this checkout

`artifacts/README.md:5-14` describes preserved Qwen histories, evaluations,
logs, and archives, while `.gitignore:43-45` excludes them. In this checkout,
`artifacts/` contains only its README; there are no `models/` directories and
no `scheduler_logs` tree. `scripts/repo_inventory.py` consequently reports
zero hero history files, zero rows, zero accepted updates, and zero artifact
directories. `CLAIMS.md:16` and `ARCHITECTURE.md:161-177` currently present
those historical results as locally available evidence, but the claim cannot
be independently checked from this working tree. `PROJECT_OVERVIEW...md:356`
correctly says there are 27 active configs. The 25-domain figure refers to the
historical hero run, so the count difference is a scope distinction rather than
itself a contradiction; paper and claim text should label the historical and
current counts explicitly.

Required protocol: publish a versioned, checksum-addressed artifact bundle or
change the claim status to unavailable in this checkout. Keep the 25-domain
historical result separate from a new 27-domain registry count.

### P2: the summary distance is scale-dependent and loses scientific structure

`extract_summary_statistics` concatenates raw means, standard deviations,
autocorrelation, and FFT amplitudes without scale normalization
(`core/sbi_engine.py:167-191`). Euclidean distance therefore lets high-unit or
longer trajectories dominate low-unit states; FFT amplitude also changes with
sample count. The code has no measurement-noise model, heteroscedasticity,
state weighting, irregular-time handling, observation model, initial-condition
uncertainty, or input forcing in the canonical ODE path. Very short trajectories
can produce an empty non-DC FFT slice (`:186-188`) rather than a structured
failure.

The evaluator checks finite outputs and declared parameter bounds, but does not
enforce state positivity, conservation laws, units, dimensional consistency,
monotonicity, or asymptotic behavior. The prompt asks for derivative clipping
(`scripts/run_domain.py:683-690`) but the validator does not require it. A
candidate may therefore win by exploiting time-dependent curve fitting while
remaining syntactically valid.

Required protocol: make an observation model and normalized distance explicit
per domain; add mechanistic invariants and state/parameter constraints; score
forecast and perturbation behavior; and penalize complexity or report the full
Pareto frontier rather than selecting solely by fit.

### P2: model selection lacks parsimony, robustness, and calibrated uncertainty

The outer-loop replacement condition is a strict scalar improvement
(`scripts/run_domain.py:879-916`). There is no minimum effect size, replicate
seed test, complexity penalty, baseline comparison, or uncertainty-aware
decision. `build_diagnostic_packet` reports medians and unweighted percentiles,
but no identifiability rank, posterior correlation, profile likelihood,
posterior predictive coverage, or sensitivity analysis. This makes a compact
equation that is scientifically stable indistinguishable from a flexible
interpolating curve at acceptance time.

### P2: application scripts are not yet a discovery framework

The repository has 38 `core/real_data` modules, 124 `run_*.py` scripts, and 124
tests, but no common `ApplicationAdapter`/`ExperimentPlan` interface. The
source gates, synthetic controls, and classical baselines are valuable
provenance work, but representative result records explicitly declare
`abc_smc_calls: 0` and `llm_calls: 0` (for example,
`scripts/run_uci_isolet_real_baselines.py:159` and
`scripts/run_uci_wisdm_real_baselines.py:263-264`). The macro-F1 table in
`docs/RESULTS.md:3-18` is therefore evidence for narrow classification
baselines and data-boundary controls, not evidence that the LLM/ABC engine
discovered new mechanisms. The project needs one shared adapter contract that
can express trajectories, interventions, grouped subjects/cells, inputs,
holdouts, observation masks, and scientific invariants before these campaigns
can become a coherent automated-science benchmark.

## Static confirmations

- **Canonical test selection:** `scripts/run_domain.py:421-426` selects
  `test_metrics.mse` whenever `held_out=True`; the behavior is asserted by
  `tests/test_run_domain_config.py:47-50`. The canonical `Makefile:test`
  target includes the generated-code, SBI, sandbox, real Warfarin, domain
  configuration, model-matrix preflight, baseline, and stress checks
  (`Makefile:15-136`).
- **Full residual prompt path:** held-out evaluation fits on the prefix, then
  simulates the full time course and computes train/test metrics
  (`core/sandbox_eval.py:729-746`). The same full observed/predicted arrays
  are passed into `build_diagnostic_packet` (`:751-761`), which computes
  `residual_summary` (`:239-242`). The next proposal prompt inserts that
  diagnostic packet, including residual summary, warnings, and failure modes
  (`scripts/run_domain.py:77-88`, `:657-675`). Thus the held-out suffix can
  influence later proposal text through full-trajectory residual diagnostics.
- **Warfarin mode mismatch:** `real_warfarin_pkpd` declares
  `evaluation_mode="dense_masked"` while configuring the normalized CSV and
  subject-holdout JSON (`core/domain_configs.py:599-611`). The evaluator
  dispatches only `subject_holdout` and `cell_holdout`
  (`core/sandbox_eval.py:626-639`); `dense_masked` falls through to the generic
  `data/{domain}_ground_truth.npy` and `data/{domain}_time_points.npy` branch
  (`:641-657`). The configured Warfarin adapter therefore cannot be reached by
  the canonical runner under its declared mode.

## Bounded checks

Passed from the system Python (`/opt/homebrew/bin/python3`):

```text
python3 -m compileall -q core scripts tests
python3 -m tests.test_generated_code
python3 -m tests.test_run_domain_config
python3 scripts/run_domain.py ecology --family qwen3_coder_next \
  --model Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --endpoint http://127.0.0.1:8000/v1/chat/completions --seed 7 --dry-run
python3 scripts/repo_inventory.py --json
curl http://127.0.0.1:8000/v1/models
```

The system interpreter lacks JAX, Diffrax, and Pytest, so the JAX-dependent
tests do not run in that interpreter. A separate bounded runtime audit used a
temporary isolated environment and reported six targeted tests passing across
`tests/test_run_domain_config.py`, `tests/test_run_model_matrix_preflight.py`,
`tests/test_generated_code.py`, `tests/test_sandbox_eval.py`, and
`tests/test_sbi_engine.py`. No full `make test`,
gauntlet, model matrix, or end-to-end ABC-SMC campaign was launched.

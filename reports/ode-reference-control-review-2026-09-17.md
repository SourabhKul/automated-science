# Independent review: ODE reference controls

Review date: 2026-09-17. I reviewed `core/ode_reference_controls.py`,
`tests/test_ode_reference_controls.py`, `core/abc_smc_reference.py`, and the
acceptance specification in `reports/dynamical-inference-control-plan-2026-09-14.md`.
I did not edit implementation or tests, run the six-cell pilot, call the model
endpoint, or commit changes.

## Decision

**Approve the corrected same-configuration six-cell rerun; preserve the
original pilot as invalid target-predictive evidence.** The pre-pilot
numerical ODE, finite-target, noise, transition-weight, incomplete-status,
deadline, and receipt mechanisms passed review. The original pilot completed
but its finite-target observation-predictive quantiles were invalid because
the target CDF omitted the finite-target density weight. That defect was
repaired, the original receipt set was retained unchanged, and one frozen
same-configuration rerun completed. The corrected target and particle
predictive fields independently pass. This approves the bounded known-model
control only; it does not extend the plan's claims to canonical SBI/BDSS
paths, coverage, or scientific discovery.

## Evidence that passed

- `uv run --no-project --with pytest --with numpy --with scipy pytest -q tests/test_abc_reference.py tests/test_ode_reference_controls.py`: **21 passed** after the re-review changes. Python compilation and `git diff --check` also passed.
- The noncentral-χ² implementation agrees with `scipy.stats.ncx2.cdf` on an expanded grid of thresholds and noncentralities (maximum absolute error `1.39e-15`). The independent Poisson-series test covers the required `(x, lambda)` values `(4, 9) x (0, 0.5, 8, 40)`.
- Independent adaptive quadrature using SciPy for all six prescribed data cases (`k=0.35, 0.8`, seeds `11, 29, 47`) and `epsilon=3, 2` agrees with the production Gauss-Legendre target to at most `6.66e-16` in the normalizer and `4.44e-16` in the mean. The current target uses `x/2` for the chi-square Gamma argument. A factor-of-two error in that argument was found during the first review pass and was corrected before these checks.
- Fixed-step RK4 starts at the original time origin, including for a single requested future time. Comparing to `exp(-k t)` at rates `[0.05, 0.35, 0.8, 1.5]` and times through `5` gives maximum absolute error `9.94e-8`, below the declared `1e-6` tolerance and far below measurement scale `0.05`.
- The simulator owns a separate seeded noise generator and ignores the proposal RNG. Repeated calls with the same simulator seed reproduce the same noise sequence across simulator instances while successive calls differ; solver failure occurs before noise draws. A manual scalar Gaussian-mixture recomputation on a focused 24-particle, two-population cell gives maximum log-proposal error `4.44e-16` and maximum normalized-weight error `2.08e-17`.
- The exact reference is now separately labeled `exact_gaussian_likelihood` and uses `exp(-lambda/2)`; the finite target is labeled `finite_epsilon_abc`. On an independent dataset, finite-ε=2 and exact means were `0.3958602024541807` and `0.39316647396206655`, respectively. Densities are zero outside `[0.05, 1.5]`. Weighted latent and noise-convolved prediction summaries pass the unequal-weight fixture.
- Forced `max_steps` failures are counted as `max_steps_exceeded` with no noise draw, and an unattainable discrepancy returns `status="incomplete"` with `termination_reason="attempt_budget_exhausted"` within the attempt budget.

## Re-review of previously blocking behavior

`run_ode_decay_control_cell` now gates `parameter_diagnostics` and
`particle_prediction` on `reference["status"] == "complete"` (current lines
943--969). An independent call with `target_samples=3`,
`epsilon_schedule=[8.0]`, and `max_attempts_per_population=1` returned the
following even when one proposal was accepted (data seed `12345`):

```text
status=incomplete, termination_reason=attempt_budget_exhausted,
accepted=1, parameter_diagnostics=unavailable, particle_prediction=unavailable
posterior_summary=False for both
```

An already-expired deadline returned `status=incomplete`,
`termination_reason=wall_budget_exhausted`, zero attempts, and unavailable
particle summaries. The independent finite-epsilon and exact-target
calculations remain available as references, separate from particle claims.

The generic reference now rechecks the stop callback immediately after a
simulator call and before discrepancy evaluation (current lines 452--469).
With an independent simulator sleeping 50 ms and a 20 ms deadline, it
returned `incomplete/wall_budget_exhausted` with one attempted and simulated
call but zero accepted particles. A separate comparison with an always-false
callback produced identical accepted parameters, distances, weights, and
transition rechecks to the no-callback run, so the timing hook does not alter
reference math when the deadline is not reached.

## Pilot-readiness recheck

`scripts/run_ode_reference_control.py` now predeclares six serial cells (three
data seeds for each of `k=0.35` and `k=0.8`) with distinct data, proposal, and
simulator-noise seed triples. Its default is 400 particles, 50,000 attempts
per population, epsilon schedule `[3, 2]`, and a 540-second cap; CLI
validation rejects a cap above 600 seconds. An expired-cap driver smoke
produced six incomplete placeholders, zero completed cells, no posterior
summaries, and durable `partial_result.json` plus `artifact_manifest.json`
without entering a cell.

A separate monkeypatched driver smoke (no scientific simulation) completed all
six cell iterations and verified six per-cell JSON receipts. Every manifest
entry matched its persisted byte count and SHA-256 digest. The current cell
receipts include population weights, ESS, epsilon, attempts, simulator calls,
out-of-support and failure counters, per-population elapsed time and solver
failure categories, seeds, solver configuration, and the independent target
summaries. The driver performs atomic writes after each cell and for the
partial aggregate.

The six-cell pilot was not run by this reviewer, and no pilot receipt or
posterior claim is being approved by the pre-pilot checks alone.

## Original pilot receipt audit — blocker (preserved)

The original pilot receipt set is present under
`artifacts/ode_reference_control/20260917T104026Z/` and the aggregate reports
record six complete cells, 400 accepted particles in both populations,
`epsilon=[3, 2]`, 50,000 attempts per population, zero solver failures, and
the declared separate seed streams. The seven manifest entries all match
their recorded byte counts and SHA-256 hashes; the aggregate JSON is byte
identical to `partial_result.json`. Independent recomputation found maximum
finite-target normalizer error `2.78e-16`, target-mean error `3.33e-16`,
transition log-density/weight error `8.88e-16`, weighted latent-mean error
`0`, and particle observation-predictive-quantile error `1.95e-12`.

However, in the pre-fix source, `FiniteEpsilonODETarget._observation_predictive_cdf`
(then lines 520--524) integrated the conditional Normal CDF over the
parameter interval without multiplying by `self._density_values(grid)`. It
therefore did not compute the observation predictive distribution under the finite-
epsilon target. In representative receipt
`cell_00_k035_data11.json`, the recorded target observation quantiles at
`t=4` are `[-0.120771, -0.090440, -0.077557]`; independent target-weighted
quadrature gives `[0.192237, 0.273668, 0.356962]`. The maximum absolute error
over all six cells, both future times, and all three probabilities is
`0.434519`. The corresponding target latent means/quantiles agree to about
`6e-17`/`3e-15`, so the defect is isolated to the observation-predictive
target CDF. It is not a finite-epsilon versus exact-likelihood difference.

The receipt labels correctly distinguish `finite_epsilon_abc` from
`exact_gaussian_likelihood`, and the exact reference itself remains
independently correct. The invalid field is specifically the finite-target
`target_prediction.observation_predictive_quantiles`; no pilot agreement or
downstream predictive claim should use it until the corrected rerun.

## Corrected rerun receipt audit — approved

The one authorized rerun is retained under
`artifacts/ode_reference_control/20260917T104954Z_corrected_targetcdf/`, with
aggregate reports
`reports/ode-reference-control-pilot-2026-09-17-corrected-targetcdf.json` and
`reports/ode-reference-control-pilot-2026-09-17-corrected-targetcdf.md`. The
checkpoint is recorded in
`reports/research-ledger/2026-09-17T1051Z.md`. It has `status="completed"`,
six of six declared cells complete, two populations per cell, 400 accepted
particles per population, epsilon `[3, 2]`, 50,000 attempts per population,
the declared 540-second cap, and the same RK4 settings and six data/proposal/
simulator-noise seed triples as the original run. Every population terminated
with `target_reached`. Across the six cells there were 37,649 proposed
attempts, 37,593 simulator calls, 56 out-of-support retries, 150,372 fresh
noise draws, and zero solver failures.

The corrected seven-entry artifact manifest verifies all recorded byte counts
and SHA-256 values. Its SHA-256 is
`4d941416567420d4bbe352ad766e7509ceb0c6296e571a6e89d40567ec9b89e3`; the
corrected `partial_result.json` and aggregate JSON are byte-identical with
SHA-256
`9488fa7a896c69faffe1a542aec198243bcaf85b74e95a6fabe3dbbec91f9c05`.
The corrected Markdown report has SHA-256
`4f9f4526d9efd40f8142325321f1dc9e582c7d35a4ca66e381ca15bbeac85f42`.
The original raw partial aggregate/JSON report remains at
`7b47d0010f5ff2f3d9cdf6edc939303063391ad5339fa0ae7f3fce111dd960af`, its
original manifest remains at
`c63e2911164ca6cdf4f920e8ffb053c2ec3dac1b349fd9b9b9b0789d33db927b`, and
both Markdown reports retain `4f9f4526d9efd40f8142325321f1dc9e582c7d35a4ca66e381ca15bbeac85f42`.

I independently compared each corrected cell with its original counterpart.
Accepted parameters, all distances, accepted and normalized weights,
effective sample sizes, ancestor indices, proposal populations, and transition
receipts are exactly identical; only target observation-predictive quantiles
and wall-clock timing fields differ. Thus the repair did not change inference,
adapt the schedule after seeing outcomes, or perform post-selection tuning.
An independent Gaussian-mixture recomputation over all six second-population
receipts gives maximum normalized-weight error `3.47e-18` and maximum stored
proposal-log-density error about `1.0e-15`, consistent with the receipt
rechecks.

Independent SciPy `ncx2`/adaptive-quadrature calculations over all six data
sets give maximum finite-target normalizer error `2.78e-16`, finite-target
mean error `3.33e-16`, exact-likelihood normalizer error `2.78e-16`, and
exact-likelihood mean error `3.33e-16`. Corrected finite-target latent means
and latent quantiles agree to `5.55e-17` and `3.22e-14`; density-weighted
finite-target observation-predictive quantiles at `t=[4,5]` agree to at most
`4.94e-14`. Recomputed weighted particle latent means and observation-
predictive quantiles agree to `1.11e-16` and `4.81e-14`, respectively. The
receipt semantics distinguish target latent/noise-free summaries from target
observation predictions and particle latent/noise-convolved summaries.

All corrected receipts label the sampled target `finite_epsilon_abc` with
epsilon `2.0`. The separate reference labels itself
`exact_gaussian_likelihood` with epsilon `null`; its finite-target and exact
means differ in every cell (absolute differences from `0.00184` to
`0.00886`). Therefore the corrected predictive agreement is not an
accidental conflation of finite-epsilon ABC with the exact Gaussian
likelihood. The original negative target observation quantiles remain in the
original receipt as an auditable historical failure and are excluded from this
approval.

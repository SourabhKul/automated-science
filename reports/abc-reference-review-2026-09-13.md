# ABC reference review

Date: 2026-09-13

Scope: `core/abc_smc_reference.py`, `core/abc_reference_controls.py`, and
`tests/test_abc_reference.py`. The historical `core.sbi_engine` path was also
checked for unchanged behavior. No implementation files were edited by this
review and no Git push was performed.

## Evidence

- `uv run --no-project --with pytest --with scipy --with jax --with diffrax pytest -q tests/test_abc_reference.py tests/test_sbi_engine.py`: **12 passed** (including the custom-support regression).
- The reference run uses an explicit non-increasing epsilon schedule, fresh
  ancestor/proposal draws inside the retry loop, unconditioned Gaussian mixture
  densities, no clipping, and saved population weights/covariance/counters.
- Independent one-dimensional quadrature (normal CDF plus adaptive numerical
  integration in the review script) was compared with six stochastic controls
  (`y = 0.1, 0.5, 0.9`; `epsilon = 0.1, 0.025`; 1,000 accepted particles).
  Mean errors were respectively `0.0005, -0.0019, -0.0074, 0.0026, -0.0034,
  -0.0005`; all runs completed. These are finite-epsilon ABC targets, not exact
  posterior checks.
- A separately recomputed mixture log density matched every saved generation-1
  value with maximum absolute difference `0.0`. Recomputing normalized weights
  from saved `log_prior_density - log_proposal_mixture_density` differed by at
  most `6.94e-18`.
- Boundary control recorded 33 out-of-support retries for 40 accepted particles;
  accepted particles had no zero/one boundary atoms. An unattainable criterion
  stopped after exactly 7 attempts with `status=incomplete`, an explicit
  `attempt_budget_exhausted` reason, and an empty partial population. A
  sequential two-generation failure retained the completed first population and
  returned the empty second population rather than substituting the first.
- The legacy engine tests pass unchanged. A simulator-stream smoke produced 500
  distinct standard-normal draws with adjacent correlation `-0.0316`.
- The bounded pilot in `scripts/run_abc_reference_controls.py` completed all
  18 cells (three seeds for each of the six scalar cases) in 1.229 seconds;
  every cell reached 400 particles and remained within the recorded
  descriptive ESS-scaled diagnostics. Its read-only oMLX `/v1/models` probe
  returned HTTP 200 and advertised the fixed Qwen model; it made no generation
  request.
- The follow-up sequential pilot in
  `reports/abc-reference-sequential-control-pilot-2026-09-13.{json,md}` used
  the explicit schedule `[0.2, epsilon]` for all 18 cells and completed both
  populations in 2.519 seconds. All 36 population receipts have normalized
  weights; generation-1 out-of-support retries totaled 9,017, final ESS ranged
  from 386.79 to 397.49, and every cell reached 400 final particles. The
  saved receipts include both population weights, covariances, and counters.

## Findings

1. **Resolved in the final implementation.** An earlier draft attempted
   generation-zero shape validation before inferring the dimension for
   `in_support`-without-`bounds`. The implementation now infers and validates a
   non-empty one-dimensional prior draw first; the focused regression test
   completes 5/5 particles through this path.

2. **Resolved in the final test update.** The scalar acceptance test now has an
   independent test-side Gauss–Legendre normal-CDF quadrature and checks its
   normalizer, mean, CDF, and quantiles against the control diagnostics. The
   control helper remains useful for the runnable pilot, but the acceptance
   assertion no longer relies solely on its returned target object.

## Verdict

**Approved for the bounded opt-in Gaussian ABC-SMC reference milestone.** The
required proposal/density correspondence, support rejection, log-space weight
propagation, finite-epsilon distinction, independent target validation,
general-support initialization, and bounded-failure behavior are supported by
direct outputs above. No broad research run should be based on an incomplete
result without checking its explicit `status`/`complete` fields.

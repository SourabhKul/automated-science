# ABC-SMC repair: scientific acceptance plan

Prepared by the primary planner on 2026-09-13. This specifies the next numerical milestone; it does not claim an implementation or executed experiment. Coding and execution remain assigned to Luna max.

## Decision

Establish one mathematically consistent Gaussian proposal reference before evaluating BDSS. Keep the existing implementation available for historical replay with its limitations labeled. The immediate target is correctness of parameter inference, not a new inference-method publication or a switch away from ABC-SMC.

Sequential importance weighting must use the proposal distribution that actually generated particles. Standard ABC population methods supply the reference framework. [Toni et al.](https://arxiv.org/abs/0901.1925), [Beaumont et al.](https://arxiv.org/abs/0805.2256).

## Explicit reference contract

For population t with normalized previous weights w and Gaussian kernel K, define the unconditioned mixture q(theta) = sum_j w_j K(theta | theta_j). Draw a fresh ancestor and a fresh proposal on every attempt. Reject out-of-prior-support proposals without clipping, then simulate and accept only when the declared discrepancy is below epsilon_t. Weight accepted proposals in proportion to prior(theta)/q(theta), evaluated stably in log space, then normalize. For a uniform prior the constant prior density cancels within its support.

Resampling the ancestor on every retry matters: repeatedly drawing within bounds from a fixed ancestor instead would create ancestor-specific truncated kernels, requiring a different mixture density. Clipping also creates probability mass on boundaries and does not implement this reference. Covariance regularization and any scale adaptation must be fixed for each population and used identically in draws and density evaluations.

Generation zero draws from the prior and uses the same stated discrepancy/acceptance rule. Persist epsilon per population, proposed/accepted/out-of-support/failed simulation counts, actual weights, effective sample size, and termination reasons. A bounded attempt budget must return an explicit incomplete population, never a silently substituted successful posterior.

## Cheap numerical controls

1. **Analytic finite-epsilon ABC target.** Let theta be uniform on [0,1], simulate Y ~ Normal(theta, 0.1), observe y=0.1, and accept |Y-y| <= epsilon. The target density is proportional on [0,1] to Phi((y+epsilon-theta)/0.1) - Phi((y-epsilon-theta)/0.1). Compute its normalization and CDF with independent one-dimensional quadrature. Compare weighted empirical CDF, means, and quantiles against this target. Include epsilon values 0.1 and 0.025, and repeat at y=0.5 and y=0.9. These cases exercise the interior and both support boundaries. Errors must be assessed against Monte Carlo variability, not hardcoded exact equality.
2. **Boundary behavior.** Confirm the reference's out-of-support draws are rejected rather than accumulated exactly at zero or one. Treat any boundary atoms introduced by a proposal implementation as a different distribution requiring explicit handling.
3. **Weight propagation.** Require saved weights to reproduce the reported means, quantiles, and predictive summaries. If an unweighted sample is desired, explicitly resample according to the weights and record the resampling seed.
4. **Resource failure.** Make an intentionally unattainable acceptance criterion hit the attempt budget. It must produce a structured failure with partial evidence, not endless resampling or a finite-looking substitute score.
5. **Small ODE integration.** After scalar controls, run one simple exponential-decay model with a declared observation-noise model. Compare finite-epsilon ABC with a reference calculation and distinguish approximation error from simulation error. Keep this separate from the original unchanged-scaffold smoke.

Begin with small populations and a few independent seeds to estimate runtime and sampling variation; choose confirmatory tolerances and sample counts from development evidence before freezing the implementation checks. Finite-epsilon ABC targets differ from exact Bayesian posteriors. Do not demand exact-posterior calibration at a nonzero epsilon or label failure to do so as automatically a coding bug.

## BDSS admission criterion

Document its actual transition measure, support, normalization, and evaluable log density before assigning Bayesian importance weights. A beta-distributed step radius and random direction are not a Gaussian kernel. If a correct density is unavailable, retain BDSS only as an explicitly heuristic search arm without posterior-calibration claims. Compare validated strategies on the same targets, discrepancies, epsilon schedules, independent seeds, simulator-call budgets, and wall time.

## Next operator handoff

Complete or diagnose the user's current-scaffold smoke first. Then implement the reference and controls in a scoped change; obtain an independent Luna review of the sampling/weight correspondence. The primary planner will inspect numerical evidence before authorizing a larger scientific campaign within the standing local-compute authorization.

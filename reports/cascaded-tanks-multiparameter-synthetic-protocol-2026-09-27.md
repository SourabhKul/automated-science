# Cascaded Tanks multi-parameter synthetic control protocol — 2026-09-27

**Status: primary scientific specification, awaiting independent review and
execution.** This is a synthetic methods control, not authorization for a
real-data Qwen/ABC run or development/official-test scoring. It extends the
one-free-parameter control in
[`2026-09-26T1952Z.md`](research-ledger/2026-09-26T1952Z.md). The
official tank test and consumed Silverbox multisine condition remain
unavailable for this work.

## Question and target

The earlier H-only control held every nuisance parameter at truth. This
control asks whether the checked ABC-SMC wrapper can recover several
effective parameters and make prospective predictions when initial states
are unknown, whether its approximate intervals cover known truth across
repeated noisy *measurement realizations*, and whether adding nuisance
freedom changes the earlier cross-family abstention. The toy equations are
the discrete S0/O2/C2 surrogates in the inference design, not calibrated
physical equations. Do not interpret H, flow coefficients, or any model
preference as a tank mechanism.

Use the existing `run_cascaded_tanks_synthetic_abc_smc` wrapper and checked
Gaussian ABC-SMC reference. Its prior/proposal/weights operate in a
normalized unit cube with an affine decoder. For O2, the entire rectangular
prior must satisfy `max(x2_0) <= min(ceiling)`; the declared bounds below do.
The API calls H `ceiling`. No clipping, partial-posterior promotion, or
post hoc tolerance changes are allowed. This experiment is separate from
the eventual real-data numeric protocol: do **not** copy its bounds or
tolerances into that protocol by default.

## Frozen synthetic truth, data, and inference

Both O2 and C2 truths use `a=0.5`, `c=0.4`, `p=0.5`,
`x1_0=x2_0=0.5`, `H=3.0`, sample interval four seconds, and the simulator's
pre-transition observation convention. The 78-sample fit input is 24
samples of `u=8`, then 54 of `u=0`. A separate 60-sample prospective input
is 12 samples of `u=3`, 12 of `u=8`, then 36 of `u=0`. Each future trace
starts from its own state after all 78 fit transitions; no reset or
future-output initializer is allowed. The generative state paths are
deterministic except for the declared training measurement noise.

For each truth family, create four independent *measurement-noise*
realizations on the same underlying 78-sample state path. Add iid
`Normal(0,0.05)` to each training observation, using NumPy default RNG
seeds `4100..4103` for O2 and `4200..4203` for C2. These eight replicates
are independent noise realizations, **not** independent physical systems or
inputs. Fit each replicate to its generating family, freeing only `p`,
`x1_0`, and `x2_0`, in that parameter order. Fix `a=0.5`, `c=0.4`,
`ceiling=3.0`. Independent uniform bounds are `p ∈ [0.35,0.65]`,
`x1_0 ∈ [0,1]`, `x2_0 ∈ [0,1]`. ABC seeds are `5100..5103` for O2 fits and
`5200..5203` for C2 fits, matched by replicate suffix.

Run four additional **noiseless**, more flexible fits on each true family
using free `p`, `x1_0`, `x2_0`, and `ceiling`, in that order, with the same
first three bounds and `ceiling ∈ [2.5,3.5]`; fix `a,c` at truth. Fit O2
truth with O2 seed 6100 and C2 seed 6300; fit C2 truth with C2 seed 6101
and O2 seed 6301. These two cross-family fits test whether nuisance
freedom changes the earlier H-only zero-accept result. They are not a
Bayes-factor or real model-selection experiment.

Finally, generate one deliberately misspecified O2 training observation
path as the noiseless O2 output plus direct feedthrough `0.15*u[k]`.
Fit O2 with the same four free bounds, seed 6200. This effect is absent
from the fitted simulator. Its prospective comparison target also adds
`0.15*u[k]` on the declared 60-step future input, so the mismatch persists
under changed forcing. Record whether the fit completes and, if so,
the high-input versus zero-input training residual means from its weighted
median trajectory. Define signed residual as observed minus pointwise
weighted-median prediction; the high-input group is `k∈[0,24)` and the
zero-input group is `k∈[24,78)`. A completed fit is not proof that the mismatch was
detected; an incomplete fit is not proof of the particular omitted cause.

All 13 fits use 32 target particles, two ABC populations with fixed
full-trajectory RMSE tolerances `[0.30,0.15]` in synthetic output units,
and a maximum of 4,096 attempts **per population**. Simulator limits are
100,000 steps and `1e12` magnitude. Use the wrapper's fixed normalized
uniform prior and Gaussian proposal/weight defaults. This tolerance is a
declared control choice, not a noise-calibrated likelihood. Record attempts,
acceptance and failure categories, terminal ESS, and weighted posterior
parameter correlations wherever all populations complete (null if either
weighted variance is zero). If any fit is incomplete, report its
diagnostics with no posterior summaries or prospective prediction.

## Declared diagnostics and limits

For each complete three-free-parameter fit, calculate the weighted
5th–95th marginal interval and whether each true parameter lies inside.
Report the inclusion count across four noise realizations per family;
that tiny count is a diagnostic, **not** a calibrated 90% coverage estimate.
For **every complete fit**, reconstruct each terminal particle's 78-step
latent state, then forecast all 60 prospective inputs. Require all particles
to yield complete finite forecasts; otherwise record no aggregate score.
Use the existing `weighted_quantile` left-inverse empirical-CDF convention
for the 5th, 50th, and 95th percentiles with the unchanged ABC weights.
Compare the weighted-median free-run forecast with the appropriate
generating-family future observation under the same prospective forcing,
using RMSE, and report the fraction of 60 truth outputs inside the pointwise
weighted 5th–95th particle-prediction envelope. For ordinary cases, that
truth is the noiseless observation trace, so this is conditional
noise-free trajectory coverage and omits future measurement noise. For the deliberate
feedthrough mismatch, the future observation target includes `0.15*u`,
so its interval fraction is an observation-mismatch diagnostic instead.
The 60 time points are dependent in every case.

For the two complete **same-family, correctly specified** four-free-parameter
fits, report parameter intervals, truth inclusion, ESS, and posterior
correlations. For the two cross-family fits, the generating vector is not a
meaningful truth target inside the wrong-family posterior; report fit
status, ESS and correlations if complete, and prospective forecast only if complete, without
truth-inclusion or mechanism claims. The deliberately misspecified fit also
has no correctly specified parameter-coverage target; report its fit status,
ESS and correlations if complete, forecast if complete, and the declared
residual contrast.
A failed forecast must not be repaired by dropping particles. Do not report
model probabilities or Bayes factors from these ABC summaries.

The operator must save a pre-run, immutable settings receipt and source/code
hashes before outcomes; use one worker, at most 600 seconds wall time and
2 GiB sampled RSS (document macOS enforcement limits and potential
between-poll overshoot), and preserve every failed launch or fit. Check
processes first. Keep all runner code and detailed arrays under ignored
`artifacts/evaluations/cascaded_tanks_multiparameter_synthetic/`; only a
compact independently reviewed report may enter Git. Do not access real
source archives, development targets, official test fields, oMLX, or the
network for this control. A separate Luna reviewer must independently
replay at least one case of each type and audit all fit-status/weight/
coverage claims before the primary uses the outcome.

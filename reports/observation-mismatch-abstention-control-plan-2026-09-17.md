# Observation mismatch and abstention control

Primary scientific specification, 2026-09-17. This is a pre-registered simulator control, not a scientific claim and not an LLM comparison. It follows the validated ODE-backed ABC-SMC reference control.

## Question

Can the system distinguish a known observation-model mismatch from a mechanism change, and can it abstain when a more flexible mechanism has no practically meaningful predictive support?

The target behavior is not to identify a named mechanism from one noisy trajectory. It is to preserve competing explanations, state when they are unresolved, and propose a discriminating measurement only when the models make a predeclared observable disagreement.

## Experiment A: observation-model mismatch

Generate a latent decay trajectory:

\[
x(t)=\exp(-kt),\quad k=0.65,
\]

with observations

\[
y(t)=\frac{x(t)}{c+x(t)}+\epsilon,\quad c=0.35,\quad
\epsilon\sim N(0,0.025^2).
\]

Use a frozen time grid containing a high-signal calibration region and a low-abundance forecast region. Split observations by predeclared time roles: train for parameter inference, validation for choosing between the two fixed candidate families, and a sealed final suffix for one evaluation after selection. Initial condition and future times are declared protocol inputs, not estimated from final observations.

Compare only these explicit hypotheses:

1. **Correct observation model:** latent exponential decay plus saturating observation map, fitting \(k,c\).
2. **Misspecified mechanism surrogate:** direct observed exponential \(y=A\exp(-kt)+\epsilon\), fitting \(A,k\).

Both may fit a narrow time interval. The scientific output is limited to a measurement-model validity boundary on this simulator. It must not call the saturation parameter an identified physical sensor mechanism outside this known generator.

The system may choose the correct family only when its validation predictive improvement exceeds a predeclared practical margin and the sealed final result is then reported once. It must retain the alternative and its forecasts in the receipt. A failure to separate the families is a valid result.

## Experiment B: unresolved nested-mechanism negative control

Generate direct noisy decay data from

\[
y(t)=\exp(-kt)+\epsilon,\quad k=0.65,\quad\epsilon\sim N(0,0.05^2).
\]

Compare a one-rate exponential with a two-rate mixture

\[
y(t)=w\exp(-k_1t)+(1-w)\exp(-k_2t)+\epsilon.
\]

The two-rate model contains the one-rate case and can absorb noise. Freeze the validation rule before data generation: a two-rate model is not promoted unless it improves validation RMSE by at least one noise standard deviation divided by the square root of the validation-count *and* its predictive advantage reproduces on the sealed suffix. Otherwise report **unresolved / retain simpler model**, including both model forecasts and the proposed discriminating condition.

The discriminating condition is a future time in a declared low-signal range only when the posterior-predictive median disagreement exceeds two observation-noise standard deviations. Otherwise the recommendation must explicitly be "no informative measurement under this protocol." The control does not permit post-hoc searching for a time that makes one model appear better.

## Inference and evaluation requirements

- Use the reviewed opt-in Gaussian ABC-SMC reference or a transparent numerical baseline only as a labeled comparator. Do not use historical `SBIEngine` or BDSS for posterior claims.
- For each model, persist weights, ESS, epsilon/attempt/counter diagnostics, solver and observation-model configuration, seeds, posterior-predictive draws, and complete/incomplete status. Parameter differences across model families are not comparable evidence by themselves.
- The discrepancy must operate on observed values through each model's declared observation map. A model may not fit latent states reconstructed from final observations.
- Validate model-selection and abstention logic with mocked deterministic posteriors as well as stochastic simulations. Vary sealed outcomes while holding development artifacts fixed; selection must not change.
- Run three independent generated data seeds for each experiment after implementation review. Describe errors across the fixed cells; do not label nearby time points or adaptive candidates as independent replicates.
- Preserve all negative and unresolved outcomes. No Qwen generation is required for this control; a later fixed-Qwen experiment may propose from a constrained version of the same declared hypothesis grammar.

## Acceptance and limits

Passing Experiment A means the receipt preserves the observation map and correctly scores observed predictions. Passing Experiment B means the workflow can refrain from promoting the more flexible two-rate explanation under the frozen rule. Neither validates real kinetic mechanism discovery, ABC model probabilities, novelty, or a real experimental intervention.

Only after both controls and an independent review should a Qwen-guided constrained hypothesis test use this decision policy. The next step after that is a real-data source audit with independent units, not a broad benchmark sweep.

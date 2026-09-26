# Cascaded Tanks inference design proposal — 2026-09-26T14:07Z wake

**Status: scientific design, not a frozen run protocol.** No tank Qwen request,
real-data ABC fit, validation score, or official test attempt is authorized by this
document. The source-only methods-control track is
`cascaded_tanks_methods_v1_20260926`; the historical external-repeat failure
still applies to discovery claims. The [source contract](cascaded-tanks-source-resolution-proposal-2026-09-26.md)
and [current ledger](research-ledger/2026-09-26T0906Z.md) provide the exact
archive, split, receipts, and negative-result history.

## Question and limits

The immediate question is whether a fixed-Qwen proposal and a checked
ABC-SMC inference loop can make a complete, deferred-target scored forecast on
a second real dynamical system while honestly retaining model ambiguity. The
original benchmark already describes Bernoulli-like outflow, upper/lower
overflow, an uncalibrated capacitive sensor, process noise, and unknown initial
states. A lower-tank voltage record alone cannot establish which physical
pathway produced a ceiling or identify absolute tank heights. Good predictive
performance would be a **methods-control** result, not discovery of overflow.
The benchmark's official test is one record, not an external replication.

Use only `uEst/yEst[0:768]` for parameter fitting and Qwen context. The
input-only `uEst[768:1024]` can define a prospective forecast, but the
corresponding `yEst` suffix must be materialized by a separate scorer only
after every candidate has produced a complete forecast. The development
suffix is contiguous with training in the same realization and its input-only
record has a lower-input excursion; it is not an independent replication.
Its score is descriptive model selection, not a replicated effect. Freeze
all choices before accessing its targets. Reserve official `uVal/yVal` for
one final attempt only after structure selection and an independently
reviewed refit/final scorer.

## Competing model families for a controlled pilot

Use an explicit **discrete-time, sensor-voltage-equivalent surrogate** at the
observed 4-second interval. For training or free-run forecasting, `u_k` acts
over the transition from state `k` to `k+1`; output `y_k` is read from state
`k` before that update. This is inspired by, but not identical to, the
continuous physical model in the [original benchmark manuscript](https://cris.vub.be/ws/portalfiles/portal/24190624/CascadedTanks_Maarten.pdf).
Let `x1,x2 >= 0` be effective states rather than calibrated heights. The
common soft-flow core is

```
soft_loss2 = c * sqrt(x2)
x1_next = max(0, x1 + p*u - a*sqrt(x1))
x2_raw  = max(0, x2 + sqrt(x1) - soft_loss2)
```

The floor at zero is part of the declared dry-tank surrogate, not an
after-the-fact numerical repair. `a,c,p` are positive per-sample effective
coefficients. The coefficient of `sqrt(x1)` entering `x2` is **fixed to 1**
as a scale convention. Without that convention, simultaneously rescaling
`x1` (including its initial value), `a`, `p`, and its separate gain into `x2`
leaves the observed trajectory unchanged, so those parameters cannot be
inferred separately. This removes one exact gauge symmetry; it does not
establish practical identifiability of `a`, `p`, or the hidden initial state.
`x2` is in output-voltage-equivalent units and `x1` is in the corresponding chosen
latent gauge; these coefficients **must not** be interpreted as physical
flow constants. Unknown starting `x1` and `x2` are nuisance
parameters/initialization variables. The common no-hard-effect family **S0**
uses `x2_next=x2_raw`, `y=x2`. Two alternatives have different memory after
a high excursion:

- **O2 (lower-state cap surrogate):** `x2_next=min(H,x2_raw)`, `y=x2`.
  Excess state is discarded, so recovery starts from at most `H`. Its
  initial-state support also requires `x2_0 <= H`; an above-cap starting
  state is an invalid O2 configuration, not silently clipped.
- **C2 (sensor ceiling surrogate):** `x2_next=x2_raw`, `y=min(H,x2)`.
  Hidden state may remain above `H`, so recovery can lag after input falls.

The source reports hard overflow and a nonlinear, uncalibrated sensor, but it
does not specify either ideal `min` rule. `H` is an effective threshold in the
output coordinate, **not** an estimated tank height. The deterministic map
omits the reported stochastic upper-overflow split, process noise, measurement
noise, nonlinear sensor transfer, and possible actuator behavior. These remain
live alternatives. O2 and C2 are observationally identical if the trajectory
never crosses `H`; even with a crossing they separate only if subsequent
recovery exposes their different hidden-state memory. **Do not promote O2
over C2 as a physical mechanism based on one voltage trace.** Include a
simple linear dynamical or persistence comparator with the same allowed
training information; any literature score comparison must match the source,
initialization, free-run metric, and training-data allowance.

Two Luna operators independently checked the source adapter's **training-only**
outputs and found one exact 10.0 V plateau of 27 samples at indices
`[150,176]`, followed by one qualifying low-input recovery interval
`[178,191]` under the declared training-input Q25 = 2.053 V rule: a maximal
`u<=Q25` interval with `y>=9.95` in the 16-sample lookback including onset
and then `y<9.5` during that same interval. The first output below 9.5 V
occurs at index 185. Only one interval met this criterion among 11 low-input
runs in one correlated record; its cause is unidentified.
Because it is far from the 768/256 boundary, the development input may not
excite the O2/C2 memory difference. Before materializing development
targets, report a frozen input-only forecast-separation diagnostic across
the competing fitted families. If it cannot distinguish their predictions
under a training-only noise/scale criterion fixed in the future protocol,
record the structural comparison as **unresolved** regardless of which
family has a slightly lower development RMSE. This does not block reporting
honest predictive scores or a methods-control failure.

The fixed Qwen's bounded task should be to choose one *testable augmentation*
from `{O2,C2,abstain}` and state the anticipated development-only signature,
assumptions, and a counterprediction. It receives the published source
description as a curated text excerpt **without published test plots or
outcomes**, the training-only EDA, the S0 equation, and optionally the
development **input-only** suffix for prospective forcing; it receives no
development target or official test data. Freeze whether that input suffix is
included before generation. Save exact request, response, model ID, endpoint, chat-template
settings, parser decision, and SHA-256 receipts. A malformed proposal or
abstention is a terminal no-proposal result, not a license to reprompt until
one family wins. S0 and the simple comparator run regardless; unchosen
alternatives remain unresolved. This is a same-Qwen workflow, not a model
ranking.

## Inference and evidence semantics

Use the opt-in `core.abc_smc_reference.run_gaussian_abc_smc_reference`, not
the legacy clipped/BDSS path. Each family must have a fixed prior sampler,
matching log density and support, normalized parameter coordinates, the
same Gaussian-mixture proposal used in the weight denominator, and no
boundary clipping. The initial population is prior rejection; later weights
are proportional to prior density divided by the previous weighted proposal
mixture density. Freeze nonincreasing finite tolerances, particle counts,
attempt caps, seeds, discrepancy, failure/trajectory limits, and resource
budgets **before any real fit**. Report ABC weights, ESS, acceptance/failure
counts and posterior-predictive spread; these finite-tolerance distributions
are model-conditional approximations, not calibrated probabilities that O2 or
C2 is true. Do not report a Bayes factor from insufficient summaries; the
[ABC model-choice limitation](https://arxiv.org/abs/1102.4432) is directly
relevant here.

For the pilot, use an explicitly defined training-trajectory discrepancy in
sensor volts, including the chosen alignment, starting-state treatment,
standardization learned on the training prefix, and handling of repeated
ceiling values. A pointwise deterministic distance ignores stochastic
overflow and serial measurement errors; retain that as an assumption and
inspect residual structure on training only. A nonfinite or magnitude-bound
crossing must terminate that simulation with a categorized receipt; no
clipping, particle dropping, or post-selection renormalization is allowed.
Before trusting uncertainty, test synthetic parameter recovery and
predictive coverage at several known settings, including both O2 and C2,
and evaluate an analytically tractable or high-accuracy control for the
transition density and simulator. Include cross-family synthetic fits in
both directions: a no-threshold-crossing input **must remain unresolved**,
while a crossing followed by sufficiently long recovery should test whether
the chosen discrepancy can distinguish the two memory patterns. Report
posterior parameter correlations and equivalently fitting settings under the
fixed gauge. Passing existing ABC unit tests does not
certify tank inference or identifiability.

For development forecasts, carry each fitted particle's latent state from
the full training input through the boundary at index 768. Never restart at
an arbitrary development initial state or use development targets to
initialize. Require a complete finite forecast from every particle before
opening `yEst[768:1024]`; otherwise record an unresolved candidate with no
score. Predeclare the weighted ensemble forecast, comparator metric,
minimum improvement rule, and tie/failure policy. A structure selected on
development may then be refit on all 1,024 estimation points under a frozen
refit rule; its development score does not become independent after refit.

The separate final stage may use only `uVal` and the benchmark-authorized
first 50 `yVal` points for a **predeclared state-only initializer**. It may
adjust latent starting states under a fixed rule, never model family,
threshold, dynamics coefficients, priors, or other parameters. State and
report whether this uses the author's same-initial-state assumption or the
current loader's test-specific 50-sample initializer convention; the two
choices are not interchangeable. Test two-state initializer recovery and
uncertainty synthetically, including ceiling plateaus. All parameters and
the initializer algorithm must be frozen before this stage. Under the
declared alignment, `uEst[767]` produces state 768 before the first
development forecast, and `uVal[49]` produces state 50 before scoring
`yVal[50]`.
It must claim a one-use marker before reading the test record, forecast all
required particles and fail terminally on any nonfinite trajectory, then
materialize `yVal[50:]` only if the forecasts succeed. The official test
record is consumed even by a failed attempt, as with Silverbox. The current
development loader deliberately has no test accessor/scorer; a new one
requires an independent leakage review.

## Implementation gate and decisions still to freeze

Implement the simulator and source-isolated inference wrapper against
**synthetic** fixtures first. Independently check the time alignment,
nonnegative-state rule, starting states, observation ceiling versus state
overflow, deterministic replay, proposal weights, forced numerical failures,
attempt exhaustion, and latency/memory. The existing source adapter must
remain unchanged except for a separately reviewed defect repair. Its
importable constructor token is a convention, not cryptographic provenance;
every later scorer must recheck the source/receipt chain.

Before a real Qwen/ABC run, publish a distinct frozen protocol ID and manifest
with numeric prior bounds, fixed input/output scaling, discrepancy, epsilon
schedule, candidate and comparator budgets, Qwen prompt, train/development
scorer, threshold/tie policy, initializer, full-estimation refit, seed list,
runtime versions, source/code hashes, run ID, and a single-use test rule.
These values are deliberately **not** chosen here: they must be informed by
training-only scale and synthetic recovery, then independently reviewed
without development targets or test outcomes. No result from this proposed
design can be called a scientific breakthrough without new independent
evidence and a current primary-literature comparison.

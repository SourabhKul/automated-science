# Silverbox controlled development contract

Decision: 2026-09-25. This is a methods integration plan on an existing
benchmark, not a new physical discovery or a confirmatory analysis. The
published Silverbox train-plus-validation record is the only data source for
this phase. The official test outputs remain sealed except for the stated
50-sample initialization windows exposed by the existing adapter.

## Evidence and question

The official record has measured input `u`, measured output `y`, and a sample
interval of `1/610.35` seconds. The canonical trajectory evaluator currently
forwards only `t,y` to an autonomous simulator and its legacy `SBIEngine`.
The checked `core.abc_smc_reference` sampler is separate. A controlled-system
contract must carry `u` into every simulation and use the checked reference
with a train-only discrepancy. Merely passing Silverbox `y` into the
canonical runner would omit the forcing signal and would not test the intended
method.

Development-only inspection found 65,062 finite published train/validation
samples, with input RMS about 0.0230 and output RMS about 0.0545. The 8,192
sample block below has similar RMS. The proposed validation interval has a
narrower output range than the training interval, so it is an integration and
interpolation check, not an extrapolation or independent-replication test.

## Phase 1: bounded data and simulator boundary

- Preserve absolute source indices and native sampling. Use only source
  `[40650,48842)` from `train_val_multisine` (8,192 samples).
- Predeclare training `[40650,46794)` (6,144 samples), a 256-sample unused
  gap `[46794,47050)`, and adaptive validation `[47050,48842)` (1,792
  samples). No metric from validation may set parameters or ABC thresholds;
  validation may choose a candidate after train-only fitting.
- For a first computationally bounded fit, use four fixed 256-sample training
  windows at train-relative starts `0,1536,3072,4608`. They span different
  amplitudes; each has 50 observed samples reserved for state initialization
  and 206 predicted samples for a train discrepancy. Window selection is
  fixed now, before model fitting. Record that these windows cover only part
  of the full training range.
- The validation input is known. Its first 50 outputs may initialize state;
  the remaining 1,742 outputs are selection targets. A model must free-run
  under the recorded input after initialization. A failure or nonfinite
  trajectory is a failure, never silently clipped into range.
- Keep the source contract's development-facing loader distinct from its
  scorer-only test loader. Phase 1 must not import or call the scorer-only
  loader, inspect test suffixes, or make any final test claim.

This phase implements and independently reviews the controlled data/simulator
interface and its source-index/role guards. It does not run ABC, Qwen, or a
fit. The validation target array remains available on the validation-series
object for later candidate selection; the simulator call does not receive it.
This interface does not enforce the policy that validation metrics must not set
fit parameters or ABC thresholds. That policy, including an auditable fit and
selection receipt, must be specified and checked before any fit. No validation
or final outcome has been observed for a candidate.

## Subsequent freeze before first fit

Predeclare two interpretable families, including a simple linear forced
baseline and one constrained nonlinear competitor, with units, support,
observation map, and state-initialization semantics. Use the exact fixed
local Qwen model for a small candidate-proposal step under a frozen grammar
and request budget. Both families receive the same declared simulation
budget; any Qwen-generated proposal must pass source and simulator checks
before fitting. Specify train discrepancy, observation noise, priors, ABC
epsilon schedule, population size, attempt cap, predictive summary,
practical validation margin, seed streams, and wall/memory limits in a
reviewed protocol *before* a fit. The reference ABC-SMC sampler returns
incomplete if a population cap is hit; do not alter thresholds after seeing
candidate outcomes.

After development selection, freeze model code, parameter distribution,
normalization, initialization, solver, and validation receipt. Only then may
the scorer load one official test condition, use its permitted first 50
outputs for initialization, and score its sealed suffix once. The full arrow
test and its no-extrapolation subset overlap and cannot count as independent
replicates. The Silverbox archive is one device/benchmark source, so even
multiple test records do not establish a new physical mechanism.

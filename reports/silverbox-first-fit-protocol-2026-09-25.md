# Silverbox first-fit protocol — freeze before execution

Status: revised after independent protocol review on 2026-09-25, **no Qwen
request, ABC run, validation score, or sealed score has been made under this
protocol**. Implement and independently review the full protocol before the
first fit. The experiment is a bounded
methods integration control on a known benchmark, not a discovery claim.

## Question, data roles, and source

Test whether one constrained same-Qwen nonlinear proposal plus checked
ABC-SMC fitting can beat a forced linear baseline on development data while
preserving a locked final evaluation. Use the reviewed CSV Silverbox adapter
with native `dt=1/610.35 s`. Use only the development block, split, unused
gap, and four fixed 256-sample training windows in
`silverbox-controlled-development-contract-plan-2026-09-25.md` and
`core/real_data/silverbox_controlled.py`. Each window provides 50 observed
initialization values and 206 free-run training targets. Validation is the
fixed 1,792-sample interval, with 50 initialization values and 1,742 targets.
It is a later segment of the same record, not an independent experiment.
Only the locked scorer may read the official test multisine output after the
first 50 samples. No arrow test is part of this first fit.

## Training-only evidence used to set the pilot bounds

An exploratory training-only ordinary least-squares fit on the 824 training
targets gave `(a1,a2,b,c)=(1.4631070,-0.9361293,0.4190676,-0.00213506)`
for a forced observed-output AR(2) recurrence. The one-step residual RMS was
0.00112682 and free-run RMS across the four windows was 0.00719364. Adding
`y[t-1]^3` gave a training-only coefficient of about -1.48029 and free-run
RMS 0.00482064, with a substantially higher raw design condition number.
One-step linear residual correlations with `y^3`, `u^3`, and `u*y` were
approximately -0.291, +0.050, and +0.067. These are selection-motivating
development observations, not validation results, significance tests, or
evidence for a physical Duffing mechanism.

## Hypotheses and observation contract

The linear observed-scale hypothesis is

`yhat[k] = a1*yhat[k-1] + a2*yhat[k-2] + b*u[k-1] + c`.

The competing hypothesis adds exactly one term
`d * s_y * feature(k-1) / s_feature`, where Qwen chooses `feature` from
`{yhat[k-1]^3, u[k-1]^3, u[k-1]*yhat[k-1]}`. `s_y` is the RMS of the 824
training targets and `s_feature` is the RMS of that feature using observed
training lags at `k-1 = 49..254` for predicted steps `k = 50..255` in each
of four windows. Pooled RMS is `sqrt(sum(value^2)/824)` with no per-window
reweighting. Both scalers are positive, training-only, computed once, and
persisted. Within each episode, `yhat[0:50]` are the measured initializer;
every later `yhat` uses the model's own preceding predictions and the
recorded input. Reject missing, wrong-length, nonfinite, or trajectories with
`max(abs(yhat)) > max(1.0, 10 * max(abs(training observed y)))` as simulator
failures; compute the bound once from the four training windows and never
clip predictions. The observation map is
identity on measured output. Unmodeled sensor noise and state ambiguity are
limitations. No physical latent-state interpretation is licensed.

Parameter support is the fixed independent product-uniform box
`a1∈[1.35,1.57]`, `a2∈[-1.05,-0.82]`, `b∈[0.30,0.54]`,
`c∈[-0.005,0.001]`; the nonlinear coefficient has `d∈[-0.10,0.10]`.
These are empirically anchored search supports chosen from training-only
exploration. The resulting ABC particles must not be called a calibrated
Bayesian posterior or used as evidence of model probability. Keep the
linear and nonlinear hypotheses separate, and retain an invalid or failed
nonlinear proposal as an explicit negative outcome.

## Fixed Qwen proposal step

Make at most one serialized chat request to
`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` at the local
`http://127.0.0.1:8000/v1/chat/completions` endpoint. Preflight the exact
ID and use the exact frozen system and user messages below. Do not name the
benchmark or expose validation/final outcomes in the request:

```text
SYSTEM: You propose one compact nonlinear extension to a forced observed-signal recurrence. Use only the training facts provided. Return exactly one JSON object with keys term_id and falsifying_prediction, no markdown or other text. term_id must be one of y_cubed, u_cubed, u_y_product. falsifying_prediction must be one short sentence about an input or amplitude condition under which the proposed term should matter.

USER: A measured signal y responds to a recorded input u at 610.35 samples/second. On 824 training prediction steps in four separated windows, a linear recurrence y[k]=a1*y[k-1]+a2*y[k-2]+b*u[k-1]+c fitted a1=1.4631070, a2=-0.9361293, b=0.4190676, c=-0.00213506. Training one-step residual RMS was 0.00112682 and free-run RMS was 0.00719364. Correlation of one-step residuals with observed-lag features was -0.290589 for y[k-1]^3, +0.0503233 for u[k-1]^3, and +0.0669048 for u[k-1]*y[k-1]. The training input lay roughly in [-0.08,0.10] and the output in [-0.21,0.21]. Choose exactly one feature as the next explicit hypothesis. The allowed term_id mapping is y_cubed -> y[k-1]^3, u_cubed -> u[k-1]^3, u_y_product -> u[k-1]*y[k-1]. Do not infer a named physical mechanism from these observations.
```

The request JSON fixes `model` to the exact Qwen ID, these two messages,
`temperature=0`, `max_tokens=256`, and
`chat_template_kwargs={"enable_thinking":false}`. Hash and persist the
complete request payload before dispatch; impose a 120-second request cap.
Parse the complete assistant content as JSON with exactly the two required
string keys and no extra keys. `term_id` must be one of
`{"y_cubed","u_cubed","u_y_product"}`;
`falsifying_prediction` must be 1–240 Unicode characters and nonblank.
Archive the prediction as **unscored and selection-independent**: it is a
proposed diagnostic for future work, not a test in this pilot. Record the
request payload/hash, raw response or transport failure, timestamp, HTTP
status, returned model ID, finish reason, usage if available, and parser
outcome in ignored atomic receipts. A mismatched model ID, invalid JSON,
truncated finish, timeout, duplicate JSON key, trailing non-JSON content,
or other failure is terminal with no retry, generated code, repair, or
fallback term. It marks the nonlinear path incomplete and blocks sealed
scoring, even if the linear control could otherwise fit.
This is a one-proposal integration test, not a Qwen superiority claim.

## Checked ABC-SMC fit and development decision

Use `core.abc_smc_reference.run_gaussian_abc_smc_reference`, never the
historical `SBIEngine`. For each hypothesis, the simulator returns all four
free-run training suffixes; discrepancy is pooled RMSE over their 824
targets, in measured-output units. Simulator failure has infinite
discrepancy and is counted. The calibration and fit entry point must receive
only the tuple of training windows, training scalers, and frozen protocol;
it must not receive the broader development object or validation target.
Persist fitting receipts before the separate selection step reads
validation targets. Use independent fixed RNG streams per
hypothesis, and persist their numeric seeds. Before fitting, draw exactly
256 uniform-prior calibration parameters per hypothesis using separate fixed
seeds `26092501` (linear) and `26092502` (nonlinear), simulate on training
windows only, and set that hypothesis's two epsilon thresholds to the 50th
and 20th empirical quantiles of its finite training discrepancies using
NumPy's `method="linear"`. Record all failed
calibration draws. If fewer than 128 finite draws occur, mark that
hypothesis incomplete; do not alter its support, thresholds, or calibration
budget after observing that outcome.

Each ABC population targets 64 accepted particles with at most 512 proposal
attempts. Use ABC RNG seeds `26092511` (linear) and `26092512`
(nonlinear), with explicit reference-kernel settings `covariance_scale=2.0`,
`lambda_noise=0.01`, and `nugget=1e-9`. The schedule is the two calibrated
non-increasing thresholds;
retain exact support rejection, proposal kernel, normalized weights, ESS,
attempt/failure counts, and incomplete status from the checked reference.
Use at most one local fit process at a time; cap the complete pilot at 600 s
wall time and 2 GiB additional fit-process memory, and write atomic
per-stage receipts with a run ID. No previous population may silently stand
in for an incomplete final population. The calibration and ABC budget are
the same for linear and nonlinear hypotheses; different parameter dimensions
and thresholds must be disclosed. Comparing their ABC weights as Bayes
factors is forbidden.

For each complete final population, simulate every parameter particle on the
fixed validation input with only its first 50 measured outputs as
initializer. At **each predicted timepoint**, sort the 64 trajectory values
and take the first value whose cumulative normalized particle weight is at
least 0.5 (stable sort for ties). This pointwise weighted median is the
forecast; do not first take a median parameter vector. Use precisely the
same operation after lock on the final test. If any particle simulation
fails, is nonfinite, has the wrong length, or breaches the fixed divergence
bound, record an incomplete forecast and unresolved decision with no sealed
score; do not drop or renormalize particles. The selection metric is
free-run RMSE across all 1,742 suffix samples. Compute the simple
last-initialization-output persistence forecast on the same suffix. If either
ABC fit is incomplete, either model's forecast fails, validation RMSE is
nonfinite, or both model forecasts fail to beat persistence,
record unresolved and do not access sealed outputs. Otherwise, promote the
nonlinear hypothesis only if its validation RMSE is at least 5% below the
linear RMSE. Retain the linear hypothesis in all other cases and preserve
both development receipts. The 5% margin is an arbitrary practical pilot
threshold fixed before validation, not a test of statistical significance.
An ordinary least-squares linear fit and its simulation cost may be reported
as a training-only sanity baseline; no matched-compute superiority claim is
allowed from this pilot.

## Locked final gate and claim limits

Freeze the selected model structure, parameter particles and weights,
scalers, initializer rule, simulator code/hash, all RNG and solver settings,
validation decision, source hash, and development receipt. The development
receipt must bind the archive SHA-256, source indices, exact Qwen request and
response hashes and parser outcome, all calibration draws/failures and
quantile thresholds, both ABC population receipts/weights/ESS/counters,
simulation failures, validation trajectories/scores, selection rule and
outcome, code revision/hash, and run ID. Raw arrays may remain in ignored
local artifacts with a compact tracked hash manifest.

An independently reviewed **new multisine-only scorer interface** must
expose only the official `test_multisine` record to this protocol. The
current `load_silverbox_for_scoring()` returns all test records and is too
broad for this gate. Before reading the sealed suffix, the scorer must
verify the frozen development receipt and exact source/code/parameter
hashes. It may then consume the known multisine input and permitted first
50 measured outputs and score one locked weighted-median forecast over the
remaining 21,638 samples. Persist an atomic final receipt with source,
input, target and forecast hashes, exact RMSE, failure status, and link to
the locked development receipt. No final diagnostic may revise the model
or thresholds. If the development decision is unresolved, the scorer must
not run. The full arrow and its subset are excluded to avoid
overlapping-confirmation claims.

The fixed Qwen choice may reflect memorized benchmark knowledge despite
the anonymized prompt. The small related development windows do not establish
uncertainty coverage. A better validation or sealed RMSE would be a bounded
method-control observation on one device, not a new scientific law.

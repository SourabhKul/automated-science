# S0: one bounded training-only tolerance control

Protocol `ct_s0_epsilon_control_v1_20261021`.
Run `ct-s0-epsilon-control-20261021-v1`. Primary freeze before execution;
independent Luna design/command review is required. This new training control
does not restart the completed development run or revive official scoring.

Question: can one added lower-epsilon population reach a known training-fit
region under the same reference kernel and a finite attempt budget? Compare
its training distances and cost with the unchanged matched point optimizer.
No forecast, score, official field, LLM request or new data acquisition.
A complete population meets its threshold by construction; the nontrivial
outcomes are completion/failure, costs and retained sample diagnostics, not
a new demonstration that a stricter threshold lowers admitted distances.

## Fixed inputs and algorithm

Use the same archive/training hash and safe column-limited reader as the
original study. Take only training_u_est/training_y_est, 768 samples. Ignore
returned future inputs. Freeze actual archive/training/source/runtime hashes
before fitting. Require Python 3.14.3, NumPy 2.4.2 and psutil 7.2.2; stop rather
than install or substitute a different runtime. Use one worker, BLAS threads 1.

Use S0 seed 11, the existing uniform prior and bounds, x2_0=5.205,
max_magnitude=1e6, observe-before-update, full trajectory RMSE and exact
original RNG streams: PCG64 with SeedSequence
[20261014,0,0,11,stream], stream 0 for calibration and stream 1 for ABC. Recompute all 256 calibration draws; charge them again.
Use their exact linear q25/q10 as the original two epsilons. Append
`1.25 * saved_common2048_S0_training_rmse`, where the pinned reference is
0.8554666760441879 and the resulting epsilon is 1.0693333450552349. It must
not increase the schedule. This is explicitly training-conditioned and
chosen after development scoring; it cannot supply untouched confirmation.

Call the existing generic Gaussian reference directly: 64 particles per
population, 8,192 proposals per population, covariance_scale=2.0,
lambda_noise=0.01, nugget=1e-9, same prior/support. No clipping, changed kernel,
failed-particle dropping, invented continuation or partial posterior. Rerun
from the start; the API has no continuation state. Save every returned
population, including partial rows and diagnostics on an incomplete result. Encode a
nonfinite partial-population ESS as JSON null, preserving its raw arrays in
NPZ; do not convert bad distances or particles into valid evidence.

The saved original S0/11 prefix costs 256 calibration+320 generation 0+239
generation 1 simulator calls=815. The new population gets 8,192 proposals.
Freeze an ABC simulator-call cap of 9,007, including calibration and failed
calls; out-of-support proposals cost attempts, not simulator calls. Stop
without a point fit if calibration or either original population does not
match the saved prefix, or if a resource stop occurs. Prefix checks cover all
common saved parameter/distance/weight/density/covariance arrays (matching
undefined-value positions) and logical counters; not elapsed times or
unrecorded rejected vectors. Do not claim equality for missing fields.

If that prefix matches and no resource stop occurred, run the existing point
search with allowance equal to actual calibration+ABC simulator calls. Do
this even if the third ABC population is incomplete only from its attempt limit;
report that partial result as failed to reach the declared target, not as a
replacement full population. Use the same point initialization/support/search
and save its arrays and cost. Use deterministic point seed label 0; it is
not an independent random seed. No cross-run callback cache or free repeated
prefix. At most 18,014 new calls and 13,834,752 training steps. Combined
fitting becomes at most 38,634 calls; original 585 forecast replays stay separate.
The refit reserve is untouched. All other incomplete outcomes or an ABC exception skip the point fit and
stop the control. Save the exception/counters, calibration arrays and any
returned population evidence. If the API raises before returning partial
arrays, record those arrays as unavailable; never substitute the old prefix
or invent partial rows. No second control after any failure.

## Minimal operation and evidence

Use a new exclusive ignored run directory under
`artifacts/cascaded-tanks-week-training-control/`. Rebind only the new process's
RUN_DIR and declared cooperative limits in process memory; no original source,
settings file, manifest, receipt or marker may change. Actual overrides must
be saved explicitly. Require 120 s cooperative wall stop from calibration start,
1 GiB sampled process RSS and 1 MiB run outputs. Check resources before callbacks
and between proposals; check expected serialized output size before writes.
Record total command elapsed time separately from the calibration-start
numerical clock. A bounded 768-step callback may finish before stopping;
report actual time and sampling limits, not a continuous external kill promise.
Stop the whole control on resource breach; retain all available arrays/counts.
Pattern ResourceStop cannot return partial evaluation arrays: record that limit.

Use a single direct existing-API command of at most 80 readable lines, including
its new helpers. Do not hide another module, introduce product files, install
dependencies or rebuild monitoring. Reuse existing guards/writers. If the
scientific and failure record cannot fit this small scope, cut the optional
control and close the week. Keep product source at 994 runtime and 299 test lines
unchanged. Save actual command/protocol/input/runtime pins before fitting.
A different Luna reviews exact bytes and independently checks results.

Report complete/incomplete status, each population's epsilon/accepted rows,
minimum and weighted training distances only for complete normalized populations,
ESS and support/proposal/simulator counters, matched point best and actual costs.
Check old prefix equality, new support/weights and all original files unchanged.
No calibrated uncertainty, model probability, forecast improvement, new mechanism
or inference-seed replication follows. The current v1 official stage stays closed.

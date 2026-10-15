# Cascaded Tanks: common training budget control

At 2,048 simulator calls per family, O2 and C2 reach almost the same training
error: 0.785439 and 0.786292 RMSE. S0 reaches 0.855467. These errors are lower
than the old, shorter point searches. Independent replay confirms every
minimum, counter and shared trace prefix. This is training evidence only.
No new forecast or target score was made. The original development choice
stays unresolved, and the official test stays sealed.

| Family | Best at 256 calls | At 512 | At 1,024 | At 2,048 | Improvement over old longest run |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 1.829837 | 1.324156 | 0.855467 | 0.855467 | 5.84% |
| O2 | 1.949528 | 1.578755 | 1.153797 | 0.785439 | 35.05% |
| C2 | 1.945378 | 1.577662 | 1.154683 | 0.786292 | 38.07% |

Each entry is the minimum training RMSE within the first k callbacks. The
old longest runs used S0=881, O2=934 and C2=831 calls. Each new path matches
its old path exactly through all those calls, for both normalized coordinates
and objectives. All repeated callbacks are charged. These are deterministic
budget controls, not new optimizer seeds or physical replications.

All three searches stopped at their 2,048-call allowance. S0 first reached
its final minimum at call 1,016 and made no later improvement. O2 and C2
first reached theirs at 1,944 and 1,908. All three final fits touch both the
a=0.2 lower bound and c=1.5 upper bound; p and x1_0 remain interior.
The results show that the old short search budgets limited training fit.
They do not establish convergence, a global optimum, a physical parameter
limit, or a benefit from widening support. O2/C2's close training errors also
do not establish equal held-out performance or identify the ceiling mechanism.

## Execution and independent check

The [frozen protocol](cascaded-tanks-training-budget-control-protocol-2026-10-15.md)
kept the original box, simulator, initialization, search rule and 768 training
points. It reused the existing fit and exclusive-save helpers. No product
source, settings, test file or dependency changed. The new run is
`ct-point-budget-control-20261015-v1`, separate from the completed forecast
study. The safe reader did not expose development output targets or official
fields; only training inputs and outputs entered these fits.

The run completed once at 15:07:24.828088–15:07:35.473430Z, exit 0. Each
family used 2,048 callbacks and 1,572,864 simulator steps, with no failures,
support rejects or unused allowance. Total: 6,144 calls and 4,718,592 steps.
Child elapsed time was 10.645304 seconds; supervisor elapsed time was
11.1786 seconds. Ten resource samples found peak worker-tree RSS 52,084,736
bytes. The 120-second, 1-GiB RSS and 1-MiB run-file caps passed. No external
stop or retry occurred. No LLM request or data acquisition was needed.

Cumulative fitting now costs 20,620 calls. Counting the original full stage
and this control conservatively gives 38.735304 seconds. The original 585
forecast replays remain separate and unchanged. The reserved refit budget
was not used. Raw arrays remain ignored. All 96 original study files remained
byte-identical; their aggregate inventory hash is
`f6cf331f3cae9d6d1ba8a6be99069d9115490eb65b1016b9369d0ca0a303a46b`.

A separate Luna recomputed metrics from saved arrays. It matched all new
minima and complete old-prefix comparisons, with no replay errors. It read
only training checkpoints and receipts. The source archive, forecasts and
scores were not reopened for analysis. Compact public copies include the
[summary](cascaded-tanks-training-budget-control-evidence-2026-10-15/summary.json),
[independent replay](cascaded-tanks-training-budget-control-evidence-2026-10-15/independent-replay.json),
[freeze](cascaded-tanks-training-budget-control-evidence-2026-10-15/freeze.json) and
[terminal record](cascaded-tanks-training-budget-control-evidence-2026-10-15/terminal.json).
No raw input/output trajectories or evaluation arrays are published.

The monitor has a recorded limitation. Final progress clears active_model
but leaves active_arm_calls=2,048. The supervisor rejects that combination
and conservatively charges the full 6,144 calls with its exact-count flag
false. The completed child terminal and saved arrays independently establish
exactly 6,144 callbacks. No cost is missing for this run. This does not test
its count recovery after an external kill. The flag was not repaired or rerun.

Freeze SHA-256:
`ee4727d6b2436fd8756a65deb11de94755e65618858b32f02911f319924cd67d`.
Terminal SHA-256:
`94b9a9701426075fb3df1e3400af0334abe758b4f1b7a0c44b863f6cec9fd288`.
Summary SHA-256:
`e6a59d2d65e48a7a3f8760df85ea8a15e1e71f907ac4b7d00a52d5ab3c2f672e`.
Independent replay SHA-256:
`edbd49434b17cd3424348e078930ac6dcce49238bc0b326107ee525673ec2d0c`.
Local evidence is under
`artifacts/cascaded-tanks-week-training-control/ct-point-budget-control-20261015-v1/`
and its sibling `-monitor/` directory. The freeze record pins the unchanged source
and operator command; the ledger records supervisor pins and negative reviews.

## Decision

Do not widen the box or spend the official test to complete the pipeline.
The support warnings remain, and lower training error has not been tested
as a new prediction. Preserve the [original result](cascaded-tanks-week-development-results-2026-10-15.md):
ARX was best, matched point fits beat ABC in all nine pairs, and the ceiling
mechanism stayed unresolved.

The next useful question is whether ABC's broad retained tolerances leave a
large training-fit gap at a common cost. First use its saved calibration and
particle receipts to measure that gap; then decide whether one small tighter-
tolerance control is worth fitting. Keep support and simple controls fixed.
Any such fit needs a separate frozen protocol and all spent calls must count.
A changed method cannot turn the consumed suffix into untouched confirmation.
Refit and official scoring remain separately reviewed stages. A report of
these limits is an acceptable week outcome; no scientific breakthrough,
calibrated uncertainty, model probability or LLM benefit is claimed.

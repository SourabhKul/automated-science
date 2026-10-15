# Cascaded Tanks: common training-search budget control

Protocol `ct_point_budget_control_v1_20261015`.
Run `ct-point-budget-control-20261015-v1`. Primary design before fitting.
Independent Luna design and exact-command review are required before launch.

The saved-receipt audit found all nine point searches still improving when
their paired budgets ended. This control tests a larger common training-call
allowance while keeping support, models, initialization and search unchanged.
It does not test prediction or the physical cause of a ceiling.

## Fixed design

Run one deterministic path for each of S0, O2 and C2, in that order, with at
most 2,048 unique simulator callbacks per family. The recorded seed label is
0, not a random seed or replication. Reuse the existing `_fit_pattern` and
`_save_arm` helpers in a fresh process. Change only the process-local
`runner.RUN_DIR` to the new output directory below. Do not mutate settings,
helper source, original manifest or original run files. No new product module,
CLI, scorer, test file, model proposal or dependency is needed.

Use exactly the original four-parameter box, fixed x2_0/H, magnitude rule,
observe-before-update convention, center/corners, mesh and stop rule. The
safe loader supplies the same 768 training input/output points. It also
returns already-permitted later inputs; do not use them. It does not expose
development suffix targets or official fields. No forecast or score call is
authorized. Do not read the completed study's forecasts or score files.

All new callbacks count, including repeated prefixes and failed simulations.
Use shared `total_calls=[0]` and one shared start time. Maximum new calls:
6,144, each at most 768 steps. Charge them to the remaining development budget,
not the reserved refit allowance. Original fitting spent 14,476 calls; the
combined maximum becomes 20,620 of 299,520 development calls. The original
585 forecast replays stay separate and unchanged. There is no free restart
or cross-run callback cache. This is a new budget control, not a continuation
or rewrite of the completed prediction study.

One CPU worker, BLAS threads=1, at most 120 seconds, 1 GiB sampled worker-tree
RSS and 1 MiB new run files. Include the entire numerical fit-stage time in
cumulative development wall time (original stage 28.090 seconds). Stop the
whole control on any resource breach; do not start later families. Preserve
successful prior-family checkpoints and terminal cause/counters. The reused
helper cannot return partial evaluation arrays on ResourceStop: record that
limitation and actual shared calls rather than inventing arrays or retrying.

## Freeze and evidence

Use separate directory
`artifacts/cascaded-tanks-week-training-control/ct-point-budget-control-20261015-v1/`.
It must be absent before an exclusive create. Write a small frozen metadata
record before fitting with run/protocol IDs, output path, current HEAD,
actual source/protocol hashes, settings, interpreter/package versions, source
receipts, arm order, budgets and repeated-prefix cost rule. This is an
experiment record, not a new authority or replay schema. Independently check
the actual record and exact short operator command before starting fits.

Bind the original archive SHA-256
`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`
and training SHA-256
`0eadc6c62e7f4e3dea76311dff037a7fee1c21fda6fac20e9363310dcf900d61`.
Reuse and verify the original manifest's code digests for runner, scorer/
settings, controlled reader, simulator, pattern search and imported reference.
Use Python 3.14.3, NumPy 2.4.2 and the existing psutil monitor. Save exact
invocation, timing, costs, resource samples and terminal outcome. Do not use
the original manifest freezer, which is bound to the completed study ID.

Save the returned per-family summary and arrays with existing exclusive
writers. Detailed poll/mesh/restart events are not retained by this helper;
do not claim to have them. Report actual calls and unused allowance if all
starts terminate early. No automatic rerun after any incomplete control.

## Analysis and decision

A different Luna recomputes minima/costs from saved arrays, verifies each new
trace's shared prefix against the old longest trace of its family, and checks
all original files remained unchanged. Compare best-so-far training RMSE
(minimum among the first k callbacks) at 256, 512, 1,024 and the actual final
callback, plus stop reasons, final parameters and
bound flags. Report the new best versus the old longest-run best. No estimate
of a global optimum or physical bound follows from these three paths.

Use the results to decide whether further search is still improving within
the box, whether boundary diagnosis remains necessary, or whether this control
failed. Keep the original development selection unresolved. Any tighter ABC
schedule, changed support, refit or official evaluation needs a new frozen
protocol. Publish only independently reviewed compact summaries and honest
limits through the sole Git owner. Raw traces stay ignored.

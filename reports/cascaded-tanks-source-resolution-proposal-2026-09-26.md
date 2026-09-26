# Cascaded Tanks source resolution and proposed methods control — 2026-09-26

**Status: independently reviewed source interpretation and bounded development
adapter, with one successful estimation-only source smoke; inference protocol
remains a proposal, with no model run.** Track ID:
`cascaded_tanks_methods_v1_20260926`. The objective is a second,
physically independent system for testing whether the fixed local Qwen plus
checked ABC-SMC workflow can produce a complete, honestly evaluated forecast.
It is not a search for a previously unknown overflow mechanism.

## Source identity and the historical blocker

The existing `data/real/cascaded_tanks/manifest.json` records a failed Phase 23
**external-repeat** contract and prohibits an adapter or discovery campaign
under that contract. Keep that record unchanged. Its fixed archive is
`CascadedTanksFiles.zip`, versioned DOI
[`10.4121/12960104.v1`](https://data.4tu.nl/articles/dataset/Cascaded_Tanks_Benchmark_Combining_Soft_and_Hard_Nonlinearities/12960104/1),
7,520,592 compressed bytes, published MD5
`f832776ed1efdca4572338f467b7baec`, local SHA-256
`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`,
and CC BY-SA 4.0. The local manifest reports one 1,024-row estimation pair
`uEst/yEst`, one 1,024-row `uVal/yVal` pair, and a 4-second sample interval.
Those names alone do not establish scientific roles.
An audited, bounded format-only pass found the actual CSV wire header is
`"uEst","uVal","yEst","yVal","Ts",` followed by LF. All 1,024 logical
data rows have five unquoted fields plus an empty sixth field from a trailing
comma. `Ts` is nonempty and equals 4.0 only on the first logical row; it is
empty in the other 1,023 rows. One additional empty physical line follows
the data. The historical manifest lists the logical five-column schema after
CSV normalization; it does not describe this wire grammar. No measurement
values were displayed or used in the format audit; the target-deferred
adapter must pin this layout rather than infer it from normalized headers.
The official portal's 11,676,168-byte "unzipped" figure is the sum of its
three separately listed attachments, not a verified expanded size of this
ZIP's internal members. An operator must measure actual archive-member sizes
before extraction. The existing local SHA-256 above is a historical manifest
claim to recheck against the already acquired bytes, not a fresh hash from
this proposal.

The [official benchmark page](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks)
describes supplied **estimation and test** records. The
[official loader README](https://github.com/MaartenSchoukens/nonlinear_benchmarks)
calls `Cascaded_Tanks()` as `train_val, test`, gives the test a 50-sample
state-initialization window, and says the train record can be split for
development while the test cannot guide modeling. Its
[source at commit `f9fb388`](https://github.com/MaartenSchoukens/nonlinear_benchmarks/blob/f9fb3883086870a27b31917ccde1f78c95d53cb2/nonlinear_benchmarks/benchmarks.py#L113-L136)
maps `uEst/yEst` to `train_val` and `uVal/yVal` to `test`. Thus the historical
manifest's "no *separate* test record" is true only in the stricter sense that
there is no **third** independent repeat beyond these two records. It is not
the official benchmark's interpretation. This proposal does **not** satisfy
the earlier Phase 23 multi-record external-repeat requirement; it creates a
new, explicitly one-test **methods-control** track. There is no claim of
independent replication across tank rigs or operating campaigns.
For this separately named track only, this independently reviewed source
resolution supersedes the historical manifest's unscoped `prohibited` list
**only for a bounded source adapter and methods-control development workflow**.
It does not clear the earlier external-repeat or scientific-discovery gates,
does not authorize a broad daily campaign, and does not make the official
test reusable. Implementation must require this track's explicit protocol ID
instead of silently treating the old manifest's failure status as passing.
Pin official loader version **1.0.1** and the source commit above: older
published loader versions used a four-sample initializer, whereas the current
benchmark convention uses **50**. Do not silently mix those scoring protocols.

The [original benchmark manuscript](https://cris.vub.be/ws/portalfiles/portal/24190624/CascadedTanks_Maarten.pdf)
reports a two-tank pump system, 1,024-point multisine estimation and test
inputs sampled every 4 seconds, an unobserved upper state, an observed lower
level, stochastic overflow, sensor nonlinearity, and unknown initial states.
The output is a capacitive-sensor voltage rather than a calibrated physical
height. These are measurement and identifiability limits, not nuisances to
hide. The benchmark's soft/hard nonlinear behavior is already described;
[prior sequential Monte Carlo work](https://arxiv.org/abs/2210.14684)
also used a two-tank overflow system. A good score here would be methods
evidence only, and any comparison with that work needs careful protocol
matching rather than a novelty assertion.
The [current benchmark baseline paper](https://www.cambridge.org/core/journals/data-centric-engineering/article/benchmarking-for-nonlinear-system-identification-submission-platform-and-baseline-results/BD06B2C87096B4A700EE173A298A0FEC)
reports CT test RMSE in sensor volts, including LTI state-space 0.600, ARX
0.682, polynomial NARX 0.417, LSTM 0.452, and optimized LSTM (OLSTM) 0.427 under its
standardized protocol. Older literature results can use different splits;
baseline comparisons must match source version, split, initializer, free-run
metric, and training-data allowance before asserting superiority.

## Proposed frozen information boundary

Do not use the old manifest's `validation_input/output` labels as authority.
For this track, the only adaptive data are the official estimation record.
Create a single internal development split *within* its 1,024 time points:
training indices `[0,768)` and validation indices `[768,1024)`. The model
may use the training output through index 767 to initialize an autonomous
forecast of the 256-point validation suffix; validation targets must be
deferred until all candidates have completed forecasts. This chronological
split is within one multisine realization and is **not** an independent
replication. It is a provisional choice to review against input periodicity,
state-initialization assumptions, and effective sample size before any run.
If review changes it, issue a new protocol ID before looking at validation
outcomes.
After development chooses one model and all choices are frozen, a separately
specified **refit on all 1,024 official estimation points** may use both the
development training and validation outputs, followed by the single official
test. That is ordinary use of `train_val`, but the refit's calibration,
seeds, ABC budget, posterior/ensemble rule, and failure gates must be frozen
before it starts; it cannot reopen model selection. Compare official baselines
only when their allowance to train on all 1,024 points is matched or the
difference is stated. A model selected using the last 256 points cannot call
that same suffix independent validation after refitting.

Reserve the official `uVal/yVal` record for **one** locked final attempt.
Before that attempt, no proposal, prompt, calibration, fit, diagnostic,
selection, or operator should parse `yVal` beyond the benchmark-authorized
first 50 initializer points, inspect the test input for model choice, or use
the test score. Prefer to withhold even the test input until the final scorer.
The final scorer must verify the frozen run/provenance/selection receipts,
claim its single-use marker before accessing the test record, load only
`uVal` and the first 50 `yVal` values for initialization, forecast **every**
particle without clipping or dropping failures, then materialize the remaining
target values and compute the declared free-run RMSE excluding the 50-point
initializer. If any particle fails, emit a terminal failed receipt with null
target hash and RMSE, as in the Silverbox control. The
[official reporting rules](https://www.nonlinearbenchmark.org/results-reporting)
require the provided train+validation/test split, free-run simulation, and no
test-driven structure or parameter tuning. This track must not relabel an
adaptive development score as official test performance.

Archive hashing and metadata inspection do not constitute outcome inspection.
The adapter must not numerically materialize held-out `yVal` suffix values in
development. A byte-oriented, column-limited reader or equivalent independently
audited deferred-target path should avoid the earlier Silverbox CSV behavior
that tokenized the unused output column. Synthetic sentinel tests should show
that changing the sealed suffix cannot change any development artifact and
that a failing forecast cannot reach target materialization. Raw data and
bulk receipts stay ignored; the active expanded/downloaded research-data set
must remain below 50 GB. The local manifest already records the archive,
so check its presence/hash and current working-set size before any download;
do not acquire a duplicate.
The archive and official loader make all test values publicly accessible, so
this is a **procedural blind** enforced by interfaces and receipts, not a
secret holdout. Do not use the general-purpose official loader in the
development runner if it materializes the whole test output array.

## Model and inference gates before a run

The Silverbox AR(2), cubic feature, scale, priors, stability margin, split,
and scorer are not transferable tank physics. Use a separate typed two-state
input/state/observation contract with explicit units, initial-state and
sensor assumptions, a robust simulator, and competing mechanisms that can be
distinguished on development data. Treat known Bernoulli outflow and overflow
as published baselines, not Qwen discoveries. Retain unresolved alternatives
when lower-level voltage alone cannot identify the hidden upper state or
overflow pathway. The primary will freeze a bounded hypothesis set, priors,
ABC distance/epsilon schedule, model-selection rule, uncertainty summary,
failure policy, compute cap, matched-compute baselines, and exact Qwen prompt
before generation. Reuse the checked ABC-SMC proposal/weight reference only
with its mathematics intact; test the new simulator/observation mapping on
synthetic known-parameter controls. Any later stability constraint that
changes support requires fresh calibration and a new protocol ID.

Next gates, in order: independently review this source reinterpretation and
split; inspect the already recorded archive without exposing test targets and
verify exact bytes/schema/size; implement and independently test a dedicated
deferred-target adapter; freeze and review the tank-specific inference and
single-Qwen protocol plus runtime/source/code manifest; run one bounded
development experiment; independently audit its receipts; then decide whether
a single locked official test attempt is eligible. No new Qwen, ABC, or final
run occurs on the strength of this proposal alone. The consumed Silverbox
multisine condition remains outside this track.

The first three adapter smoke attempts stopped safely at, respectively, a
legitimate ZIP directory entry, the actual quoted/trailing-comma CSV header,
and the scalar `Ts` column being empty after its first row. Each prompted a
format-only audit, synthetic-fixture repair, and independent code review; none
produced a development object or score. The fourth smoke used the final
reviewed adapter on the pinned archived ZIP and returned 768 training input/
output samples and 256 validation inputs, with no target suffix in its output.
Its archive SHA-256 is the one above; adapter SHA-256 is
`d651c1a149c702342498ecc63d13c47a803e424fb93938225c5f71cf4b949014`,
contract SHA-256 is
`c92dad6bcf731c1171172861ca85fae5d9b69dff5a45783b61f3eeeeda017723`,
and training/forecast-input stage receipts are
`0eadc6c62e7f4e3dea76311dff037a7fee1c21fda6fac20e9363310dcf900d61`
and `dbffbfc30b0b2ae9a0f7b48a850e57f7113909c50bc8fcb6d84a109e7508c486`.
The reviewed synthetic suite passed 39 tests and Ruff. This source smoke is
an adapter-control result, not a parameter fit or final test.

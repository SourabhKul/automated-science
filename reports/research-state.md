# Automated science: current research state

Updated: 2026-09-26T14:07Z wake. This file records observed state, not inferred execution.

## Latest checkpoint: 2026-09-26T14:07Z

The Cascaded Tanks official test targets have not been parsed, materialized,
or scored; the source adapter only skips their interleaved bytes. The last
Silverbox multisine final is a consumed terminal failure with no RMSE. A
fresh Luna preflight found no active research job and the exact required
Qwen ID at the local oMLX endpoint; clean `main == origin/main` at
`288cbc3e543cdb73a02fb82a8c9b16621b6d4956` before this wake's edits.
The old Silverbox PID lock is stale and unheld. Detailed process, source,
resource, and failure evidence is in
`reports/research-ledger/2026-09-26T1407Z.md`.

The primary wrote `reports/cascaded-tanks-inference-design-proposal-2026-09-26.md`
for a distinct discrete-time S0/O2/C2 methods-control design. Independent
Luna review led to a fixed latent-state scale convention, explicit
cross-family identifiability controls, honest surrogate labels, and a
state-only 50-sample test-initializer gate. **This is not a frozen inference
protocol:** numeric priors, epsilon schedule, Qwen prompt, selection/refit,
run ID, and final scorer remain to be specified and independently reviewed.
No tank Qwen request, real-data ABC fit, development target score, or official
test attempt occurred.

One Luna operator built only a synthetic S0/O2/C2 simulator and tests in
`core/real_data/cascaded_tanks_models.py` and
`tests/test_cascaded_tanks_models.py`; the author reported 20 focused tests,
Ruff and formatting passing. A separate Luna reviewer independently
inspected its equations, indexing, failure semantics, and synthetic
no-crossing/crossing/recovery and dry-floor controls, found no blocking
defect, and cleared **synthetic-only** synchronization. The reviewer could
not rerun tests in its own PATH; the passing runtime result is the author's.
An ABC caller-array glue bridge has since been implemented and independently
reviewed against the checked Gaussian reference. Its synthetic one-parameter
control, forced-incomplete control, and seeded replay passed 3 focused tests
in both author and reviewer environments. Its provenance explicitly says
caller-array origin is unverified, and its intended scope is a synthetic
integration control; it does not establish O2/C2 parameter recovery or a
real-data ABC path. No real-data inference path or deferred scorer exists.
Only reviewed compact
work should be synced; the next scientific gate is synthetic known-parameter
and cross-family control, then a genuinely frozen, reviewed real-data
protocol before any run.

Two independent Luna training-only passes found one 27-sample exact 10.0 V
plateau at indices `[150,176]` and one declared high-to-low-input recovery
episode `[178,191]`, with output below 9.5 V at index 185. Later development
**input-only** forcing contains 54/256 values at or below the training-input
Q25. The episode is far from the development boundary; O2/C2 structural
identifiability remains unresolved, and the future protocol must report
pre-target forecast separation before any development outcome is seen.
Counts, formulas, scope, and hashes are in the 14:07Z ledger. No validation
or official test output was inspected.

## Latest checkpoint: 2026-09-26T1100Z

The reviewed Cascaded Tanks adapter, tests, and compact source evidence were
pushed non-force as content commit
`493e33bec0c31a33b9748d1c747dcaa742cbbd11` on `main`; a fresh fetch
confirmed `HEAD == origin/main` and a clean worktree immediately afterward.
No raw data, receipts, artifacts, or secrets were staged. The source adapter
milestone is complete; no Qwen, ABC, validation score, or official final test
has run on Cascaded Tanks.

Two Luna operators independently reproduced development-only descriptions
from the adapter's training `uEst/yEst` and later `uEst` inputs. Training
sensor output ranged 2.9116–10.0 V; a ceiling at 10 V is only a hypothesis.
The later development input minimum, 0.40937 V, is below the training-input
minimum, 1.2099 V; 228/256 later inputs fall within the training-input range,
and the two highest of ten training-amplitude bins are absent. Training-only
overlap correlation peaks near 112 seconds but does not identify a pure delay
under strong serial correlation. No held-out outputs or official test fields
were used. Exact formulas, numbers, limits, and hashes are in
`reports/research-ledger/2026-09-26T0906Z.md`. The pending step is a separately
frozen tank-specific model/inference/validation/refit protocol with independent
review before any generation or fit. The consumed Silverbox multisine
condition remains unavailable for confirmation.

## Latest checkpoint: 2026-09-26T1042Z source-adapter gate

The independently reviewed Cascaded Tanks development adapter is implemented
locally under `core/real_data/cascaded_tanks_controlled.py`. It pins the
official v1 archive and returns only 768 estimation-record training input/
output samples plus 256 later estimation inputs, with development targets and
official test fields deferred. A real source smoke on the previously verified
archive succeeded under adapter SHA-256
`d651c1a149c702342498ecc63d13c47a803e424fb93938225c5f71cf4b949014`
and contract SHA-256
`c92dad6bcf731c1171172861ca85fae5d9b69dff5a45783b61f3eeeeda017723`.
The 39 focused synthetic tests and Ruff passed; a separate Luna reviewer
approved the final source grammar and target-skip behavior. Three earlier
fail-closed real-source smoke attempts exposed ZIP directory, CSV header,
and scalar-`Ts` wire-layout mistakes. They are preserved with code hashes and
precise failure stages in `reports/research-ledger/2026-09-26T0906Z.md`.
No Qwen, ABC, validation score, or official final test has run on Cascaded
Tanks. Its official test is a single public record and remains a procedural
blind; the older multi-repeat gate is still failed. Next is Git sync of
reviewed code/evidence, then a separately frozen and independently reviewed
tank-specific inference/proposal/refit protocol. The Silverbox multisine
condition remains consumed.

## Latest checkpoint: 2026-09-26T0906Z

The consumed Silverbox multisine final remains a terminal failed attempt with
no RMSE or mechanism claim. A fresh Luna preflight found no active research
job; oMLX advertised the exact required Qwen model, and pre-edit
`main == origin/main` at `6b7d7ab63c8d00f7765a332fea0c7327908b124c`.
No Qwen generation, ABC run, or final scoring occurred in this wake.

The official Cascaded Tanks loader maps source `uEst/yEst` to `train_val` and
`uVal/yVal` to its one held-out test, with 50 initializer samples. The older
local manifest correctly fails a **stricter multi-record external-repeat**
gate but mislabels the canonical official test role. It remains unchanged.
The primary's separately named one-test methods-control source resolution is
`reports/cascaded-tanks-source-resolution-proposal-2026-09-26.md`, track ID
`cascaded_tanks_methods_v1_20260926`; an independent Luna review cleared the
bounded source-adapter gate only. It does not clear inference or a final test.
The existing 7,520,592-byte archive was found in a dated backup, not this
active checkout; byte hashes match the historical and official records.
Metadata-only ZIP inspection found 7,625,951 expanded member bytes and no
path/size alarm. No data member or test outcome was opened. Total active
research data/artifacts plus the archive are about 69.8 MiB, below 50 GB.

The next work is an independently reviewed, target-deferred development
adapter, followed by a real estimation-only source smoke and a separately
frozen tank-specific Qwen plus ABC-SMC protocol. Published overflow physics
and prior SMC results make Cascaded Tanks a methods control, not a discovery
claim. Precise sources, limits, resource observations, and pending gates are
in `reports/research-ledger/2026-09-26T0906Z.md`. This documentation and
adapter work are local pending independent review/Git synchronization.

## Latest checkpoint: 2026-09-26T0504Z–0526Z

The reviewed V2 implementation was synchronized to `origin/main` as content
commit `b0cbac4885dab84b9a43b8b1f75c68eb52704221`. A later status-only
commit left clean local `HEAD == origin/main` at
`0db503edb9ec33dc1d831b9306cd40959b3de41a` before the run. A distinct
Luna reviewer verified the ignored exact-byte V2 manifest (SHA-256
`b830357ba4225621d84193bf1bea7b788ac426e75d48d54645e2391db397a26f`)
against that revision, eight code hashes, pinned runtime, official archive,
and exact Qwen ID. No competing research job was present. One mistyped CLI
command failed in argparse before the workflow or Qwen request; it is logged
separately and did not create a run.

One corrected V2 development workflow ran as
`silverbox-20260926T050458Z-03859cef3530`. The exact Qwen proposed `y_cubed`.
Both hypotheses completed two 64-particle ABC populations with the frozen
2,048-attempt cap. The second populations used 553 linear and 773 nonlinear
attempts; terminal ESS was 55.6702 and 56.9518. Validation RMSE was
0.0193194470 linear, 0.0182095781 nonlinear, and 0.0601643587 persistence.
The nonlinear family cleared the frozen 5% promotion rule by 5.7448% and was
selected. Independent Luna audits found no blocking receipt, reference-kernel
arithmetic, or selection discrepancy. These are development scores from one
block/device and do not establish a new physical mechanism.

After separate pre-final receipt audits, one locked multisine-only scorer
invocation produced a **terminal failure**, not a score: three selected
nonlinear particles (zero-based 25, 44, 53) generated nonfinite trajectories
under the held-out multisine input. The scorer did not drop particles or
compute a weighted-median forecast; final RMSE, forecast hash, and target hash
are null. Independent receipt audits confirmed one final-start marker and one
terminal receipt, both content-hash valid, and no second attempt. The scorer
returned before numeric target-suffix materialization, though its CSV reader
may have transiently tokenized suffix-output strings while scanning inputs.
The multisine condition is now consumed for confirmation. Do not retry or
tune against it. This result supports a robustness failure of the locked
ensemble, **not** a comparative predictive score or publishable discovery.
The precise evidence and limits are in
`reports/research-ledger/2026-09-26T0405Z.md` and
`reports/silverbox-v2-development-and-locked-final-2026-09-26.md`. The new
compact reports were synchronized as content commit
`1910e6ea856acb3368ded87109ac5b251e4f1f14` on `main`; raw archive and
receipts remain ignored.

The next step is development-only generic stability diagnostics, followed by
an independently reviewed, separately frozen protocol on a new confirmation
condition. The post-failure review found 69/256 nonlinear calibration and 60
nonlinear ABC simulation failures, while all 64 selected particles forecast
the development validation segment. Candidate diagnostics and their
prior/ABC-target implications are recorded in the 2026-09-26 ledger; none
has been implemented or used for a new run. Cascaded Tanks is an
independent-system candidate, pending official source/split/size audit. The
five-hour thread heartbeat is ACTIVE; it does not authorize adapting to or
reusing the consumed Silverbox multisine input.

## Latest checkpoint: 2026-09-26T0405Z wake

The first real Silverbox development run remains unresolved: both final ABC
populations exhausted the frozen v1 512-attempt cap and no validation metric or
sealed score was produced. Luna's fresh read-only preflight found no competing
research job, a clean local checkout equal to `origin/main` at
`0bc1c80cb354babfcfaf1a7fe7bfcc27b9c31b92` before v2 edits, ample local
memory/disk, the unchanged official archive digest, and HTTP 200 from oMLX
advertising the exact fixed Qwen ID. The stale v1 lock-file path is not an
active process or OS lock. Details are in
`reports/research-ledger/2026-09-26T0405Z.md`.

An opt-in v2 implementation for proposal ID
`silverbox_first_fit_v2_2048_20260925` and 2,048 ABC attempts/population was
developed locally. It carries a caller-frozen manifest of reviewed Git
revision, clean tree, archive, code hashes, and runtime through the fit,
selection, terminal, and multisine-only final gates. V1 retains its 512 cap
and historical receipts; no old result is relabeled. Independent read-only
review found a pre-Qwen archive rehash gap; the author fixed it and a synthetic
archive-mutation test now blocks proposal dispatch. Reviewers approved the
result after 114 focused runner/scorer tests and 113 inference/protocol tests,
respectively. Those are control tests, not evidence that v2 inference is
correct or that it will complete on real data. No v2 Qwen request, ABC fit,
validation selection, or sealed score has run. The reviewed code and compact
report were synchronized as content commit
`b0cbac4885dab84b9a43b8b1f75c68eb52704221` on `main`; a later status-only
commit recorded this synchronization, and a fresh fetch verified that local
`HEAD` and `origin/main` matched with a clean worktree. Only after that can
the exact run manifest be frozen and one v2 pilot executed. The proposed protocol is
`reports/silverbox-second-development-protocol-proposal-2026-09-25.md`.

## Latest checkpoint: 2026-09-25T1802Z wake

The reviewed Silverbox development bridge and multisine-only final scorer were
implemented locally. The bridge's first independent review caught a broad
loader that parsed sealed multisine test outputs during development; Luna
replaced it with a bounded loader and an independent reviewer approved its
source, receipt, and deadline gates after 61 offline tests. The scorer's
source/proposal gates and later terminal-run gate were independently approved
after 23 synthetic tests. The final scorer has **not** been run on real data.
The reviewed package was synchronized by the one-owner Git workflow as
content commit `3928793d286b9fc1f8a176f0e6779d19b75becae` on `main`.

One real development pilot ran under the frozen first-fit protocol as run
`silverbox-20260925T191245Z-7e7d6af42abe`. The exact local Qwen model was
available and returned one successful `y_cubed` proposal. Both train-only
calibrations and first ABC populations completed, but the fixed 512-attempt
second populations reached only 58/64 linear and 42/64 nonlinear particles.
The run ended `unresolved/fit_incomplete` in 12.857 seconds, below the 600 s
and 2 GiB caps. There was no selection receipt or sealed score; the
incomplete populations were not promoted. A Luna read-only audit verified the
source/Qwen/fit receipt hashes, counts, terminal process absence, and no
selection or final receipt. The auditor had authored the first-fit module,
so that implementation check is coupled. Details and evidence paths are in
`reports/research-ledger/2026-09-25T1802Z.md`.

The audit found a narrower boundary weakness: the fit API received only train
windows and made no validation score, but the controlled loader had already
materialized validation target values in memory before the fit. Thus the
run's `validation_accessed:false` flags mean no validation selection or
scoring, not that values were never loaded. Luna repaired the controlled
development path to retain only validation input and its permitted 50-output
initializer until a complete frozen fit is verified. The selector forecasts
all particles before loading the target suffix; eager validation objects are
rejected. An independent reviewer found and the author fixed incomplete
receipt checks and an ABC epsilon-rejection counter error in this gate; the
final read-only review approved it after 45 synthetic tests. This repair has
not been exercised in a new real fit. No new fit or sealed evaluation should
start until a revised attempt/proposal protocol is frozen from training-only
diagnostics. The proposed version is
`reports/silverbox-second-development-protocol-proposal-2026-09-25.md`,
methodologically reviewed as a proposal, not yet implemented or run. A
512-attempt cap was the immediate limiting factor; no scientific discovery
is claimed. The content commit above was pushed without force to
`origin/main`. A later status-only commit recorded this synchronization; a
fresh fetch verified that local `HEAD` and `origin/main` matched on `main`,
with a clean worktree.

## Latest checkpoint: 2026-09-25T1301Z

The reviewed Phase 1 controlled development boundary is synchronized to
`origin/main` at `a8b43477e82df7c323fc30b109a0d1a8a38e6215` with a clean
worktree at the start of this wake. The local oMLX service is running; a
read-only preflight again advertised the exact fixed Qwen model. There was
no active research job or recent checkpoint to resume.

Using only the four fixed training windows, a Luna operator measured a
forced observed-output linear AR(2) free-run RMS of 0.00719364 and a
training-only cubic extension RMS of 0.00482064. These are exploratory
development fits, not validation or independent evidence. The primary agent
proposed `reports/silverbox-first-fit-protocol-2026-09-25.md` for one
constrained Qwen proposal and checked ABC-SMC fit. Independent methodological
review approved the revised protocol. The bounded Qwen proposer and
train-only checked-ABC development fit/selection components were implemented
and independently approved after 29 focused tests passed. The proposer now
preflights the exact model and returns a receipted term choice; the fit
requires that receipt hash. A reviewed orchestration bridge must still
validate the successful proposer receipt and its term match before the first
real fit. A separate multisine-only scorer remains a later gate. No Qwen
generation, ABC run, validation selection, or sealed evaluation has occurred.
The fit's hard memory guard uses `psutil`; that optional runtime package is
absent from `requirements-freeze.txt` and the default checkout environment.
The focused suite passes when `psutil` is supplied in an ephemeral test
environment, while without it the fit correctly stops as incomplete before
calibration. A reviewed environment/dependency integration must ensure
`psutil` is available before any real pilot. No live Qwen request, ABC fit,
validation score, or sealed access has occurred.
See `reports/research-ledger/2026-09-25T1301Z.md`.

## Synchronization verification: 2026-09-25 13:53 UTC

The approved seven-file protocol and development implementation package was
pushed as content commit
`973bab4f30c094aa40fa74d69f140182bc7dd486`. A fresh fetch confirmed that
exact commit on `origin/main`; the working tree was clean after the push.
The focused proposer/first-fit suite passed 29 tests in an ephemeral `uv`
environment with `pytest`, `numpy`, and `psutil`; scoped Python compilation
and staged whitespace checks passed. The `psutil` availability requirement
and the absence of a run are recorded above and in the ledger. No Qwen or ABC
run, validation scoring, or sealed data access occurred during synchronization.

## Latest checkpoint: 2026-09-25T0808Z

The Silverbox controlled-development boundary and its focused tests were
independently reviewed. At synchronization start, local `main` and fetched
`origin/main` matched at `191a25d7daa7c2af7fe2679a16cf516aedf2480f`; no other
Git synchronization or research job was active. The exact focused command
`uv run --with pytest --with numpy python -m pytest -q
tests/test_silverbox_controlled.py` passed all five tests. The tests use a
temporary index-coded archive. No raw dataset, Qwen generation, ABC fit,
candidate fit, or sealed evaluation was used in this check.

The Phase 1 interface fixes the source ranges, training windows, initialization
length, measured input, and native sampling interval. Its simulator call omits
the target arrays and reports simulator exceptions, malformed shapes, or
nonfinite predictions as failures. The validation `target_y` remains
accessible on the validation-series object for candidate selection, so this
boundary does not enforce a policy preventing validation metrics from setting
fit parameters or ABC thresholds. That fit/threshold policy and its audit
receipt remain future pre-fit work. No end-to-end leakage-proof claim, real-data
inference result, or scientific discovery is supported.

The canonical runner still uses the autonomous legacy `SBIEngine`; it has not
been connected to the controlled Silverbox interface or checked ABC-SMC
reference. A reviewed controlled fit/selection/scorer contract is still needed
before any inference run. The model family, priors, discrepancy, ABC schedule,
and selection margin remain unfrozen.

## Prior checkpoint: 2026-09-25T0301Z

The independently reviewed Silverbox source contract, 2026-09-24 next-milestone
plan, and active GPT-6 Luna delegation policy were pushed in content commit
`35831429ea8c1e23d6c7804e333223fdcc71dc8e`, followed by whitespace
cleanup `191a25d7daa7c2af7fe2679a16cf516aedf2480f`. At this wakeup, local
`HEAD` matched `origin/main` at the latter commit and the worktree was clean.
Raw data remain ignored. The seven focused Silverbox tests and archive hashes
were verified by independent review before synchronization.

The installed oMLX service was stopped; one bounded restart restored it. A
subsequent read-only `/v1/models` request returned HTTP 200 and advertised the
exact `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` ID. No Qwen generation,
ABC fit, or sealed evaluation was run. The integration audit established that
the canonical trajectory runner is autonomous and still uses legacy
`SBIEngine`; the checked ABC-SMC reference and controlled Silverbox input
cannot pass through it as-is. At that checkpoint, a dedicated controlled-system
development and scorer route was the next implementation gate. No real-data
inference claim was supported. Development-only characterization and the first controlled
data-boundary implementation scope are recorded in
`reports/silverbox-controlled-development-contract-plan-2026-09-25.md` and
`reports/research-ledger/2026-09-25T0301Z.md`. The model family and ABC
protocol are not frozen; no fit or generation has been run.

## Latest checkpoint: 2026-09-24T2250Z

The independent final review approved the Silverbox source adapter, focused
tests, manifest, published split contract, archive hashes, and development vs
scoring boundary. The reviewer reported all seven focused tests passing. The
dataset license and reuse terms remain unstated; this permits internal source
auditing but leaves redistribution and public benchmark claims unresolved.

A read-only runtime snapshot at 2026-09-24 22:50 UTC could not connect to
`http://127.0.0.1:8000/v1/models`, so the exact fixed Qwen model ID could not
be confirmed as available. The process scan found no active ABC, Silverbox,
science, oMLX, Qwen, fitting, proposal, experiment, or research jobs. No
fitting, ABC-SMC, Qwen generation, or sealed evaluation was launched during
this source synchronization. The endpoint status blocks Qwen experiments and
does not block review or synchronization of the offline adapter.

Before synchronization, local `HEAD`, local `origin/main`, and remote
`origin/main` all resolved to `4b8822ea8f39e052adf91f3d60eaa9602c634c36`.
The reviewed adapter, metadata, and the planning checkpoint
`reports/real-data-discovery-next-milestone-2026-09-24.md` are the active
bounded scope for the sole Git owner. The five-hour automation is verified
active on this task with `gpt-6-luna` at maximum reasoning. Delegated coding,
review, tests, experiments, monitoring, and Git synchronization use that
allocation; research runs remain fixed to the local Qwen checkpoint in
`AGENTS.md`.

The ordered research gates are in the linked next-milestone plan. First
synchronize the approved offline source contract and record its source-term
and CSV-precision limits. Then verify the local endpoint and exact model ID,
pre-register all development choices, run a bounded real-data integration
pilot, and score the locked result once.

## Latest checkpoint: 2026-09-18T1107Z

The independently approved observation-mismatch and nested-mechanism control
pilot ran exactly once, serially, using the unchanged six predeclared cells
(both experiments with data seeds 11, 29, and 47), in 6.8 seconds under the
540-second cap. Fresh preflight found local `HEAD` and `origin/main` equal at
`029b1e2fb592c774041ce7b30240db2854bce25f`, no competing research workload,
16 CPU cores, 128 GiB RAM, and 447 GiB free disk. A read-only `/v1/models`
request returned HTTP 200 and advertised
`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`; no generation request was
made.

All six cells and model receipts completed. The run used the explicitly
labeled deterministic likelihood-weighted grid reference, not canonical SBI,
BDSS, or Qwen. It preserved normalized grid weights, ESS, counters, seeds,
solver and observation-map configuration, predictive draws, development and
final hashes, alternatives, and unresolved outcomes. Experiment A selected
the correct saturating observation map for seeds 11 and 47 and recorded
`unresolved_retain_both` for seed 29; the unresolved cell did not run a sealed
score. Experiment B recorded `unresolved_retain_simpler` for all three seeds;
seed 47 passed the validation margin but failed sealed reproduction. The fixed
future-time measurement rule returned
`no_informative_measurement_under_this_protocol` in all B cells.

Compact evidence is in
`reports/observation-mismatch-abstention-control-pilot-2026-09-18.md` and
`reports/research-ledger/2026-09-18T1107Z.md`. Raw receipts remain ignored at
`artifacts/observation_mismatch_control/`, with artifact-manifest SHA-256
`1cd0c016cfd4c74ae3f16dcd62c837590e057ab3065370377cd18bbb184dc3f1`. No
rerun, setting change, commit, or push was made after this pilot; root must
inspect the evidence before Git synchronization.

## Latest checkpoint: 2026-09-17T1051Z

The 2026-09-17T1040Z pre-pilot inspection found local `HEAD` and `origin/main` equal at `e6c6008b19628fcb7d57ca4c79b91ae1a5d6f79d`, no competing ODE/ABC/pilot processes or checkpoints, 16 CPU cores, 128 GiB physical memory, and 447 GiB free disk. A read-only HTTP-200 `/v1/models` response advertised the exact fixed Qwen model `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`; no generation request was made. The previous 2026-09-15 workspace-credit failure remains recorded in `reports/research-ledger/2026-09-15T2258Z.md`.

The reviewed ODE-control implementation is present in `core/ode_reference_controls.py` with focused `tests/test_ode_reference_controls.py`. Its dependency-light checks and the existing ABC-reference checks passed before the pilot. Independent review found and the implementation corrected a central-chi-square `x/2` scaling blocker before the pilot; the optional direct SciPy regression matches thresholds 4 and 9 and noncentralities 0, 0.5, 8, and 40. A separately labeled exact Gaussian-likelihood quadrature reference was added, with support-zero checks and tests requiring finite-epsilon and exact means/quantiles to differ. The pre-pilot commands, fixed-step RK4 settings, independent target check, bounded-failure checks, and limitations are recorded in `reports/research-ledger/2026-09-17T1011Z.md` as historical evidence.

The reviewer-required status gate, serial six-cell driver, atomic per-cell/aggregate partial receipts, per-population timing and solver-failure counters, and global wall-deadline propagation were implemented and independently approved before the pilot. A post-pilot audit found that only the finite-target `target_prediction.observation_predictive_quantiles` field omitted target-density weights; the original run/report/hash is preserved and that field is invalidated. The implementation now uses density-weighted Normal-CDF quadrature, with an independent CDF/quantile regression and explicit target-versus-particle latent/observation-noise semantic checks. The reviewer approved exactly one same-config rerun, `20260917T104954Z_corrected_targetcdf`: six predeclared cells, 400 particles/population, epsilon `[3, 2]`, 50,000 attempts/population, and a 540-second cap. It completed 6/6 cells in 34.481593 seconds; stochastic inference receipts match the original exactly, every population reached 400 particles, all solver-failure counters were zero, and no cap or failure occurred. Corrected per-cell finite-epsilon comparisons, separate exact Gaussian references, corrected target predictive quantiles, counters, timing, and hashes are recorded in `reports/research-ledger/2026-09-17T1051Z.md` and `reports/ode-reference-control-pilot-2026-09-17-corrected-targetcdf.md`; corrected raw receipts remain ignored under `artifacts/ode_reference_control/20260917T104954Z_corrected_targetcdf/`. These are known-model control diagnostics for six independently generated datasets, with no coverage or scientific-discovery claim. Root accepted both the scoped implementation and corrected evidence; the designated Git owner is now synchronizing this reviewed scope.

## Latest checkpoint: 2026-09-15T2258Z

The last verified pushed commit is `e6c6008b19628fcb7d57ca4c79b91ae1a5d6f79d`, recording canonical Qwen response provenance. Its reviewed smoke completed two full Qwen responses, zero repairs, and two finite candidate evaluations; the first was selected on validation MSE 1.9346842034702103. Separate frozen final evaluation returned finite MSE 9.470892248597332. See `reports/research-ledger/2026-09-14T0646Z.md` and `reports/qwen-completion-review-2026-09-14.md`. These are bounded synthetic integration results, not calibrated inference or discovery evidence.

The next ODE-control milestone is blocked by Luna workspace credits. The initial implementation worker, its one bounded recovery, and the independent reviewer reported: "Your workspace is out of credits. Ask your workspace owner to refill in order to continue." The recovery performed preflight and began an untracked `core/ode_reference_controls.py`; this partial source has not been independently reviewed or verified by tests. No completed ODE pilot or new inference result has been verified. Preserve this draft and inspect it before resuming; do not treat it as accepted implementation.

The primary scientific specification is `reports/dynamical-inference-control-plan-2026-09-14.md`. The pre-pilot numerical choice is RK4 maximum step 0.05, with absolute solution-error tolerance 1e-6 over t<=5 at prior-boundary and interior rates, to be checked independently before any pilot. The six-cell pilot remains capped at 600 seconds total with explicit partial/incomplete receipts. Resume only once a Luna worker can run; keep the existing code/review ownership policy and do not substitute another research LLM.

Local `HEAD` and `origin/main` were verified equal at `e6c6008` after the failure. This state update and the failure ledger are local documentation pending Luna Git synchronization; the unreviewed source draft must not be pushed as completed work. Earlier checkpoint prose below is historical.

## Fixed direction and authority

- Local research model: Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed on oMLX at http://127.0.0.1:8000/v1.
- The primary assistant owns scientific planning and synthesis. Luna max teams own code, tests, experiment execution/monitoring, and Git synchronization.
- The automation is attached to the existing task every five hours. Local simulation compute and GitHub updates of completed reviewed changes are authorized. Keep raw data, bulk artifacts, environments, and secrets out of Git.
- Git maintenance remains single-owner: after independent review, the designated operator synchronizes only the completed scoped changes, checks the remote state first, and never force-pushes. Raw or bulky experiment artifacts, local environments, and secrets remain excluded.

## Verified evidence

- Initial review and six targeted passing tests are documented in reports/2026-09-07/. They do not establish scientific validity or full end-to-end operation.
- The 2026-09-13 heartbeat's GET /v1/models returned HTTP 200 and listed the exact Qwen ID. No new chat-generation performance claim is made from that listing.
- The initial smoke and Git synchronization workers had terminated with: "Your workspace is out of credits. Ask your workspace owner to refill in order to continue." A single bounded recovery was then launched after confirming no active experiment process.
- The recovery completed with return code 0 in 43.154943 seconds. It executed the configured ecology seed plus two sequential Qwen proposals through the unchanged `scripts.run_domain` and `core.sandbox_eval` paths: 3 evaluator subprocesses, 2 Qwen calls, all finite, no repairs, no timeouts, and no active experiment process remaining afterward.
- The exact evidence bundle is `artifacts/scaffolding_smoke/20260912T174040-0700/`, with local synthetic data, prompts/responses, candidate code, diagnostics, stdout/stderr, timing/status records, and SHA-256 ledger. The permanent result is `reports/scaffolding-smoke-2026-09-12.md`; the reviewed replay harness is `scripts/run_scaffolding_smoke.py`.
- The seed remained incumbent: adaptive held-out MSE 1402.122559 versus 2453.491455 for both proposals. This suffix score is adaptive validation, not independent generalization. No scientific discovery is claimed.
- The oMLX status endpoint still reported exactly one loaded model, `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`; no model load or unload operation was requested. No external or paid compute was used.
- The canonical `trajectory_train_validation_sealed_v1` boundary is implemented behind an explicit opt-in through the existing `scripts.run_domain` and `core.sandbox_eval` paths. It separates train fitting, validation selection, and sealed final evaluation, binds frozen code/parameters/solver/time-origin settings to a development receipt, and records finite or failed final receipts. Focused boundary, SBI, and run-config tests pass (`11 passed`); compileall and diff checks pass.
- The exact-Qwen smoke used a temporary synthetic 60/20/20 trajectory and recorded `chat_template_kwargs={"enable_thinking": false}`. The seed froze with validation MSE `71.36812249334737`; two proposals and two repairs were syntax-invalid. The run made four LLM requests because the pre-fix repair cap was per proposal, exceeding the intended total of two proposals plus one repair. This is recorded as an overrun in `reports/evaluation-boundary-smoke-2026-09-13.md`; no rerun was made after the global-cap fix.
- The separate frozen final evaluator then returned a finite terminal receipt for the seed incumbent (MSE `126.09971677703511`) without a model request, verifying the development receipt and sealed-manifest binding. This is synthetic protocol/runtime evidence only and carries no scientific quality claim.
- A durable ignored copy of the smoke inputs, synthetic-data recipe, run log, receipts, final receipt, and 21-file SHA-256 inventory is preserved at `artifacts/evaluation_boundary/canonical_qwen_boundary_smoke_20260913/` (manifest SHA-256 `1bdf9cd7fc3f2e73efafdecc0e6d72e7733e4e29187ef14e407b8bb47e2c33a5`).
- The 2026-09-14T0145Z wake performed read-only process/model/remote checks, reread the required guidance and audits, launched no new Qwen request, and found only the pre-existing `omlx-server`. The approved boundary scope is being synchronized by the sole Git owner; the pre-sync remote and local `HEAD` both resolved to `0d5c7c6daa01b7490a3ecaa0ea2b1b630b944ef5`.

## Current work

The bounded recovery is complete. Its synthetic-data deviation, exact budgets, and known scorer/sampler limitations are documented in `reports/scaffolding-smoke-2026-09-12.md`; no scientific source implementation was changed. The timestamped raw bundle remains local under the ignored `artifacts/` tree, while the report and replay harness are reviewable project files. The reviewed project files were committed as `f2ee1cc1e0a95ea8c3bd9ac9aacc725c76ae72d8` and verified pushed to `origin/main`.

The scoped evaluation-boundary implementation and smoke evidence are documented in `reports/evaluation-boundary-smoke-2026-09-13.md`, `reports/research-ledger/2026-09-13T2044Z.md`, and `reports/research-ledger/2026-09-14T0145Z.md`. Raw model artifacts and temporary manifests remain outside Git. The canonical global repair cap is now one repair for the whole run, with a focused regression; the real smoke predates that fix and is retained as observed evidence. Parent and independent review approved the final diff, and the designated Git owner synchronized it in content commit `e1b7d40d7caaed17b208424a1e49ee88bd75152e`, verified at `origin/main` with a clean worktree.

The primary planner produced `reports/inference-repair-acceptance-plan-2026-09-13.md`: an explicit proposal/weight contract, finite-epsilon analytic test target, boundary/weight/failure checks, and criteria for evaluating BDSS. The opt-in reference milestone is now implemented separately from the historical runner in `core/abc_smc_reference.py` and `core/abc_reference_controls.py`, with focused checks in `tests/test_abc_reference.py`. It requires an explicit epsilon schedule and finite per-population attempt budgets, resamples an ancestor on every proposal attempt, rejects out-of-support proposals without clipping, uses the exact fixed covariance for both draws and mixture densities, normalizes log-space prior-over-mixture weights, and returns population diagnostics, weights, and ESS. The canonical `core.sbi_engine.SBIEngine` path remains unchanged and is not registered to this reference by default. Independent review approved the scoped milestone, which was pushed to `origin/main` in content commit `88a1d1d6839a88a69170ddbdecd616c1d90f46ee`.

The single-population rejection baseline is recorded in `reports/abc-reference-control-pilot-2026-09-13.md` and `.json`; it completed 18 cells (three seeds for each of `y={0.1,0.5,0.9}` and `epsilon={0.1,0.025}`) in 1.229 seconds with 400 accepted particles and a 50,000-attempt cap per cell. The sequential pilot is recorded separately in `reports/abc-reference-sequential-control-pilot-2026-09-13.md` and `.json`; each cell used the explicit two-population schedule `[0.2, epsilon]`, persisted both populations' counters/weights/covariance receipts, completed all 18 cells in 2.519–2.530 seconds across identical bounded runs, and recorded 68–1,112 out-of-support transition retries per cell with final ESS 386.8–397.5. Independent fixed Gauss-Legendre quadrature supplied the finite-epsilon target; ESS-scaled mean, quantile, and empirical-CDF errors are descriptive Monte Carlo diagnostics, not formal weighted-SMC confidence claims. The read-only model preflight returned HTTP 200 from `/v1/models`, advertised `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`, and made no generation request.

## Next decisions

Follow the ordered gates in `reports/real-data-discovery-next-milestone-2026-09-24.md`: complete the reviewed source sync; check the local oMLX endpoint and exact Qwen ID before any generation; freeze the development split, candidate families, observation model, ABC-SMC schedule, budgets, baselines, failure criteria, and scoring contract; then run the bounded integration pilot and evaluate the locked outcome once. Keep the five-hour automation attached to this task, inspect research processes before future runtime work, and preserve the fixed-Qwen research policy. Keep raw data, local environments, and secrets out of Git.

No scientific breakthrough is claimed by this checkpoint; the smoke integration result and the reviewed `origin/main` synchronization are verified locally and linked above.

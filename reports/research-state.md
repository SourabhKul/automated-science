# Automated science: current research state

Updated: 2026-09-25 after Silverbox Phase 1 boundary review and focused validation. This file records observed state, not inferred execution.

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

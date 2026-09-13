# Automated science: current research state

Updated: 2026-09-13 after the bounded smoke recovery, opt-in ABC reference controls, and sequential scalar pilot. This file records observed state, not inferred execution.

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

## Current work

The bounded recovery is complete. Its synthetic-data deviation, exact budgets, and known scorer/sampler limitations are documented in `reports/scaffolding-smoke-2026-09-12.md`; no scientific source implementation was changed. The timestamped raw bundle remains local under the ignored `artifacts/` tree, while the report and replay harness are reviewable project files. The reviewed project files were committed as `f2ee1cc1e0a95ea8c3bd9ac9aacc725c76ae72d8` and verified pushed to `origin/main`.

The primary planner produced `reports/inference-repair-acceptance-plan-2026-09-13.md`: an explicit proposal/weight contract, finite-epsilon analytic test target, boundary/weight/failure checks, and criteria for evaluating BDSS. The opt-in reference milestone is now implemented separately from the historical runner in `core/abc_smc_reference.py` and `core/abc_reference_controls.py`, with focused checks in `tests/test_abc_reference.py`. It requires an explicit epsilon schedule and finite per-population attempt budgets, resamples an ancestor on every proposal attempt, rejects out-of-support proposals without clipping, uses the exact fixed covariance for both draws and mixture densities, normalizes log-space prior-over-mixture weights, and returns population diagnostics, weights, and ESS. The canonical `core.sbi_engine.SBIEngine` path remains unchanged and is not registered to this reference by default. Independent review approved the scoped milestone, which was pushed to `origin/main` in content commit `88a1d1d6839a88a69170ddbdecd616c1d90f46ee`.

The single-population rejection baseline is recorded in `reports/abc-reference-control-pilot-2026-09-13.md` and `.json`; it completed 18 cells (three seeds for each of `y={0.1,0.5,0.9}` and `epsilon={0.1,0.025}`) in 1.229 seconds with 400 accepted particles and a 50,000-attempt cap per cell. The sequential pilot is recorded separately in `reports/abc-reference-sequential-control-pilot-2026-09-13.md` and `.json`; each cell used the explicit two-population schedule `[0.2, epsilon]`, persisted both populations' counters/weights/covariance receipts, completed all 18 cells in 2.519–2.530 seconds across identical bounded runs, and recorded 68–1,112 out-of-support transition retries per cell with final ESS 386.8–397.5. Independent fixed Gauss-Legendre quadrature supplied the finite-epsilon target; ESS-scaled mean, quantile, and empirical-CDF errors are descriptive Monte Carlo diagnostics, not formal weighted-SMC confidence claims. The read-only model preflight returned HTTP 200 from `/v1/models`, advertised `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`, and made no generation request.

## Next decisions

1. The scoped reference implementation and scalar evidence have independent approval. Decide whether a small ODE control is warranted before treating ABC-SMC outputs as posterior evidence. The scalar controls do not establish calibration of the canonical discovery runner.
2. Correct adaptive-validation/final-test separation and reserve an untouched sealed trajectory, replicate, or intervention condition before scaling discovery runs.
3. Add semantic/canonical-form deduplication as an efficiency diagnostic for algebraically equivalent proposals before paying another full inference budget; treat this as an ablation target, not a claim of general symbolic-equivalence solving.
4. Keep the five-hour heartbeat quiet while state is unchanged; on each wakeup inspect active processes first and preserve any failure evidence without substituting another research model.
5. Keep raw smoke artifacts, local environments, and secrets out of Git; the earlier reviewed smoke report, state, and replay harness remain synchronized on `origin/main` at `f2ee1cc1e0a95ea8c3bd9ac9aacc725c76ae72d8`, and the reviewed reference milestone is synchronized in content commit `88a1d1d6839a88a69170ddbdecd616c1d90f46ee`.

No scientific breakthrough is claimed by this checkpoint; the smoke integration result and the reviewed `origin/main` synchronization are verified locally and linked above.

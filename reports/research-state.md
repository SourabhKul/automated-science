# Automated science: current research state

Updated: 2026-09-13 after the bounded smoke recovery and verified GitHub synchronization. This file records observed state, not inferred execution.

## Fixed direction and authority

- Local research model: Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed on oMLX at http://127.0.0.1:8000/v1.
- The primary assistant owns scientific planning and synthesis. Luna max teams own code, tests, experiment execution/monitoring, and Git synchronization.
- The automation is attached to the existing task every five hours. Local simulation compute and GitHub updates of completed reviewed changes are authorized. Keep raw data, bulk artifacts, environments, and secrets out of Git.

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

The primary planner produced reports/inference-repair-acceptance-plan-2026-09-13.md: an explicit proposal/weight contract, finite-epsilon analytic test target, boundary/weight/failure checks, and criteria for evaluating BDSS. This is a scientific specification, not implemented or tested code.

## Next decisions

1. Use `reports/inference-repair-acceptance-plan-2026-09-13.md` to implement and test the finite-epsilon analytic, parameter-recovery, boundary, weight, and failure checks before treating ABC-SMC outputs as posterior evidence.
2. Correct adaptive-validation/final-test separation and reserve an untouched sealed trajectory, replicate, or intervention condition before scaling discovery runs.
3. Add semantic/canonical-form deduplication as an efficiency diagnostic for algebraically equivalent proposals before paying another full inference budget; treat this as an ablation target, not a claim of general symbolic-equivalence solving.
4. Keep the five-hour heartbeat quiet while state is unchanged; on each wakeup inspect active processes first and preserve any failure evidence without substituting another research model.
5. Keep raw smoke artifacts, local environments, and secrets out of Git; the reviewed report, research-state updates, and replay harness are already synchronized on `origin/main` at `f2ee1cc1e0a95ea8c3bd9ac9aacc725c76ae72d8`.

No scientific breakthrough is claimed by this checkpoint; the smoke integration result and the reviewed `origin/main` synchronization are verified locally and linked above.

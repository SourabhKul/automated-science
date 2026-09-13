# Bounded scaffolding smoke experiment

Requested on 2026-09-12 to create concrete evidence before the next automation cycle. The user subsequently changed the recurring cadence to every five hours; this immediate smoke remains authorized.

## Question

Can the existing local system carry one small synthetic dynamical problem through Qwen proposal generation, code validation, numerical simulation, ABC-SMC-style parameter fitting, scoring, and durable artifact recording?

This tests integration, not discovery, calibrated Bayesian uncertainty, or independent generalization. The previously identified sampler and evaluation limitations remain in scope as limitations, not fixes to slip into this run.

## Design

- One stable synthetic domain, preferring the current ecology configuration.
- Evaluate the seed model and at most two Qwen structural proposals.
- Use only `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` on local oMLX; record the request and any thinking-mode override or repair.
- Fixed seed 7, small explicitly recorded particle counts, at most a few inference generations, one generation request at a time.
- The Luna operator selects feasible exact budgets after checking current interfaces and machine state. Cap the experiment at ten minutes and individual candidates with explicit timeouts. Never inherit the large historical default budgets.
- Prefer unchanged canonical code. If missing data or provider configuration requires a temporary smoke harness, isolate it, record the deviation, and call the actual existing evaluator/inference functions. Do not overwrite existing datasets.
- Existing held-out suffix scores are development/validation diagnostics, not untouched test evidence.

## Evidence required

| Stage | Evidence |
| --- | --- |
| Source and setup | Revision, environment, exact commands/configuration, seed, data generation and hashes |
| Proposal | Actual Qwen request/response, extracted model, validation outcome, repair history |
| Simulation and inference | Real evaluator output, finite/failure status, particle/generation budgets and fitted results |
| Scoring | Seed and candidate scores with metric definitions and comparison scope |
| Completion | Exit status, wall time, logs, artifact paths, actual remaining job status |

A completed candidate may fail scientifically or fail to beat the seed while still demonstrating a functioning scaffold. Rejection or timeout is retained. A mocked test, successful API call, or dry-run alone does not satisfy the end-to-end question.

## Handoff

Save machine-readable artifacts under `artifacts/scaffolding_smoke/`, summarize the observed result in a dated report, and update `reports/research-state.md` so the scheduled automation can resume from evidence. Its next decision should address the earliest observed integration failure, or proceed to evaluator/inference corrections if the scaffold works. Do not scale to a large campaign on the basis of this smoke alone.

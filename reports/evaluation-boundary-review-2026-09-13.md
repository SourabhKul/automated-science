# Evaluation-boundary code review

Review date: 2026-09-13

Verdict: **approved for bounded smoke/readiness**. The canonical train/validation development path and the separate sealed final evaluator satisfy the reviewed information-flow checks. This is an implementation review, not a claim that legacy ABC uncertainty or scientific predictive quality has been resolved.

## Scope

I read `AGENTS.md` and `reports/evaluation-boundary-acceptance-plan-2026-09-13.md`, then reviewed the changed paths in `core/evaluation_boundary.py`, `core/sandbox_eval.py`, `scripts/run_domain.py`, `scripts/run_final_evaluation.py`, the boundary tests, and the Makefile test hook. The review followed the actual data flow from manifest loading through SBI fitting, validation selection, prompts and diagnostics, frozen selection, and final scoring. No source code, model, or data was changed by this review; only this report was added.

The canonical worker now requires a development manifest exposing `train` and `validation` and rejects a manifest exposing `final`. Its SBI engine receives only training observations and times. Validation is scored after fitting and drives canonical selection. The legacy `--held-out` path remains separate and retains its historical adaptive suffix semantics.

The freeze records candidate code and parameter hashes, parameter semantics, the development receipt, split metadata, the declared time origin, and the solver/initial-condition contract. The sealed manifest embeds the exact development receipt. The final command verifies those bindings, validates the frozen code and parameters, integrates from the frozen original time origin, and has no proposal, repair, tuning, or LLM call path.

## Evidence

Focused regression command, using `/tmp/automated_science_boundary_venv/bin/python`:

```text
17 passed
```

The command was:

```text
/tmp/automated_science_boundary_venv/bin/python -m pytest -q \
  tests/test_evaluation_boundary.py tests/test_abc_reference.py tests/test_sbi_engine.py
```

`git diff --check` and `py_compile` passed for the changed Python files.

Independent adversarial checks also passed:

- Replacing only sealed final outcomes with `1e90` left the actual canonical candidate's fit result, train metrics, validation metrics, and development receipt unchanged.
- Changing validation outcomes changed validation MSE and the canonical selection score while leaving train fitting metrics unchanged.
- Canonical diagnostics and prompt context contained only train/validation receipt roles and no final sentinel.
- Seed, candidate, and forced repair calls all forwarded the canonical protocol, the same development manifest, and the bounded 120-second timeout.
- Passing a sealed manifest from a different development protocol was rejected through the embedded development-receipt binding.
- An analytic decay model with a single late final timestamp matched integration from the frozen original `t0/y0` (MSE below `1e-8`); it did not restart at the final timestamp.
- Non-finite final predictions produced a persisted failed receipt with `prediction.finite: false`. Finite values whose metric calculation overflowed produced a persisted failed receipt with null non-finite metrics. The receipt hash covered the added provenance fields, including the final manifest hash.
- Direct receipt construction with a non-finite prediction now forces `status: failed` and records the failure.

## Remaining scope limits

The boundary is an API/input boundary. The manifests and partition files share a filesystem, so this review does not approve an OS-level isolation claim. The caller must keep final paths out of development-worker inputs.

The legacy runner and legacy grouped adapters still have their existing data-loading and adaptive-validation behavior. The narrow canonical integration does not establish that real grouped subject/cell normalization or other adapter-specific initial-state procedures are leakage-safe. No claim is made about those paths here.

Per the task restriction, I did not call the research model endpoint. A fixed-Qwen development-loop run remains separate smoke evidence for the implementing/root agents to collect. The legacy ABC reference uncertainty remains outside this boundary verdict.

## Follow-up: canonical repair budget and fixed-Qwen request forwarding

The follow-up patch moves the canonical repair counter to run scope. The focused regression `test_canonical_repair_budget_is_global` forces both proposals to fail and verifies the canonical sequence is exactly two proposal calls plus one repair call total; the second proposal does not receive a repair attempt. The counter is incremented only when a repair request is actually issued, so failed or skipped repair paths cannot silently reset the run-level budget.

The focused suite now reports **18 passed** with the same bounded command above. An offline `requests.post` capture also confirmed that canonical `call_llm` sends the fixed model and endpoint, the 120-second timeout, and `chat_template_kwargs: {"enable_thinking": false}` in the request payload. The orchestration regression confirms the same template kwargs are forwarded on proposal and repair calls. No live Qwen request or additional smoke run was made for this follow-up.

Follow-up verdict: **pass; no acceptance blocker found**. The previously observed two-proposal/two-repair smoke overrun is covered by the run-level regression and is not evidence against the patched behavior.

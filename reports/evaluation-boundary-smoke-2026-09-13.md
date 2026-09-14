# Canonical evaluation-boundary smoke

Smoke date: 2026-09-13 UTC. This is protocol and runtime evidence on a bounded synthetic trajectory, not a scientific result.

Before the run, `GET http://127.0.0.1:8000/v1/models` returned HTTP 200 and advertised the exact required model `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`. The only pre-existing matching workload was `omlx-server` (PID 47976); no `run_domain`, `sandbox_eval`, or final-evaluator worker was active. The same check after the run and after final evaluation found only that server process.

The development command was:

```text
/tmp/automated_science_boundary_venv/bin/python scripts/run_domain.py ecology \
  --model Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --evaluation-protocol trajectory_train_validation_sealed_v1 \
  --protocol-manifest /tmp/automated_science_boundary_qwen_smoke_20260913/inputs/development_manifest.json \
  --tag-prefix canonical_qwen_boundary_smoke_20260913 \
  --epochs 2 --seed 17 --max-tokens 256 \
  --target-samples 2 --generations 1 --initial-particles 4
```

The manifest contained 18 train, 6 validation, and 6 sealed-final points. Its original time origin was `t0=0.0`; sealed times started at `4.9655172413793105`. Canonical requests recorded `chat_template_kwargs: {"enable_thinking": false}` in the run configuration, proposal receipts, frozen selection, and search receipt. The run used the existing `scripts.run_domain` and `core.sandbox_eval` paths with serialized generation. The observed run window was about 41.9 seconds from run configuration creation (`21:17:06Z`) to the final run log mtime (`21:17:48Z`), below the 600-second overall budget.

The seed completed and remained the frozen incumbent: median distance `95.94964599609375` and development validation MSE `71.36812249334737`. Qwen returned two proposals and two repairs, all rejected before candidate evaluation: proposal/repair 1 had an unterminated string literal (line 22), and proposal/repair 2 had an unclosed `[` (line 18). The `max_tokens=256` cap may have contributed to truncated code; `finish_reason` was not returned or persisted, so this is a possible cause rather than a model diagnosis. This actual run therefore made four LLM requests, exceeding the intended total smoke allowance of two proposals plus one repair. The overrun is retained as evidence; no Qwen rerun was made to erase it. Response token usage was not returned or persisted by the existing `call_llm` path. No proposal candidate evaluator ran, and no final outcome entered search artifacts.

The separately invoked final command was:

```text
/tmp/automated_science_boundary_venv/bin/python scripts/run_final_evaluation.py \
  --selection models/canonical_qwen_boundary_smoke_20260913_ecology/frozen_selection.json \
  --final-manifest /tmp/automated_science_boundary_qwen_smoke_20260913/inputs/sealed_final_manifest.json \
  --domain ecology \
  --output /tmp/automated_science_boundary_qwen_smoke_20260913/final_receipt.json
```

It made no model request and returned a successful terminal receipt. The frozen seed prediction was finite, with final MSE `126.09971677703511` and RMSE `11.229413020146472` over 12 values. The receipt bound the development receipt hash `83ed17a1c144c6d72865b484e92835ee2078c7b5e8665d38e521d035b95c336d`, sealed-manifest hash `c3b6a3195258c5ef627d1d38ddcc91f55728bcc30055801a07deff39066b6939`, and final receipt hash `1f26a5db3fa19322b67267e5dcb779b2962cf96811552955de1b94aead54d6bf`.

Focused validation after the cap fix passed: `pytest` over `tests/test_evaluation_boundary.py`, `tests/test_sbi_engine.py`, and `tests/test_run_domain_config.py` reported `11 passed in 4.02s`; compileall and `git diff --check` also passed. The boundary tests cover sentinel invariance, development-only prompt and fit inputs, canonical seed/candidate/repair forwarding, the global one-repair cap, frozen parameter/time-origin binding, cross-protocol rejection, non-finite prediction and overflow receipts, and the separate final evaluator. The legacy sandbox test still depends on the absent historical `data/ecology_ground_truth.npy`; this smoke does not claim all legacy adapters are repaired.

Raw run logs and ignored model artifacts remain local under `models/canonical_qwen_boundary_smoke_20260913_ecology/`; temporary manifests and the final receipt remain under `/tmp/automated_science_boundary_qwen_smoke_20260913/`. A durable ignored copy, including the synthetic-data recipe, all 21 source-file checksums, run log, model receipts, inputs, and final receipt, is at `artifacts/evaluation_boundary/canonical_qwen_boundary_smoke_20260913/`; its artifact-manifest SHA-256 is `1bdf9cd7fc3f2e73efafdecc0e6d72e7733e4e29187ef14e407b8bb47e2c33a5`.

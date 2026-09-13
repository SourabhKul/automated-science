# Collaboration guidance

- The primary agent owns scientific brainstorming, planning, and synthesis.
- Delegate coding, code reviews, tests, experiment execution, and monitoring to `gpt-5.6-luna` subagents at maximum reasoning effort.
- Delegated work must receive bounded specifications, and the primary agent must inspect the returned evidence.

## Fixed-model research policy (2026-09-12)

- Research runs use only `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` through the local oMLX OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1` (chat completions at `/v1/chat/completions`). Do not develop or run cross-model LLM comparison matrices for this research program.
- Same-Qwen workflow ablations remain allowed. Prompts, inference strategies, priors, proposal kernels, data splits/adapters, diagnostics, seeds, and compute budgets may be compared while the model ID and oMLX endpoint stay fixed; label these as same-Qwen ablations.
- Near-term research focus is data exploration and ABC-SMC-style scientific discovery.
- The Luna maximum-reasoning allocation remains the team policy for delegated coding, code review, tests, experiment execution, and monitoring. It does not authorize changing the fixed research model.

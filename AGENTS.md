# Collaboration guidance

- The primary agent owns scientific brainstorming, planning, and synthesis.
- Delegate coding, code reviews, tests, experiment execution, monitoring, and Git synchronization to `gpt-6-luna` subagents at maximum reasoning effort (active allocation effective 2026-09-24).
- Delegated work must receive bounded specifications, and the primary agent must inspect the returned evidence.

## Fixed-model research policy (user override, 2026-11-02)

- Research runs use exactly `qwen3.8-flash-next-iq3_s` through the existing Strata OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1` (chat completions at `/v1/chat/completions`). No API key is required. Verify the served ID before generation. Do not use aliases, the old 27B/8000 service, cross-model matrices or automatic fallback.
- The primary agent is PI of a persistent autoresearch lab. Researchers propose ideas and independent peers review them; the PI selects, scopes and judges scientific work. Candidate failure triggers a pivot, not program shutdown. Follow `research_lab/charter.md` for the current loop, reviewed stage budgets and progress metrics.
- Same-model workflow and ABC-SMC/BDSS proposal ablations are allowed under a frozen target, checked weighting, reference controls and fair resource accounting. BDSS is an inference strategy, not a scientific mechanism.
- Luna maximum reasoning remains required for coding, code reviews, tests, execution, monitoring and Git. The user explicitly authorizes subagent research proposals and peer review. This does not authorize executing unreviewed local-model output or changing user servers.
- Preserve prior failed studies, user edits and protected holdouts. The restarted agenda supersedes the old closed two-family planning scope; it does not erase history or silently reopen old evaluations.

## Git synchronization policy

- Luna owns keeping GitHub up to date with completed, independently reviewed code, documentation, and compact evidence within the authorized scope.
- Use one Git owner for a synchronization step: inspect the remote first, sync only the reviewed scoped changes, and verify the pushed content commit on `origin/main`.
- Never force-push. Keep raw or bulky experiment artifacts, local environments, and secrets out of Git.

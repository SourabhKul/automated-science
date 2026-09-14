# Canonical Qwen completion provenance review

Review date: 2026-09-14

Verdict: **approved for the bounded 2048-token smoke; no code acceptance blocker found.** This review covers the local mocked call path and canonical orchestration. It does not claim that a live Qwen generation is valid or scientifically useful; the live request remains a separate bounded smoke step for the implementing/root agents.

## Scope and findings

I read `AGENTS.md`, the evaluation-boundary review and acceptance plan, the current `scripts/run_domain.py` diff, the canonical repair path, and `tests/test_llm_response_provenance.py` alongside the existing boundary tests.

The canonical path now passes an opt-in metadata sink through `invoke_llm_with_metadata`, while legacy proposal and repair paths call `call_llm` with their original arguments and continue to receive the original string-or-`None` API. The sink records the HTTP status, success/failure status, `finish_reason`, usage when present, elapsed time, request timestamps, raw response text, SHA-256 of that raw text, and error detail. Malformed JSON, missing choices, non-string content, and empty canonical content become recorded failures instead of escaping as uncaught extraction errors. HTTP failures preserve their status and body/hash.

Canonical proposal and repair receipts are appended immediately after each request and atomically written to `proposal_prompt_receipts.json`. The receipt metadata is therefore associated with the call that produced it, including failed calls; it is not read from a shared “last response” slot. The existing run-level `canonical_repairs_used` counter remains in force, with two proposal slots and at most one repair across the run. The canonical development manifest remains the only input to search, so final outcomes are not added to prompts or response metadata by this patch.

## Evidence

The focused suite passed independently:

```text
/tmp/automated_science_boundary_venv/bin/python -m pytest -q \
  tests/test_llm_response_provenance.py tests/test_evaluation_boundary.py \
  tests/test_abc_reference.py tests/test_sbi_engine.py
23 passed in 4.60s

python3 -m tests.test_run_domain_config
SUCCESS: run-domain family segregation is stable
```

`py_compile` and `git diff --check` also passed.

Additional mocked checks covered the requested failure matrix without contacting the research endpoint:

- A successful response with `finish_reason="length"` preserved the partial content and truncation metadata. A successful response with no `usage` preserved `usage: null` without failing.
- HTTP 429/500 responses, malformed JSON, an empty `choices` list, non-string content, empty content, and a request timeout all returned the legacy failure value and produced status, elapsed, raw body, raw hash, and error metadata where available.
- A canonical two-epoch sequence using `max_tokens=2048` produced exactly `proposal`, `repair_1`, `proposal` (three calls), with per-call response markers and statuses retained in the matching receipts. No second repair was issued.
- In a separate run, I replaced only the sealed final observation file with a `999999` sentinel. The persisted proposal and repair prompts contained no sentinel, while response markers `call_index=[0,1,2]` remained correctly associated with their three receipts. This checks both the final-outcome boundary and per-request persistence.

The repository-wide `pytest -q` collection is not a useful gate in this environment: unrelated baseline modules require unavailable `sklearn`, and `tests/test_qwen.py` performs an unguarded live request to the absent legacy `localhost:1234` endpoint. Those collection errors are outside this patch. No live Qwen request was made during this review, per the bounded review instruction.

## Post-review bounded smoke

The separately authorized smoke completed under
`artifacts/evaluation_boundary/canonical_qwen_response_provenance_smoke_20260914T0657Z/`.
Its plan records the fixed model and endpoint, `max_tokens=2048`, two proposal
slots, one global repair slot, a 120-second request bound, and a 600-second
parent bound. The run made exactly two proposal requests and no repair request.

Both receipts are HTTP 200 with `finish_reason="stop"`, usage present (320 and
354 completion tokens), elapsed times of about 14.42 and 15.28 seconds, and
raw-body SHA-256 checks that recompute successfully. The first proposal was
accepted on development validation MSE `1.934684`; the second was rejected at
`4602.476561`. The search receipt retains
`final_outcomes_available_to_search=false` and `final_data_hash=null`, and the
persisted prompts contain no sealed-outcome sentinel. This is successful
transport and bounded-orchestration evidence only, not a scientific quality or
generalization claim.

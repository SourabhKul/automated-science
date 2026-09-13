# Runtime audit: local oMLX endpoint

Date: 2026-09-07 13:17–13:21 PDT  
Repository revision: `6ff92f2` (`Clean up repository and improve documentation`)

## Scope

This bounded audit checked the user-provided local OpenAI-compatible endpoint,
the requested model ID, the repository's existing LLM integration points, and
targeted tests. It did not load or unload any model, launch a scientific
experiment, change scientific code, commit, or push.

## Endpoint evidence

`GET http://127.0.0.1:8000/v1/models` returned HTTP 200 from `uvicorn` and
advertised the exact requested model ID:

`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`

The response also advertised `mlx-community--Qwen3.5-35B-A3B-4bit` and
`nightmedia--Qwen3.5-122B-A10B-Text-mxfp4-mlx`. The requested model reported
`max_model_len: 262144` and `owned_by: omlx`.

A direct `POST /v1/chat/completions` using the exact model ID, a one-message
prompt, `temperature: 0`, and `max_tokens: 8` returned HTTP 200 in 0.84 s and
the server honored the token limit. It did not return the requested literal
`OK`: because the model's default thinking mode was enabled, the short output
was internal prose and ended with `finish_reason: length`. This proves
transport/model reachability, not code generation readiness.

The oMLX status endpoint `GET /v1/models/status` returned HTTP 200. At audit
time it reported three known models and one resident model: the requested
Qwen model. This resident state was pre-existing and was left untouched as
requested. The repository's LM Studio-style native probe
`GET /api/v1/models` returned HTTP 404, while `GET /admin/api/models` returned
HTTP 401 (admin authentication required).

## Repository integration findings

`scripts/run_domain.py` already accepts `--model` and `--endpoint`; a single
domain can therefore target oMLX with an explicit endpoint such as
`http://127.0.0.1:8000/v1/chat/completions` and the exact model ID. Its default
family IDs and endpoint remain LM Studio values (`localhost:1234`).
The corresponding `ecology --dry-run` resolved the exact oMLX model and
endpoint without creating a run or contacting the server.

`scripts/run_model_matrix.py` discovers models through `/v1/models`, but its
normal path also calls LM Studio native `/api/v1/models/load` and
`/api/v1/models/unload` before and after each model. Those calls are
incompatible with the observed oMLX surface. The smallest matrix-level
adaptation is an explicit provider mode that skips LM Studio residency calls
for oMLX while retaining `/v1/models` discovery and chat completions. Do not
map oMLX to the current `native_base_url()` helper.

The Qwen model reports thinking enabled by default through oMLX. The existing
payload builders (`run_domain.call_llm`, `run_model_matrix.preflight_model`,
and the smoke script) do not set `chat_template_kwargs`. A request carrying
`"chat_template_kwargs": {"enable_thinking": false}` returned a clean
HTTP-200 response in 0.74 s. Applying that field to the matrix preflight
request produced code-shaped output, but the first code response still used
dict-style `args['k']`; the existing single repair request was required before
validation succeeded. The repaired preflight record had `ok: true`,
`finish_reason: stop`, and detail `validated dynamics(t, y, args) and metadata
preflight module`. The smallest generation adaptation is a configurable
per-provider or per-run chat-template kwargs field, defaulting to
`enable_thinking: false` for this code-generation workflow.

## Test evidence

The system Python is `/opt/homebrew/bin/python3` 3.14.3 with `requests` and
NumPy available; JAX, Diffrax, and Pytest were absent from that interpreter.
Compilation and dependency-light tests passed directly:

```text
python3 -m compileall -q core scripts tests                 PASS
python3 -m tests.test_run_domain_config                      PASS
python3 -m tests.test_run_model_matrix_preflight             PASS
python3 -m tests.test_generated_code                         PASS
```

The JAX-dependent tests initially failed only because JAX was absent. A
temporary isolated venv at `/tmp/automated_science_runtime_audit_venv` was
created with `pytest`, `jax==0.9.2`, and `diffrax==0.7.2`; no system-wide
packages were changed. The targeted integration/kernel set then passed:

```text
6 passed in 8.10s
tests/test_run_domain_config.py
tests/test_run_model_matrix_preflight.py
tests/test_generated_code.py
tests/test_sandbox_eval.py
tests/test_sbi_engine.py
```

The exact isolated environment setup and test commands, together with the
sanitized request/response evidence, are captured in
[`runtime-evidence/endpoint-preflight.json`](runtime-evidence/endpoint-preflight.json).
For convenience, the setup and test commands were:

```bash
audit_venv=/tmp/automated_science_runtime_audit_venv
if [ ! -x "$audit_venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$audit_venv"
fi
"$audit_venv/bin/python" -m pip install --disable-pip-version-check --quiet \
  'pytest>=8' 'jax==0.9.2' 'diffrax==0.7.2'
"$audit_venv/bin/python" -m pytest -q \
  tests/test_run_domain_config.py \
  tests/test_run_model_matrix_preflight.py \
  tests/test_generated_code.py \
  tests/test_sandbox_eval.py \
  tests/test_sbi_engine.py
```

The full `make test` suite and an end-to-end ABC-SMC/gauntlet were not run;
they were outside this bounded audit scope.

## Limitations

The endpoint probe did not verify a full candidate evaluation or model-matrix
run. The observed preflight behavior is model- and server-setting-specific;
other resident models may have different chat-template defaults. The report
therefore supports the explicit endpoint/model pairing and the two smallest
integration changes above, but not a claim that the unmodified multi-model
matrix is oMLX-compatible.

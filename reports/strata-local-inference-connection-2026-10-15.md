# Local inference connection templates — 2026-10-15

Endpoint status below was captured at 2026-10-15 09:10:54 UTC with GET calls. This note was shortened at 09:13 UTC. No POST, load, unload, generation, or research-data read was performed for this connection check.

| Endpoint | Exact model ID | GET status |
|---|---|---|
| oMLX `http://127.0.0.1:8000/v1` | `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` | `/health` and `/v1/models` returned 200; healthy, listed, 0 loaded, max length 262144. |
| Strata `http://127.0.0.1:8080/v1` | `qwen3.8-flash-next-iq3_s` | `/health` and `/v1/models` returned 200; listed but unloaded, max context 131072, API key disabled. |

The current tank study remains on the fixed oMLX Qwen ID. The user has separately authorized a future Strata development comparison. Keep it to three fixed tasks, two attempts per task, and both exact model IDs. The hard cap is 12 total HTTP requests, including preflights and repairs. Count those first; do not add separate “OK” smoke calls.

## Manual connection templates

These examples are templates only. They send one synthetic prompt if an operator later runs them. The `jq` check fails unless the response reports the exact requested model. Never use `default` or a model alias.

oMLX chat calls are supported by historical smokes, but this GET-only check did not live-verify its route. Keep this as a template pending the bounded future protocol:

```sh
curl --fail --silent --show-error --connect-timeout 5 --max-time 600 \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed","messages":[{"role":"user","content":"Return exactly OK."}],"max_tokens":8,"temperature":0,"stream":false}' \
  'http://127.0.0.1:8000/v1/chat/completions' \
  | jq -e --arg want 'Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed' '.model == $want'
```

Strata's chat route and request fields are verified in local source and docs, but no request was sent. A chat request to an unloaded model triggers Strata's lazy load, so this template is an operator-initiated load as well as a request:

```sh
curl --fail --silent --show-error --connect-timeout 5 --max-time 600 \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"qwen3.8-flash-next-iq3_s","messages":[{"role":"user","content":"Return exactly OK."}],"max_tokens":8,"temperature":0,"stream":false}' \
  'http://127.0.0.1:8080/v1/chat/completions' \
  | jq -e --arg want 'qwen3.8-flash-next-iq3_s' '.model == $want'
```

Strata can fall back to its configured model for an unrecognized requested ID. Use the exact ID above and stop on a response-ID mismatch. Its launcher defaults to a 900-second idle-unload timer; `STRATA_TTL=0` disables that timer in the script. The active runtime timer was not exposed by GET. No client-triggered load/unload automation is configured; the server's lazy load still applies to a manual chat request.

The JSON connection record is [local_research_llm_endpoints.json](/Users/sourabh/Documents/Codex/automated_science/configs/local_research_llm_endpoints.json). Route references: [Strata server chat route](/Users/sourabh/Strata/serve/server.py:5807), [model selection](/Users/sourabh/Strata/serve/server.py:3197), [API request example](/Users/sourabh/Strata/docs/DETAILS.md:740), and [control-route notes](/Users/sourabh/Strata/docs/DETAILS.md:1975). The launcher settings are in [serve_strata.sh](/Users/sourabh/Strata/serve_strata.sh:6).

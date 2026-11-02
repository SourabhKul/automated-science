# Exact-byte addendum: local ideation client

**Decision:** GO for the bounded two-call ideation stage only. This is not
approval for raw-data access, the ecology metadata gate, fitting, simulation,
or scientific model selection.

## Exact bytes reviewed

| Item | SHA-256 |
|---|---|
| `scripts/run_lab_local_ideation.py` | `e692535f5e553438319dedecf5d7ce537ef4b9a2516b530278187d830dddeec5` |
| `research_lab/protocols/restart-local-ideation-2026-11-02.md` | `1032837c2fd79917290e27865b5252ee0f0cc26048cda287ddb7af9eed7821e8` |
| `research_lab/requests/restart-2026-11-02/ecology.json` | `732a802841892933ed05503e783eb2c4acd22d7f0f3593e957a8472d119b5de4` |
| `research_lab/requests/restart-2026-11-02/hydrology.json` | `effc508d4d4e5733ba24e418fea766d15b81ec7ee92e949fcf731f28fb0690f5` |

The earlier request-pin mismatch is resolved. The current ecology hash matches
the client pin. The requests pin the exact requested model and stage. The
protocol limits the run to two serialized chat attempts, 1,500 output tokens
per request, 120 seconds per request, 300 seconds total, 512 MiB sampled
client RSS, and 8 MiB of retained plus temporary artifacts. The client does
not claim to bound inference-server memory.

## Resolved issues

- The client disables proxies, rejects redirects, and checks the final URL
  against the fixed loopback endpoint.
- It records pre-run bindings and checks protocol, request, and client hashes
  before it marks a successful batch complete.
- Final receipt failure changes the status to failure and returns nonzero.
- Missing or null usage becomes the literal `unknown`.
- Only `finish_reason == "stop"` can count as complete. Length-truncated
  output is retained as a failed attempt, with no retry.
- The ecology prompt now names the known `prey` status field and states that
  the complete arm map is unknown. It does not assume four arms.

One low-probability budget branch remains a minor accounting defect:
`fetch()` returns four values if the total deadline has already expired at
entry, while callers unpack five. The exception handler records a failed
stage and returns nonzero, so it cannot report completion or send another
request. No request was sent on that branch. It is not a blocker for this
bounded ideation stage; record the exact budget reason and `sent=false` if a
later cleanup touches it.

The current requests are compact, use only supplied source facts, and do not
include observations or published numeric estimates. A returned idea is not
source evidence or PI approval. A failed/refused response stops this batch
and leaves the lab open for its next action. No universal posterior
certificate is required for this ideation stage.

No endpoint query, model call, code execution, test, raw-data read, or server
change occurred during this review.

# Static review: first local ideation batch

**Decision:** HOLD before generation.  
**Review scope:** charter, current `AGENTS.md`, ideation protocol, two request
JSON files, and the exact client script. No server query, model call, data
read, code execution, or test run occurred.

## Exact bytes reviewed

| Item | SHA-256 |
|---|---|
| `scripts/run_lab_local_ideation.py` | `a284eb623995a0e3c6a4c64610897e824e0faa0859a6788d43c617ad7caf083f` |
| `research_lab/protocols/restart-local-ideation-2026-11-02.md` | `1032837c2fd79917290e27865b5252ee0f0cc26048cda287ddb7af9eed7821e8` |
| `research_lab/requests/restart-2026-11-02/ecology.json` | `241484f9e4b5ad8140b31caa26eae1f6382afe8770ab7d160b41eba111594e9d` |
| `research_lab/requests/restart-2026-11-02/hydrology.json` | `effc508d4d4e5733ba24e418fea766d15b81ec7ee92e949fcf731f28fb0690f5` |

The script's model and endpoint match the new policy. It requires the exact
served ID from `/models`, uses no API key, and sends no fallback request. The
protocol's two-chat-attempt, 1,500-token, 120-second-per-request, 300-second
total, 512 MiB sampled client RSS, and 8 MiB retained-plus-temporary limits
are below the charter's general ceilings. The receipt correctly says that
the client RSS sample excludes the inference server.

The persistence rule is also sound: a failed finite batch stops that batch
and starts another lab action. It does not end the program. This stage is
ideation only. It does not authorize ecology data access, a source gate, a
fit, or an ABC-SMC run. The existing science review gives the ecology lead
GO only for a separate source/measurement-contract gate; see
[`peer-review.md`](peer-review.md). That gate still needs its required
independent reviews and PI decision.

## Required bounded fixes

1. **Block redirects.** `urlopen` follows redirects by default
   (`scripts/run_lab_local_ideation.py:58`). A local endpoint could redirect
   the prompt to another host. Use an opener that rejects every redirect, and
   confirm the final URL remains the exact `127.0.0.1:8080` endpoint. Treat a
   3xx response as a failed attempt.

2. **Recheck frozen inputs before completion.** The client checks request and
   protocol hashes once (`:91-95`), then never checks them again. Rehash the
   running client, protocol, and both request files after the network work
   and before marking the batch complete. Compare request/protocol hashes to
   their pins and compare the client hash to its start-of-run digest. The
   operator must separately verify that the start-of-run digest equals the
   reviewed client digest in this table. If any binding changes, record a
   failed stage and do not report completion.

3. **Do not return success when the final receipt fails.** The `finally`
   block only prints a receipt-write error (`:133-138`); the in-memory status
   can still be `completed_two_calls`, so the CLI can return zero without a
   completed receipt. Set a failure status and nonzero exit code if final
   receipt publication fails. Do not print a completed status in that case.

4. **Normalize absent usage to unknown.** `result.get("usage", "unknown")`
   returns JSON `null` as Python `None` (`:117`). Map both a missing key and
   `null` to the literal `"unknown"`; retain reported usage when present.

5. **Reject length-truncated output.** The client records `finish_reason` but
   still accepts a schema-valid JSON answer with `finish_reason == "length"`
   (`:115-126`). Mark that response incomplete, count the attempt as
   failed, preserve its bytes, and stop the batch without retry or repair.

## Request correction

Before refreshing the request hash and client pin, add the known `prey` field
to the ecology source facts. The
[Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.3kb72) lists
`prey`, `predator`, and `parasite` status. State that the complete arm map is
unknown and must be checked. Do not let the model assume a four-arm design.
Keep the request's existing limits: no raw rows, no published estimates as
priors, and no blind-discovery claim.

Hydrology's request correctly names the existing basin split and unresolved
`A:<` flag, and it forbids choosing windows from held-out outcomes. The
returned ideas will still need peer review and source checks. An LLM response
cannot fill a missing source fact or approve a scientific gate.

## GO conditions

After the five code fixes and the ecology prompt correction, submit the new
script, protocol, and request hashes for one more static pass. Do not broaden
this client or add retries. Only an updated static review can give GO for the
two-call ideation stage. GO for that stage will not approve the separate
ecology metadata gate or any scientific analysis.

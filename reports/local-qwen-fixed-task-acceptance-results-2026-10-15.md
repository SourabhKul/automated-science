# Local Qwen: structured output worked; semantic checks mostly failed

Both endpoints returned valid JSON with the exact requested model ID on every
first attempt. Independent answer checks found errors in five of the six
cells. The new Strata model can generate structured responses, but this small
check does not establish better scientific reasoning or readiness to replace
the study instrument. The original endpoint also made substantial errors.

The [frozen protocol](local-qwen-fixed-task-acceptance-protocol-2026-10-15.md)
used three synthetic tasks. No research data or sealed target entered either
prompt. All responses were HTTP 200 with finish_reason=stop, exact schema and
correct task tags. These transport/format successes are distinct from the
semantic results below.

| Task | Original oMLX endpoint | New Strata endpoint |
| --- | --- | --- |
| Release arithmetic | Full pass: retained [8,4,2], capped [5,2.5,1.25], distinguishable | Fail: capped [8,4,2] and distinction=false |
| Abstain and propose a probe | Correct abstention; fail retained probe [11,5.5] | Correct abstention; fail retained probe [10,5] |
| Diagnose toy support failure | Correct support action, coefficient 2 and RMSE 0; fail real_mechanism_identified=true | Same: correct support action/coefficient/RMSE; fail real_mechanism_identified=true |

For task 2 the correct retained probe is [10,8]; both models correctly gave
capped [10,5] and said the original constant record identifies neither model.
For task 3 the synthetic equation has exact a=2 under expanded support and
zero error, but it identifies no real physical mechanism. All three tasks
require real_mechanism_identified=false. A complete task passes only if every
frozen requirement passes. Only task 1 on the original endpoint fully passes.
These six cells do not support a general model ranking, statistical superiority
or a statement about real discovery performance.

The operator's terminal complete status means it finished the planned calls.
It does not mean every scientific answer passed. All errors were valid-format
semantic errors, so the frozen rule correctly provided no repair or answer
feedback. The three unused repair slots cannot be used to tune this result.
No generated code was executed and no downstream numerical fit was run.

## Execution, identity and cost

Exact model IDs:

- Original: `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`, oMLX port 8000.
- New: `qwen3.8-flash-next-iq3_s`, Strata port 8080; no aliases were used.

Every request matched the frozen system/task text, temperature 0, output cap
1,024 tokens and enable_thinking=false. The run started at 19:28:47.924720Z,
lasted 31.4962 seconds, and exited 0 before its 21:08:53Z deadline. It used
six POSTs plus the three charged preflight GETs: 9/12 calls, with no repairs,
timeouts or ambiguous execution. No startup script or lifecycle POST ran.
The later generation responses, not a new status GET, establish that both
exact instruments responded during this check.

| Task | Endpoint | Request seconds | Prompt/completion tokens |
| --- | --- | ---: | ---: |
| Release | oMLX | 3.132 | 198/94 |
| Release | Strata | 17.141 | 198/41 |
| Abstain | oMLX | 3.600 | 230/75 |
| Abstain | Strata | 2.888 | 230/96 |
| Failure | oMLX | 2.795 | 254/68 |
| Failure | Strata | 1.884 | 254/42 |

Total usage was 1,364 prompt and 416 completion tokens. First Strata latency
may include a cold load; original residency was not established. The services,
checkpoints and quantization differ. These timings are descriptive observations,
not an isolated speed comparison. This check also differs from the earlier
study's failed 768-token request in task text and token cap; it does not explain
or repair that failure.

Maximum sampled operator RSS was 34,390,016 bytes, sampled only before/after
requests. Saved run files total 20,236 bytes; maximum raw response 873 bytes.
The 1 GiB sampled-operator, 1 MiB saved-output and 32 KiB body caps passed. Those
operator limits do not bound the existing model servers. A post-run process
snapshot observed Strata worker RSS 60,534,358,016 bytes and original service
RSS 16,919,707,648 bytes; these are snapshots, not peak memory measurements.
No research data was acquired. Study source/settings and simulator/forecast
costs remain unchanged. The fixed-model policy remains in effect for research.

## Independent assessment and next step

A different Luna read all raw request/response records and checked arithmetic,
abstention and restraint against the rubric frozen before generation. It also
verified prompt/settings/identity, request numbers 4–9, the three counted GETs,
usage, limits and no repairs. Receipt integrity errors were empty. No additional
endpoint, source-data or target access was needed for that assessment.

Summary SHA-256:
`1cbe614d24e085d8d81205612b3277cb89656bd43599978e899b82431ac8dfcb`.
Invocation SHA-256:
`385b9a9a1a091fb0cf08c67b18fd5f070e622289f373e9610432331e3df46b50`.
Independent assessment SHA-256:
`da0fbf60942321d393976b0c209fed9c6c875c2e97d517691acf8c178af481c8`.
Compact public records are under
[acceptance evidence](local-qwen-fixed-task-acceptance-evidence-2026-10-15/summary.json),
including the [independent assessment](local-qwen-fixed-task-acceptance-evidence-2026-10-15/independent-assessment.json).
Exact one-off operator bytes and held candidates remain in ignored local
evidence. No product source or test module was added.

Keep the new connection available, without claiming a research-model upgrade.
Any future local proposal must be checked by explicit equations and numerical
or analytic tests before it affects a scientific claim. The six-cell result
supports that verification requirement; it does not measure the benefit of a
new verifier or diagnose the source of the models' errors. Further generation
needs a new scoped development study, not an extension of this completed
acceptance run. The present numerical v1 configuration stops before official
scoring as recorded in the [ABC diagnosis](cascaded-tanks-abc-training-receipt-audit-results-2026-10-15.md).

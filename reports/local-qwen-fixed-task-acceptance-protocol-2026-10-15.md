# Local Qwen: fixed-task acceptance check

Protocol `local_qwen_acceptance_v1_20261015`.
Run `local-qwen-acceptance-20261015-v1`. Primary freeze before generation;
independent Luna design review is required. This is a small workflow check,
not a model ranking or a test of scientific discovery. It uses only the
synthetic tasks written below. No research dataset, forecast or sealed target
is an input. This is the one-time acceptance exception expressly authorized
by the user on October 14, outside the fixed-model research fits and model
selection prohibited by AGENTS.md. Do not replace the study instrument from
these six cells alone.

## Instruments and limits

Use exact IDs, never aliases:

- Original: `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`,
  `http://127.0.0.1:8000/v1/chat/completions`.
- New: `qwen3.8-flash-next-iq3_s`,
  `http://127.0.0.1:8080/v1/chat/completions`.

Use the three GET calls already made in the 19:08:53Z preflight: original
models, Strata health, Strata models. Bind the saved response record/hash;
these are the first three calls of this acceptance allowance, not free calls.
Both exact IDs are listed. Make no additional preflight or lifecycle calls
within this run; validate returned model IDs on every generation response. An unloaded Strata model may load through its first
chat request. Do not run the setup script, use lifecycle POSTs, change server
settings, or choose another model. Require the returned top-level model ID to equal the requested ID exactly.
A missing, unparseable or different model ID fails identity validation; do not score it as the named model's answer.

Serialize all generation. Fixed order: task 1 original/new, task 2 original/new,
task 3 original/new. Use temperature 0, max_tokens=1,024, timeout 180 seconds,
`chat_template_kwargs={"enable_thinking":false}`. No provider-specific grammar
or response-format switch. Both receive the same system/task text and JSON
shape. Record unsupported API fields as failures, not silent parameter changes.

At most 12 endpoint calls total, including the three recorded preflights,
transport failures and repairs. This leaves six first generation attempts and
at most three repairs. At most two generation attempts per cell. Run all six
first attempts in the fixed order before any repairs. Then repair eligible
cells in the same order until the three-repair allowance is gone; label any
remaining eligible cell unrepaired_due_to_budget. Stop two hours after the
19:08:53Z preflight (21:08:53Z deadline), even if calls remain. At operator
start convert the remaining UTC allowance to a monotonic deadline; each POST
timeout is min(180 seconds, remaining). The GETs are already complete within
this same wake; do not repeat them or issue an unbounded GET. Persist each numbered
request before POST and its raw response/error afterward. Use one worker,
BLAS threads 1 for the small operator, at most 1 GiB sampled operator RSS and 1 MiB saved
request/response metadata. Cap each raw HTTP body at 32 KiB. The 1 MiB limit is cumulative saved
run bytes. Before each call reserve room for a full 32 KiB response plus
8 KiB metadata; stop before a call/write would exceed the total cap. On an
oversized response save only a bounded prefix and a cap-breach record, mark
it truncated, and stop further calls. Never validate a truncated prefix. Sample the
operator before and after each request; report that sampling limit. Stop
further calls on any cap breach. The server is an existing separate workload; record
its observed process/RSS and state where available. This check is not a claim
that model-server memory is bounded by the operator cap. No simulator calls,
package installation, generated-code execution or new product module is needed.

Use a small one-off request command, target 100 readable lines. Save its exact
bytes/hash locally with prompt/protocol/config/runtime hashes. No new test,
HTTP framework, CLI or authority schema. Save raw responses under ignored
`artifacts/local-qwen-acceptance/local-qwen-acceptance-20261015-v1/`.
Use an exclusive new directory and preserve all partial/failed calls; do not
silently restart the run. Log usage, latency, finish_reason, returned ID,
HTTP/validation status, and original text including any reasoning field.
Do not publish secrets or use client credentials in evidence.

## Common system message and repair

System message, exact text:

> You are checking a small synthetic scientific task. Use only the supplied assumptions. Return one JSON object with exactly the requested keys. Do not include Markdown, code, private reasoning, or an introduction. Do not claim evidence about a real physical system.

Attempt 1 uses the exact task text below. A cell gets attempt 2 only if HTTP,
identity, completion or strict JSON/schema validation failed. Wrong scientific content in
valid JSON is a scored failure, not a reason for another try. For HTTP errors
or ID mismatch, repeat the original request once. For incomplete output or invalid JSON/schema,
append this same sentence to the original user text, without the old response:

> Your previous response was incomplete or failed JSON or schema validation. Return only the requested JSON object with exactly the requested keys and value types.

No answer-specific or semantic feedback. Do not increase tokens or timeouts.
If attempt 2 fails, retain that failure and move to the next cell. Record first-
attempt success and eventual success separately. Never treat a repair as a
new independent task or claim sample-based statistical superiority.

## Fixed user tasks

Task 1 exact text:

> Synthetic release task. Model R updates latent z to 0.5*z+u, retaining that value, and observes min(10,z). Model C updates z to min(10,0.5*z+u) and observes z. Current latent states are R=16 and C=10; both currently observe 10. Apply inputs [0,0,0]. Report the three observations immediately AFTER these updates for each model. Return exactly {"task":"release","retained":[number,number,number],"capped":[number,number,number],"first_release_distinguishes":boolean,"real_mechanism_identified":boolean}.

Task 2 exact text:

> Synthetic identification task. Model R updates latent z to 0.5*z+u, retains it, and observes min(10,z). Model C updates z to min(10,0.5*z+u) and observes z. Both start at z=4. The only measured record is ten constant inputs u=2 with ten observed values y=4, without noise. Does this record identify which model produced it? As a separate proposed probe, start again at z=4, apply [14,0], and calculate observations immediately AFTER each update for both models. Return exactly {"task":"abstain","identified_from_record":boolean,"probe_inputs":[number,number],"retained_probe":[number,number],"capped_probe":[number,number],"real_mechanism_identified":boolean}.

Task 3 exact text:

> Synthetic training failure task. A noiseless toy model y=a*x has training x=[1,2,3] and y=[2,4,6]. Its fitting support is a in [0,1]. An ABC run retains a=[0.2,0.4,0.6], while a point search reaches a=1. Which of these two next actions is informative: "training_support_control" (check the same toy equation under a in [0,3]) or "consume_final_test" (open a reserved final target to choose a)? Give the exact training coefficient under expanded support and its training RMSE. State whether tightening ABC tolerance within [0,1] can attain zero training RMSE. Return exactly {"task":"failure","action":string,"expanded_a":number,"expanded_training_rmse":number,"zero_rmse_with_old_support":boolean,"real_mechanism_identified":boolean}.

The word number in these shapes specifies a JSON numeric type, not a literal
string. Use no additional keys. Strict JSON means json.loads of stripped
content, with finite numbers, exact keys/types and array lengths; no fence
extraction or content repair by the operator. Require finish_reason="stop" to pass. Every other finish_reason is a
retryable completion failure, including a truncated answer whose content
otherwise parses. Do not silently discard these failures.

## Frozen assessment and interpretation

A different Luna checks identity, parsing and arithmetic from raw receipts.
The expected task tags are release, abstain and failure for tasks 1/2/3.
A wrong tag in otherwise valid-schema JSON is a semantic failure, with no
repair. Within numeric tolerance 1e-9, task 1 expects R=[8,4,2], C=[5,2.5,1.25],
first_release_distinguishes=true. Task 2 expects identified_from_record=false,
probe_inputs=[14,0], retained_probe=[10,8], capped_probe=[10,5]. Task 3 expects
action="training_support_control", expanded_a=2, expanded_training_rmse=0,
zero_rmse_with_old_support=false. All real_mechanism_identified flags must
be false. Score exact structure, correct calculation/abstention and claim
restraint separately; a complete task passes only if all checks pass.

Report all six cells, each call/repair, failures, wall/usage and model IDs.
No aggregate ranking, significance, independent replication, real-data fit
benefit or scientific breakthrough follows. No downstream numerical fit is
part of this check: its failure task has an analytic answer. That deliberate
scope cut avoids growing engineering or spending sealed data on model choice.
Any later real proposal/fit benefit needs its own matched-budget study.

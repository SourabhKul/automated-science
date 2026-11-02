# Independent methods review: persistent lab restart

Scope: operational design only. I read `AGENTS.md`, the current entry in
`reports/research-state.md`, the October 31 count decision and ledger, the
October 30 scientific-question screen, and the existing run/receipt APIs. I
did not inspect raw observations, call a model, run code, or browse for new
sources.

## Adversarial findings

1. **Treat the new request as a new active scope.** The latest state closes
   its old bounded planning branch. Keep that record intact. Start a new
   restart state that stays open across failed proposals and experiments.
   An experiment may end as rejected, failed, or resource-limited. That must
   move the lab to `pivot_required`, not to a program-level closed state.

2. **Keep the record file-based and small.** Use one current-state file with
   the active proposal ID, stage, frozen-plan hash, last receipt, status, and
   next action. Keep each proposal, each independent review, and each frozen
   plan as separate immutable files. Write one receipt per run. Build the
   scoreboard from receipts. Add one append-only event line only for a stage
   change, decision, or stop. Do not make a report and ledger entry for every
   unchanged wake. The current `reports/research-state.md` is already a long
   history; do not grow it into a live event stream.

3. **Separate proposal, review, and PI selection.** The proposal author cannot
   review that proposal. A reviewer must see the question, source evidence,
   proposed measurement model, reference, leakage boundary, and resource
   plan, but no held-out outcomes. The reviewer returns explicit blockers and
   GO/NO-GO. Only the root PI selects. Freeze the plan after review and before
   any observation read, fit, or outcome score. Bind proposal, review, plan,
   code revision, and input-manifest hashes in the receipt.

4. **Honor the exact model override.** New research calls must use
   `qwen3.8-flash-next-iq3_s` at `http://127.0.0.1:8080/v1` with no auth.
   Do not silently reuse the old 27B/8000 endpoint. In particular,
   `scripts/run_domain.py` hard-codes the old model and endpoint for its
   canonical protocol, so it is not a valid route for this scope without a
   separately reviewed change. Capture requested and returned model IDs,
   endpoint, prompt/response hashes, usage, finish status, error, and time.
   Count failed calls and timeouts against the call cap. Do not call the model
   during protocol preparation.

5. **Gate data, reference, and leakage before fitting.** Before selection,
   proposals may use source metadata and existing methods documentation only.
   A frozen plan must name the source/version/license/hash, experimental unit,
   measurement process and assumptions, inclusion rules, train/validation/
   sealed-final roles, and who may read each role. The proposed target must
   have a tractable reference with an independently checked implementation,
   declared support, diagnostics, and tolerances. Stop that experiment if any
   gate is unsupported. Never use final outcomes to edit a prompt, prior,
   threshold, model, or split. Preserve B2a, raw inputs, and historical
   holdouts as-is.

6. **Set separate hard budgets.** For the proposal batch, a useful small cap
   is at most three Qwen requests, 2,048 output tokens and 300 seconds per
   request, and 15 minutes total; every attempt consumes budget. No automatic
   repair or unbounded retry. For a selected experiment, freeze numeric
   limits for model calls, simulator/likelihood calls, wall time, peak memory,
   retained artifacts, and transaction peak before launch. Include temporary
   writes and a reserved final-receipt allowance in the artifact cap. Do not
   inherit unused calls or costs from a closed study. A cap breach records a
   resource stop and pivots; it does not erase partial evidence.

7. **Keep scientific and software scores separate.** The scoreboard needs
   the frozen scientific endpoint and uncertainty/diagnostic outcome, plus
   separate engineering fields for validity, failures, calls, wall time,
   memory, and bytes. A software test pass is not scientific success. A good
   fit or predictive score is not proof that the reference or measurement
   model is valid. Do not make a composite score that lets one hide the other.
   Give every proposal and run a new ID; retries must resume the same frozen
   plan and saved state, not turn a weak result into a new run.

## Candidate screen

I do not see a strong ready-to-run data question in the reviewed evidence.
The newest question screen found no eligible pilot. The count lead failed its
measurement contract; the yeast branch is deferred. Phase 73 UCI Wine Quality
has a source-gate record, but its split reserves white wine as an external
file and has no observed fit. A new claim there would need a defensible
experimental unit, rating model, and within-source reference; I would not
use its protected external file or raw rows as a restart shortcut. Treat it
as a proposal-screen candidate only if a new independent review can close
those gaps without touching protected material. For now, let the bounded
proposal cycle find a distinct source-backed question and reject weak ideas
without closing the lab.

## Relevant evidence

- `reports/research-state.md` records the closed October 31 branch and no
  eligible selected pilot.
- `reports/scientific-question-screen-2026-10-30.md` and
  `reports/count-measurement-contract-decision-2026-10-31.md` document the
  screened leads and measurement blockers.
- `scripts/run_domain.py` contains the legacy fixed model/endpoint check and
  `invoke_llm_with_metadata`; `append_prompt_receipt` records hashes, usage,
  timing, and raw response text.
- `core/run_config.py` writes basic run provenance. It does not represent
  proposal review, PI selection, pivot state, or a science/engineering
  scoreboard.
- `data/real/uci_wine_quality/source_file_split.json` identifies the existing
  source/external file roles. I read metadata only.

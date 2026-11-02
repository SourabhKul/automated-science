# Exact-byte review: autoresearch lab ledger

**Decision:** GO for supervised initialization, proposal/review registration,
and status/scorecard use. HOLD on selecting a proposal ID that has already
been consumed by a stage receipt. This review does not approve a scientific
launch. The CLI is a ledger and scorecard, not launch authority.

No CLI command or test ran in this review.

## Exact bytes and current state

| Item | SHA-256 |
|---|---|
| `scripts/autoresearch_lab.py` | `088cad0b27b3e4e248848c30ba52bba5b2adb9dc1e38a9b624a57ab82b25b9e1` |
| `tests/test_autoresearch_lab.py` | `fa44236dfdc8d3885cb2b5f766b85f7e15e3ed7403fbfc8b3e7590ce88c53a75` |
| `research_lab/charter.md` | `a337668efa67c361a5cd5a224e7aa6a16a2de4d0160d64ca83a630262bee1c03` |
| `research_lab/state.json` | `f18491166095fcd43a0076feefb2b1114788bb21253391e8a02fd0d4181f0d5e` |
| `research_lab/scorecard.md` | `0faf3aee2ec484756d17cbdbb2afaeb296c17454b3e6f47810bae6895edb4d91` |
| `research_lab/events.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The initial state binds the current charter hash and exact model/endpoint. It
is open, has no active experiment, and has no registered records. The empty
journal agrees with its zero count and null head. The scorecard's counts and
cost totals match that state. Failed or inconclusive receipts set
`pivot_required`; the program remains open. Source-gate and numerical-control
receipts cannot claim a supported or falsified scientific outcome. Proposal
reviews bind to exact proposal bytes, require two distinct latest GO reviews
with no blockers, match the proposal stage, and reject proposer self-review.
Experiment receipts bind the protocol, code list, and receipt review.

The earlier test API mismatch is fixed in the reviewed test bytes: the tests
use `mkdirs()` and `MODEL`, match current validation messages, and tolerate
the existing protocol directory. This is static inspection only. It is not a
test result.

## One required selection guard

`select()` checks that only one experiment is active, but it does not reject
a proposal ID already present in `selections` (lines 286–307). After a failed
receipt sets `pivot_required`, the same proposal can be selected again if its
old GO reviews remain current. This conflicts with the scorecard's rule that
advancement needs a new proposal and can repeat a failed lead without a new
review.

Before using `select`, reject any proposal ID already in `state["selections"]`.
Require a revised or stage-specific proposal with a new ID and new reviews.
Add one focused test that records a failed receipt and verifies a second
selection of that ID fails while the program remains open. This guard does
not stop the lab; it sends the operator to a new proposal.

## Limits the ledger does not enforce

These checks stay with the PI and the exact external protocol/reviews. Do not
claim that the CLI enforces them:

- It binds the protocol and code files, but not the data/source manifest.
  Each empirical receipt must list the source version and input-manifest
  hashes. The source gate must bind the raw-file hashes or the verified
  metadata receipt that carries them.
- It does not store code-author identity. The code reviewer must be checked
  against the actual code author and executor in the external code review.
- Proposal `budget` only has to be a JSON object. The CLI does not verify
  numeric caps or compare them with actual costs. A frozen protocol and its
  independent review must set the caps. An over-cap run must still be
  recorded as failed or incomplete and must pivot.
- Costs cover registered experiment receipts only. The fields are simulator
  calls, forecast calls, LLM calls, wall time, peak RSS, and artifact bytes.
  The CLI does not count likelihood calls, failed attempts, model tokens, or
  transaction bytes. Keep those in the exact run receipt and report unknowns
  as unknown. The scorecard points to
  `research_lab/restart-status.md` for ideation and software-check costs, but
  that file is absent at this hash. Create it when those costs are recorded,
  or change the link to the actual evidence location.
- The charter limits the active proposal queue to three. This CLI has no
  proposal lifecycle field or queue cap. The PI must manage that queue
  manually until the state model has a small, clear active/terminal status.
- `selected_by` is recorded as `root_pi`; the CLI does not authenticate the
  person who runs the command. The PI must run selection. A CLI selection
  records the decision but cannot authorize a launch.

`load()` safely refuses a broken state/journal hash chain. It has no automatic
recovery for a partial multi-file commit. Keep that as a manual HOLD and
preserve files; no crash-recovery framework is needed for this scope.

## Operating boundary

Use the ledger to preserve proposals, reviews, stage receipts, pivots, and
registered-receipt costs. Static GO covers the bounded test command
`python3 -m unittest discover -s tests -p test_autoresearch_lab.py`. No test
has run, so this is not a pass. If it passes, GO covers supervised
initialization, proposal/review registration, and status use. Do not use
`select` for a proposal ID that already has a receipt.
Before a later scientific stage, obtain the stage-specific reviews and keep
source manifests, measurement gates, code-author separation, model/input
hashes, and hard caps visible in the external protocol and receipt. No source
gate GO is a fit GO.

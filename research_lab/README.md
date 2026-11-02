# Autoresearch lab

Read `charter.md`, `state.json`, `scorecard.md`, `restart-status.md` and the
latest receipts before a cycle. The hourly app heartbeat resumes this chat.
The machine, app and local Strata service must be available for local work.

The PI selects the question and next stage. Luna max agents propose, review,
implement, test, run and check evidence. Use exact local research model
`qwen3.8-flash-next-iq3_s` at `http://127.0.0.1:8080/v1`.

The stdlib CLI is a supervised ledger. From the repository root:

```sh
python3 scripts/autoresearch_lab.py status --format markdown
python3 scripts/autoresearch_lab.py register-proposal path/to/proposal.json
python3 scripts/autoresearch_lab.py register-review path/to/review.json
python3 scripts/autoresearch_lab.py select proposal-id --rationale 'PI decision'
python3 scripts/autoresearch_lab.py record-experiment path/to/reviewed-receipt.json
```

Input paths must stay inside the repository. Registration writes canonical
immutable records; reviews bind their canonical SHA256. Do not edit registered
records. Use a new ID for a revision or stage. Keep exact protocol, source
manifest, reader, target and code hashes in the external stage evidence.

A selection records the PI decision. It is not launch authority. Before an
empirical stage, independently review measurement, split, target, controls,
input hashes, code-author separation and numeric caps. Record failures and
unknown costs. Review actual receipts before accepting a result. The current
CLI does not enforce every scientific, source or budget rule.

Keep bulky/raw files under ignored storage. Preserve stopped runs and all
partials. A state/journal mismatch is a manual HOLD: preserve both files and
review the interrupted transaction. Do not overwrite evidence to force a
restart. Failed candidates lead to a smaller test or another proposal.

Software check:

```sh
python3 -m unittest discover -s tests -p test_autoresearch_lab.py
```

Run checks only when needed by a new change or unresolved issue. Do not
replay consumed research runs to fill a cycle. The first IQ3 batch is closed;
its truncated answer and costs remain recorded, with no retry allowance.

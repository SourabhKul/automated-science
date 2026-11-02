# Static addendum: bounded lab CLI check

**Decision: GO to run the specified bounded unit test command.** If it passes,
GO to supervised ledger bootstrap and registration of the reviewed source
cards. This is a bookkeeping GO only. It does not select a proposal, authorize
data acquisition, or approve a scientific experiment.

## Exact bytes reviewed

| Input | SHA-256 |
|---|---|
| `scripts/autoresearch_lab.py` | `38070425699f6db349ff11c65e9fc859b6883b21698eef2a62e6d85c341fc296` |
| `tests/test_autoresearch_lab.py` | `fd4d6d790f8ec3b93d3718803387354847e30da300d70f872a0702ce08d3ba27` |
| `research_lab/charter.md` | `a337668efa67c361a5cd5a224e7aa6a16a2de4d0160d64ca83a630262bee1c03` |
| `research_lab/state.json` | `f18491166095fcd43a0076feefb2b1114788bb21253391e8a02fd0d4181f0d5e` |
| `research_lab/events.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `research_lab/scorecard.md` | `0faf3aee2ec484756d17cbdbb2afaeb296c17454b3e6f47810bae6895edb4d91` |

The inspected source has the new duplicate-selection guard. It rejects another
selection of a proposal already used and directs the PI to register a new
proposal or stage. The regression test records a failed receipt and asserts
that the same proposal cannot be selected again. Existing checks also cover
proposal and review hashes, self-review, distinct reviewers, receipt binding,
unknown costs, path safety, and continued open status after a failed stage.

The test command may now run:

```sh
python3 -m unittest discover -s tests -p test_autoresearch_lab.py
```

After a passing test, the CLI may be initialized or resumed and used to
register the exact reviewed proposal and review records, then produce status
and scorecard output. Preserve existing state. Do not treat a CLI selection as
launch authority.

## Limits that remain manual

The CLI is a supervised ledger and scorecard. It does not launch work or
automatically enforce source-manifest identity, code-author independence,
protocol-specific numeric caps, or complete costs outside registered
experiment receipts. The exact external protocol and independent reviews must
bind source inputs, code, budgets, and transaction bytes before any stage is
run. A source-gate GO remains limited to that source stage. A later fit needs
its own reviewed proposal, design, code, and receipt.

This was static review only. No imports, tests, CLI commands, model calls, or
experiments were run for this addendum.

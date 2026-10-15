# Cascaded Tanks: search budget and boundary diagnosis

Independent replay reproduced every summary exactly. This audit used saved
training receipts only: zero new simulator calls, fits, predictions or target
reads. The completed development outcome stays unresolved. Official scoring
is not authorized by this report.

All nine point searches stopped during a poll at their assigned call budget.
They were still finding better training fits near that limit. Each family's
three seed-labeled runs followed the same deterministic search path for every
shared call. Their different final fits reflect different paired budgets,
not independent optimizer randomness.

| Family | Paired seed | Calls | Best training RMSE | First call attaining it | Improvement over 90% prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 | 11 | 831 | 1.269643 | 825 | 6.57% |
| C2 | 29 | 799 | 1.300751 | 799 | 6.20% |
| C2 | 47 | 637 | 1.472648 | 636 | 3.85% |
| O2 | 11 | 735 | 1.370813 | 734 | 5.34% |
| O2 | 29 | 765 | 1.338787 | 763 | 5.91% |
| O2 | 47 | 934 | 1.209211 | 934 | 3.03% |
| S0 | 11 | 815 | 0.947888 | 814 | 8.72% |
| S0 | 29 | 841 | 0.921058 | 840 | 9.07% |
| S0 | 47 | 881 | 0.908483 | 874 | 6.72% |

Here B=ceil(0.9*N); improvement is `(best_first_B-best_final)/best_first_B`.
Each search made 13–17 strict new minima after B. This is evidence of late
progress, not a statement of what an unlimited search would attain. Every
stop reason is `budget_exhausted_poll`, and N equals its allowance.

At common callback budgets the training results are:

| Family | Best in first 256 calls | Best in first 512 calls |
| --- | ---: | ---: |
| C2 | 1.945378 | 1.577662 |
| O2 | 1.949528 | 1.578755 |
| S0 | 1.829837 | 1.324156 |

All three within-family seed pairs have bit-exact equal coordinates and
objectives through the shorter run. The common-budget table therefore
does not need seed averaging. These are training errors; they do not predict
an unmeasured improvement on the development suffix or official record.

All nine final c values are at the original upper bound 1.5; no other fitted
coordinate is at a bound. That flag did not stop the search. Budget limits
and boundary contact can coexist. Neither proves that the box is the cause
of prediction error or that widening it helps. The earlier broad-ABC-tolerance
warning also remains a separate question.

## Method and scope

The [protocol](cascaded-tanks-training-receipt-audit-protocol-2026-10-15.md)
was narrowed before execution. The first unexecuted draft covered ABC as well
as point search and exceeded the small command budget. Primary removed the
ABC section instead of growing engineering. The executed command has 80
readable lines and changes no product runtime or test file. Its elapsed time
was 0.00838 seconds; reported peak process RSS was 36,405,248 bytes. It used
Python 3.14.3 and NumPy 2.4.2, under 120 seconds/1 GiB/1 MiB-output limits.

The inputs were the original manifest and 18 pattern JSON/NPZ files from
`ct-memory-week-dev-20261015-v1`. The command checked stored normalized
coordinates, objectives, minima, physical parameter mapping, callback counts
and bounds. A different Luna used separate calculations, not the author
command, to reproduce the summaries. All 19 input hashes matched; maximum
numeric discrepancy was zero. No archive members, forecast or score files
were read. The original study artifacts remained unchanged.

Manifest SHA-256:
`1981072a7557a87bba893fb1f71ae7f051bcad40f5651b70b80821229f6de029`.
Command SHA-256:
`7fcd90c6bd6ff988f1c1a88f4a2dca34b989c26677970212e8d41ed696db3b0c`.
Summary SHA-256:
`a7c06947109613f160a2fd1cf2dffb59ad934932e42b364667eed69e6a6fc97e`.
Independent replay SHA-256:
`aed18b18a457db50c40a1e3d4c9935334682e18a708875835c043da0629e5960`.
Local analysis evidence is under
`artifacts/deferred_work/ct-training-audit-20261015/`; raw traces stay ignored.
Compact public copies are the
[summary](cascaded-tanks-training-receipt-audit-evidence-2026-10-15/summary.json),
[run metadata](cascaded-tanks-training-receipt-audit-evidence-2026-10-15/run-metadata.json)
and [independent replay](cascaded-tanks-training-receipt-audit-evidence-2026-10-15/independent-replay.json).

## Decision

Do not interpret the c boundary as a reason to widen support yet. First
assess a small equal-budget training control with the original box and search
rule. Reuse existing helpers, preserve all charged calls, and freeze/review
any actual fit before launch. Keep the [original result](cascaded-tanks-week-development-results-2026-10-15.md)
and its paired comparisons unchanged. No independent replication, mechanistic
discovery, calibrated uncertainty or new predictive gain follows from this audit.

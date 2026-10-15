# Cascaded Tanks: training receipt audit

Protocol `ct_memory_training_audit_v1_20261015`, reduced before execution.
Output ID `ct-training-audit-20261015-v1`. Primary design, before execution.
This is a post-score diagnostic of saved training evidence. It is neither a
new fit nor an independent prediction test. Keep the completed development
decision unresolved and the official test closed.

## Questions

1. Did the deterministic point search finish improving, or did its paired
   call allowance stop it? An optimum at a bound alone does not answer this.
2. Do point-search seed labels represent different search paths or different
   truncation lengths of the same deterministic path?

No result here identifies why development forecasts differed. It can choose
the next controlled experiment and prevent a premature prior-box change.

## Inputs and limits

Read only the original manifest and the 18 pattern fit JSON/NPZ checkpoints under
`artifacts/cascaded-tanks-week/ct-memory-week-dev-20261015-v1/arms/fit/`.
Manifest SHA-256 is
`1981072a7557a87bba893fb1f71ae7f051bcad40f5651b70b80821229f6de029`.
Its source content commit is `5997cc2c0498b3ff435499c049fb5976f4b2f1e2`.
Verify its settings and the nine family/seed pairs. Record an input hash list.
The analysis does not use current HEAD as the original execution HEAD.

Do not read archive members, source outputs, forecast or score files. Do not
run simulators, optimizers, inference, generation, scoring or resampling.
Zero new simulator calls; no research data acquisition. Use Python 3.14.3
and NumPy 2.4.2, one worker, at most 120 seconds and 1 GiB sampled RSS.
Write at most 1 MiB to the separate ignored audit output directory. Keep all
original files unchanged. Stop on malformed/missing evidence; do not repair it.

No new library, CLI, runtime module, test file or dependency is needed.
Use one small receipt-processing command, at most 80 readable Python lines,
with the exact command saved as invocation metadata. This is data analysis,
not an exception to the two-runtime-file/1,000-line engineering ceiling.
A different Luna reviews the command and independently recomputes its
summaries before publication. This read-only saved-array analysis needs no
separate pre-execution approval gate. No new test suite is required.

## Fixed calculations

For each pattern arm, read normalized `evaluations` and `objectives`, plus
physical `best_parameter`. Check N equals recorded callback calls; shapes,
finite coordinates in [0,1], and objective/parameter minima agree with JSON.
Preserve failures as infinite objectives; NaN is invalid evidence. Map
coordinates with the original manifest bounds. Use first-index tie handling.

- Report N, stop reason, cache hits, final best RMSE and first attainment
  call (one-based) of that minimum. Report the best objective in the first
  256 and 512 calls.
- Let B=ceil(0.9*N). Report best-of-first-B minus final best and that decrease
  divided by best-of-first-B, plus the count of strict new minima after B.
  These describe progress; zero change does not certify convergence.
- Compare evaluation and objective prefixes exactly for all three within-
  family seed pairs, through their shorter N. Report equality separately.
  This is a search-path check, not physical replication.
- Report final parameters on bounds, with absolute tolerance `1e-12` and no
  relative tolerance. A bound hit alone does not show that widening it helps.

The initial draft also covered ABC distance, tighter-epsilon and support-mass
summaries. Its 172-line unexecuted command could not fit the 80-line analysis
ceiling without code golf. Primary removed that entire section and reduced
the prefix grid before any analysis ran. The unused draft stays as local
evidence, not an approved module. Do not execute it. This audit now addresses
search progress and seed/path meaning only. ABC tolerance diagnosis remains
a separate question; the earlier reviewed broad-tolerance warning is unchanged.

Save a compact summary and timing/resource/input-hash metadata separately.
Luna operators do all command work and independent replay. Primary interprets
the evidence, records limits and chooses the next scientific protocol.
One Git owner publishes only reviewed compact summaries/report/protocol.

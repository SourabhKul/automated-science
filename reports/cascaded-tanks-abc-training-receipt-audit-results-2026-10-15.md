# Cascaded Tanks: stored ABC tolerance and training-fit gap

Independent replay found a large training-fit gap in the stored ABC particles.
None of the nine final populations has any weighted mass within 1.25 times
its family's separate 2,048-call point-fit RMSE. The thresholds were
1.069333 for S0, 0.981799 for O2 and 0.982865 for C2. Those point controls
are descriptive reference levels, not compute-matched ABC comparisons.
No new fit, forecast, target read or LLM call was needed for this audit.

The old paired point comparisons do match simulator calls, including
calibration. Only C2/29, C2/47 and O2/47 have positive final weighted mass
within 1.25 times their paired point-fit RMSE: 1.69%, 7.60% and 2.77%.
The other six have zero mass there. This measures the stored finite
populations, not the probability that an unobserved parameter region exists.

| Family/seed | Final epsilon | Minimum particle RMSE | Weighted mean RMSE | Weight ESS | Mass near paired point |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2/11 | 2.4075 | 1.7287 | 2.1920 | 55.85 | 0% |
| C2/29 | 2.4661 | 1.4984 | 2.2323 | 51.90 | 1.69% |
| C2/47 | 2.5459 | 1.6078 | 2.2749 | 56.69 | 7.60% |
| O2/11 | 2.4359 | 1.8538 | 2.2376 | 56.25 | 0% |
| O2/29 | 2.4797 | 1.9133 | 2.2403 | 36.13 | 0% |
| O2/47 | 2.3959 | 1.4698 | 2.1799 | 48.61 | 2.77% |
| S0/11 | 2.9360 | 1.9674 | 2.6856 | 50.25 | 0% |
| S0/29 | 2.8971 | 2.0869 | 2.6863 | 58.92 | 0% |
| S0/47 | 2.9046 | 1.5387 | 2.6142 | 57.05 | 0% |

The mean column averages particle training distances with their weights. It
is not the training error of a mean trajectory. Weight ESS describes weight
concentration; a high ESS does not imply a close fit or calibrated uncertainty.
The 1.25 factor defines this descriptive audit, not a new hypothesis test.

## Receipt checks and cost

All 2,304 calibration distances were finite, with no failed calibration
outcomes. Each original epsilon exactly matches its frozen linear calibration
quantile: q25 then q10. Final populations contain 64 finite, supported
particles each, with nonnegative normalized weights. Recomputed ESS values
match receipts. All recorded ABC population failure counters are zero.
These checks do not establish inference calibration or validate every proposal
density calculation; the audit uses saved summaries and accepted particles.

Generation 1 support rejects range from 61.36% to 72.20% of proposals.
Accepted/proposed ranges from 5.68% to 10.14%. The denominator includes
out-of-support proposals; those consume proposal attempts but no simulator
calls. For example C2/47 has 631 proposals, 453 support rejects, 178 simulator
calls and 64 accepted particles. Do not confuse proposal counts with simulator
cost. The receipts retain aggregate rejection counts, not rejected vectors.

The audit read 31 hashed inputs: the original manifest, ABC fit JSON/NPZ
checkpoints, old paired point summaries and new common-budget point summaries.
It used an 80-line one-off analysis with a 120-second alarm and outputs below
1 MiB; peak reported process RSS was 37,715,968 bytes. A different Luna used
separate calculations on the same training receipts. All input hashes and
metrics matched, with maximum numeric discrepancy zero and no replay errors.
No source archive, forecast or score values were read. Existing inference
and forecast costs are unchanged: 20,620 fitting calls and 585 forecast replays.

Summary SHA-256:
`0bb8b63f41089f06c4974d7bb1c2e09be7accf4e4b8fceb855b58b397e81d4e6`.
Independent replay SHA-256:
`9b9dac9dec9088b831d294e2c5727237c4a6eb79fafb941c03d005e09e440dc7`.
Analysis script SHA-256:
`b479985eb3112229b6452626cd238f34965dbe1a0e4c2b406c86050ff4aa7c79`.
Raw checkpoints stay ignored. Compact public copies are the
[summary](cascaded-tanks-abc-training-receipt-audit-evidence-2026-10-15/summary.json)
and [independent replay](cascaded-tanks-abc-training-receipt-audit-evidence-2026-10-15/independent-replay.json).

## Scientific decision

The final thresholds were internally consistent with the frozen protocol,
but broad relative to known training fits. This supports considering one
smaller-tolerance training control; it does not show why the forecasts were
poor or guarantee that reducing epsilon helps. Support rejection and the
point fits at a/c bounds are separate warnings. Do not declare either the
ABC reference or these model families scientifically disproved by this audit.

A source-only feasibility check found the generic reference accepts another
non-increasing epsilon, with preserved partial-population failures. The week
wrapper has no schedule override, so a third population needs a separately
frozen small control; do not restart the completed full study or mutate its
settings. Keep any engineering inside the existing ceiling, or reduce scope.
No new fit is authorized here. Retain ARX and point controls, all failures and
actual costs. The original development decision remains unresolved. Official
scoring stays sealed pending scientific eligibility and a separate protocol.

For the current v1 configuration, primary stops progression to refit and
one-use official evaluation. The frozen development rule found no selected
family with useful predictive improvement over the simple controls. This is
a scientific prioritization decision, not a software launch failure or a new
predeclared rule. The public development result and its negative diagnosis
are the present methods deliverable. A future revised study must justify its
scientific change and evaluation plan before asking the official record a
new question. The available training-only tolerance control remains optional.

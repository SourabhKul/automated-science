# Proposed small paired-plate pilot

Primary decision after the October 29 metadata-helper checks: narrow the
first pilot to prediction of a known strain's second technical assay plate
from its first plate. No new-strain or biological-population claim is in scope.
No fitting or observation parsing is authorized by this design memo.

## Cohort and target

Choose the first four nonempty unambiguous ASCII labels in a row-major scan
of the original strain layout, skipping literal `EMPTY` and empty cells.
The stored, CRC-checked 8 by 12 table gives:

| Well | Raw label |
| --- | --- |
| B2 | DBVPG6765 |
| B3 | SK1 |
| B4 | Y55 |
| B5 | YPS128 |

This metadata rule precedes any OD-value inspection. Preserve raw labels;
do not use the published filtered map or replace a failed record with a
better-looking strain. The encoding-unknown label elsewhere is not selected.
No alias merge or assertion of four independent biological cultures follows.
Other isolate/derivative labels remain outside this pilot; a later held-out-
strain study still needs a full biological grouping map.

Use only the `glucose + galactose` directory: filename metadata list 271
plate1 and 271 plate2 files. These counts are not curves or independent n.
Training is the selected wells on plate1; evaluation is the same wells on
plate2, conditional on matching raw metadata and the stated strain map.
The intended targets are whole paired trajectories, not random time-point
splits. All single-carbon files and derived growth-rate files stay out of
training, tuning and evaluation. No pure-sugar biological control claim is
made in this small pilot.

Before observations, a separate reviewed metadata-only reader must confirm
the exact raw-file roster, channel/OD layout, plate IDs and assay timestamps.
Use assay time, not filesystem mtime. Prefer elapsed hours from the first
validated training-plate timestamp, using that same origin for plate2.
Do not substitute the source script's commented inoculation time if the
metadata do not support it. Missing, ambiguous or duplicate records stop the
stage; no silent row dropping or value-driven selection.

## Measurement scope

The initial question is raw-OD assay prediction with an explicit curve
intercept. It is narrower than calibrated biomass or blank-corrected growth
inference. The source helper takes a supplied background, floors low corrected
values, then logs them; it does not estimate background from blank wells.
Do not inherit that transformation or the old fixed argument automatically.

For the proposed raw-OD curve, `b` is a training-plate curve intercept, not a
measured blank correction. Prediction on plate2 assumes that OD offset and
gain transfer across the pair. Plate1 alone cannot identify a new plate2
offset or gain. Do not fit either to plate2 before scoring. This assumption
must be explicit, challenged in model checks and stated as a limit on claims.
Exact media-only well identities remain unresolved. Their values cannot be
used or called calibrated blanks. If this transfer assumption is unsuitable,
stop this design and resolve physical controls before fitting; do not change
it after seeing the score.

## Required inference freeze

Compare OD-scale one-phase and ordered delayed-second-phase curves. Retain
the zero-second-amplitude/equal-phase nesting cases and handle label exchange
in support rules. Neither curve preference nor a particle weight identifies
GAL induction or another molecular cause.

The existing checked ABC-SMC callback API can accept these curves. Its
Gaussian parameter proposal is not an observation likelihood. A cheap point
fit and a declared Gaussian-likelihood weighting control must precede or
accompany ABC. Finite prior Monte Carlo likelihood integration needs weight
ESS and fixed draw-sensitivity checks; it is not an exact or calibrated
posterior. Do not use the specialized one-parameter exponential ODE reference
as if it were a generic yeast reference.

Still to freeze before any fit: exact curve equations/parameter support,
Gaussian error structure (including shared plate/time errors), noise scale
and training-only rule, objective/score, simulator and likelihood
call accounting, ABC discrepancy/tolerance schedule, seeds, output freeze,
new wall/RSS/artifact/call limits, reference-quality gates and stop rules.
Any small adapter needs independent design and exact-byte code review. Do
not inherit tank budgets, expand its closed source or build a new framework.

## Proposed marginal error model

An independent Luna max design review accepted the following mathematical
option, with fitting still on HOLD. For well k on plate p, let

`y[p,k](t) = f[k](t; theta[k]) + u[p](t) + e[p,k](t)`.

Here `u[p]` is a zero-mean shared plate process with covariance
`sigma_p^2 * exp(-abs(t-s)/ell)`. The errors `e[p,k]` are independent Gaussian
errors with variance `sigma_e^2`, independent of the plate processes. Assume
independent plate processes across plates. A common instrument drift across
both plates would violate that assumption.

Each well can use only its own plate1 data. Its marginal Gaussian covariance
is `sigma_e^2 I + sigma_p^2 K_ell`, built from actual elapsed-time differences.
This deliberately leaves other wells unused when learning a plate effect.
It loses information but gives a coherent marginal prediction. Plate2
prediction integrates parameter uncertainty and a new independent plate
process; it must not condition on plate2 values before the prediction freeze.

The four predictions remain correlated through shared plate errors. Report
four descriptive trajectory scores. Do not multiply their likelihoods into
a joint score, count them as independent cultures or use them for population
inference. A long process length scale can resemble an offset and confound
the curve intercept. Prespecify `sigma_p`, `ell` and `sigma_e` before any OD
inspection; no numerical values are frozen here. A likelihood control must
include both the residual quadratic and covariance log determinant. This
review establishes a possible error structure, not calibrated noise, an
exact posterior or permission to fit.

## Metadata gate completed; fitting remains closed

The separate raw-record and receipt-only time stages completed on October 29.
All 542 selected file identities have the allowed channel and a saved assay
timestamp. No OD values were parsed. See
`yeast-diauxic-raw-metadata-results-2026-10-29.md` for exact receipt hashes,
format, full-scan limits and time checks. These stages must not be rerun.

Use the first plate1 timestamp, `7/19/2013 5:10:58 PM`, as the common clock
origin and express differences in hours. The timestamp format is numeric
month/day/year with a 12-hour clock. The timezone is unknown. Treat this as
a shared, stable local-clock coordinate, not validated time since inoculation.
Retain the distinct, irregular assay times. Plate2 begins 113 seconds after
that origin and ends 185 seconds beyond the last plate1 timestamp. The
prediction task includes that short extrapolation; do not trim it after
seeing values. Unique times after sorting do not independently prove clock
stability or acquisition order. No OD matrix shape has yet been validated.

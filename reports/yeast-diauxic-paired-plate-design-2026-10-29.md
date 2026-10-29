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

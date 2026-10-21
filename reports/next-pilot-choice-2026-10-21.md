# Next pilot: yeast growth across assay plates

Primary decision at the October 21 09:17Z wake: select a conditional yeast
assay-transfer pilot and retain a bounded source-inventory design. Do not restart the closed tank study. No real fit follows
from this choice alone. The question is whether one-phase versus two-phase
growth descriptions transfer across whole assay records after measured
background correction. This is a retrospective methods question. Its value
is a test across experimental records with a stated measurement model, not
rediscovery of diauxic growth or identification of a molecular pathway.

## Three opportunities considered

| Opportunity | Significance and novelty headroom | Data and independence | Identifiability and local cost | Decision |
| --- | --- | --- | --- | --- |
| More tank conditions | Physical state/measurement ambiguity matters, but the current question is already unresolved | The official source describes estimation and test records; I did not establish a new non-official independent rig or intervention set | Existing code is cheap; more slices from the same rig do not repair independent validation | Defer; preserve the unused official record |
| Enzyme inhibition | Competing rate laws and a tractable reference are attractive; established inhibition analysis leaves limited novelty for a reproduction | The 50-BOA paper reports both simulated data and reuse of experimental graph data; independent raw replicate provenance was not established in this scan | Low-dimensional analytic laws are cheap, but digitized points need a different measurement-error account | Defer this source rather than treat simulated/digitized rows as new experiments |
| Yeast diauxic growth | A direct test of model transfer and measurement sensitivity; biological novelty remains unproven | A small raw-OD subset, control conditions and multiple strain records are described; plate and alias mapping need verification | Analytic curve baselines and checked ABC are feasible if schema and grouping are clear | Select only the bounded inventory now |

The [official tank page](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks)
describes its supplied records. This search did not establish that no other
tank data exist. The [enzyme paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12141731/)
and [author repository](https://github.com/Mathbiomed/50-BOA) distinguish data
and analysis examples. Their published findings are prior art, not ours.

The selected [Wang et al. paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002041)
reports 43 strain labels, two assay plates from a shared inoculation,
media-only wells and an OD linearity check. Some strain names are aliases.
The [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.39h5m)
lists Dataset 1.zip at 1.59 MB as raw OD CSVs; the entire collection is
4.72 GB. Only the small subset is in scope. Published plots and biological
conclusions are already known to us, so a new computational holdout here
cannot be called untouched independent scientific confirmation.

## Scientific boundaries for a later fit

Candidate process alternatives are a single resource-limited growth phase,
and an additional delayed growth phase. Initially use explicit bounded curve
forms as phenomenological descriptions; a curve preference cannot establish
GAL induction, sugar consumption or another unmeasured molecular mechanism.
A mechanistic resource model needs actual substrate observations and its own
identifiability check. OD offset, scale, blank drift and shared plate effects
are competing measurement explanations, not terms to discard after fitting.

The experimental record is a culture/plate trajectory, never each time point.
Two plates with a shared inoculum are paired assay repeats, not two independent
biological experiments. Group duplicate isolate labels before splitting;
strain relatedness also limits population claims. No random row split. A
later protocol must state whether it predicts a new plate conditional on a
known strain, or a new strain group. These are different estimands. Pure-sugar
controls must receive a fixed training or evaluation role before values are
parsed. Do not pool them merely to make a larger nominal sample.

A fit protocol must lock the exact cohort, group/plate split, preprocessing,
model equations/support, controls, discrepancy/tolerances, simulator and
proposal costs, failure rule and forecast freeze before outcome scoring.
Retain a cheap point-fit/analytic likelihood reference under an explicit
measurement-error assumption. Check ABC against that reference; do not call
a finite-tolerance particle cloud calibrated. One fixed original-Qwen proposal
may be useful after training metadata are ready, but no generation is needed
for the inventory. Invalid or unsupported suggestions cannot alter a score.

## Immediate executable scope

A future source inventory may run only after its separate review. Verify access,
bytes, hashes, archive members, headers and group/control metadata without
parsing OD rows. No data fit, curve visualization, target score or local-model
request. If source access or metadata cannot resolve records and grouping,
stop at an exact blocker. Do not buy data, query authors or add infrastructure.
Keep the closed week's 994 runtime/299 test files unchanged. A later fit needs
its own small engineering decision; it cannot silently inherit unused budget.

## Analytic measurement and model check

Let latent biomass follow logistic growth, and let measured optical density
be linear in that biomass:

\[
N(t)=\frac{K}{1+(K/N_0-1)e^{-rt}},\qquad y(t)=b+sN(t).
\]

For any positive c, replace (K,N_0,s) with (cK,cN_0,s/c). The ratio K/N_0
and the entire measured trajectory are unchanged. Thus this model cannot
separately identify absolute biomass scale and OD sensitivity from OD alone.
Use identifiable OD-scale amplitudes in a phenomenological fit, or supply an
independent calibration for absolute biomass. More time points do not remove
this structural symmetry. This is an algebraic property, not a data result.

For candidate two-phase curves, define f(t;r,t0)=1/(1+exp(-r(t-t0))) and
compare y=b+A f(t;r,t0) with

y=b+A1 f(t;r1,t1)+A2 f(t;r2,t2).

If r1=r2 and t1=t2, the two-phase curve is exactly the one-phase curve with
A=A1+A2; allocation between A1 and A2 is unidentified. Swapping both phase
labels also leaves y unchanged. A2=0 is another nested one-phase boundary.
A later fit must handle these degeneracies explicitly, retain the simpler
control and define label/support rules before examining outcomes. Particle
weights within a model are not probabilities of the two process descriptions.
These checks restrict a later fit; they do not establish a new growth law.

## Actual access decision

Official metadata API responses were HTTP 200: source version 1, public,
CC0-1.0. Dataset 1 is 1,594,286 bytes, MD5
`40bcd862ebf977036a3d9d5c9b33a74c`; README is 4,777 bytes, MD5
`693ff0c43eb2245892837305af955b85`. The API download of README returned
`Unauthorized, must have current bearer token`; the one linked-stream
fallback returned 403. Separately, primary's browser tool attempted the normal
ZIP link and the README link; both returned 403. Luna did not request the ZIP.
No archive bytes or OD rows were obtained; no fit or LLM request followed. Public metadata does not establish public
unauthenticated file access. No credential or mirror workaround is authorized.

Primary HOLD: no further denied requests, inventory command or fitting now.
Plate/strain/alias/control mapping remains unresolved. Clearing this source
blocker requires legitimate access to only Dataset 1.zip and its README,
through Dryad's normal permitted path or files supplied by the user. Do not
make a new request while this HOLD is unchanged. After access changes, verify
the above sizes/checksums before reviewed header inspection. The full collection is not required. If mapping still cannot
support the declared unit, reject the pilot before fitting.

Current Python 3.14.3 has NumPy 2.4.2 and pandas 2.2.2 but no SciPy. Existing
NumPy RK4 and checked ABC APIs are available; a yeast reader is absent. These
are source/runtime observations, not permission to install packages or expand
closed-week code. Any later runtime and fit stage needs a small frozen scope.

The compact [feasibility receipt](yeast-diauxic-source-evidence-2026-10-21/feasibility.json)
records Luna's own observations only, not primary's separate browser attempts.
An initial independent factual review held the docs because those actors were
not distinguished. This correction preserves the failed primary ZIP request;
it does not turn the blocked attempt into an acquisition or a fit.

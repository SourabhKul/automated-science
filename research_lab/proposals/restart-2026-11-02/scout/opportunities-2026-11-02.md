# Restart screen: two scientific opportunities

Status: research leads only. No data were downloaded, no observation rows were
opened, and no dataset or analysis was selected. The published findings below
are prior art. Any later result must be framed as a held-out model test unless
new evidence supports a new scientific claim.

## Recommendation

Start with the predator–parasite microcosm as the stronger empirical design.
It has a factorial intervention, replicate microcosms, raw counts, and known
sampling exposure. Use the hydrology option as a second, distinct-domain lead
if the goal is cross-basin generalization. It has a larger forcing and
measurement contract to clear.

## 1. Consumer–resource ecology: parasite effects in a predator–prey food web

**Falsifiable question.** Across the predator-present / predator-absent by
parasite-present / parasite-absent experiment, can a shared predator-response
law predict held-out *Paramecium* and *Didinium* trajectories after the model
accounts for parasite effects on host abundance? Does the held-out evidence
require parasite-status-specific predator attack or handling parameters?

**Why try it.** This is a compact mechanistic test of density-mediated versus
trait-mediated pathways. The experiment changes predator and parasite
presence independently. A small ODE and a raw-count likelihood could test
whether mechanism changes improve prediction across complete microcosms. The
primary paper already reports its own analysis, including a direct feeding
test. That result is known; a reanalysis cannot be presented as a new finding.
The plausible contribution is a predeclared, unit-level predictive test with
an explicit count model. Novelty is not established.

**Primary sources and observed repository evidence.**

- Paper: [Banerji et al., Journal of Animal Ecology (2015)](https://doi.org/10.1111/1365-2656.12317).
- Data: [Dryad record 10.5061/dryad.3kb72](https://doi.org/10.5061/dryad.3kb72).
- The metadata lists three files totaling about 1.71 MB. It describes
  replicate IDs, treatment labels, observation day, species, count, and
  sampled volume for ciliate counts. A second file records Serratia colony
  counts, dilution, and plated volume. The data page describes a factorial
  predator/parasite presence design. No file or row was opened.

**Units and measurement contract.** The experimental unit is a microcosm
(`repID`), not a species-by-day row. Ciliate observations are counts in a
recorded sample volume. Serratia observations are colony-forming units with
recorded dilution and plated volume. Use these exposures as offsets in a
Poisson count model, with a negative-binomial sensitivity for clumping or
other overdispersion. Microscopy and plating do not provide a stated
calibration error model in the inspected metadata. Verify sampling removal,
replacement, and zero-count handling from the paper methods before any fit.

**Small simulator and controls.** Start with three states: bacterial resource,
Paramecium, and Didinium. Compare a shared predator functional response with
one nested alternative that lets parasite status change the predator response.
Keep the parasite effect on host growth as a separate, predeclared term.
Controls are the no-predator/no-parasite trajectory, a logistic resource and
host model, and a standard Rosenzweig–MacArthur or Lotka–Volterra fit. A
direct likelihood or profile fit is the cheap reference for ABC-SMC. Use
ABC-SMC only after it agrees with that reference on the frozen low-dimensional
target; treat BDSS as a same-model proposal ablation, not a scientific
mechanism.

**Split.** Group by whole microcosm. Within each of the four factorial arms,
assign replicate IDs by a fixed hash to fit and holdout sets; never split
timepoints or species rows. Proceed only if each arm has enough independent
microcosms to leave at least one whole unit out while retaining training units.
If not, use leave-one-microcosm-out prediction and report its limits. The
published experiment has been analyzed already, so these are computational
holdouts, not untouched confirmation.

**Cost and identifiability risks.** The listed files are small. A three-state
ODE over the experiment is cheap per proposal, but particle cost depends on
the number of distinct microcosms and accepted tolerance. Do not set that
budget until the replicate map and trajectory lengths are checked. Make zero
Qwen calls during source/reference review; a later locked run could use at
most one fixed-model proposal call plus one repair. Attack, handling, host
growth, and mortality can compensate for one another. Extinction and low
counts weaken identification. Counts from the same microcosm are correlated.

**Smallest decisive experiment.** Freeze one four-arm, complete-microcosm
split and compare the shared-response model against the nested
parasite-specific-response model. Score joint held-out count log likelihood
and species trajectories, with a count-calibrated reference and the simplest
resource/consumer control. Report posterior predictive checks by arm and
microcosm; do not select on pointwise time rows.

**Strong reject criterion.** Stop before fitting if the raw contract cannot
confirm the independent replicate IDs, per-sample volume/dilution, or all
four intervention arms. Reject the mechanism claim if treatment-specific
predator response is not identifiable from host growth and mortality, or if
it does not improve locked whole-microcosm prediction beyond measurement
uncertainty. Any accepted result remains a reanalysis of previously published
data.

## 2. Hydrology: catchment recession after rainfall

**Falsifiable question.** In warm, rain-dominated CAMELS-US basins, does a
two-timescale recession model predict held-out post-storm streamflow better
than a single linear reservoir, with parameters that transfer to separate
basins? This tests whether one storage time scale is enough to describe the
observed recession shape; it does not establish a new hydrologic law.

**Why try it.** Recession behavior constrains how catchments release stored
water after rain. The small candidate models are interpretable and have
tractable simulations. The test can use separate basins as units and whole
time blocks as event-level holdouts. Recession laws are well studied, so the
plausible value is a strict cross-basin, uncertainty-aware falsification test.
Novelty is not established.

**Primary sources and observed repository evidence.**

- Dataset paper: [Addor et al., Hydrology and Earth System Sciences (2017)](https://doi.org/10.5194/hess-21-5293-2017).
- Original dataset: [NCAR CAMELS record, DOI 10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D).
- Versioned source record: [Zenodo record 15529996](https://zenodo.org/records/15529996).
- The tracked project manifest records a 3.41 GB version 1.2 archive, CC BY
  4.0, daily forcing and USGS discharge, with 563 basins passing its source
  quality gate. It records forcing in mm/day, degrees C, W/m², mm, and Pa,
  flow in cubic feet per second, and drainage area in km². The existing split
  lists 36 train, 12 selection, and 16 external basins. These are metadata
  facts only; this screen did not read discharge or forcing rows. See
  [`data/real/camels_us/manifest.json`](../../../../data/real/camels_us/manifest.json)
  and [`source_basin_split.json`](../../../../data/real/camels_us/source_basin_split.json).

**Units and measurement contract.** Treat a gaged basin as the independent
spatial unit and retain daily dependence within each basin. Rainfall is a
gridded forcing; discharge is USGS gauged flow. The model needs drainage area
conversion and a likelihood for positive flow, with an explicit treatment of
zero, missing, and censored values. The local metadata lists QC flags `A`,
`A:e`, `M`, and an additional `A:<` token that the source readme does not
document; 135 finite observations across four basin files use that token.
Resolve or exclude the affected files before fitting. The source basin set
uses HCDN-2009 criteria, but the local contract applies no further human-impact
filter. Do not call the selected basins pristine or treat spatial similarity
as randomized intervention.

**Small simulator and controls.** Compare a one-state linear recession,
`dQ/dt = -kQ`, with a two-reservoir recession whose fast and slow components
drain at different rates. Use direct exponential/log-flow regression as the
cheap reference. Then test a small conceptual rainfall–runoff model on a
predeclared storm/recession window. A standard four-parameter model such as
GR4J is a useful external baseline if its forcing and units are supported.
Do not infer storm windows from held-out outcomes.

**Split.** Preserve the existing basin split: 36 training, 12 model-selection,
and 16 external basins. Within each basin, use contiguous calendar blocks for
fit, selection, and a single locked late-period score, subject to metadata-only
coverage checks. Keep complete storm/recession windows together. If the
existing groups or data coverage fail this plan, stop and ask the primary PI
to revise the design; do not quietly reshuffle the recorded external set.

**Cost and identifiability risks.** The release is large (3.41 GB). A
recession-only reference is very cheap; a daily simulator over 36 basins and
decades can become expensive under ABC-SMC, so start from short, predeclared
event windows and measure cost before any particle campaign. Use zero Qwen
calls during source and analytic review; a later fixed-model run needs at
most one proposal call and one repair. Fast/slow reservoirs can be
unidentifiable when their time scales overlap. Rainfall timing and amount,
catchment area, snow, evaporation, gauge error, and human water management
can mimic extra storage.

**Smallest decisive experiment.** On source-quality-eligible, warm-rain basins
from the frozen split, fit one- and two-timescale recession models on training
windows and compare locked predictions on complete held-out windows and the
external basin set. First compare the ABC target against direct likelihood or
profile fits on the one-state control. Report basin-level uncertainty and
leave-one-basin-out sensitivity; do not count daily measurements as
independent replicates.

**Strong reject criterion.** Reject the pilot if flow QC, unit conversion,
storm/recession timing, or human-impact screening cannot be fixed before
outcomes are scored. Reject a two-storage mechanism claim if the two rates
are not separately identified or if the extra reservoir fails to improve
locked basin-level predictions beyond gauge/forcing uncertainty.

## Scope notes

The review used public paper and repository metadata only. No raw source file,
observation table, local array, or archived dataset was opened. No code,
simulator, ABC-SMC run, BDSS run, or local model call was used. Root PI owns
which lead, if any, advances to a separate protocol.

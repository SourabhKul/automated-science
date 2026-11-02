# Independent peer review: two restart opportunities

**Decision date:** 2026-11-02  
**Scope:** review the two opportunity cards only. This review authorizes a
source and measurement-contract gate for ecology. It does not authorize raw
data access, model generation, code, a fit, or an ABC-SMC run.

I read the opportunity memo, the local CAMELS manifest and split, the Dryad
metadata page, the CAMELS Zenodo record, and the CAMELS attributes paper
abstract. I did not download data or open observation rows. The PI reports
that the Banerji paper abstract and MLE table have already been exposed. Treat
the ecology study as known-result reanalysis. Do not describe it as blind
discovery or use its published numerical estimates to set priors, thresholds,
or targets.

Scores run from 0 to 5. For compute fit, 5 means the proposed first test is
cheap within a small fixed budget. Scores describe each idea as written.

| Opportunity | Scientific value | Data readiness | Identifiability | Falsifiability | Compute fit | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Predator–parasite microcosms | 3 | 2 | 1 | 4 | 2 | **GO, limited to source and measurement-contract review** |
| CAMELS recession | 2 | 1 | 2 | 4 | 1 | **HOLD; retain as a later pivot candidate** |

The low identification and cost scores for ecology apply to the proposed
three-state ODE and ABC comparison. A four-coefficient factorial outcome
model could provide a cheaper, narrower first test after the gate. It would
not identify attack or handling rates. The low CAMELS cost score applies to
the proposed broad daily-data campaign. A recession-only fit could be much
cheaper, but its source, quality, and window contract is not ready.

## 1. Ecology: GO for the source and measurement-contract gate only

The primary [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.3kb72)
backs the small-source and schema claims. It lists three files totaling
1.71 MB. Its usage notes describe ciliate rows with `repID`, treatment,
species, sampled volume, and count. They describe Serratia rows with plate
volume, dilution, and colony counts. The record also labels the data as the
main experiment. These fields make a later count-model review possible. They
do not establish the units, sampling mechanism, or independence assumptions
needed for a Poisson likelihood.

The Dryad record's abstract states that predator and parasite presence were
manipulated and reports the study's conclusions. The cited paper is already
published ([Banerji et al.](https://doi.org/10.1111/1365-2656.12317)). These
facts support a compact reanalysis with a predictive target. They do not
support a new biological-discovery claim. The separate morphology/trajectory
file is not a new independent experiment by itself.

### Main blockers to the proposed ODE

- The source schema includes a `prey` field in addition to `predator` and
  `parasite`. The paper abstract says predator and parasite were crossed, but
  the memo must not assume that the complete design has only four arms.
  Confirm all treatment combinations and their meaning from the methods
  before any data read or fit.
- `repID` is the right candidate unit, but the source page alone does not
  show whether IDs match across ciliate and Serratia files, whether a
  microcosm is independent, or how many independent units lie in each arm.
  Repeated days and species rows are not independent replicates.
- The source lists sample-volume fields, but does not state their units or
  how microscopy and plating produced the counts. Verify dilution factors,
  plate replication, zeros, detection limits, and sample removal or
  replacement. A volume field alone does not prove Poisson thinning or justify
  a log-volume offset.
- A three-state resource/host/predator model with parasite-dependent host
  effects and predator response has confounded growth, attack, handling,
  mortality, and resource effects. Extinction and low counts make this worse.
  It is too large for the first test. Do not select the model or its priors
  from published MLEs.

### Authorized next scope

Proceed with a bounded, row-free review of the official methods and data
documentation. Resolve the unit, complete treatment map, sampling and
plating procedures, units, exposure meaning, repeated sampling, and count
handling. Confirm the source version, file identity, and reuse terms. Do not
download the files or inspect rows in this stage. Record each fact as
**source-confirmed**, **inferred**, or **unknown**. If a key fact remains
unknown, mark this candidate `HOLD` and move to a new proposal cycle. Do not
close the lab.

After that gate, a lower-dimensional first experiment could test whether
parasite status changes the predator effect on *Paramecium* abundance. Use
one predeclared common endpoint, or one predeclared summary per microcosm if
no common endpoint exists. Fit four treatment coefficients for predator,
parasite, and their interaction, using one microcosm-level outcome per unit.
If the source contract supports it, use a count likelihood with the measured
sampling exposure; compare Poisson with a predeclared overdispersion
sensitivity. Direct likelihood is the cheap reference. Hold out complete
microcosms within confirmed treatment arms if the replicate counts support
it. This tests a net treatment interaction and its prediction. It does not
test a change in a predator attack or handling parameter.

Do not demand a universal global-accuracy certificate before this cycle can
proceed. Freeze an independent reference, quantitative checks, and honest
limits for the selected small target. If a later dynamic model fails those
checks, report that result and pivot; do not build an unbounded repair effort.

## 2. CAMELS: HOLD

The official [CAMELS v1.2 record](https://zenodo.org/records/15529996)
confirms daily meteorological forcing and streamflow, an original set of 671
HCDN-2009 basins, and a 3.4 GB core time-series archive. The full Zenodo
record lists about 14.6 GB across the core archive, optional model outputs,
and attribute files. The local
[manifest](../../../data/real/camels_us/manifest.json) records a 563-basin
eligible set after its source-quality gate, while the local
[split](../../../data/real/camels_us/source_basin_split.json) fixes 36
train, 12 selection, and 16 external basins. The existing external split is
protected. Do not reopen it, reassign it, or use it to tune windows.

The source contract is not ready. The local manifest records QC tokens `A`,
`A:e`, and `M` in the README, plus `A:<` in 135 finite observations across
four basin files. Its meaning is unresolved. The proposed question also
needs a fixed rule for rainfall-driven recession windows, flow unit
conversion, drainage-area use, zero or missing flow, snow and human-impact
scope, and dependence within basin. The project tree reviewed here contains
the manifest and split, not the 3.4 GB archive. Acquiring and parsing that
archive is a material new cost.

If this lead returns later, start with the small analytical comparison
`Q(t)=Q0 exp(-k t)` (two parameters) versus
`Q(t)=Af exp(-kf t)+As exp(-ks t)` (four parameters) on short windows chosen
from precipitation alone. Fit direct likelihood or profile references first.
Only use windows with a frozen dry-period rule. Overlapping rates and a small
slow-component amplitude make the four-parameter model weakly identified.
Do not begin with full daily rainfall-runoff simulation or ABC-SMC. Do not
touch the protected external basins. Keep the unresolved QC token as a
specific blocker, not as permission to invent an interpretation.

This is a **HOLD**, not a rejection of hydrology and not a stop to the
research program. Revisit only after the ecology source gate or a later
proposal cycle, with an explicit archive and compute budget and an approved
train-only split.

## Prompt brief for a later two-idea proposal call

This is a proposed prompt contract only. No Qwen call was made. Use the exact
requested local model and endpoint: `qwen3.8-flash-next-iq3_s` at
`http://127.0.0.1:8080/v1`, no auth. Make one request for exactly two cards,
maximum 2,048 output tokens, 300-second timeout, and no repair or retry.
Provide only the source packets below. Do not include raw data or published
numeric estimates. The proposal call has a fixed 1-call cap, a 300-second
request cap, and a 15-minute total cap. Cap all new prompt, response, and
receipt artifacts at 256 KiB, including temporary writes, with 32 KiB reserved
for the final receipt. The model may suggest later experiment budgets, but
those are estimates for PI review, not approval.

**Required source packets**

1. Ecology: Dryad DOI `10.5061/dryad.3kb72`, paper DOI
   `10.1111/1365-2656.12317`, 3 files / 1.71 MB, metadata fields described
   above, with the exact unresolved unit, treatment, and sampling questions.
2. Hydrology: CAMELS v1.2 DOI `10.5065/D6MW2F4D`, Zenodo record
   `15529996`, core archive 3.4 GB, local source gate 563 eligible basins,
   fixed train/selection/external split, and unresolved `A:<` flag. State
   clearly that external basins are sealed.

**Prompt instructions**

> Using only the two source packets and the supplied primary-source links,
> return exactly two proposal cards, one for each source. Do not rank them or
> select a pilot. Do not browse, request, infer, or summarize raw observation
> values. Published findings are prior art; label each idea a reanalysis and
> do not use published estimates as priors, thresholds, or expected answers.
> Separate source facts, inferences, and unknowns. If a required fact is
> missing, mark the gate as blocked rather than filling it in. Keep the first
> empirical test to a simple 2–4 parameter model with an independent
> likelihood/profile reference. Do not start with a multistate ODE, broad
> daily simulator, ABC-SMC campaign, or code. Never propose use of protected
> external outcomes. Root PI selects after independent peer review.

Each card must contain: (1) falsifiable question and narrow estimand; (2)
primary source URLs and the exact supported facts; (3) experimental unit and
measurement assumptions; (4) the smallest 2–4 parameter first test and
reference; (5) the observation-free source/measurement gate; (6) leakage and
split rules; (7) key identifiability risks and one clear falsifier; (8)
suggested call, wall-time, memory, and artifact limits for a later experiment,
marked as estimates for PI review; and (9) `GO`, `HOLD`, or `REJECT` with a
short reason. Do not set a scientific outcome threshold from published
results. Output valid JSON only.

The independent reviewer must score each card before the root PI chooses.
The Qwen proposer cannot review or approve its own output.

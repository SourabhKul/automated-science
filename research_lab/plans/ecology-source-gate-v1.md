# Provisional first-stage plan: ecology source and measurement gate

PI candidate for cycle restart-2026-11-02. This is a source gate, not a
scientific fit or permission to infer a parasite mechanism. Selection needs
two scoped non-author reviews. Exact acquisition/reader code needs separate
review before it runs. Outcomes remain sealed until a later reviewed plan.

## Scientific target to qualify

Test whether a shared predator response can predict complete held-out
microcosms under parasite/predator interventions, compared with a narrowly
specified alternative. Published findings and fitted coefficients are known;
never use them as priors, calibration or desired answers. The first decision
is whether the records support any small falsifiable test at all. Do not
start the proposed three-state, many-parameter ODE by default.

## Exact source scope

Use only the author Dryad deposition DOI 10.5061/dryad.3kb72, version shown
as October 27, 2015. Its metadata names `p-d-h_raw_data3.csv` (about 32 KB)
and `p-d-h.serra3.csv` (about 19 KB). Do not acquire the morphology/trajectory
file, figures, notebooks or other datasets. Resolve exact version/file IDs,
byte counts, license and checksums through the normal public metadata API
before any file GET. No credential, mirror or denied-route retry.

If an identical verified local file is already registered, reuse its bytes.
Otherwise permit at most one public metadata lookup and one GET per selected
file, with total source bytes at most 1 MiB, each file at most 256 KiB, and
no redirects or authentication workaround. Hash complete bytes, compare
published size/digest when supplied, and preserve originals under ignored
`data/research_lab/`. Record every failure. Before/after acquisition measure
active data/artifacts and retain the program's below-50-GB working-set bound.
No package installation or deposited-code execution.

## Permitted interpretation

Freeze the schema from the deposition's usage notes before parsing. Decode
header and only metadata/measurement fields: day, repID, treatment, prey,
predator, parasite, species, sampled volume; for bacterial records, plated
volume and dilution. Treat `count` and `Colonies` as opaque, unconverted tokens and immediately
discard them. A standard CSV reader may decode/tokenize complete rows; do
not claim that the outcome strings were never decoded. They must never be
converted to numbers, retained in interpreted output, summarized, plotted,
fitted, scored or sent to a model. Handle quoted CSV with a bounded standard
parser; invalid schema stops.
No row deletion from outcome values and no published extinction exclusions.

Produce a metadata-only map of replicate IDs, treatment labels, species,
days and exposures. Use the already inspected public deposition usage notes as the methods
context for this gate. No extra paper/methods request is included in the
three-request budget. Units or sampling/removal/replacement facts absent
from that context must remain unresolved and block fit readiness; they do
not authorize another request or an inferred calibration.
The paper has a prey-presence flag as well as predator/parasite flags; do not
assume a four-arm roster or independent units from a filename. Do not treat
species/day rows as independent cultures.

## Decision and limits

A useful next fit requires a documented microcosm unit, intervention map,
time origin, exposure units/process and enough whole replicates. First map
ALL combinations of prey/predator/parasite labels and every repID. The
predeclared candidate roster is prey=yes crossed with predator=no/yes and
parasite=no/yes. Report prey-absent and all other combinations separately;
do not drop sparse arms or rename labels to pass.

For a metadata-coverage pass, require all four candidate arms to be present,
at least two documented microcosms per arm, and at least three distinct
PARA days for every candidate microcosm. In predator-present arms also
require at least three distinct DIDI days per microcosm. Report SERRA day
coverage and missingness per arm/repID, and whether its map joins the ciliate
map; do not invent a join or discard unmatched records. A missing arm,
sparse coverage, ambiguous labels or unresolved units/process yields HOLD
for fit readiness. Save the entire map, including sparse/failed units.
These are screening minima, not power, biological independence or model
identifiability. A fit and final unit split need a separate reviewed design.

Stage budget: one client process, 90 seconds processing, 256 MiB sampled
client RSS, 4 MiB retained/temporary output including a reserved receipt,
zero simulator/likelihood/forecast/model calls, and the three network requests
above. Any unmet contract yields HOLD or rejection with evidence. No numeric
model experiment, count parsing, final split, kernel ablation or holdout score
is authorized by this plan. A stopped source gate triggers a new lab next
action; it never closes the program.

Required outputs: source/metadata/code/plan hashes, request status and byte
accounting, opaque-outcome scope proof, replicate/exposure map, unresolved
facts, resource receipt, independent factual review, and PI go/hold/reject
for a later separately frozen small experiment. Keep raw files out of Git.

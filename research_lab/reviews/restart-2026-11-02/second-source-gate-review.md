# Independent second review: ecology source gate

**Decision:** HOLD before execution  
**Stage scope:** source/metadata gate only  
**Reviewed proposal SHA-256:** `82081d9f58a153c7627bc76ed13bb05b46b65cb3b29aae85316c3fbf254e30e7`

I reviewed the charter, source-gate plan, scout memo, and first peer review. I did not access the Dryad record or files, inspect rows, run code, or use a model. The source facts below come from those project documents; this review establishes no new metadata or scientific result.

| Criterion | Score (0–5) | Basis |
|---|---:|---|
| Scientific value | 3 | A narrow predictive reanalysis may be useful; the published result is prior art. |
| Data readiness | 3 | The two named CSVs are small, but units, sampling, and the full treatment map remain unresolved. |
| Identifiability | 2 | This gate cannot establish parameter identification; future dynamic-mechanism claims remain at risk. |
| Falsifiability | 4 | It sets concrete data thresholds, but arm and day coverage rules need clearer application. |
| Compute fit | 5 | The file and process caps are reasonable for a metadata-only pass with no numerical calls. |

## Current source-gate blockers

1. **The request cap does not cover all planned source checks.** The three stated requests cover one Dryad metadata lookup and two file GETs. The plan also requires checking the paper's methods for sampling, removal, and replacement. Name a versioned methods source and include its access in the request cap, or name an already available copy by hash. If those facts stay unknown, report them as unknown and hold fit readiness.
2. **Define how the arm threshold applies.** First map every observed combination of prey, predator, and parasite labels. State which predeclared candidate arms the “at least two microcosms per arm” rule covers. Report all other combinations; do not drop sparse arms after seeing the map just to pass the threshold. If the design does not match the candidate roster, hold for PI scope review.
3. **Make the day threshold check longitudinal coverage.** “Three distinct planned observation days” could be met across the full dataset while a candidate arm or its units lack useful repeated coverage. Report day coverage by arm and `repID`, plus missingness, and define the pass rule before reading metadata rows.
4. **Describe parser handling precisely.** A standard CSV reader may decode or tokenize `count` and `Colonies` fields. Say that these tokens are immediately discarded without numeric conversion, output, summary, plotting, fitting, or scoring. Keep the ban on deleting rows based on those values. The implementation still needs separate code review before it runs.

## Future fit risks, outside this review's authorization

- Sample-volume and dilution fields do not by themselves establish units, an exposure offset, plate replication, count handling, or sampling/removal effects. Do not treat them as a count likelihood exposure unless the methods support that use.
- `repID` must identify an independent microcosm, and any link between the ciliate and bacterial files must be documented. Species-by-day rows are not independent units.
- Two units per arm and three days are screening minima, not evidence of power or model identifiability. Any later split must keep whole microcosms together, with no timepoint or species-row leakage. The gate itself is not approval to fit.

The selected scope is appropriately narrow: DOI `10.5061/dryad.3kb72`, only `p-d-h_raw_data3.csv` and `p-d-h.serra3.csv`, with the morphology/trajectory file excluded. The acquisition, size, and compute limits are prudent. I recommend resubmission after the four source-gate rules above are made explicit. This HOLD does not reject ecology or close the lab.

# Ecology v2 source check

The local workspace has no saved Dryad API response, exact ecology version or
file IDs, exact CSV byte sizes/digests, or verified CSV header schema. The
restart notes summarize the official metadata and usage notes in prose. A
separate Dryad API example for another DOI shows dataset and version-file
metadata as separate resources; its IDs do not apply here.

Local summaries mention the literal terms `repID`, `prey`, `predator`, and
`parasite`, and describe ciliate treatment/day/species/count/sample-volume
fields plus bacterial colony-count/dilution/plated-volume fields. They do not
bind those terms to exact header roles. These are source-summary terms, not
verified CSV headers. Exact header aliases, outcome-header spelling, label
encodings, sample-volume units, dilution convention, sampling/removal,
microcosm independence, and cross-file joins remain unknown.

Keep the parser fail-closed. Freeze only a separately reviewed field/alias
table; preserve literal labels; stop on unknown, duplicate or ambiguous
schema. Do not infer a header role from the row values or metadata terms. These
unknowns block fit readiness if unresolved; the source gate may report them
without changing its coverage rules.

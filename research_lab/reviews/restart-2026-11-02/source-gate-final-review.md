# Independent final review: ecology source and measurement gate

**Decision: GO for the bounded source/metadata stage only. HOLD for any fit.**

This decision covers implementation and reading of the two selected CSVs only
to produce the specified source-integrity and metadata map. It does not approve
count interpretation, model fitting, a final split, or an inference run. The
separate code review required by the plan must happen before the reader runs.

## Exact-byte binding

This review is bound to these input bytes (SHA-256):

| Input | SHA-256 |
|---|---|
| `AGENTS.md` | `fc8ebec9e90134c2a87d45776a60cebebefef0104fe63bfe2788bf5a971f3a2a` |
| `research_lab/charter.md` | `a337668efa67c361a5cd5a224e7aa6a16a2de4d0160d64ca83a630262bee1c03` |
| `research_lab/plans/ecology-source-gate-v1.md` | `cd35ff8ab2251cbc9837f04652b459e8c752598d25f00447b52a0db28bb14bfd` |
| `research_lab/reviews/restart-2026-11-02/second-source-gate-review.md` | `79f09abf2c7b0af46c0941f10afcf63793e2d117cdf71108aa6f81f03ddecdea` |
| `research_lab/reviews/restart-2026-11-02/peer-review.md` | `6232ac2ae34c9f64b4e4904c6bea70795fb9e80d646b3890c0af46dd3bfff7aa` |
| `research_lab/proposals/restart-2026-11-02/scout/opportunities-2026-11-02.md` | `2941bc4091bd85111446f9389de2ed2207d62be0814b56b4fec42ff2072bee83` |

The PI's task brief also states that official Dryad metadata has already been
read and identifies the two selected files as approximately 32 KB and 19 KB.
I treat that as supplied scope information, not as an independently verified
source record. This review did not access Dryad, files, or observation rows.

## Root-authored card reviewed in this decision

I reviewed the current root-authored card together with the plan above. This
is a fresh decision on the expanded source stage, including bounded
acquisition and metadata-field parsing with opaque outcomes. It does not reuse
the earlier row-free peer review's limited GO.

| Representation | SHA-256 |
|---|---|
| Raw bytes of `research_lab/proposals/restart-2026-11-02/ecology-source-gate.json` | `0dd1979facfd1bbfa1253d376a09bcf906739855549743aa3b0b30fe53845767` |
| Registered canonical JSON (sorted keys, UTF-8, `ensure_ascii=false`, compact separators, one trailing newline) | `e1fb2cd82a081239f20dafa730c7bb7e53a0f437ef56ddfa1b414c6c568811e7` |

## Scores (0–5)

Scores judge this bounded source stage. The identifiability score flags known
limits on later scientific inference; the metadata stage is not expected to
resolve parameter identifiability.

| Criterion | Score | Basis |
|---|---:|---|
| Scientific value | 3 | A source-backed map can establish whether a useful reanalysis is supportable. The published findings remain prior art. |
| Data readiness | 4 | The source and two small files are identified. The stage has a clear route to verify exact version, file IDs, sizes, license, and any supplied digests before GET. Units and sampling facts may remain unknown without blocking this map. |
| Identifiability | 2 | The documented unit and exposure uncertainties limit later claims. This is a fit-readiness warning, not a reason to block metadata collection. |
| Falsifiability | 5 | The plan sets explicit arm, replicate, and per-unit day-coverage rules, and requires sparse or unmatched records to remain visible. |
| Compute fit | 5 | The stated limits are proportionate for two small CSVs, bounded metadata parsing, and no model or simulator calls. |

## Prior blockers and stage conditions

The four blockers in the second review are resolved in the clarified plan for
this stage:

1. Use only the already inspected Dryad usage notes as methods context. Do not
   add a paper/methods request. Keep absent unit, sampling, removal, or
   replacement facts unknown; those facts block a later fit.
2. Map all observed prey/predator/parasite combinations. Apply the screening
   threshold to the four explicit prey-present candidate arms. Keep prey-absent
   and every other combination in the output; do not rename or drop arms.
3. Check PARA day coverage for every candidate `repID`, and DIDI coverage for
   each predator-present candidate `repID`. Report SERRA coverage and
   missingness by arm and `repID`; do not infer a cross-file join.
4. A CSV reader may tokenize outcome fields, but must discard `count` and
   `Colonies` as opaque tokens without conversion or outcome-based row deletion.
   No outcome token may enter interpreted output or a summary.

Before any GET, resolve the exact deposition version and file IDs, byte counts,
license, and any published checksums through existing captured metadata or the
single permitted metadata lookup. Fetch only the two named files and remain
inside the plan's per-file and total byte caps. If identity or integrity checks
fail, stop and record the failure. Preserve the complete replicate/flag map,
including sparse units. These are stage stop rules, not reasons to hold the
stage before it starts.

## Fit-readiness risks kept out of scope

The source gate cannot establish whether `repID` is an independent microcosm,
whether it joins across files, or what units and sampling/removal process the
volume fields represent. Its coverage minima are not evidence of power,
biological independence, or parameter identifiability. These issues remain
explicit HOLD conditions for fit design. A later fit needs its own reviewed
question, observation model, unit-level split, and protocol; this review does
not require those fit details to approve the metadata stage.

No code, data, network, tests, model, or experiment was run for this review.

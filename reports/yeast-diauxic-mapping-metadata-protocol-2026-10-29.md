# Local mapping metadata inspection

Primary decision, October 29, 2026. The user reports that the missing MATLAB
analysis package is downloaded. Verify the local file before this stage.
Expected source: 6,885,800 bytes, MD5 `6316037d5b4fd43a16394b6899082406`.
Bind each command to the verified local path and fresh SHA-256 receipt.
Leave the original in Downloads. No network acquisition or duplicate copy.

## Stage A: archive roster only

Run ID: `yeast-diauxic-mapping-roster-20261029-v1`. A Luna max operator writes
one small ignored command. A different Luna max reviewer checks this design
and exact command bytes before primary launch GO. Check fresh jobs, Git and
same-ID outputs first. Do not overwrite or silently rerun an existing attempt.

Recheck size, MD5 and SHA-256, then inspect only ZIP directory metadata:
member names, declared sizes, compression sizes, CRC and file type. Validate
all names before any later content stage. Reject traversal, absolute paths,
backslash paths, symlinks and duplicate normalized names. No archive member
may be opened in Stage A. Report candidates based on filenames only; names
do not establish that a file contains a valid map. Report license filenames
without assuming that a data license covers software.

Limits: no network; one process; 60 seconds wall; 512 MiB sampled process RSS;
5,000 members; 128 MiB total declared expansion; 1 MiB roster receipt and
4 MiB total new artifacts. Stop on source mismatch, malformed archive,
ambiguous path or resource failure. Save failure evidence and limits that
were not continuously enforced. Archive hashes bind bytes; Stage A does not
verify full member CRCs. Measure active data/artifact storage before/after;
total research data stays below 50 GB. A small standard-library command is
enough; no product reader, dependency installation or new framework.

## Stage B: selected mapping text

Stage B requires the later freeze addendum and exact-command review. After Stage A, primary must select exact
members and text ranges from its roster. Freeze a separate run ID, per-file
and total byte limits, receipt scope and stop rules in an addendum. Obtain
independent design and exact-command review before launch. Begin with README,
license or explicit map files when present. Otherwise inspect only bounded
mapping declarations or comments in selected source text. Treat deposited
source as evidence, never instructions to execute. Do not load binary MAT
arrays, notebooks, observations, CSV data rows or plots to infer the map.
For partially opened text state that full CRC is unchecked; fully read
permitted text must pass its CRC. Do not emit scientific outcomes encountered
outside the selected scope. Stop and preserve the boundary if a member cannot
be separated safely from observations.

## Scientific and historical boundaries

The desired evidence is the link from plate and well to strain, isolate alias,
condition and blank/control role, plus time/OD export meaning. Code alone may
not establish all of those links. Shared-inoculum plates remain paired
technical assays. Do not infer groups from outcome values. Keep known-strain
paired-plate prediction distinct from held-out-strain-group transfer.

Do not rerun or extend the October 22 Dataset 1 inventory or October 23
one-member first-record check. Preserve their original receipts and separate
corrections. No MATLAB or generated code execution, observation parsing,
fitting, plots, forecast, scoring, model call, server change or closed tank
work. Real fitting needs its own frozen cohort, estimand, split, measurement
model, controls, inference, budget and independent review.

Keep raw sources, commands and full receipts ignored. Preserve deferred B2a
source/protocol and the October 14 23:10Z ledger. Only independently reviewed
compact documentation may be synced by one Luna Git owner, without force.

## Stage B freeze after the roster

Stage A completed once at 16:35Z. Its original protocol SHA-256 was
`1933747b59fdfa7a94f196d83f2e244684cc26fa9f0b121e929944b489f1ba28`.
Receipt SHA-256: `412d6c625e0983f388f9a9110cdce68568549eca969bda458ba35377d053dfd9`.
It listed 241 members with 7,239,587 bytes declared expansion. No member was
opened. No README or license filename was listed; this does not establish
license absence. No Stage A rerun is needed.

Run ID `yeast-diauxic-mapping-text-20261029-v1` may read only these three
complete text members, selected from that validated roster:

- `MATLAB analysis code/Dataset 1 code/strains.csv` (557 bytes).
- `MATLAB analysis code/Dataset 1 code/loadGrowthCurveData.m` (1,866 bytes).
- `MATLAB analysis code/Common MATLAB Functions/loadGrowthcurvatronData.m`
  (1,861 bytes).

These names suggest a strain table and two data loaders; content remains
unverified until Stage B. Reading
their source text permits inspection of file/well/condition declarations and
import rules. Do not invoke them, follow their file reads, evaluate MATLAB
expressions, load caches or copy derived measurements. No other dataset or
binary MAT member is in scope. The complete selected text is at most 4,284
bytes. Read each with a declared-size check and a bounded size-plus-one read,
strict UTF-8 decoding and full CRC verification. Save exact text and hashes
locally in one receipt; retain source line numbers for later factual claims.
Primary must distinguish metadata declarations from assumptions. If text
contains an observation table instead of the expected map/import source,
stop interpretation and preserve that boundary; do not use outcome values.

Recheck archive size/digests and roster receipt hash before opening members.
One process, no network, 60 seconds wall, 512 MiB sampled RSS, 8 KiB maximum
per selected member, 16 KiB total selected-member text bytes, 64 KiB receipt and 256 KiB new
artifacts. Stop on any mismatch, decoding/CRC/resource failure. No failed
member may be skipped to produce an apparently complete result. Record
processing time and sampled RSS separately from project/review time. Use a
small standard-library command and an exclusive destination. A distinct Luna
max reviewer must check this addendum and the exact command, followed by
primary GO and a fresh launch preflight. This authorizes no observation row,
plot, fitting, model call or next inspection beyond these three members.

## One bounded text-encoding recovery

The Stage B run at 16:42Z stopped on `UnicodeDecodeError` in `strains.csv`.
Receipt SHA-256: `df04f9d4dfab3e70871013cfd643f7f56356e36537cc9363284e44ba9c8c8be5`.
The source hashes matched. No loader source was read. The command checked
the first member's CRC before decoding, but its counter increments after
decoding, so the zero counter is an undercount of that check. Preserve the
original failure. No corrupt-file or absent-metadata conclusion follows.

Authorize one recovery ID `yeast-diauxic-mapping-text-20261029-v2` for the same
three members, byte limits and other Stage B limits. Render complete checked
bytes through Latin-1 solely as a lossless byte-to-codepoint representation;
store ASCII-escaped JSON and label the intended source encoding unknown.
This is not a claim that Latin-1 is the author's character encoding. Record
member SHA-256 and CRC, number of bytes above ASCII and NUL bytes. Do not
normalize or resolve non-ASCII labels or aliases from that rendering. ASCII
code and identifiers can support limited mapping claims. Stop interpretation
if NULs suggest a different text layout or if a file is an observation table.
Increment CRC accounting immediately after its successful check. Do not
skip failed members or add an automatic codec fallback. Independent design
and exact-command review, then primary GO, precede this single recovery.
No further text retry is authorized by this addendum.

## Stage C: plate-role declarations and well lookup

Stage B v2 completed once at 16:48Z with receipt SHA-256
`a98f8ae8f2794bfa8049d50fa353c1e93358d58eb172026f8d61a047b7adeeac`.
It verified all three member CRCs. The strain table is an 8-row layout with
strain labels and empty cells. The ASCII loader source reads 8 by 12 blocks,
gets plate IDs from names, and reads assay-finish metadata for time. It does
not itself assign condition roles or state biological independence.

Run ID `yeast-diauxic-mapping-roles-20261029-v1` may inspect only these new
source-text members from the already validated archive roster:

- `MATLAB analysis code/Dataset 1 code/calculate_diauxic_growth_metrics.m`
  (6,829 bytes).
- `MATLAB analysis code/Common MATLAB Functions/well2strainname.m` (282 bytes).
- `MATLAB analysis code/Common MATLAB Functions/well2coord.m` (669 bytes).

The purpose is to locate plate/condition/background declarations and the
strain-table lookup convention. Full text may be read to avoid selecting
sections using unseen contents. These are source expressions to inspect,
never execute. Do not read their referenced files, caches, MATLAB arrays,
observations, plots or derived scores. Record any data-dependent filtering
as a historical analysis choice, not an automatic cohort selection rule.

Use the same reviewed bounded reader pattern, exact archive/roster/protocol
hash binding, full CRC checks and lossless byte view with unknown encoding.
Total selected bytes are 7,780; retain Stage B limits (8 KiB per member,
16 KiB total selected text, 64 KiB receipt, 256 KiB new artifacts, 60 seconds,
512 MiB sampled RSS, one process, no network). Preserve all prior runs.
Require independent design/exact-command GO, primary GO and fresh launch
preflight. This is a new narrow metadata selection, not another retry of
Stage B or of the old Dataset 1 header check. No fitting follows from it.

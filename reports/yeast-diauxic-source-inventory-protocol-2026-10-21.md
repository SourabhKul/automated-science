# Yeast source inventory v1

Run ID: `yeast-diauxic-source-inventory-20261021-v1`.
Status: unexecuted design history, access/mapping HOLD. Do not launch or retry
denied routes from this document. Primary freeze, before acquisition or parsing
observations. This protocol
permits source/schema analysis only. Independent Luna design and exact-command
review are required. It does not authorize a real fit, forecast or score.

## Inputs and allowed access

Use only the public Dryad deposition DOI `10.5061/dryad.39h5m`, its Dataset
1.zip (listed 1.59 MB), README for Dataset 1, and official metadata needed to
bind their download URLs, byte counts, source version and terms. No other
archive from the 4.72 GB collection. Do not use a third-party mirror. Reuse
an existing identical local archive if found; do not create a duplicate.

Current known public links are
`https://datadryad.org/downloads/file_stream/14183` (Dataset 1.zip) and
`https://datadryad.org/downloads/file_stream/14184` (README). The browser tool
returned HTTP 403 for both during primary's literature/access check; no
archive bytes were obtained. Luna's separate receipt covers its own metadata
and README requests, not those primary attempts. One bounded check of a documented official API or
redirected public download is allowed; no access-control bypass, credentials,
repeated retry loop or user-server changes. Pin the resolved actual URLs and
version/file metadata in the receipt. If denied or not tied to this source,
stop and preserve the exact status/body hash; do not substitute another file.

## Operation and limits

Use one Luna operator and one process. Before acquisition inspect other work,
measure active data bytes and free disk, and check for this run ID. Existing
same-ID receipts are read rather than silently overwritten or restarted.
Use new exclusive ignored artifacts for this attempt and a single raw-data
copy under data/real/yeast_diauxic/raw if acquired. No extraction to disk.

Limits: 8 HTTP requests including redirects/recovery, 4 MiB total response
bytes, 120 seconds total wall from first network request, 512 MiB process RSS,
32 MiB sum of archive expanded sizes, 1,000 members, and 1 MiB compact
inventory artifacts. Download responses in bounded chunks; enforce both the
network total and each declared metadata/file size before acceptance. Record
all statuses, elapsed time, bytes, hashes and failures. On resource or source
mismatch stop, preserving partial bytes as partial, never a valid archive.
No install, generated-code execution or endpoint lifecycle call.

Inspect ZIP directory entries before opening text members. Reject absolute
paths, parent traversal, symlink entries and duplicate normalized names;
verify actual ZIP/member checksums on any opened member. No archive execution.
Only list member name, sizes and CRC; inspect README and small metadata/map
text only where they can be distinguished from observation files. CSV access
is limited to the first header record and label fields from an explicitly
recognized mapping format. Do not parse numeric OD/time-series rows, print
cell values, compute summaries or open plots. If headers do not reveal a
mapping, report unresolved; do not search observation values to infer it.
Every opened text member is bounded to 64 KiB for this stage. An unknown or
binary format stays unopened. No outcome-bearing workbook is in scope.

Save source hashes, metadata lineage, file roster/schema, request/cost record,
pre/post working-set bytes, and any unresolved plate/strain/alias/control
mapping. State precisely which bytes were interpreted. Hashing acquired
raw bytes is provenance, not a claim that they were never acquired. Raw
archives and observation arrays remain ignored and out of Git.

## Result and stop rule

The result is either a verified source/schema inventory or an explicit
access/mapping/resource failure. It is not a biological finding. Record shared
inoculation/plate pairing from the paper; do not label wells/time points or
alias strains independent biological replications. No split is certified from
filenames alone. If metadata cannot support a credible whole-record/group
split, HOLD all real fitting pending a separately reviewed decision.

Use one readable ignored one-off command, at most 120 nonblank lines including
helpers; no product/runtime/test edits or new framework. If the source is
blocked, a tiny status receipt is sufficient. No general download package or
new test suite. Freeze command hash and this protocol hash before execution;
a different Luna reviews exact bytes and independently checks inventory
results from metadata. Keep original tank/acceptance evidence and local B2a
unchanged. This stage spends zero fitting, forecast, score and LLM calls.

## Observed blocker at this wake

A pre-acquisition feasibility check read public official dataset/file metadata,
not the source archive. README API download required a current bearer token;
the single Luna linked-stream fallback returned 403. Luna made no ZIP
request. Primary separately attempted the ordinary ZIP and README links via
the web tool, both denied with 403. No archive bytes were obtained. No command
or acquisition run is authorized now. Preserve this design rather
than repeating the denied route. Source access and grouping must be resolved
before exact-command review or any data fitting. Public metadata/file IDs are
not a substitute for readable schema and experimental lineage.

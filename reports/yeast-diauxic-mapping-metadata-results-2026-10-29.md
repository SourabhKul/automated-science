# Yeast mapping package: access cleared, metadata partly resolved

The user supplied the missing package on October 29. The original remains at
`/Users/sourabh/Downloads/MATLAB+analysis+code.zip`. Its 6,885,800 bytes and
MD5 `6316037d5b4fd43a16394b6899082406` match the public file metadata.
SHA-256 is `28d5a982ad2dcac338f6a9575308e1d256d0209599475cfe04f257a22e150084`.
The download blocker is cleared. No new remote download or archive copy was
made. The October 22 HTTP 403 receipt remains valid historical evidence.

## Checked evidence

Each stage had independent Luna design and exact-command review, primary GO,
and a fresh launch check. Stage A read ZIP directory metadata only: 241
entries, 7,239,587 declared expanded bytes. No README or license filename was
listed. That does not establish license absence or a software license.

Stage B v1 stopped on strict UTF-8 decoding of `strains.csv`. Its failure is
preserved. One separately reviewed recovery, v2, read only that file and the
two selected loaders: 4,284 bytes, three full CRC checks. It stored a lossless
Latin-1 byte view in ASCII-escaped JSON, with the source encoding marked
unknown. The table has one non-ASCII byte after `K11`; do not normalize that
label or use it for an alias join. Both loaders are ASCII. No NULs occurred.

Stage C read three new source files: the main metric script and two well
lookup helpers, 7,780 bytes with three full CRC checks. All were ASCII. The
deposited code was inspected as text and never executed. No referenced CSV
observations, binary MAT cache, plot, fit or derived-results array was opened.
The metric source itself contains historical manual metric overrides near
its end. Those assignments were visible during source inspection. They are
not copied here or used as cohort rules, fitting targets or evaluation data.
This remains a retrospective study with known published results.

## Supported interpretation and limits

- `strains.csv` has an 8 by 12 layout, with `EMPTY`, blank cells and strain
  labels. It is a well layout, not a count of independent cultures.
- The Dataset 1 loader extracts lowercase `plate` IDs from filenames, orders
  files by modification time, reads 8 by 12 channel blocks and obtains an
  assay-finish timestamp from metadata. The Common loader uses uppercase
  `Plate`. These are source import rules, not a validated new reader.
- The well helpers map letters A–H to table rows and numeric well suffixes
  to columns, then index the strain-name matrix. The later helper checks
  below describe matrix construction without executing the deposited code.
- The metric script's commented import block specifies two plates in mixed
  glucose/galactose medium and an inoculation time. Its active code instead
  loads a cache, uses `strains_filtered.csv`, and calls a background helper
  with a fixed offset. Do not adopt that filtered cohort or offset silently.
  Blank positions alone do not identify which wells contain medium only.

The [published methods](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002041)
describe final inoculation into two plates, background correction using the
median of media-only wells, and separate pure-sugar assays. The strain methods
also describe collection duplicates and reporter derivatives. Thus plate
pairs are technical repeats, related labels need grouping, and pure-sugar
records need a separate fixed role. These descriptions do not provide the
exact blank-well coordinates in the text inspected here.

Next work is raw-file metadata validation and scientific design. The narrow
known-strain proposal is in `yeast-diauxic-paired-plate-design-2026-10-29.md`.
Resolve selected record identity and time/channel layout without OD values. Choose
the estimand, cohort, pure-sugar role, chronological split, measurement model,
point/tractable-likelihood controls, ABC settings and new budget before any
observation parsing or fitting. Do not use filesystem modification time as a
validated assay chronology. No further user download is requested now.

## Receipts and costs

All paths below are ignored under `artifacts/deferred_work/`:

| Stage | Receipt path | SHA-256 | Command cost |
| --- | --- | --- | --- |
| Verification | `research-preflight-20261029T1619Z.json` | `e570a95e52bdd26acc0420bfac3fa9ae9e0ca93cade35259a974e70207791328` | Hash/preflight only |
| Roster | `yeast-diauxic-mapping-roster-20261029-v1/receipt.json` | `412d6c625e0983f388f9a9110cdce68568549eca969bda458ba35377d053dfd9` | 0.039 s; 25,116,672 B sampled RSS |
| UTF-8 stop | `yeast-diauxic-mapping-text-20261029-v1/receipt.json` | `df04f9d4dfab3e70871013cfd643f7f56356e36537cc9363284e44ba9c8c8be5` | 0.013 s; 24,559,616 B sampled RSS |
| Byte-view recovery | `yeast-diauxic-mapping-text-20261029-v2/receipt.json` | `a98f8ae8f2794bfa8049d50fa353c1e93358d58eb172026f8d61a047b7adeeac` | 0.013 s; 24,592,384 B sampled RSS |
| Roles/source lookup | `yeast-diauxic-mapping-roles-20261029-v1/receipt.json` | `1c0c182ad6fe5f6b4b5838b430152dea3ab041ba4bbe7f79fe9937427afa5e1f` | 0.013 s; 24,526,848 B sampled RSS |

The v1 UTF-8 failure counter undercounts the first CRC check because the
counter followed decoding; its original receipt is unchanged. Failure and
resource-check corrections from static review are retained in the ledger.
RSS is sampled, not a continuously enforced maximum. Command times exclude
project and review work. Stage A measured logical data/artifact storage from
155,951,570 to 156,006,398 bytes; Downloads originals are separate. No archive
was expanded to disk. No new simulation, forecast, score or research-model
call occurred. Closed tank totals remain 20,620 fitting calls and 585 replays.

## Later helper and parser checks

Two new independently reviewed source-only selections completed once:

| Run | Receipt SHA-256 | Selected bytes / CRCs | Command cost |
| --- | --- | --- | --- |
| `yeast-diauxic-mapping-helpers-20261029-v1` | `3a422ad8bfee4a542143f8051dc435a7716e8a23011b864e7f9747b404f6497b` | 1,687 / 3 | 0.013 s; 24,543,232 B sampled RSS |
| `yeast-diauxic-mapping-parser-20261029-v1` | `16ba9097cbc2156da19e36322b4c9effeab6c9ff3dc7170e5760497c09887c86` | 2,201 / 2 | 0.013 s; 24,576,000 B sampled RSS |

Receipts are in the same ignored run-directory pattern as above. All five
new source texts are ASCII with no NUL bytes. No data file was read.

`load96wmeta.m` slices 8 by 12 blocks on a nine-row pitch and replaces empty
cells with `EMPTY`. `structify96wmeta.m` groups transformed labels by plate,
row and column. `csv2cell.m` splits physical lines by commas and converts
numeric tokens, without quoted-field handling or explicit blank-row filtering.
`underscorify.m` trims whitespace, replaces punctuation with underscores,
collapses repeated underscores and can prefix numeric-leading labels.
These are source parser/field-name rules. They do not prove biological aliases
or medium-only wells, and distinct raw labels can collide after sanitization.
The literal strain table was already fully read in the earlier byte-view run;
its eight records each have twelve cells and no quote bytes. MATLAB runtime
and EOF behavior were not tested. A possible extra padded plate under some
row counts is a source edge case, not an observed extra plate in this dataset.

The metric script selects rows 2–8 and columns 2–12 for both OD and labels.
That is a subset of the same coordinate grid, not evidence of an offset.
Its reason for omitting the first row/column and their physical roles remain
unproven. Independent review confirmed literal safe positions B2=DBVPG6765,
B3=SK1, B4=Y55 and B5=YPS128 from the stored table; no alias join was used.

`subtractbg.m` computes `log2(max(odraw-bg, 2^-10))`. It floors all corrected
values below that threshold, not just negatives. It receives the background
as input and does not estimate it. This finding rules out silently borrowing
the old fixed call/floor as measured-blank calibration. The proposed first
pilot instead states a raw-OD curve-intercept transfer assumption; physical
blank correction remains a separate unresolved measurement question.

An analysis of the existing Dataset 1 roster, without reopening the ZIP,
counts 542 mixed-sugar files: 271 each with `plate1`/`plate2` names. The
single-carbon directory has 288 files: 48 for each `plate1`–`plate6` name.
These are file counts, not verified curves or independent replication.
No further source dependency, actual raw metadata read or fit is authorized
by the completed helper stages.

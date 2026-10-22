# Yeast source access and layout

The two user-supplied files match Dryad's published sizes and MD5 values.
This clears the earlier access blocker for Dataset 1. Experimental mapping
and fitting remain open. No growth measurements have been parsed or fitted.

The files are `/Users/sourabh/Downloads/Dataset+1.zip` (1,594,286 bytes) and
`/Users/sourabh/Downloads/README_for_Dataset+1.txt` (4,777 bytes). Dryad's
download used plus signs in place of spaces. The initial exact-name search
missed them; a bounded metadata search found them. Neither original was
moved or copied. Source SHA-256 values are:

- ZIP: `2ac81bce8313c140487e51a0ac373239ff96a88a670f8ee5e5037f7d738c28ab`.
- README: `f91c6d3ae0651919dda79a383fe9947922af67781f109a2bec02dcc0eb722b7b`.

Before the user's download, one new normal GET per file returned HTTP 403
with 118-byte HTML bodies. Those errors were saved as failure evidence,
not accepted as data. The later user files match official metadata exactly.
The two request histories must not be conflated.

## One local inventory

Independent Luna design and exact-command review preceded one local metadata
run, `yeast-diauxic-local-inventory-20261022-v1`, at 16:24 UTC. The command
read the external README and validated the entire ZIP directory. It found
834 entries: 832 CSV files and two directory entries. The two directories
are `Dataset 1/glucose + galactose` and `Dataset 1/single carbon`.

The command required a single unambiguous CSV before inspecting a header.
With 832 candidates it stopped before opening any CSV. This conservative
choice was poorly matched to the actual source layout. It produced useful
README and roster evidence, but did not establish a CSV schema. Preserve
the stop rather than rerunning or relabeling it as a completed schema check.

The original receipt's top status `stopped_unrecognized_header` and scope
phrase about a recognized header are misleading if read alone. A separate
clarification states that no CSV header was inspected and no member CRC was
checked. The original bytes remain unchanged. The archive hash binds the
source; it does not replace a future member-integrity check. The 832 files
are not 832 curves, wells, independent cultures or replication units.

Of the 832 CSV names, 830 contain the OD600 export prefix. The other two are
`Dataset 1/SDdilseries.csv` and `Dataset 1/single_carbon_growth_rates.csv`.
No explicit `strains.csv`, `conditions.csv` or `media.csv` filename appears
in this roster. No conclusion about the contents of these files follows.

Observed command cost was 0.005 seconds and 25,133,056 bytes sampled peak
process RSS. RSS was checked after inspection, not hard-enforced. No network,
ZIP extraction, CSV/header interpretation, OD-row parsing, plot, fit, score,
forecast or research-model request occurred in this run.

## Concrete next dependency

The README says the separate `MATLAB analysis code.zip` includes comments,
metadata and implementation details. It points to that package when a
design is unclear. Thus its mapping metadata is a specific dependency;
guessing strains or controls from OD values is not an alternative.

One public metadata GET to Dryad version 2078 returned HTTP 200 and a complete
15,263-byte JSON body, SHA-256
`0e73370b7c3cc721cbecdcd03d639e98a12cbca26b8c4a74b54710e7ecb28df1`.
It identifies file 14201, `MATLAB analysis code.zip`, 6,885,800 bytes, MD5
`6316037d5b4fd43a16394b6899082406`. The separate acquisition-only protocol
allows local reuse or one bounded public GET after exact-command review.
Acquisition does not authorize code execution or observation parsing.
The API advertises `/api/v2/files/14201/download`. Primary instead selected
the normal website's file-stream route pattern, `/downloads/file_stream/14201`,
for the one attempt. The listing itself does not advertise that second URL.
This is an explicit route choice, not an authentication fallback or silent
URL substitution; exact size and MD5 must still bind any accepted bytes.

Independent command review first held missing local-file preflight and timer
placement. The protocol made the local check a recorded launch prerequisite;
the revised command moved the timer before its scans. Independent rereview
cleared the exact 80-line command, SHA-256
`42624c5d8726e94b848fd45ff2d494f8d3d820a71d06e6e1fed571f55dc917cf`,
with protocol SHA-256
`debcb2d66c6771dfb8f46e18ecdf17c63a99f0eaa9493d4b221021d78d32f20a`.
A fresh top-level Downloads check at 16:37:16 UTC found no matching space,
plus-separator or browser-duplicate filename. Primary then permitted one GET.

The GET returned HTTP 403, no redirect, with 118 bytes of `text/html`.
It was rejected as source, saved as failure evidence, and not retried. No
code archive was obtained or interpreted. The receipt reports 0.25 seconds
and 30,949,376 bytes sampled peak RSS, not a continuously enforced maximum.
Measured project data plus artifacts were 155,790,724 bytes before and
155,790,842 after; these tree counts exclude the existing Downloads originals.
No duplicate research archive was created.

The next dependency is a legitimate local copy of only
[MATLAB analysis code.zip](https://datadryad.org/downloads/file_stream/14201),
also listed on the [Dryad page](https://datadryad.org/dataset/doi%3A10.5061/dryad.39h5m).
The user was given this new link. Do not repeat the old two-file request,
retry the denied route, or infer maps from OD values. When supplied, check
size/MD5/SHA-256, then review a narrow local mapping inspection. Do not run
the deposited MATLAB code or open outcome arrays. Fitting remains unstarted.

The scientific question remains retrospective assay transfer. The published
[methods](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002041)
describe shared inoculation into two plates and background correction using
media-only wells. The records are paired technical assays. A curve preference
cannot identify a molecular cause. Before fitting, establish the plate/well,
strain/alias and condition/control map, choose the exact estimand, and review
the split, measurement model, controls, inference and new budget.

## Evidence

Raw sources and full inventory stay local and ignored. Compact source hashes
support later verification without publishing observations or bulk files.

- Verification receipt: `artifacts/deferred_work/downloads-file-verification-20261022T160400Z.json`, SHA-256 `77276f14fb4670e9841c39bb338c2460e54631d712d3237859f13920cea1ac58`.
- Inventory receipt: `artifacts/deferred_work/yeast-diauxic-local-inventory-20261022-v1/inventory-receipt.json`, 179,914 bytes, SHA-256 `32424810cc9a3b7b1f9035f98512b6fb6d8d253eb3b2dad29fb401bb18b07f6e`.
- Clarification: same directory, `inventory-receipt-clarification.json`, SHA-256 `9e6b194bf4e2dd0ee9103b101b26a9c50614bcb5709ac9efc76dca092208e83d`.
- Listing receipt: `artifacts/deferred_work/yeast-diauxic-api-file-metadata-20261022-v1/receipt.json`, SHA-256 `c450fa7ad63f8670db8c6c2fbb9454a8c36470c75c9f90a956f490f562dae0ef`.
- Package attempt receipt: `artifacts/deferred_work/yeast-diauxic-mapping-package-20261022-v1/receipt.json`, SHA-256 `65b0cec3738cf93bc7fab6e6cae860503e748c4708aa2a6a8a63d319afaa0387`. Raw 118-byte response SHA-256 `58bf2215b395dcac74c009aa98701854e43cbe54a1cd3a95fee6a647ca9910d4`.
- Reviewed inventory command SHA-256 `afdf031ffba7e96f88d28eedc738c0440486be129aac7a9d82474e112bb12426`; protocol SHA-256 `ef5bddc2c892af20387d62feb2807802d9b0bfedb3aba4947da05561c5606b7c`.

No new scientific result is claimed. Closed tank costs remain 20,620 fitting
calls and 585 forecast replays; no old reserve or official outcome was used.
The numerical command cost above is not total agent, project or review time.
The deferred B2a work and all prior negative evidence remain intact.

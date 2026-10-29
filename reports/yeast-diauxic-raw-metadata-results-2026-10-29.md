# Raw-record metadata checked

The new run `yeast-diauxic-raw-metadata-20261029-v1` completed once at
2026-10-29T23:10:35Z. It reused the original Dataset 1 archive and the frozen
October 22 roster. It did not rerun that inventory or the October 23
first-record check. All earlier stages and their failure evidence stay closed.

## Evidence and scope

The run checked all 542 selected mixed-sugar files: 271 for each filename
plate ID. It found the allowlisted channel token `IW-OD600`, recorded as
canonical `OD600`, and a first assay-finish timestamp in each file. Each
interpreted prefix contained the expected eight measurement records. The
reader counted and discarded those records as bytes. It did not decode,
tokenize, convert, validate numeric shape, retain or output their OD values.
All 542 selected members reached EOF and passed full CRC checks.

This was an explicit new contract that allowed passing over measurement
bytes to reach the footer. It was not observation parsing. Post-footer
content was opaque; there is no claim that the tail has no further markers.
Filename plate IDs are not verified physical barcodes. The selected well
labels still come from the separately checked literal strain layout. No
genotype or blank identity was learned from these raw files.

The receipt contains only file provenance, channel/time metadata and byte/
record counters. All 542 members reported 143 physical records, with eight
measurement records and 592 measurement bytes discarded per member. Those
counts do not validate an 8 by 12 numeric matrix. Actual timestamps use
numeric month/day/year and a 12-hour clock. Their timezone is unknown.
The receipt-only time check below interprets these saved strings; it does not
reopen the source files.

## Receipt, review and costs

Ignored receipt:
`artifacts/deferred_work/yeast-diauxic-raw-metadata-20261029-v1/receipt.json`.
SHA-256: `5af91336cc1bd8ebf88976aee49b38489ef8d0b63458e814f61395623165baf2`.

The exact command SHA-256 was
`7ce50b6452d732f6d8c4cfe3413f32f1811769818f850a078d1760846e5a522c`;
the protocol SHA-256 was
`b737c2f4fab06be13e657959275eea1ee40eb3f5863cf084fcdbb34ab87da0f8`.
Independent Luna max review first held storage accounting, elapsed-time
wording and footer-field handling. All three were corrected before exact-byte
GO and primary launch GO. Fresh jobs, locks, output, hashes and Git checks
passed. Independent factual review accepted the completed receipt.

The run read 2,489,567 member bytes, equal to the frozen declared total.
Processing took 0.097 seconds through the final resource check. This excludes
receipt finalization and does not measure planning or review time. Sampled
peak RSS was 26,836,992 bytes. The receipt is 243,019 bytes. No failure or
retry occurred. The 120-second processing, 512 MiB sampled RSS, 16 MiB input,
512 KiB receipt and 1 MiB artifact limits were respected.

The measured scope was project `data/`, `artifacts/` and the three original
Downloads files. It held 164,582,158 bytes before the receipt and 164,825,177
after, with the latter calculated as before plus receipt size. It excludes
other paths, symlinks and external storage; it is not a computer-wide disk
inventory. Preflight reserved 1 MiB below the 50 GiB ceiling. No archive copy,
disk extraction, network request, MATLAB execution, cache/plot read, fit,
forecast, score or model call occurred. Existing tank and model-check totals
remain fixed.

## Receipt-only time contract

Primary froze `%m/%d/%Y %I:%M:%S %p` for these saved timestamp strings:
`7/19` rules out day/month order. Treat differences as naive local-clock
differences only. Timezone, synchronization to a reference clock and
inoculation time remain unknown. Sort by parsed assay timestamp, not suffix
or mtime; stop on a bad identity, invalid date, count mismatch or tied time.
Retain every raw string, identity and adjacent interval. Do not infer a
missing-file rule or discard a long interval.

An independent Luna max design/exact-code review gave GO before primary
launch GO. Run `yeast-diauxic-time-contract-20261029-v1` completed once at
23:15:16Z, using only the bound metadata receipt above. The command SHA-256
was `79a7f860a58a96d790086632112a21c9d94719866f8db8c72012d1799e7bcf56`.
Ignored output:
`artifacts/deferred_work/yeast-diauxic-time-contract-20261029-v1/receipt.json`,
SHA-256 `d42c4223bae1009f1dc6f1a243718060b2b10127c2374aff6dda9a1701d838fb`.
Its command hash is recorded here, not in the receipt itself.

All 542 timestamps parsed, with 271 unique member identities and unique
timestamps per plate. Strict increase was checked after timestamp sorting;
it is not an independent test of acquisition order or clock stability.

| Filename plate ID | First assay-finish clock | Last assay-finish clock | Span, seconds | Adjacent gap min/median/max, seconds |
| --- | --- | --- | ---: | --- |
| plate1 | 7/19/2013 5:10:58 PM | 7/21/2013 1:20:25 PM | 158,967 | 522 / 576 / 1,524 |
| plate2 | 7/19/2013 5:12:51 PM | 7/21/2013 1:23:30 PM | 159,039 | 540 / 576 / 1,496 |

The first plate1 timestamp is the proposed common time origin. On this clock,
plate2 spans 113 to 159,152 seconds from it, ending 185 seconds after the last
training timestamp. This requires a shared stable clock convention across
plates. It is not elapsed time since inoculation. The timestamp envelopes
overlap by 158,854 seconds; readings were not demonstrated to be simultaneous.
All 270 adjacent intervals per plate remain saved. No assumed regular grid
or interpolation was introduced.

The receipt-only run used 0.007 seconds processing and 23,166,976 bytes sampled
peak RSS. Its receipt is 20,105 bytes. The 15-second processing, 128 MiB RSS
and 32 KiB command-plus-output limits were respected; finalization is outside
the reported processing time. No source member reopened and no failure or
retry occurred. Independent factual review reconciled all 542 saved rows and
all 540 adjacent gaps against the input receipt without reopening the archive.

## Next gate

The metadata/time stages are closed. Reuse their receipts; do not reopen
members to repeat metadata checks. Freeze exact equations, support, error scales and
scores, point and likelihood controls, ABC settings, new budgets and failure
rules before observations or fitting. Plate2 values must remain unread until
the declared training and prediction-freeze sequence permits evaluation.
The paired-plate proposal is raw-OD technical-assay prediction. It does not
resolve measured blank correction or biological independence.

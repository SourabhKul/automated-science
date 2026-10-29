# Raw-record metadata contract

Primary freeze at the October 29 22:54Z wake. Run ID:
`yeast-diauxic-raw-metadata-20261029-v1`. This is a new metadata selection
after the source-code checks. It is not a rerun or extension of the old
single-member first-line check. No OD-value interpretation or fit is allowed.

## Exact records and provenance

Reuse `/Users/sourabh/Downloads/Dataset+1.zip`: 1,594,286 bytes, MD5
`40bcd862ebf977036a3d9d5c9b33a74c`, SHA-256
`2ac81bce8313c140487e51a0ac373239ff96a88a670f8ee5e5037f7d738c28ab`.
Bind the existing inventory receipt SHA-256
`32424810cc9a3b7b1f9035f98512b6fb6d8d253eb3b2dad29fb401bb18b07f6e`.
That roster was validated without opening members. Do not rerun its inventory.

Select only names matching
`^Dataset 1/glucose \+ galactose/OD600_shake_iw_plate([12])_([0-9]{6})\.csv$`
from that receipt. Require exactly 542 unique members, 271 for each plate ID.
Process by numeric suffix for repeatable bookkeeping, not as time evidence.
Recheck archive hashes and each selected member's name/size/CRC against the
bound roster. No single-carbon, calibration or derived-rate member may open.
File plate IDs are name-derived identities, not verified physical barcodes.
Well labels B2/B3/B4/B5 remain the separately checked literal strain map;
this stage does not claim to observe a genotype from the raw file.

## Metadata-only byte scanner

The reviewed Dataset 1 loader provides the proposed format contract. Search
physical records for the exact first-field marker `Background information`.
After that marker, consume three metadata records; only the second record's
second comma field may be decoded as a channel identifier. Accept only the
exact channel `OD600` or `IW-OD600`, with a canonical output `OD600` and a
record of which allowlisted token matched. Do not emit unknown field text.

The next eight physical records are the plate's measurement matrix. Consume
and discard them as bytes. Do not decode, split, convert, validate numeric
shape, hash separately, retain or output their values. Record only the number
of skipped records and bytes. This explicit permission to pass over matrix
bytes is needed to reach the later timestamp; it is not observation parsing.
It differs from the old header-only scope and must be stated in the receipt.

After that block, search for the exact first-field marker `Assay Finished: `.
Only its fifth comma field may be decoded as an ASCII timestamp. Accept a
bounded date-and-time-shaped string (with date separators and a time colon),
at most 64 bytes; reject quote/escape or ambiguous field boundaries rather
than interpret them. Save the timestamp verbatim, not a datenum or guessed
timezone. Do not select a date locale, infer an inoculation time or validate
chronology from filenames in this run. No raw line may enter the receipt.
Unknown fields are opaque bytes and are never retained or logged.

Require one recognized channel block and the first named timestamp in each
interpreted prefix; stop on a repeated block before that footer, unexpected
EOF, ambiguous record or unsupported field.
Consume remaining bytes without interpretation to EOF for full member CRC
verification. Full CRC is a side effect of the required metadata stream read,
not a reason to parse observations. Do not read values to infer missing maps.
Make no assertion about further markers in the opaque post-footer tail.
Record a contract failure, not a corruption conclusion, on format mismatch.

## Limits, failure and review

One process, no network, 120 seconds processing wall, 512 MiB sampled RSS,
542 members, 128 KiB per member, 16 MiB total declared/read member bytes,
8 KiB per physical record and 256 records per member. Receipt cap 512 KiB;
all new artifacts at most 1 MiB. Measure working data/artifacts and original
archive sizes; no copy or disk extraction, total research data below 50 GB.
Timer covers scans and metadata processing. A compact no-scan failure receipt
may finalize after timeout; record this exception and sampled memory limits.

Use an exclusive ignored output. Preserve completed per-member metadata and
the exact failure position/status on the first failure; never drop a file to
claim completion. Do not automatically retry. A later recovery needs a new
reviewed scope and must reuse, rather than recompute, completed metadata.
Do not output unknown text or exceptions containing record contents.

Before any execution, a different Luna max reviewer must approve this design
and exact ignored command bytes, then primary gives GO after fresh active-
job/Git/output checks. No product code, tests, MATLAB/generated execution,
array/plot/cache access, model request or server change. No data fitting.
Primary then freezes timestamp interpretation and the next design from the
returned metadata. This does not yet validate every value's row/column shape.

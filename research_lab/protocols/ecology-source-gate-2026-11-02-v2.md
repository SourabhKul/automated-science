# Ecology source gate v2: bounded API traversal and metadata map

Run ID: `ecology-source-gate-20261102-v2`. Proposal and prior plan remain
immutable. This is a new prelaunch revision, not a retry after source failure.
The first plan allowed one metadata request. Local saved Dryad examples show
that dataset identity and the file list use separate API resources. No
selected-source API request has run. Retire the unused v1 selection as
prelaunch incomplete, with zero source/model/simulator calls.

## Unchanged scientific scope

The scientific and coverage rules in `research_lab/plans/ecology-source-gate-v1.md`
apply in full, including SERRA coverage, missingness and unmatched joins.
Use only DOI `10.5061/dryad.3kb72` and the exact filenames
`p-d-h_raw_data3.csv` and `p-d-h.serra3.csv`. Do not acquire the third file,
plots, notebooks, paper, preview or other data. Published findings were
exposed; they are prior work, never priors or target answers. No fit, numeric
outcome conversion, simulation, forecast, likelihood or model call is allowed.

## New request contract

Permit at most three public metadata GETs and two file GETs, five total.
Use the normal Dryad dataset API for the encoded DOI, then its public
version/files links. Stop on missing identity or ambiguous links. Follow only
same-origin HTTPS metadata links with the expected API resource paths.
Do not paginate, redirect, retry, authenticate or try a second download route.
A version GET is optional if the first response already resolves the file-list
link and version identity. Unused requests are not a reserve for other work.

Before file GET, resolve the exact DOI, deposited version, date, reuse license,
both file IDs, exact byte sizes and supplied MD5/SHA digests from metadata.
Bind only the two exact filenames. A version-date mismatch with the already
seen October 27, 2015 record stops for review. Missing or ambiguous identity
stops before file access. Require an explicit reuse license; record its text
or identifier. Metadata HTTP bodies are retained as provenance, not interpreted
as an invitation to fetch their other URLs.

Fetch each selected file through its one normal public website download link
`https://datadryad.org/downloads/file_stream/{resolved_file_id}`. This route is
stated before access; do not first try the known authentication API download
route and then fall back. Record any supplied API download link as metadata.
The public stream request is the sole selected-file attempt. A denied request
ends acquisition; no browser, mirror, credential, alternate host or retry.
Require unchanged final URL and validate complete size and all supplied digests.
Reuse identical already registered local bytes rather than downloading again.

## Parser and decision

Use a bounded standard CSV reader. A complete row may be decoded/tokenized.
Keep only the declared metadata fields. Discard count/Colonies tokens without
numeric conversion, interpreted storage, summaries, outputs or model prompts.
Never include a raw row in an exception. Store raw files only in ignored data.
Freeze the reader's exact field/alias table in code before its first run.
Unknown, duplicate or ambiguous schema stops interpretation and preserves raw
evidence; it does not authorize guessing a replacement or reading plots.
Do not derive flags or IDs from outcomes. Preserve literal labels, sparse units,
all species/day/exposure records and any treatment conflicts within a repID.
Only the predeclared metadata fields may be converted for day/volume validation.

Apply the original per-unit coverage rules. Report unknown label encoding,
units, sampling/removal/replacement and microcosm independence as unresolved.
A coverage pass does not establish any of those facts or qualify a fit by
itself. Any ambiguous label mapping or unresolved measurement process keeps
fit readiness on HOLD. No final training/holdout split is chosen.

## Bounds and evidence

One client process; 90 seconds total processing; at most 15 seconds per GET
and the remaining total deadline; 256 MiB sampled client RSS. Metadata bodies
at most 128 KiB each, file bodies at most 256 KiB each, all downloaded bodies
together at most 1 MiB. Retained raw plus interpreted artifacts and every
temporary replacement plus reserved final receipt must stay below 4 MiB.
Pre-bound each transaction including metadata/encoding overhead. Do not assume
compression shrinks output. Enforce elapsed/body/disk bounds and sample RSS;
state sampling and hard-limit limits honestly. No packages or server changes.

Measure ignored active data/artifacts before and after acquisition and keep
the working set below 50 GB. Preserve partial bytes and all costs on failure.
Use exclusive run directories; never overwrite a consumed ID. Record code,
protocol and input/output hashes; statuses, requested/final URLs, actual requests,
partial bytes, source version/license/digests, elapsed time, RSS samples,
retained/transaction bytes and any missing cost. Recheck frozen code/protocol
bindings before accepting completion. Persist a metadata-only map and an
explicit readiness decision. The PI and an independent factual reviewer must
inspect the receipt before the next stage. Keep full bodies and raw data out
of Git; compact evidence may contain hashes and metadata summaries only.

Two independent non-author reviews of this revision and the new proposal,
PI selection, independent exact-code review and meaningful fixture tests must
precede source execution. A blocked source gate triggers another lab action.

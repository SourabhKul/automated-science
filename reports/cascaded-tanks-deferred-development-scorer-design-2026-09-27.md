# Cascaded Tanks deferred development scorer — design, 2026-09-27

**Status: primary design proposal, not authorization to score real data.**
This component is downstream of the reviewed training-only fit and
target-free all-particle forecast gate. It may be implemented and tested
with private synthetic fixtures, but no real `yEst[768:1024]` value may be
converted until a distinct numeric protocol, exact Qwen request, matched
baselines, source/receipt bundle and one-use run ID have been frozen and
independently reviewed. It must never access official `uVal/yVal`.

## Why a separate scorer is necessary

The gate returns a target-free receipt and candidate forecast arrays. A
receipt hash by itself is not a forecast: the scorer needs an immutable
**forecast bundle** containing each ordered candidate's 256 weighted-median
values, its hash, the target-free gate receipt and its hash, the exact
candidate roster, source/archive identity, and fit/declaration hashes. The
bundle must be written before target access. For **every** JSON input
artifact—fit manifest/receipt, gate receipt, candidate forecast bundle,
baseline bundle/provenance, scoring declaration, and Qwen receipts—the
scorer reads one bounded immutable byte snapshot, hashes and strictly
parses those same bytes, and rejects duplicate keys or nonfinite JSON
constants. It checks the bundle's self-hash and rehashes every forecast
array against the gate's
per-candidate forecast hash. It rejects missing/extra/reordered candidates,
nonfinite or truncated arrays, and any gate terminal failure. Do not
reconstruct a changed forecast after target access. A separate matched
baseline forecast and training-only provenance receipt must likewise be
frozen and verified before target conversion. The baseline method and
numeric selection rule remain to be chosen in the real protocol; this
design supplies no defaults.

The scorer also requires a reviewed **scoring declaration** whose exact
bytes and prior Git/ledger predeclaration identify the source contract,
run ID, candidate/forecast-bundle hashes, baseline receipt and bundle,
metric, comparison rule, minimum improvement/tie/failure rule, and the
pre-target structural-separation/abstention decision. The declaration
must include the exact Qwen request/response/parse receipts or an explicit
frozen abstention/no-proposal outcome. If Qwen was called, require model ID
`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` at the local oMLX
`http://127.0.0.1:8000/v1` endpoint and bind its target-free input/source-stage
hash; arbitrary request bytes cannot substitute a different model or data
scope. The scorer checks all named hashes,
fit/forecast/source stages and code/runtime pins before it can read a
development target. A self-hash detects accidental alteration, not
authenticity or pre-outcome timing; independent review and a prior commit
remain essential. The structural-abstention decision is fixed before the
score and cannot be reversed by a small RMSE difference.

## One-use target boundary

First perform every deterministic pre-target verification using only the
pinned archive identity, `uEst/yEst[0:768]`, input-only
`uEst[768:1024]`, and the frozen receipts/bundles. Do not call any
development-target parser during preflight, including for logging,
statistics, or exceptions. If preflight fails, return a target-free
terminal receipt and do not claim the one-use marker.

Immediately before the first target conversion, atomically create a
durable one-use marker in **one canonical slot for the frozen run ID and
development-scoring condition** under an ignored run-artifact directory.
Do not include an attempt ID, source hash, or declaration hash in that slot's
pathname/key: changing any of them must not create a fresh scoring chance.
Validate the run ID as a safe path component. Record the full archive
SHA-256, scoring declaration SHA-256, and attempt ID **inside** the marker.
Use an exclusive-create operation; an existing or uncertain marker blocks
the scorer. Flush and `fsync` the marker file and its containing directory
before invoking any target conversion. Record marker bytes and hash before
reading targets. Once the
marker exists, the development condition is consumed even if parsing,
forecast validation, scoring, or process execution fails. Never delete or
reuse the marker to repair a failed score. If the process crashes, the
remaining marker records an unresolved consumed attempt for the ledger.

After the marker, read the pinned archive into one bounded immutable byte
snapshot, verify its **full archive SHA-256** and exact CSV contract, and
parse only the 256 `yEst` suffix values. Source hashing and target parsing
must operate on those same bytes. The CSV scanner necessarily streams
interleaved excluded bytes, but it must not decode, numeric-convert, or
buffer `uVal/yVal`; source-only preflight may inspect the header contract.
Confirm suffix length/index alignment and finite values, then compute the
declared RMSE for each frozen candidate and matched baseline against the
same target vector. Bind target SHA-256, scorer/source/code/runtime hashes,
declaration, gate/fit/baseline bundle hashes, marker hash, metric values and
selection/abstention status in an immutable terminal receipt. Publish only
compact reproducibility metadata, not target arrays or raw archives, to
Git. A single contiguous estimation record yields descriptive selection
evidence, not an independent replication or calibrated uncertainty.

The scorer may select a model for full-estimation refit only under the
predeclared comparison rule and after complete scores. It cannot change
candidate families, priors, simulator, Qwen parsing, threshold, baseline,
or initial-state treatment based on this outcome. A scoring failure after
marker creation leaves the run unresolved and consumed. The later official
test must use a separately reviewed, single-use final scorer and state-only
50-point initializer; no development component may access it.

## Synthetic review gates

Private synthetic tests must prove zero target-parser calls for malformed
or changed gate/fit/baseline/declaration receipts, missing or failing
particles, source mismatch, and candidate/forecast hash mismatch. Test
same-inode mutations of every JSON input artifact and the archive around
hash/parse, with same-byte snapshot behavior. A failed preflight must leave
the canonical marker absent. A successful synthetic score must claim and
`fsync` the marker file and directory **before** the target-parser hook,
score the exact frozen arrays, and
refuse a second attempt even after a forced post-marker exception or
process-style interruption. Vary hidden synthetic targets while keeping
training and later inputs fixed: permitted numerical forecasts must be
invariant under separately frozen matching source declarations, while the
full archive identity and eligibility receipt hashes change. Reusing the
old source declaration must fail closed. The post-marker target/score may
then change under the distinct synthetic declaration. Test CSV sentinels
that would fail if official-test columns were numeric-converted. Have a
different Luna reviewer audit the source parser, lock timing, receipt chain,
synthetic results and claims before any production scorer is enabled.

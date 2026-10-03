# ABC6 pre-marker evidence v1 — primary protocol proposal

**Design-reviewed GO for bounded fake-only writer/scorer implementation,
2026-10-03; no code-review or launch GO.** The first independent Luna design
review held four evidence gaps. Primary narrowed the contract and an
independent rereview cleared only the attempt/failure receipt writer and
scorer integration. This proposal extends the independently reviewed Stage B1 one-use
scoring permit. It changes no synthetic truth, roster, prior, ABC kernel,
baseline, forecast, score or resource budget. Production source pins remain
`None`, and reserved `ct-abc6-20260928-v1` remains UNUSED/HOLD. It defines
only the receipt immediately after permit consumption, a catchable scorer
failure before marker publication, and the evidence needed to classify those
states. Existing deferred-score receipt schema v1 and public terminal-v3
replay remain unchanged.

## Authority and immutable wire format

The scorer is the sole writer. Its first supervised action stays
`scoring_permit.consume(execution)`. A rejected consume creates no attempt
receipt, marker or prospective targets and permits no retry under this run
ID. Scorer invocation alone does not prove that the permit was consumed;
without separately verified rejection evidence the claimed run is
`unreplayable`, not a pre-entry failure. Only after consume returns successfully,
and before any source, status, forecast or marker check, it publishes exactly
one `<receipt_root>/score-attempt.json`. Publication is an irreversible
attempt: error or ambiguous completion stops the scorer without retry,
marker creation or prospective-target materialization. The one-use run ID
remains consumed. The receipt uses the following **exact top-level keys**:

| Key | Type and invariant |
| --- | --- |
| `schema_version` | integer `1` |
| `artifact_type` | string `abc6_score_attempt` |
| `authority_bindings` | exact `ABC6AuthorityBindings.payload()` object; no subset, caller value, descriptor FD or Python object ID |
| `grant_record_sha256` | lowercase 64-hex digest of the independently verified durable grant record |
| `campaign_claim_sha256` | lowercase 64-hex digest of the anchored, independently verified campaign claim |
| `training_evidence_manifest_sha256` | lowercase 64-hex digest from the exact registered Stage A execution |
| `training_summary_sha256` | lowercase 64-hex digest of the 24-case terminal summary |
| `status_receipts` | exactly 48 ordered objects, each with exact keys `case_index` (integer 0–23), `component` (`fit` then `baseline` for each index), and `sha256` (lowercase 64-hex) |
| `forecast_artifact_sha256` | lowercase 64-hex digest of the final target-free forecast artifact revalidated by the handoff |
| `reveal_marker_relative` | exact string `artifacts/evaluations/cascaded_tanks_abc6_scoring/claims/ct-abc6-20260928-v1.claim` |
| `event` | exact event object below |

`event` has exactly `stage: "scoring_permit_consumed"`,
`monotonic_ns: int` in `[0, 2**63-1]`, `occurred_at_utc: str` in the
existing UTC format, and `clock: "host-local-monotonic-ns"`. Capture it in
the scorer immediately after `consume()` returns; the timestamp is an
observation of that boundary, not proof of durable publication. The binding
object is the already typed/frozen authority payload containing protocol,
run, manifest, approval, review, reviewed HEAD, physical and receipt roots,
root/receipt device and inode, watchdog claim, watchdog and child PID/start,
child launch/image, full source/runtime digests, and exact four role paths and
functions. Replay must compare it field-for-field with **independently
constructed expected bindings**, as required by the reviewed grant-record
contract; a self-hash is not authority. The receipt's raw digest becomes the
`score_attempt_sha256` used by later artifacts. Do not put that self-digest
inside the receipt.

The current `consume()` returns flattened `receipt_metadata()` that includes
a process-local grant FD and omits both the bindings payload and durable grant
record digest. **Do not serialize or reverse-engineer that dictionary.**
The bounded implementation must introduce a frozen, authority-minted
`ABC6ConsumedScoringContext` returned only by successful consume. It carries
the exact retained `ABC6AuthorityBindings` object, retained verified grant
record snapshot, registered execution and sealed handoff reference, and no
caller-supplied values. The scorer derives the attempt payload from this
context and its own fixed marker path. The context is an in-process authority
object, not a new persistent authority source; the receipt's claims still
require independent replay checks.

`campaign_claim_sha256` is specifically the canonical bytes of the external
claim at `ABC6AuthorityBindings.campaign_claim_relative`, namely
`artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/<run_id>.claim`.
The distinct `<receipt_root>/campaign.claim` must independently read as the
same bytes and digest. Neither value is the watchdog claim digest. For
replay, expected values come from separately opened, pinned sources:

1. Build expected `authority_bindings` from the frozen manifest/review/
   approval, separately verified watchdog claim and launch attestation, and
   current anchored roots, as in the grant-record contract. Read the grant
   record through its pinned parent, verify its raw digest against the
   terminal grant-record field, and compare its full bindings to that
   independently built object. The attempt must match the same object and
   record digest field for field.
2. Read both campaign claim locations through their separately pinned
   no-follow parents. Require identical canonical raw bytes and digest,
   matching the Stage A execution's `campaign_result.claim_sha256` at write
   time and, after process exit, the verified terminal/Stage A claim chain,
   attempt field, frozen protocol/run/manifest and expected claim path.
3. Read the exact Stage A evidence-manifest file from the frozen source-derived
   path under the pinned receipt root, recompute its raw digest, and compare
   it at write time to `execution.evidence_manifest_sha256` and at replay
   time to the independently verified watchdog terminal fit-gate evidence
   digest and the attempt field. Replay must not require the vanished
   in-memory execution object; the attempt itself is never the source of the
   expected manifest digest. Apply the same independent raw reads to all 48
   ordered fit/baseline status receipts and the 24-case terminal summary;
   compare their verified Stage A/terminal gate identities to every attempt
   row and digest.
4. Independently read the target-free forecast artifact under the anchored
   receipt root, verify its declared roster and raw digest against the
   runner's final forecast and terminal fit gate, and compare the attempt
   field. Derive the marker path from the fixed scoring-claim contract, not
   the receipt's proposed path. Missing expected sources or a mismatch is
   `unreplayable` for a claimed scorer attempt.

For a catchable error detected after a valid attempt receipt and **before
calling marker publication**, the same scorer may publish exactly one
`<receipt_root>/score-pre-marker-failure.json`. This is not emitted for a
failed or uncertain marker publication, where marker presence is ambiguous.
Its exact top-level keys are:

| Key | Type and invariant |
| --- | --- |
| `schema_version` | integer `1` |
| `artifact_type` | string `abc6_score_pre_marker_failure` |
| `score_attempt_sha256` | verified raw digest of the durable attempt receipt |
| `authority_bindings_sha256` | digest of the canonical `authority_bindings` object in that attempt receipt |
| `campaign_claim_sha256` | same verified claim digest as the attempt receipt |
| `phase` | one of `source_revalidation`, `training_evidence_revalidation`, `forecast_revalidation`, `marker_payload_preparation` |
| `reason_code` | `checked_rejection` or `unexpected_exception`; no arbitrary exception text or target values |
| `event` | exact event object below |

Its `event` has exactly `stage: "pre_marker_failure_detected"`,
`monotonic_ns: int` in `[0, 2**63-1]`, `occurred_at_utc: str`, and
`clock: "host-local-monotonic-ns"`. The scorer captures the event at the
fixed catch frame before writing this receipt. The source must make `phase`
an internal enum set before each check, never a caller-supplied label. The
failure record does not establish marker absence by assertion: replay also
checks the separately anchored marker parent. Any write/readback uncertainty
leaves the phase unreplayable. Do not alter the historical post-marker
deferred-score receipt's exact schema or treat this failure record as it.

Both files must be canonical sorted-key, compact ASCII JSON with nonfinite
values rejected, no trailing newline, exact-key validation and at most
256 KiB each. Use a pinned no-follow parent descriptor, exclusive creation,
file and parent `fsync`, mode `0600`, and bounded no-follow readback of the
named entry. Compare raw SHA-256 and stable opened/named device, inode,
mode, size, mtime and ctime before trusting publication. Reject symlink,
FIFO, copied-ancestor, duplicate leaf or changed named entry. If an API
reports failure after a valid entry may have appeared, never retry; a later
settled-filesystem replay may accept only the one canonical entry if its
independent chain verifies **and** its pinned opened file and parent directory
both `fsync` successfully before acceptance. Replay creates or changes no
artifact bytes or names; failed sync or ambiguous identity is
`unreplayable`. The existing watchdog writer is a readback
example, not an automatic authorization to reuse its terminal schema.

## Replay classification and ordering

An exclusive watchdog/run claim burns the reserved ID regardless of the
operational outcome. A verified terminal with **no campaign claim** retains
the established pre-campaign failed/incomplete rules and expects no attempt
receipt. For this v1 component, **any campaign-claimed run without a valid
attempt receipt is `unreplayable`**, even when its child exit is nonzero or
the fit gate is partial. This conservative scope does not declare that
scoring was entered; it acknowledges that current durable evidence cannot
exclude activation/consumption after a campaign claim. A future versioned
pre-entry failure/stop receipt and terminal verifier may recover `failed` or
`incomplete` for provably pre-scorer cases, but that wire contract is not
defined here and cannot be inferred from `fit_phase_gate`, child exit or the
currently unverified `campaign.failure.json`. In particular, a crash after in-memory
permit consumption but before a verifiable attempt receipt is consumed and
`unreplayable`, with no retry, marker or target generation.

A valid attempt receipt with no marker requires an independently verified
typed failure receipt to call the scorer error `failed`. If a watchdog
budget-cap or deliberate-stop intervention also exists, compare the two
captured `monotonic_ns` values **only** when both validated events use
`host-local-monotonic-ns`, the same pinned child/watchdog identities and the
watchdog intervention is exactly one of the three pre-cleanup pairs
`(budget_cap, sampled_rss_limit_exceeded)`, `(budget_cap,
wall_clock_limit_exceeded)`, or `(operator_stop, watchdog_interrupted)`.
Failure strictly earlier than intervention is `failed`; intervention strictly
earlier with no earlier error is `incomplete`; equality, missing events,
uncertain clock/identity, or a post-cleanup-only intervention is
`unreplayable` if it affects the choice. UTC is diagnostic, not an ordering
tie-breaker. A valid attempt without marker and without a failure receipt
is `incomplete` only when a verified pre-cleanup cap/stop event and terminal
partial chain justify it; otherwise `unreplayable`. A marker-present branch
requires a future versioned marker binding `score_attempt_sha256`, a
versioned terminal score-phase gate and full post-marker replay; current
marker v1 and terminal v4 do not supply those links. Operational `complete`
still does not imply the frozen synthetic scientific-readiness gate passed.

This replay table is a **required future verifier contract**, not a claim
that public replay currently accepts terminal v4 or these new receipts.
The terminal ACK verifies unchanged terminal bytes/file identity but does
not independently prove a stop event's timing; the reviewed watchdog code
and pinned runtime must be the source of the event. A self-hashed attempt,
failure or terminal file without source-backed expected bindings is never
sufficient provenance.

## Fake-only gates before implementation GO

Independent Luna max design rereview cleared the exact fields, marker path,
clock-domain assumptions and the conservative no-campaign/campaign split
**for this writer-only fake component**. A bounded Luna implementation and a
different exact-byte reviewer must use
only fake roots, fake status/forecast receipts and sentinel prospective
targets. Cover valid one-use attempt, every missing/altered evidence digest,
rejected consume, crash immediately after consume, ambiguous publication and stable-present
recovery, file/parent sync and readback failures, symlink/FIFO/ancestor
swaps, valid pre-marker failure, error after marker-publication begins,
stop-before-error and error-before-stop, equal/unknown clocks, no campaign
claim versus campaign-claimed pre-entry failure (the latter stays
`unreplayable` in this component), full pre-score chain with no entry proof,
duplicate use and stale/copied authority. Every rejected
pre-marker branch must leave marker and target sentinel untouched. Preserve
exact source hashes, tests on current and clean-base overlay, negative
outcomes, and one-owner non-force Git sync only after independent component
GO. The exact source-derived evidence path/key formulas and production
expected-binding builder are still HOLD before any public replay verifier.
Source/runtime closure, terminal/marker versioning, full replay, immutable
manifest/report/approval, and a separate launch GO remain HOLD.

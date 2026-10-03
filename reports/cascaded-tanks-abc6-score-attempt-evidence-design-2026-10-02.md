# ABC6 pre-marker scoring-attempt evidence — design draft

**Primary decision draft, 2026-10-02T15:30Z.** This defines the next fake-only
review target after the Stage B1 scoring-permit chain. It authorizes no
production manifest, campaign claim, Qwen/ABC run, protocol prospective
targets, real tank development suffix or official held-out access. The
reserved `ct-abc6-20260928-v1` remains UNUSED/HOLD. The corrected Stage B1
component was independently reviewed and synced on 2026-10-03 as
`5c3452531b267f7f0e820fd60896ae39089042b4`; that component GO does not
approve this attempt-receipt design or a campaign launch.

## Failure that this design addresses

Stage B1 consumes an in-memory one-use scoring permit before scorer source and
evidence checks, the reveal marker and prospective-target materialization. If
any of those pre-marker checks fails or the child crashes, the marker is absent
and the existing post-marker score receipt cannot prove that the permit was
consumed. A generic watchdog failure receipt cannot distinguish that branch
from a failure before scoring began. Missing marker is therefore not evidence
of an unused scoring opportunity. Every failed attempt still consumes the
reserved run under the one-use protocol.

## Proposed durable boundary

After a valid `scoring_permit.consume(execution)` and before **any** source or
evidence recheck, marker operation, or prospective-target materializer call,
the scorer should publish exactly one canonical `abc6_score_attempt_v1`
receipt beneath the anchored run root. It must be created O_EXCL through a
pinned no-follow parent descriptor, with bounded regular-file bytes, exact
canonical JSON, file and parent `fsync`, and a no-follow readback that confirms
raw SHA-256 and stable device/inode/mode/size. A failed write, sync or readback
stops scoring before marker and target materialization. No second permit,
receipt leaf or run ID may be substituted to repair it. A pre-existing leaf,
symlink, copied ancestor, FIFO or changed named entry fails closed.

The versioned receipt should bind the exact run/protocol/manifest/approval and
watchdog claim identity, anchored receipt-root and scorer-child start identity,
the verified grant-record digest, the one-use scoring activation/handoff
digest, all 48 ordered fit/baseline status digests, the training summary
digest, and the target-free forecast artifact digest. It records the fact of
permit consumption and an exact expected reveal-marker leaf. It contains no
prospective target values, scores, model-selection result, raw grant secret or
caller-supplied authority metadata. Exact field names, types, size cap and
serialized byte contract require independent review before implementation.
Do not retrofit this into the historical `schema_version: 1` deferred-score
receipt or weaken strict v3 replay. A future versioned deferred-score receipt
is a separate artifact from this pre-marker attempt receipt and from the
watchdog terminal-v4 receipt.

If the receipt publication API reports failure after a durable entry may have
appeared, the scorer still stops and does not retry. An independent settled-
filesystem replay may verify that one present canonical entry through its
pinned parent, but an absent, altered or ambiguous entry is `unreplayable`
after the run claim; it is not proof that the permit was unused. A verified
attempt receipt with no marker can support a `failed` classification only when
the terminal chain independently verifies the subsequent pre-marker failure
point and no earlier watchdog cap or deliberate stop supersedes it. The
existing `failed`/`incomplete`/`unreplayable` precedence in the terminal
decision remains otherwise unchanged. Marker-present failures consume the
condition as already frozen. A later versioned marker and terminal-v4 replay
contract must bind the attempt-receipt digest to post-marker evidence before
any production launch; this draft does not claim that link is implemented.

**Proposed crash and stop classification, still requiring design review.** A
claimed run with no valid attempt receipt is consumed and `unreplayable`,
including a crash immediately after in-memory permit consumption and any
ambiguous publication whose durable bytes cannot be verified. A valid attempt
receipt with no reveal marker is not by itself `failed`: a separately durable,
typed scorer failure point must verify the pre-marker error and match the
attempt digest and child identity; a watchdog cap or deliberate stop takes
the separately frozen precedence. If the child disappears without such a
failure point, classify the attempt `unreplayable` rather than inventing a
scorer failure. A valid marker requires its own versioned link to the exact
attempt digest, terminal identity and post-marker score chain before the
condition can be classified. This table of cases is a design constraint, not
an implemented replay path or a reason to reuse the run ID.

## Fake-only review gate

An independent Luna max design review must first freeze the exact receipt
schema, provenance fields and expected marker link; source pins remain `None`.
Then a bounded Luna implementation and a different exact-byte reviewer should
test: valid permit to durable attempt receipt to marker to fake-only target
materializer; missing/wrong/copied/reused permit with no attempt receipt;
exception after consumption but before receipt; O_EXCL collision; failure at
write, file sync, parent sync or readback; stable present entry after
ambiguous API error; partial/corrupt/extra-field/noncanonical receipt;
symlink/FIFO/ancestor swaps; pre-marker scorer check failure with a verified
attempt receipt and no marker; and post-marker failure. Every rejected path
must leave the prospective-target sentinel untouched unless the marker has
already been durably created. Run pinned-Python current and clean-base exact-
diff fake tests, independently review exact bytes, and sync only a GO-scoped
component through one non-force Luna Git owner. Full source/runtime closure,
source-backed expectations, terminal-v4 replay, immutable manifest/report/
approval and a distinct launch GO remain separate requirements.

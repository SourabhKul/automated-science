# ABC6 pre-marker scoring-attempt evidence — design draft

**Primary decision draft, 2026-10-02T15:30Z; bounded v1 writer scope was
subsequently design-reviewed on 2026-10-03.** This defines the next fake-only
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
caller-supplied authority metadata. The separate
`reports/cascaded-tanks-abc6-pre-marker-evidence-v1-protocol-2026-10-03.md`
now freezes the exact field names, types, size cap and serialized byte
contract for a bounded fake-only writer/scorer implementation. It does not
clear public replay, source-derived path/key formulas or production launch.
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

**Future pre-entry recovery proposal; outside the reviewed v1 writer scope.**
The one-use campaign claim always burns the run ID; the operational replay
classification is a separate question. A missing attempt receipt alone does
not override a verified failure or intervention *before scorer entry* in a
future version that has independently reviewed, typed pre-entry evidence. A
canonical, durable, claim/run/child-bound stage or intervention record,
verified by the terminal chain, must establish the pre-entry phase and its
order relative to any later stop. The frozen execution path must guarantee
that a verified pre-entry failure exits before invoking the scorer. A verified
pre-entry stage error before intervention is `failed`; a verified earlier
watchdog cap or deliberate stop is `incomplete`, under the existing terminal
precedence. If full training/forecast evidence exists but no durable event
proves whether scorer entry occurred, a missing attempt receipt is
`unreplayable`. In particular, a crash after in-memory permit consumption but
before verifiable attempt-receipt publication, or ambiguous publication with
unverifiable durable bytes, is consumed and `unreplayable`.

The earlier watchdog claim and later campaign claim are separate. A verified
terminal with *no campaign claim* can use the existing no-campaign-chain
failure/stop rules without an attempt receipt. Once the campaign claim
exists, current durable `fit_phase_gate` evidence alone does not prove that
scoring activation or permit consumption did not occur. The current
`campaign.failure.json` may identify some failures inside campaign fitting,
but public replay does not yet verify it and it cannot cover runner or scorer
failures after campaign return. Do not promote a child nonzero exit plus
absent marker to a claimed pre-scoring failure. Any campaign-claimed
pre-entry exception needs an independently verified typed failure or stop
record that proves the scorer was unreachable in the frozen call path;
otherwise the phase remains `unreplayable`. In the bounded v1 design, **every
campaign-claimed run without a valid attempt receipt is `unreplayable`**;
the proposed pre-entry exception is not part of its replay contract.

A valid attempt receipt with no reveal marker is not by itself `failed`: a
separately durable, typed scorer failure point must verify the pre-marker
error and match the attempt digest and child identity. A failure durably
verified before a later stop/cap is `failed`; a stop/cap verified first with
no earlier error is `incomplete`; uncertain event order or child disappearance
without a verifiable failure point is `unreplayable`. A valid marker requires
its own versioned link to the exact attempt digest, terminal identity and
post-marker score chain before the condition can be classified. These are
design constraints, not implemented replay paths or reasons to reuse the ID.

## Fake-only review gate

Independent Luna max design rereview cleared the separate v1 receipt schema
and provenance fields for bounded fake-only writer/scorer integration, while
the future marker/terminal link remains HOLD; source pins remain `None`.
A bounded Luna implementation and a different exact-byte reviewer should
test: valid permit to durable attempt receipt to marker to fake-only target
materializer; missing/wrong/copied/reused permit with no attempt receipt;
exception after consumption but before receipt (assert no retry, marker or
target materialization and `unreplayable` despite a burned run ID); verified
pre-entry error or earlier stop with no attempt receipt (future separate
schema only; v1 campaign-claimed cases stay `unreplayable`); full pre-score
artifacts with no entry proof or attempt receipt; O_EXCL collision; failure at
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

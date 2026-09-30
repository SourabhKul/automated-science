# ABC6 supervised binding amendment proposal

Status: **proposal for independent Luna review; no launch authorization**. The
reserved `ct-abc6-20260928-v1` synthetic campaign is unused and on hold. This
proposal changes only the execution and provenance contract. It must not
change the frozen truths, case roster, priors, ABC kernel, baseline, forecasts,
scores, or compute caps. No real Cascaded Tanks target may be accessed through
this work.

## Why the current reviewed pieces cannot yet connect

The campaign manifest's exact `source_hashes` key set is currently eight
paths, while its static reviewed-digest map covers five. The integrated runner
requires runner, forecast, and scorer hashes in both structures before a
campaign claim. The isolated grant primitive also requires runner, campaign,
cases, and scorer digests. Thus a watchdog-to-runner grant cannot be published
using the current manifest without weakening a closed gate. The approved child
command has no grant-descriptor locator. The grant primitive would write
`watchdog-child-grant.json` into the receipt root before campaign-fit claims,
but campaign-fit currently requires that directory to be empty. A bounded
Luna integration attempt stopped before editing on these conflicts.

## Proposed contract to review before implementation

1. **Close the source set only after wiring is final.** The manifest must bind
   the final reviewed, transitive execution sources. The current eight-path
   roster must at least add exactly these seven paths:
   `scripts/watch_cascaded_tanks_abc6.py`,
   `scripts/run_cascaded_tanks_abc6_synthetic.py`,
   `core/real_data/cascaded_tanks_abc6_authority.py`,
   `core/real_data/cascaded_tanks_abc6_forecast.py`,
   `core/real_data/cascaded_tanks_abc6_scoring.py`,
   `core/real_data/cascaded_tanks_abc6_replay.py`, and
   `core/abc_smc_reference.py`. This explicitly binds the independent replay
   verifier in the same manifest. The resulting fifteen paths are a starting
   inventory, not permission to insert provisional digests. Re-audit imports
   after wiring and before freezing; include any newly executed source.
   Keep the runtime and exact reviewed Git HEAD pinned. The campaign-fit
   module's own SHA-256 belongs in the external immutable manifest; do not
   embed it in its own static digest map. Require exact key equality between
   the external manifest source map and the final source roster, and between
   the static reviewed map and that roster excluding only campaign-fit
   itself; subset checks are insufficient. Any changed byte invalidates the
   final manifest and requires a distinct approval.
2. **Carry the grant through an inherited anonymous socket.** Preserve the
   exact reviewed child launch vector in the approval. Proposed transport is
   one inherited socket FD selected by the watchdog and located by a single
   canonical decimal `ABC6_GRANT_FD` environment value. The environment value
   is a locator, not authority: the child must validate the socket peer,
   process start/vector/images, manifest/approval/review digests, physical
   root, source/runtime hashes, expiry, one-use secret, and grant receipt
   before campaign claim. The watchdog must pass only that FD, close its child
   copy, and record the selected FD and grant digest in durable receipts.
   Direct runner invocation, a copied environment, a wrong FD, and a grant
   from a different process must fail before a campaign claim. Approval-v2
   remains sufficient for this transport: its fixed report hash binds the
   reviewed transport policy, manifest and exact watchdog/child source hashes,
   while the FD number is ephemeral and confers no authority by itself.
   Do not append a CLI argument, add an unreviewed approval field, or loosen
   the exact-vector check.
3. **Make the preclaim receipt exception capability-bound.** The watchdog
   still requires an empty receipt root before its own one-use claim and child
   launch. After attestation, the sole permitted pre-campaign receipt leaf is
   the durable, no-follow `watchdog-child-grant.json`. Campaign-fit may accept
   it only while holding the live process-bound training capability and after
   verifying the leaf's exact identity/digest through the anchored receipt
   root. Any other leaf, duplicate grant, missing grant, altered ancestor, or
   missing capability fails before campaign claim. Direct public fit/case and
   scorer APIs must separately require the appropriate one-use permits.
4. **Keep scoring dormant.** The grant does not authorize target reveal at
   receipt time. Only the attested runner may activate scoring after the full
   48 ordered statuses, campaign summary, and target-free forecast are durable
   and independently reverified. It passes a one-use scoring permit to the
   scorer and its deferred-target gate; each must consume or validate the
   same bound permit before the marker or any target generation. A failed
   post-claim or post-marker condition remains consumed; no retry under this
   run ID.

Implementation order is source/schema design review; watchdog-to-runner
transport with fake-only tests while pins remain closed; campaign/case/scorer
capability checks; complete source closure and immutable manifest; independent
fake-only end-to-end review and read-only replay; separate review report and
approval; and only then an operational GO decision. The held-out real-data
boundary and Silverbox consumed condition are unaffected. This document is a
reviewable proposal, not evidence that any gate has passed.

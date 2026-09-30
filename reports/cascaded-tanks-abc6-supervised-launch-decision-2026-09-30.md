# ABC6 synthetic campaign: supervised-launch decision

**Decision:** authorize implementation and fake-only review of a single
supervised launch path for the reserved `ct-abc6-20260928-v1` synthetic
control. This is **not** approval to claim or run that ID. Preserve the
independently reviewed six-parameter scientific protocol and the
receipt-root amendment; change no truth, roster, seed, prior, epsilon,
discrepancy, ABC proposal/weights, baseline, forecast, score, or compute cap.
Source pins remain fail-closed until the full path is independently reviewed.

## Frozen locations and authority

The candidate physical checkout root for this host is
`/Users/sourabh/Documents/Codex/automated_science`. A Luna operator must
verify it with the amendment's component-by-component no-follow traversal
at manifest freeze and again before launch; this prose does not establish
that identity. The immutable schema-v2 manifest location for this one run is
`artifacts/cascaded_tanks_abc6_synthetic/ct-abc6-20260928-v1/manifest-v2.json`;
the separate approval-record location is
`artifacts/cascaded_tanks_abc6_synthetic/ct-abc6-20260928-v1/approval-go.json`.
Both are host-local, ignored artifacts. The receipt root remains exactly
`artifacts/cascaded_tanks_abc6_synthetic/ct-abc6-20260928-v1/receipts`, and
the terminal receipt is `watchdog-terminal-receipt.json` beneath that root.
The final reviewed Git HEAD, source/runtime hashes, manifest SHA, exact
absolute launch vector, and approval-record bytes can be filled only after
the code and fake-only end-to-end checks are reviewed. Moving any fixed path
or using another checkout requires a new manifest and independent review.

The primary assistant is the operational issuer of the final GO record under
the user's existing authorization for local simulation. A separate
`gpt-6-luna` max reviewer must first provide a written GO binding the exact
manifest SHA, frozen scientific-protocol SHA, reviewed HEAD, source/runtime hashes, receipt root, launch
vector, fake-only test evidence, and unused-run preflight. A different Luna
operator executes the one approved watchdog command. `reviewer_id` and the
review report hash are provenance fields, not cryptographic authentication
against someone who controls the local checkout. The trusted-local-checkout
assumption does not excuse accidental path swaps, duplicate claims, or
unreviewed code drift.

## Ordered launch gates

1. Keep the source-pin gate closed while implementing the watchdog's exact
   root/path preclaim check and no-follow checkout/receipt/claim traversal.
   Before its one-use claim, reject a CLI receipt path unequal to manifest
   root plus frozen relative path, any existing campaign/watchdog/scorer
   claim or marker, and any nonempty receipt root. The watchdog child must
   invoke the reviewed integrated runner, not the training-only entrypoint.
   Read the manifest and approval through bounded, nonblocking, no-follow
   regular-file descriptors before claim; reject swapped paths, FIFO leaves,
   or mismatched opened-file identities.
2. Bind the reviewed watchdog and runner launch vector, all integrated
   sources (including forecast, scorer, cases and receipt I/O), runtime,
   physical root, fixed paths, and final Git HEAD into the immutable manifest.
   No generic or stale allowlist may silently enable the run. Direct public
   fit/runner/scorer entrypoints must remain fail-closed without verified
   single-use launch authorization from that manifest and approval.
3. Publish the independent review evidence, then create the separate GO
   record binding its hash and every launch identity. The Luna executor
   rechecks the exact model server only if generation is part of that frozen
   run; this synthetic control does not need an LLM request. Verify process,
   claim, marker and receipt absence immediately before one launch.
4. Use retained descriptors and bounded nonblocking regular-file reads to
   verify the complete claim → 48 status receipts → target-free forecast →
   single-use reveal/score → watchdog terminal chain. Publish, fsync and
   read back the terminal receipt. An independent replay verifier must
   distinguish complete, incomplete, failed, and unreplayable outcomes. A
   failure after a one-use claim consumes this run condition; do not silently
   retry or relocate its artifacts.

Independent Luna fake-only review must exercise wrong root/path and launch
vector before claim; symlink/FIFO and copied-ancestor swaps; alternate
entrypoints; duplicate marker/claim; terminal publication failure; and
replay of both success and failed terminal receipts. Only then may the
primary evaluate whether a final GO record is warranted. Any synthetic
parameter-recovery or coverage result remains a methods control, not a
physical discovery. Real tank development suffix and official test stay
unread under their separate protocols.

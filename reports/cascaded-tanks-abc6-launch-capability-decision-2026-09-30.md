# ABC6 synthetic campaign: supervised-child launch authorization

**Status: primary operational decision for fake-only implementation review, not approval to claim or launch.** This closes a separate path around the supervised watchdog. It changes no ABC6 scientific protocol setting or target boundary. The trusted-local-checkout assumption applies; this is an accidental-bypass and replay-control mechanism, not cryptographic protection against a user who can edit the checkout and artifacts.

## Authority transfer

The watchdog alone verifies the frozen manifest, separate approval, physical root, reviewed HEAD, complete source/runtime pins and exact child vector, then publishes its O_EXCL one-use claim. Version the current approval schema to require the independent review report's SHA-256 as provenance in addition to reviewer ID and the identities in the supervised-launch decision; old approval records cannot satisfy this run. Once the declared child has been started and its PID, process start and executable/image identities attested, the watchdog releases one fresh unpredictable grant through an inherited anonymous pipe or socket designated for that child. Do not put the grant secret in an argument, environment variable, manifest, approval or receipt. Use bounded, nonblocking acquisition with a deadline; malformed, absent, delayed or wrong-child grants fail before any training claim or target access. Persist only a digest and binding metadata, never the secret.

The child verifies the grant against the watchdog claim, manifest and approval hashes, run/protocol IDs, fixed root and receipt path, reviewed HEAD, child vector and process identity. Its grant is one use for the declared training phase and carries a distinct one-use scoring phase through the typed in-memory execution and scorer handoff. Public runner, both campaign-fit functions, and `run_abc6_training_case` require verified training authority before their global claim or production simulation; alternatively make the case-level API private and unreachable as a production entrypoint. `build_synthetic_training_bundle` may remain a source-free deterministic helper for fake/offline controls, but the reserved campaign's production bundle creation must occur only after verified training authority and must not create run artifacts beforehand. The public scorer and deferred target gate require verified scoring authority before the reveal marker or target materialization; the gate may instead be made scorer-internal. A path string, boolean, claim file alone, or a caller-constructible typed object is insufficient. Read-only manifest preflight remains available without a grant. Existing private fake-fit fixtures may have an explicitly test-only issuer that cannot open production source pins.

The scoring phase starts dormant. The supervised child retains its verified,
process-bound authority while it performs the one declared campaign; the
campaign may delegate only case-index-bound fit permits to its 24 case calls.
After all 48 fit/baseline status receipts, summary and target-free forecast
have been durably written and reverified, the runner may activate scoring
exactly once, binding that activation to the frozen receipt and forecast
digests. Only the attested runner holding the sealed process-bound capability
can activate scoring; the scorer consumes this activated authority before
claiming the reveal marker. Neither a training-only execution object nor a
caller-supplied digest grants activation. A fork, changed child identity,
duplicate case index, repeated activation or second scoring call fails before
new simulation or target materialization. A failed stage consumes the relevant
one-use run/condition; it cannot mint a replacement phase or retry under the
same run ID. This is a lifecycle contract for implementation and fake-only
review, not an implementation or launch approval.

Receipts link the grant digest, watchdog and campaign claims, manifest/approval and review-report hashes, root/runtime/source identities, child PID/start, phase consumption, 48 statuses, summary, target-free forecast, reveal marker, post-reveal score receipt and watchdog terminal receipt. A missing/invalid grant, phase reuse or failed post-claim stage consumes the condition according to the frozen one-use rules; no direct retry or alternate entrypoint may mint authority for the same run.

## Review before launch

An independent Luna reviewer must exercise direct runner, both campaign-fit APIs, the case-level training API, scorer and target-gate calls without authority and with wrong-run/hash/root/vector or replayed grants. Fake tests must check truncated, delayed and wrong-child anonymous-grant delivery, and reject any path-based FIFO/symlink fallback; one valid attested fake child should traverse fit and score once with target sentinel and no protocol prospective target. Source pins remain fail-closed until the complete integrated map is separately reviewed. This decision and its implementation do not authorize the reserved `ct-abc6-20260928-v1` campaign; terminal replay, immutable manifest, separate GO and fresh unused-run preflight remain required.

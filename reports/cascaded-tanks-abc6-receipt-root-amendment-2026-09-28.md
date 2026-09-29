# ABC6 synthetic campaign: receipt-root preflight amendment

**Status: primary scientific/operational decision proposed before any campaign claim or target generation; requires independent Luna review, implementation review, and a new immutable manifest before launch.** This adds provenance and path identity to the independently reviewed six-parameter synthetic ABC protocol. It changes no model, truth, roster, seed, prior, epsilon, discrepancy, sampler, baseline, forecast, score, or compute budget. The reserved run ID `ct-abc6-20260928-v1` is unused.

## Fixed local path contract

For this one local campaign, set `receipt_root_relative` to exactly

```text
artifacts/cascaded_tanks_abc6_synthetic/ct-abc6-20260928-v1/receipts
```

Add this normalized, traversal-free relative path, the physical absolute checkout root (`repository_root_realpath`), and the approved commit (`reviewed_git_head`) as strict top-level fields in a **schema-v2 immutable campaign manifest**. The physical root is a host-local provenance field, not a portable result path. At freeze time and again on every launch, walk both the manifest root and the actively executing source checkout root from `/`, component by component with `O_DIRECTORY|O_NOFOLLOW`; require matching physical paths and opened-root identity, then check active Git `HEAD` against the manifest's `reviewed_git_head`. A symlinked checkout is not accepted merely because it resolves to the manifest path. The manifest SHA-256 and separate GO approval must bind these fields, the code/source/runtime hashes, and the existing scientific protocol SHA. The watchdog's absolute `--receipt-directory` must equal the exact join of the frozen root and relative receipt path; command-line paths do not redefine the root. The approval's `manifest_sha256` must equal the v2 manifest hash, its `reviewed_git_head` must equal the manifest field, and its `receipt_directory` and each launch vector's `--receipt-directory` value must equal the exact root/path join; explicit root fields, if added to the approval schema, must match the manifest. A different checkout location or receipt path needs a newly frozen manifest and approval before any claim. Device and inode numbers are checked and recorded at runtime, not frozen into the manifest.

At preflight, open the physical checkout root as a directory descriptor, then traverse or create each component of `receipt_root_relative` relative to pinned descriptors with no-follow directory opens. Reject `..`, absolute or noncanonical relative paths, symlinks at any component, mismatched device/inode on recheck, a nonempty receipt directory, and any preexisting campaign claim, reveal marker, status, forecast, or terminal receipt. Directory creation and its parent must be durably synced before the watchdog claim. Keep the opened receipt-root identity in the typed campaign execution and carry it to scoring; do not derive the authoritative root from `summary_path.parent` or `Path.resolve()` on a caller-supplied receipt path.

The existing runtime claim and terminal paths are distinct from the pre-run approval record. Their fixed source-derived relative paths, with `<RUN_ID>` equal to the frozen run ID, are:

```text
artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/<RUN_ID>.claim
<receipt_root>/campaign.claim
artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/<RUN_ID>.watchdog.claim
artifacts/evaluations/cascaded_tanks_abc6_scoring/claims/<RUN_ID>.claim
<receipt_root>/watchdog-terminal-receipt.json
```

Open every parent from the same pinned checkout-root descriptor with no-follow traversal and create/read each leaf relative to its opened parent. The scorer reveal-marker parent is **outside** receipts: carry its exact directory identity from exclusive claim and fsync through scoring handoff. The campaign, watchdog, scorer and gate must hash and parse receipt files through the pinned receipt-root descriptor, and read the marker through its separately pinned claims-parent descriptor. Both are bound to the same physical checkout-root identity. Durable receipts should state the frozen relative receipt path, physical checkout root, manifest SHA, and relevant runtime dev/inode identities for local audit; dev/inode is not a cross-machine replay identifier.

Private fake tests may use an explicitly supplied physical temporary checkout root such as `/private/tmp/...` and a relative receipt path. This preserves macOS `/tmp` aliases without allowing an arbitrary symlink *inside* the declared root. A fake copied `artifacts/` ancestor present before gate opening must fail before scorer marker claim or target materialization; copied ancestors substituted after opening must continue to fail. The existing receipt-only public call, one-use marker, reentrancy, and post-marker failure-consumption checks remain mandatory.

## Decision gate

The first actual launch remains **HOLD** until a Luna author implements this contract, an independent Luna reviewer clears fake-only pre-open and post-open adversarial probes and current runner bytes, the full source allowlist/hash map is updated and reviewed, an immutable manifest and approval are reviewed at the chosen path, and the watchdog/replay chain is verified. If the chosen receipt directory is already occupied or any pin differs, preserve that finding and stop; do not silently choose a new path or reuse the reserved run ID. The synthetic run remains a methods control and cannot establish a physical discovery.

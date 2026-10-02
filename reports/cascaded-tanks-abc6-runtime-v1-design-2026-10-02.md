# ABC6 runtime identity v1 — proposed preclaim contract

**Draft, 2026-10-02T01:44Z.** This is a proposed methods-control contract for independent design review, not a final runtime pin, immutable manifest, approval or launch GO. It authorizes no ABC/Qwen run and no synthetic prospective or real/official target access. The reserved `ct-abc6-20260928-v1` remains UNUSED/HOLD. The source/role component was independently reviewed and synced at `a2276e7fbd562666a4d2311224f3fd1aa9bcf1c7`, with production static source hashes still `None`.

## Observed gap and host evidence

The current campaign-fit fingerprint has twelve fields but hashes only the Python executable and NumPy/psutil `__init__.py` files. Watchdog checks a subset, while the grant binds child/launcher image hashes and `sys.version` hash. Child authority does not independently compare the loaded NumPy/psutil/native stack. A read-only Luna probe on 2026-10-02 observed CPython 3.14.3, NumPy 2.4.2, psutil 7.2.2 on macOS 27.0 build 26A428, arm64. `proc_pidinfo` and `proc_pidpath` loaded through the `libproc` route but resolved to cached `/usr/lib/system/libsystem_kernel.dylib` (Mach-O UUID `f63bf418-8f75-34a4-aaf1-8caf2b9b6a05`); `sysctl` resolved to cached `/usr/lib/system/libsystem_c.dylib` (UUID `fba7b23e-aa60-3a90-9aa4-a7a2e0ad63ee`). The dyld shared-cache UUID was `ea2c265e-297c-39c2-8646-7d8a2dff648a`. NumPy native extensions, OpenBLAS and its libraries, psutil's native extension, the Python app and framework libpython were observable as loaded images. These are diagnostic host facts, not production hashes. `/usr/lib/libproc.dylib` is a loader route within the shared cache, not a regular standalone file to hash. The compact measurement record is `reports/research-ledger/2026-10-02T0046Z.md` (00:56Z addendum).

## Versioned object and canonical bytes

**05:38Z read-only inventory amendment.** The present NumPy install has 911
relative symlink aliases (889 package, 22 metadata); all resolve to regular
files elsewhere under the same Homebrew Cellar, including the two loaded
NumPy extensions. A blanket symlink rejection would make this contract
unusable on the observed host. An accepted alias must instead record exact
link text and a resolved, independently allowlisted versioned Cellar target;
traversal and hashing must use pinned no-follow descriptors, verify link and
target identity before and after reading, and reject escape, loops, broken
links or an unexpected root. The allowlist root, alias-resolution algorithm
and duplicate-target accounting rule are not yet frozen. The diagnostic
inventory counted 1,211 nodes, 23,395,635 total content bytes traversed by
alias (regular files plus link targets; link targets alone were 18,775,579
bytes), largest file 3,029,120 bytes, with 142
`__pycache__`/`.pyc` files totalling 4,143,933 bytes. A read-only 64 MiB
total/32 MiB file/4,096-node streaming probe took 0.091 seconds and about
5.9 MiB additional RSS on this warm host (6,144,000 bytes; peak RSS rose
5,685,248 bytes); those caps and costs are observations
or candidates, not approved production limits.

The same probe process listed 409 loaded dyld images, 389 shared-cache and 20
file-backed. A simple exact-path/unique-basename dependency resolver initially
missed eleven install names; realpath normalization uniquely mapped the two
absolute Homebrew names, leaving nine absolute system names unmapped.
Shared-cache membership alone did not identify
their owning images. Consequently the draft cannot claim a complete minimal
native dependency closure. A conservative all-loaded-image snapshot would
need separately frozen warmups and **role-specific** expected sets because
watchdog and child processes need not load identical images. This is a design
choice, not a measured equivalence. The exact dependency rule and the proposed
4,096-node, 32 MiB/file, 64 MiB total, 1,024-image and 4 MiB object limits
remain HOLD pending a role-specific probe and independent review.

The future `abc6-runtime-v1` object should have exactly these top-level keys: `schema_id`, `host`, `python`, `packages`, `native_images`, `api_symbols`. Require exact nested key schemas, exact scalar types (a bool cannot satisfy an integer field), canonical lowercase SHA-256 and UUID hex, absolute normalized paths with no `..` or NUL, unique identities and deterministic sort orders. A path alias visible to imports or loader and its resolved loaded path are distinct fields; neither silently replaces the other. All floating-point values are excluded. The byte contract is compact ASCII JSON with sorted object keys, `separators=(',', ':')`, and NaN disallowed. `runtime_fingerprint_sha256` is SHA-256 of those canonical bytes. **This draft does not yet define all nested keys, scalar types, uniqueness keys or array sort keys, so it is not a test oracle or production schema.** Freeze those exact definitions in a reviewed amendment before collector implementation beyond injected fake interfaces.

`host` records `sys.platform`, architecture and ABI, macOS product/build and kernel release/version, plus dyld shared-cache UUID. `python` records implementation, exact version/build string, SOABI/cache tag, declared executable alias, resolved executable, actual loaded app image and framework-libpython image. The file-backed Python images need stable identity and SHA-256 of bounded regular-file bytes plus Mach-O UUID and CPU type/subtype; alias paths must also be checked for unchanged resolution.

`packages` is exactly NumPy and psutil by name, with exact version, import-visible root, resolved installation root, installed distribution metadata root, a deterministic package-content tree digest and a sorted list of actually loaded native extensions (module, alias/resolved path, regular-file SHA-256, loaded Mach-O UUID and CPU type/subtype). The tree digest must use a reviewed inclusion rule covering the distribution metadata tree and package-adjacent native payloads such as `numpy.libs`, or explicitly justify their separate native-image coverage: sorted relative paths, file type/mode/size/content SHA-256; reject duplicate names, unapproved symlink targets, missing and unsupported entries. Resolve approved aliases explicitly as above, never through implicit path traversal. Walk through pinned no-follow directory descriptors, bound per-file and total bytes, and compare directory/link/file identities before and after the walk so replacement cannot create a mixed snapshot. Generated caches cannot silently enter or leave the tree; any exclusion such as `__pycache__`/`.pyc` must be written into the frozen rule and tested before final pinning. The exact inclusion rule, bounds and runtime cost are **not yet frozen**.

`native_images` is the sorted, unique loaded dependency closure relevant to the required Python/package extensions and native APIs, including OpenBLAS, its native libraries, and linked system images. Each image records loaded path, Mach-O UUID, architecture/subtype and backing kind. A regular file has a pinned no-follow resolved path, stable file identity and SHA-256. A shared-cache image instead has its per-image UUID plus cache UUID and a positive cache-membership check; requiring a nonexistent standalone file hash would be wrong. The v1 supported host is this reviewed macOS arm64 stack; other platforms, CPU subtypes or backing kinds fail closed until separately reviewed. The exact dependency-closure selection algorithm, maximum loaded-image count, serialized object size and collection cost are **not yet frozen**.

`api_symbols` records exactly `proc_pidinfo`, `proc_pidpath` and `sysctl`, each with requested loader route, symbol name, `dladdr` owner path, owner image UUID/architecture/subtype and backing kind. Cache-backed owners require both per-image and shared-cache UUIDs and positive dyld membership. Fail closed on unavailable `dladdr`, missing owner/header/UUID, duplicate or wrong architecture, failed dyld cache query, changed symbol owner, unexpected regular-file/cache branch, or a native ABI mismatch. Do not trust `find_library('proc')` as the final owner identity.

## Cross-layer binding and stage gate

After an exact schema and measured cost are independently reviewed, the immutable manifest must carry **two separately typed expected runtime objects/digests, watchdog and child, unconditionally**. The digests may coincide if frozen warmups yield equal objects, but neither role inherits its expectation from the other. Watchdog warms its declared imports and native APIs, collects its role object fresh, and compares complete canonical bytes/digest to the manifest watchdog expectation before claim. The grant binds both the verified watchdog digest and the manifest-derived expected child digest. A versioned launch-attestation sidecar carries those typed digests and is bound by its verified claim/terminal link; terminal evidence can retain them through that link rather than duplicating unauthenticated fields. The child independently collects its role object and compares its canonical bytes/digest to the manifest child expectation and grant before authority construction and again at the defined preclaim boundary. Replay derives both expected digests from independently verified immutable manifest, approval, claim and attestation facts, checks each role's equality chain through sidecar/grant/child receipts, and never adopts grant/terminal self-report as the expectation. Stable shared fields should match under an explicit schema rule, but **the two role objects need not be byte-equal**. The exact role warmups, nested schemas, alias/deduplication rules, sidecar version and receipt field names are **not yet frozen**. A new collector source file adds a seventeenth path; edits in existing rostered files change their hashes. Either case requires a new immutable manifest and approval, with import/source closure re-audited.

The immediate bounded work is **fake-provider design only** after independent review: a later frozen test oracle must cover cache-only versus regular-file image, changed symbol owner, missing/duplicate/wrong UUID, extension and package-tree mutation, alias/symlink/path replacement, extra/missing/noncanonical fields, changed grant digest, child mismatch and replay expectation substitution. Freeze exact nested fields/types, package-tree and distribution metadata inclusion, native dependency selection, supported backing branches, per-file/total-size/image-count/object-size bounds, sidecar schema and measured cost before those tests can be acceptance criteria. A read-only live-host smoke may confirm measurability and cost; it cannot become a production pin without a separately frozen exact rule and review. The full synthetic campaign, real tank work and any final outcome remain blocked by direct-call permits, complete source/runtime closure, source-backed expected bindings, terminal-v4 replay, immutable manifest, independent report, approval and separate launch GO.

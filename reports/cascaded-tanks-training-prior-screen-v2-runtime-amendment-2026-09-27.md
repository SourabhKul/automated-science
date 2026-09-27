# Cascaded Tanks training-only prior screen — v2 runtime amendment

**Status: predeclared; independent review required before execution.** This
amendment creates a distinct run because the v1 immutable pre-run manifest
named a transient `uv` Python executable that disappeared before approval.
The v1 manifest SHA-256 is
`ca167385f864aec664044d274b29c92f99e1ed9e3a588354d2186d5238546d83`.
An independent reviewer confirmed that no approval, attempt marker,
simulation, outcome, summary, or watchdog receipt exists for v1. Preserve
its manifest and deterministic draw table unchanged as a failed preflight;
do not re-label v1 as an experimental result or reuse its run ID.

The v2 scientific protocol is exactly the [v1 predeclaration](cascaded-tanks-training-prior-screen-protocol-2026-09-27.md), SHA-256
`33002007bc6fcf9027af243e42e35e5787ae308db099809d8a04a6c8655e986e`
in reviewed content commit
`a63f1a08f6cf2a49bfc314e0dd47bb48767a1bff`, **except** for the
protocol/run IDs and the runtime/manifest requirements below. The 512 paired
draws, seed, parameter box, fixed training-conditioned `x2_0` and `H`,
S0/O2/C2 simulator, all-768-point training RMSE, failure rules, summaries,
source/field boundary, one-worker 120-second/1-GiB sampled-resource limits,
and interpretation limits do not change. A different scientific setting
requires another separately reviewed protocol. Recreate the draw table
before outcomes and require its little-endian float64 C-order SHA-256 to
equal the independently replayed v1 value
`0bef4dbeb45c8b0dd4d29a0e9f46bd7ae359afe0cdf9443005dfdf900829b974`.
The matching draws do not make v1 a completed run.

Use protocol ID
`cascaded_tanks_training_prior_screen_v2_runtime_20260927` and run ID
`ct-prior-screen-20260927-v2`. Pin the resolved stable interpreter
`/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/bin/python3.14`,
Python `3.14.3`, interpreter-file SHA-256
`cbf84109626aa1013bbe408fbb9590bd0f1c1548f038b2221c6b8b87de26ca43`,
NumPy `2.4.2` at
`/opt/homebrew/lib/python3.14/site-packages/numpy/__init__.py` with SHA-256
`2e8da3e4385e79c4885b3f7324a8b957e6f01732b239e99e266a12c62a008b8d`,
and psutil `7.2.2` at
`/Users/sourabh/Library/Python/3.14/lib/python/site-packages/psutil/__init__.py`
with SHA-256
`d138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a`.
The preparer must record these resolved dependency module paths and hashes
in the immutable v2 manifest. The watchdog and child runner must use this exact
absolute interpreter path, not a temporary build path or a shell-resolved
alias. At preparation and immediately before launch, fail closed if the
interpreter file, dependency paths/versions/hashes, source archive and stage
hashes, reviewed Git commit/clean checkout, draw-table hash, or executable
code hashes differ from the manifest. Record exact command, platform and
package versions. Pin the source archive to the independently checked regular
file at
`/Users/sourabh/Documents/Projects/_backups/automated_science_20260806_pre_cleanup/data/real/cascaded_tanks/raw/CascadedTanksFiles.zip`,
7,520,592 bytes, SHA-256
`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`.
This local copy is outside the checkout and adds no download; the source
adapter must still enforce the reviewed training-output/future-input-only
contract. Never decode the development suffix output or official test.

The v2 operator must write a fresh immutable ignored pre-run manifest and
draw table under the new run ID without executing the screen. A different
Luna reviewer must independently check the manifest, actual stable runtime,
code/source hashes, draw replay, one-use attempt/failure/start receipts, and
external watchdog before GO. Only after GO may one operator launch the
512-by-three **training-only prior-predictive screen** once. Preserve any
post-attempt failure or timeout as terminal under v2. The screen is not Qwen,
ABC, validation scoring, model selection, a posterior, or a scientific
breakthrough; real tank fitting remains gated by a separate fully frozen
protocol and independent review.

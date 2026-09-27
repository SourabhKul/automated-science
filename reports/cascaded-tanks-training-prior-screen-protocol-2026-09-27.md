# Cascaded Tanks training-only prior-predictive screen — proposal

**Status: primary predeclaration for independent review, not permission to
run real ABC-SMC, Qwen, development scoring, or the official test.** This is
one bounded diagnostic on the already permitted estimation-training prefix.
It asks whether a transparent coarse parameter box places any meaningful
prior mass near that prefix before freezing an ABC tolerance schedule. Its
result may inform a separately frozen real-data methods-control protocol; it
is not a model-selection score, posterior, calibrated uncertainty, or new
physical result.

## Source and information boundary

Use only the reviewed Cascaded Tanks source adapter under its named
methods-control track `cascaded_tanks_methods_v1_20260926`. The older
generic `data/real/cascaded_tanks/manifest.json` failure remains historical
and is **not** an alternate source authority. Pin the official archive
to DOI/version `10.4121/12960104.v1` and exactly `7,520,592` bytes;
pin the existing simulator `core/real_data/cascaded_tanks_models.py` to
SHA-256 `69f4dddb435aaad8a63720073ae031a4b9c80305531a8f8aab8e2f06eaf0db12`.
Pin the archive
SHA-256 to
`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`,
the adapter SHA-256 to
`d0a693cf56250081409dcd64d6a11d60b85cf01b597574698312a40d0a2d1672`,
the contract SHA-256 to
`c92dad6bcf731c1171172861ca85fae5d9b69dff5a45783b61f3eeeeda017723`,
the training-stage SHA-256 to
`0eadc6c62e7f4e3dea76311dff037a7fee1c21fda6fac20e9363310dcf900d61`,
and the future-input-stage SHA-256 to
`dbffbfc30b0b2ae9a0f7b48a850e57f7113909c50bc8fcb6d84a109e7508c486`.
Require all five identities in the pre-run manifest and recheck them from
the same source snapshot before simulation. The loader returns
`uEst/yEst[0:768]` and `uEst[768:1024]` **input only**. No values from the
development suffix output or official `uVal/yVal` fields may be extracted
into retained field buffers, decoded, numeric-converted, logged, or scored.
The adapter may hold a bounded immutable snapshot of full archive bytes for
hash+parse and scan excluded field bytes for CSV structure without extracting
their values. Require the adapter's complete archive, contract, training, and
forecast-input hashes to match the reviewed source contract. Use the
four-second pre-transition observation convention: each `u[k]` advances
the state after output `y[k]`. Only the 768 training outputs contribute to
the screen distance.

## Candidate box and screen

Use the existing S0, O2, and C2 deterministic simulator unchanged. For this
screen only, set `x2_0=5.205` V-equivalent, the first training observation,
and `H=10.0` for O2/C2, the exact training plateau value. These are
training-conditioned modeling choices, not independent physical prior
information. They reduce dimension for a feasibility check; a final
protocol must explicitly decide whether to retain them and disclose the
conditioning. Draw **512** independent quadruples with an explicit NumPy
`Generator(PCG64(150927))` in parameter order `(a,c,p,x1_0)`, nominally
uniform on the half-open intervals `[low, high)` below (the stored float64
table is authoritative if endpoint rounding occurs):

```
a ∈ [0.2, 1.5)       c ∈ [0.2, 1.5)
p ∈ [0.05, 1.2)      x1_0 ∈ [0, 25)
```

Construct exactly `z = rng.random((512, 4), dtype=np.float64)` and for each
column `j` calculate `theta[:,j] = low[j] + z[:,j] * (high[j] - low[j])`,
using float64 arithmetic and no clipping. Draw ID is the zero-based row
index. Save NumPy/Python versions, explicit PCG64 identity, and SHA-256 of
the C-order bytes of the little-endian float64 `(512,4)` `theta` table in
the pre-run manifest before outcomes. Use the same 512 quadruples for each
family, with fixed `x2_0` and `H`
where applicable; this is paired common-random-number screening, not 1,536
independent parameter draws. The O2 box satisfies `x2_0 <= H`. Set
`max_steps=768` and `max_magnitude=1e6` for every simulator call. A failed
trajectory remains failed and receives no RMSE; no clipping, dropped
time points, replacement draw, or implicit retry. For every complete finite
trajectory calculate exactly with float64 array arithmetic

```
RMSE = np.sqrt(np.mean((y_sim - y_train)**2, dtype=np.float64))
```

No ABC acceptance, importance weight, posterior, optimizer, Qwen proposal,
or development selection is performed. Keep the shared `(512,4)` parameter
table and the `(512,3)` family-outcome table in ignored artifacts, with
hashes and failure categories.
Publish only compact counts and the following predeclared summaries per
family: complete/failure counts by category; RMSE minimum and empirical
quantiles over **complete finite traces only** at 1%, 5%, 25%, 50%, 75%,
95%, using `np.quantile(..., method="linear")`; and counts with RMSE at most
`[0.5, 1, 2, 3, 5, 10]` V. Give the paired draw IDs attaining the lowest
five finite RMSE values for each family, breaking exact RMSE ties by
ascending draw ID, without treating them as fitted parameters or mechanisms.
On only the draw IDs complete in **all three** families, report count and
empirical 1%, 5%, 25%, 50%, 75%, 95% quantiles using
`np.quantile(..., method="linear")` for each ordered signed
RMSE contrast `S0−O2`, `S0−C2`, and `O2−C2` (negative means the first-named
family has lower training RMSE). For every complete O2/C2 trace, calculate
the fraction of its 768 simulated outputs equal to `10.0` and report median
and 95th percentile using the same linear quantile method, plus counts with
at least 50% and at least 90% ceiling
outputs. These cap fractions are diagnostics for the surrogate, not evidence
of a physical threshold. Do not compute a Bayes factor or
use these counts to claim one mechanism is true.

## Execution and interpretation gate

Before execution, a different Luna reviewer must check that the loader,
simulator, frozen numeric box, seed, source boundary, metric, and resource
budget are coherent and that no development/test target is reachable. Use
protocol ID `cascaded_tanks_training_prior_screen_v1_20260927` and run ID
`ct-prior-screen-20260927-v1` exactly once. Then
one Luna operator must inspect running processes/checkpoints, save an
immutable ignored pre-run manifest with the reviewed source hashes above,
the exact simulator SHA above, actual runner SHA-256, this protocol's exact
SHA-256 and reviewed Git commit, exact runtime,
draw-table hash, seed/settings, and start time. Independently verify this
manifest before execution. Run serially under **120 seconds hard parent wall
deadline, 1 GiB sampled total RSS for the operator process and descendants,
one worker**. A separate launcher must sample that process tree every
0.1 seconds and terminate it on an observed RSS breach; note that brief
between-sample peaks can escape detection. Preserve a failed or
partial run under the same ID; never silently restart with changed settings.
Record observed wall time/peak RSS and the limits of sampled enforcement.
Have a different Luna reviewer independently replay at least the source
identity and deterministic draws/summary counts before compact evidence is
used to freeze the real ABC protocol.

The decision after this screen is not automatically to narrow the box around
the best draw. The primary will use the complete discrepancy distribution,
failure rate, parameter tradeoffs, and one-record limitations to choose and
predeclare a distinct real methods-control prior and epsilon schedule.
Counts at RMSE ≤2 V and ceiling fractions are exploratory warning
statistics, not automatic pass/fail gates. If they are unfavorable,
preserve that negative result and reconsider the **development protocol**
before any validation target is opened. The official test remains sealed and the
Silverbox multisine condition remains consumed.

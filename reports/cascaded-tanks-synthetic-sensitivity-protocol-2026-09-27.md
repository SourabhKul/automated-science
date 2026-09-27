# Cascaded Tanks synthetic sensitivity control — predeclaration

**Status: primary scientific specification; independent review required before
execution.** This is a deterministic, synthetic-only diagnostic of the
existing S0/O2/C2 voltage-equivalent surrogate. It is a distinct control
from the completed 13-case synthetic ABC-SMC batch and the real-data
training-only prior screen. It does not fit a posterior, use Qwen, rank model
families, access the real tank archive, or open development/official-test
outputs. Its result may inform a later separately frozen ABC calibration
protocol but cannot establish physical identifiability or discovery.

## Question and exact synthetic settings

Can the local output trajectory respond separately to `a,c,p,x1_0,x2_0`
and, in O2/C2, the ceiling `H`, under no-crossing versus crossing/recovery
forcing? Does a longer recovery reveal the O2/C2 memory difference that a
short horizon can hide? This is **local sensitivity**, not a global
identifiability or uncertainty-calibration test.

Use only `core/real_data/cascaded_tanks_models.py` at the reviewed Git commit
and pin its SHA-256 in the immutable pre-run manifest. Use the simulator's
pre-transition observation convention: record `y[k]` before applying `u[k]`.
Fix the common truth values `a=0.5`, `c=0.4`, `p=0.5`, `x1_0=0.5`,
`x2_0=0.5`. Set `H=1000.0` for the no-crossing case, and `H=3.0` for
the crossing cases. S0 has no `H`. Use no stochastic process or measurement
noise; `0.05` below is a declared output scale only. The three forcing
sequences, each built as exact float64 arrays, are:

| Case | Input sequence | Intended check |
| --- | --- | --- |
| N | 24 samples at `u=8`, then 180 at `u=0`; `H=1000` | No cap crossing; S0/O2/C2 truth outputs must be byte-identical and O2/C2 `H` columns must be zero. |
| S | 24 samples at `u=8`, then 54 at `u=0`; `H=3` | Crossing with short recovery. |
| L | 24 samples at `u=8`, then 180 at `u=0`; `H=3` | Crossing with long recovery. |

Run all three families on all three cases; call S0 with `ceiling=None` rather
than passing the listed `H`, which its simulator rejects.
For each simulation set `max_steps` to the exact input length (78 or 204),
`max_magnitude=1e6`, and use the existing nonnegative-state and cap/sensor
rules unchanged. A failed or nonfinite trajectory is a failed case, with no
substitution or clipping. Record complete output-array hashes, terminal
states and failure categories in ignored artifacts. For each case report
the first zero-based output index and count where O2/C2 truth outputs differ
exactly (`first_differing_index=null` when none), their maximum absolute
output difference, and each capped-family
count of outputs exactly equal to `H`. Exact output equality or a zero `H`
column is a **synthetic control**; output equality alone cannot identify
which branch was contacted in a physical system. If case N does not satisfy
its declared exact equalities, retain the failure and do not label it
no-crossing.

## Local finite-difference matrix

For each family/case, use normalized independent parameter coordinates
`z_j=(theta_j-low_j)/(high_j-low_j)`. All truth coordinates are 0.5. The
common parameter order is `(a,c,p,x1_0,x2_0)` with half-open boxes
`[0.375,0.625)`, `[0.30,0.50)`, `[0.375,0.625)`, `[0,1)`, `[0,1)`.
For O2/C2 append `H` with `[900,1100)` in N and `[2.5,3.5)` in S/L;
S0 has five columns and O2/C2 have six. These synthetic boxes are local
scaling conventions and **not** priors for a real tank fit. They keep all
central perturbations inside support and preserve O2's `x2_0<=H` rule.

For each coordinate and each `h ∈ {1e-3,1e-4,1e-5}` in that order,
simulate at `z_j=0.5+h` and `z_j=0.5-h`, all other coordinates fixed at
truth, and calculate the central finite-difference column
`J_j(h)=(y(z+h e_j)-y(z-h e_j))/(2h)`. Use float64 arrays and ordinary
NumPy arithmetic. Preserve all three raw `(T,d)` matrices in ignored
artifacts. Define the scaled matrix `A(h)=J(h)/(0.05*sqrt(T))`; the known
synthetic 0.05 V value is just a common reporting scale, not a likelihood or
Fisher-information claim. Compute singular values by `np.linalg.svd(A,
compute_uv=False)` in descending order. Report the full singular-value
vector, and descriptive relative ranks counting `s_i/s_1 > 1e-3` and
`> 1e-6` (rank zero if `s_1=0`). Report each column norm and all pairwise
absolute column cosines, using `null` for a cosine with a zero-norm column.
For each column and adjacent step-size pair, report
`max(abs(J_j(h)-J_j(h/10))) / max(1, max(abs(J_j(h))),
max(abs(J_j(h/10))))`; report `null` if either trajectory failed.
If either plus or minus trajectory for a coordinate/step fails, mark that
column invalid; do not form its central derivative. If any column is invalid
at a step size, report that full `A(h)` and its SVD/ranks as unavailable,
while retaining other valid columns and failure categories. A failed truth
trajectory makes its whole family/case unavailable. Do not replace a central
derivative with a one-sided derivative after seeing its value. Report the
step-size comparison metric numerically without a post hoc binary stability
label; variation may reflect branch switching or nonsmoothness, not a
mechanism. The explicit
`max`/`min` dry floor and cap can make derivatives undefined at a switch;
local rank can never certify global identifiability.

## Execution, resource and interpretation gate

Use protocol ID `cascaded_tanks_synthetic_sensitivity_v1_20260927` and run
ID `ct-sensitivity-20260927-v1`. Before any simulation, a Luna operator
must inspect actual processes/checkpoints and save an immutable ignored
pre-run manifest with this protocol's exact hash/reviewed Git commit,
simulator and runner code hashes, runtime executable and dependency
versions/paths/hashes, arrays/case definitions, numeric settings, and the
one-use run ID. Reuse the stable Python 3.14.3 / NumPy 2.4.2 / psutil 7.2.2
runtime from the independently reviewed prior-screen v2 amendment if and
only if the pinned files still match; never use a transient `uv` build path.
A different Luna reviewer must check the manifest, source-free code path,
finite-difference formula, no-crossing expectation, exact case roster and
watchdog **before** one launch. Use one worker, a 120-second hard parent wall
limit and 1 GiB sampled runner-tree RSS every 0.1 seconds, recording the
watchdog's own exclusion and between-sample limitation. Save timestamped,
per-PID RSS samples and process-tree totals in ignored artifacts, together
with peak, timer-start and kill semantics, so the recorded resource peak
can be independently reconstructed. Claim a one-use
attempt marker before simulations and preserve any failure or timeout as
terminal under this ID. Do not silently rerun or change step sizes after
outcomes. Detailed matrices/output arrays stay under ignored
`artifacts/evaluations/cascaded_tanks_synthetic_sensitivity/`; only compact
independently replayed evidence enters Git. A separate Luna reviewer must
recompute the arrays and summaries from the frozen settings, inspect all
failures and the exact O2/C2 divergence, and report step-size variation and
suspected nonsmoothness descriptively without a binary stability label
before the primary uses the result. No real-source access, Qwen,
ABC, validation scoring, or official-test access is permitted in this run.

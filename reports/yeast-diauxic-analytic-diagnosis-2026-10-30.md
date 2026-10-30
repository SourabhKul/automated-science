# Yeast pilot: analytic diagnosis and decision

Decision at the October 30 09:01Z wake: keep the first pilot closed.
The current finite importance reference has no demonstrated coverage for all
eight cases. The next step is a design decision about reference adequacy and
artifact transactions. This memo does not select a replacement sampler,
change a parameter or authorize a run.

The prior result remains five failed control cases and a separate artifact
stop. There was no ABC or held-out evaluation. See
`yeast-diauxic-pilot-results-2026-10-30.md`. The training receipt is
`bf7e453c0ccbddc029bdb42fc535b0a7c8c126aed741b33b2fec096005802844`;
compact all-case progress is
`fb743c1990926e5a8562a329ed211344a872e6daf3a3a3171842553b54251389`.
This diagnosis uses frozen equations, source and compact diagnostics, plus
file-size metadata. It reads no observation, NPZ payload, point-history
content, raw archive or plot. It performs no fit, simulation or numerical
check. All consumed IDs and prior evidence remain unchanged.

## What the defensive mixture guarantees

Let x be on the unit cube, with prior density one. The frozen proposal is
q(x)=0.8*q_LN(x)+0.2. It evaluates the full mixture density for every draw.
For z=logit(x), the logistic-normal density in x is

\[
\log q_{LN}(x)=\log\phi(z;\mu,C)-\sum_j\log[x_j(1-x_j)].
\]

The source uses this Jacobian sign. It does not divide a draw by only its
selected component's density. This is consistent with the mixture-weight
construction in [Owen, chapter 9, section 9.11](https://artowen.su.domains/mc/Ch-var-is.pdf).

The following bounds are derived here for the frozen Gaussian model.
For fixed positive-definite covariance Sigma and finite observations, write
L(x)=C0*exp(-Q(x)/2), where Q is nonnegative and
C0=(2*pi)^(-n/2)*det(Sigma)^(-1/2). Thus 0<L(x)<=C0. Let
Z=integral L(x) dx over the unit cube. It is finite and positive. Since
q(x)>=0.2, the unnormalized importance weight w=L/q satisfies

\[
w(x)\leq5C_0,\qquad
E_q[w^2]=\int\frac{L(x)^2}{q(x)}dx
\leq5\int L(x)^2dx\leq5C_0Z.
\]

So an infinite unnormalized-weight variance is not a valid diagnosis for
this exact ideal-arithmetic mixture/model. Its normalized posterior ratio
is bounded by 5*C0/Z, but Z is unknown and that bound can be very large.
Full support and finite variance do not ensure useful accuracy at 16,384
draws. A narrow important region can receive too few draws. The finite
self-normalized estimates remain approximations. Floating-point saturation
is a separate implementation stop, not covered by this ideal bound.

ESS and block agreement cannot prove that all important regions were found.
This limit of weight-based diagnostics is also discussed in
[Owen, section 9.3](https://artowen.su.domains/mc/Ch-var-is.pdf).
The three passing cases therefore supply no license for a survivor-only
comparison. The five failures still block downstream inference. No observed
log-normalizer is used here as an accepted evidence or model score.

## What the local matrix represents

The whitening operator W is fixed by the frozen noise covariance and times.
Let x_j=sigmoid(z_j), h(z)=W*f(x(z)), e=Wy-h(z), and
J=dh/dz. Let H_i be the Hessian of h_i. A uniform x prior becomes density
proportional to g(z)=product_j x_j*(1-x_j) in z coordinates.
The negative log target, up to a constant, is

\[
U(z)=\tfrac12 e^Te-\log g(z).
\]

Direct differentiation gives

\[
\nabla^2U(z)=J^TJ-\sum_i e_iH_i
+\operatorname{diag}\{2x_j(1-x_j)\}.
\]

The sign of the residual-curvature term is negative. The transformed-prior
term is positive. There is no covariance-derivative term here because the
noise covariance is fixed. This is a coordinate calculation, not an estimate
of a case's actual Hessian. The actual weighting target stays in x, where
the prior density is one. This coordinate calculation does not identify a
missing prior factor in the executed weights.

The frozen code forms G=J^TJ+0.25*I at a point clipped only for proposal
centering to [0.02,0.98]. It then uses covariance eigenvalues
clip(2/eigenvalue(G),1e-6,4). The finite difference is in z, so it already
includes the sigmoid chain rule. The matrix is a regularized local
Gauss–Newton scale heuristic. It is not the full observed Hessian, a proved
Laplace covariance or a coverage guarantee. Adding a missing Hessian term
is not a justified automatic repair: the full Hessian need not be positive
definite at a bounded search result, and local curvature still cannot certify
global coverage. Seven point searches exhausted their allowed calls; none
has a certified global optimum.

## Weak directions exist in the equations

These are structural properties, not identified causes of any recorded
well's failed ESS:

- At A=0, phase rates, centers and allocation do not affect the mean curve.
- At q=0, the second phase's rate and center do not affect the curve. At q=1,
  the first rate is inactive. The ordered parameterization couples c1 to c2,
  so c1 need not be inactive at q=1.
- At equal rates and equal centers, the two sigmoid values agree. The
  allocation derivative A*(S2-S1) is zero. Other phase derivatives become
  dependent; the extra parameters are not all separately identified there.
- The logit chain factor x_j*(1-x_j) tends to zero near a cube boundary.
  Changes in z then produce small local changes in x. Clipping a proposal
  center does not change the prior or prove adequate boundary coverage.

These sets lie on exact boundaries or equality surfaces with zero mass
under the continuous cube prior. Nearby regions can still have weak
curvature. A single Gaussian in z can miss curved or separated regions.
That is a possible generic failure mechanism. The permitted compact
receipts do not locate the observed cases on those sets, establish
multimodality, or isolate geometry as the cause of their weight concentration.
The same concern can arise in one-phase models through approximately
confounded amplitude, offset, rate and center over a finite time span.

## Artifact diagnosis: retention and transaction peaks differ

The consumed training run retains 59,498,890 bytes. A stat-only audit found:

| Category | Bytes |
| --- | ---: |
| Eight full point JSON histories | 50,443,503 |
| Sixteen importance block NPZ files | 9,013,194 |
| Observation NPZ files, complete and partial | 9,946 |
| All-case progress | 14,526 |
| Final receipt | 17,246 |
| Outer stdout | 475 |

These sizes use directory entries only. They do not interpret those files.
The attempted aggregate partial bundle repeats draws and weight-related
arrays already retained by block. Its fixed-shape array payload is
10,001,112 bytes before NPY/ZIP overhead and compression. This is a shape
calculation, not its deleted compressed size. With 59,481,169 bytes already
retained before final receipt/stdout and a 1 MiB receipt reserve, the new
bundle could occupy at most 6,579,119 bytes under the final-size projection.

`write_npz` serializes the full temporary file before testing the projected
size. It subtracts any old destination size when estimating replacement
size. `write_json` also subtracts the old destination in its final-size check.
An old file and its temporary replacement coexist before atomic rename.
Thus these guards are not a prior bound on peak logical file bytes during
all transactions. The deleted temporary size is unknown. The recorded stop
can occur while the receipt reserve is crossed but the hard 64 MiB total
is not. Do not infer that the hard cap was actually exceeded.

A future design needs a deterministic transaction envelope before writes:
other retained bytes + old destination + worst-case temporary bytes +
receipt/manifest reserve. Its post-commit footprint is a separate quantity.
Compression cannot be assumed to reduce size. Encoding overhead and bounded
variable text must enter that envelope. A manifest can refer to already saved
blocks instead of copying their payload into another bundle. Such choices
must retain complete failure/cost accounting and the final success barrier.
They are requirements for review, not approved code changes. Do not delete,
compact or rewrite the consumed run to recover space or enable ABC.

## Decision gates before any later study

1. Keep the x-space target and the historical failure unchanged. A future
   generic design must state what its reference can validate, including
   weak, boundary and separated regions. Unknown coverage must be an
   allowed terminal result. Do not claim an exact posterior from finite IS.
2. Choose any new reference method and its adequacy criteria before a new
   numerical study. Preserve full density/Jacobian accounting and all-case
   stop rules. Neither larger draws, a tuned covariance nor a different
   mixture weight is selected by this memo. Do not reuse this cohort to
   optimize a desired curve preference or declare it a fresh blind study.
3. Prove the retained and transaction byte envelope from bounded records,
   shapes and encoding rules. Require it within the existing intended cap,
   or reconsider feasibility before requesting a different scope. A storage
   repair alone cannot clear the five reference-quality failures.
4. Only a separate frozen protocol, independent design/code review and new
   explicit launch decision can authorize numerical checks or another study.
   This memo allocates zero new curve/model calls and opens no plate2 data.

No new download or user action is needed. All closed tank work, fixed
research-model settings and deferred B2a files remain unchanged. This is
methods diagnosis, not a scientific breakthrough or completed transfer test.

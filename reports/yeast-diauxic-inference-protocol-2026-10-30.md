# Four-well paired-plate inference protocol

Primary draft at the October 30 03:57Z wake. This freezes a small prospective
analysis of retrospective data, before OD inspection. It does not authorize
observation parsing or fitting. Independent design review, exact-code review,
bounded numerical checks and a separate primary launch GO are still required.
All completed metadata/source stages remain closed. No new framework or
change to the closed tank product is in scope.

## Question, records and order

Use the four literal wells B2=DBVPG6765, B3=SK1, B4=Y55 and B5=YPS128.
Use only the 271 mixed-sugar plate1 files for training. Predict the same wells
at all 271 plate2 assay times. These are technical-assay predictions for known
labels, not independent cultures or new-strain transfer. Do not merge labels,
replace failures, use published filters, or include single-carbon/derived
rates. Manual historical metric overrides must not enter any analysis rule.

Bind the original Dataset+1.zip SHA-256
`2ac81bce8313c140487e51a0ac373239ff96a88a670f8ee5e5037f7d738c28ab`,
raw-metadata receipt
`5af91336cc1bd8ebf88976aee49b38489ef8d0b63458e814f61395623165baf2`
and clock receipt
`d42c4223bae1009f1dc6f1a243718060b2b10127c2374aff6dda9a1701d838fb`.
The prior source/layout receipts fix well coordinates. Originals stay intact.
Reuse the saved identities and timestamp order; no new inventory/time stage.

Set t in hours from 7/19/2013 5:10:58 PM. The timezone is unknown; a shared
stable local-clock convention across plates is an assumption. Set
T=158967/3600 hours from metadata only. Retain irregular times and gaps.
Plate2 starts at 113/3600 hours and ends at 159152/3600 hours, including
185 seconds beyond training. Do not trim or interpolate after seeing values.

First read training observations under the reviewed reader contract below.
Run all eight well/model cases in well order B2,B3,B4,B5, one-phase then
two-phase. Complete the point and likelihood controls for all eight before
any ABC. If every reference-quality gate passes, run ABC for all eight.
Only a complete successful training stage may freeze predictions. Save and
hash all point parameters, weighted clouds, protocol/code/input bindings,
seeds, diagnostics, and plate2 mean predictions before any plate2 OD parsing.
A separate evaluation process must verify that immutable freeze first.
Evaluation is once only, without fitting or calibration.

## New observation reader

This is a new observation stage, not a metadata rerun. Open only the bound
plate1 members in training; plate2 members only after the prediction freeze.
Read each selected member once per stage, at most 128 KiB, 256 physical
records and 8 KiB per record; the selected total remains below 16 MiB.
Match the saved size/CRC/channel/first footer timestamp while locating the
same channel block. Read to EOF for CRC. The tail remains opaque. No other
archive member, MATLAB source, MAT cache, plot or non-OD table may open.

Require eight matrix records, each with twelve comma fields, optionally
followed by one empty trailing field. Reject quotes, escapes or another
shape rather than guess. Only row B, columns 2 through 5 may be converted
to finite real numbers. Other cells remain opaque byte fields. This checks
structural shape and the selected numeric cells, not every well's numeric
validity. Accept ordinary signed decimal/exponent notation; no NaN/Inf,
empty or missing selected cell. No clipping, smoothing, logging, blank
subtraction, outcome filter or value-derived alias mapping. A bad selected
cell stops the stage; do not drop rows or choose another well. Save the
four selected training trajectories only in ignored storage. Raw observed
values are not to be printed or committed.

## Curves and prior

Let S(v)=1/(1+exp(-v)), evaluated with a stable sigmoid. Work in independent
uniform coordinates x in [0,1]^d. The density is one on this cube. These
coordinates, not physical parameters, are the ABC and importance targets.

For one phase, d=4, map x to:

- b=-0.05+0.30*x[0] raw OD units;
- A=1.50*x[1] raw OD units;
- r=0.01*200^x[2] per hour;
- c=-0.5*T+2*T*x[3] hours.

Then f1(t)=b+A*S(r*(t-c)). For two phases, d=7, use the same b,A,r1,c1
mapping from x[0:4], q=x[4], r2=0.01*200^x[5], and
c2=c1+x[6]*(1.5*T-c1). Define
f2(t)=b+A*((1-q)*S(r1*(t-c1))+q*S(r2*(t-c2))).

The ordered centers prevent label exchange when centers differ. Equal-center
cases retain an exchange degeneracy. q=0 or 1 retains a one-phase
boundary; equal rates/centers also nest one phase. Zero total amplitude is
retained. Center order permits equality; it does not prove delayed substrate
use. The conditional c2 prior is not an ordered iid center prior. Report
boundary/weak-identification flags, not molecular parameters or model
probabilities. No prior or support change follows from the observed values.

## Working measurement hypothesis

Use raw OD with a training-only curve intercept, not measured blank correction.
Assume plate offset and gain transfer. For each well's marginal analysis,
y[p,k](t)=f[k](t;theta[k])+u[p](t)+e[p,k](t). Set sigma_e=0.005 OD,
sigma_p=0.010 OD and ell=1 hour. These are fixed working assumptions, not
measured instrument calibration. e is iid Gaussian; u is a stationary
zero-mean OU plate process with covariance sigma_p^2*exp(-abs(dt)/ell).
Plate processes are independent across plates. Common instrument drift
could violate this assumption. Do not tune noise using either plate.

Each well uses only its own training values. The four results remain
correlated through plate effects; no joint four-well likelihood or population
inference is permitted. Long-scale plate offsets confound b. No absolute
biomass calibration or GAL-induction claim follows from either curve.

Use exact scalar Gaussian innovations for covariance
Sigma=sigma_e^2*I+sigma_p^2*K_ell, with actual within-plate time gaps.
Initialize predicted state mean m=0 and variance P=sigma_p^2. At each record
use v=(y-f)-m, F=P+sigma_e^2 and gain K=P/F. Update m_post=m+K*v and
P_post=P*sigma_e^2/F. For the next gap, set phi=exp(-dt/ell),
m=phi*m_post and P=phi^2*P_post+sigma_p^2*(-expm1(-2*dt/ell)).
Whitening W maps residuals to v/sqrt(F). The log likelihood is
-0.5*(n*log(2*pi)+sum(log(F))+sum((W*r)^2)). Check it against a dense
Gaussian calculation before launch. Plate2 has a new stationary OU process,
not a continuation or fitted effect from plate1.

## Point and likelihood controls

Use the existing normalized-cube pattern-search callback, unchanged, with
negative Gaussian log likelihood and 4096 actual calls per well/model. Keep
its complete diagnostics. A finite best point is a bounded search result,
not a proven global optimum. No extra starts or local polishing.

For a tractable likelihood-weighting control, adapt one known-density
proposal using that training point. Clip its coordinates to [0.02,0.98]
ONLY to locate the proposal center z0=logit(x); do not alter the point result
or prior. Form the central-difference Jacobian of the curve with respect
to z, step 1e-4 in each dimension. Whiten its columns with W. Set
G=Jw'Jw+0.25*I. Eigendecompose G and use covariance eigenvalues
clip(2/eigenvalue(G),1e-6,4), retaining eigenvectors.

Draw from q(x)=0.8*logisticNormal(z0,C)+0.2*UniformCube. The logistic-normal
density in x is logq_LN=logN(logit(x);z0,C)-sum(log(x_j*(1-x_j))).
Inside the cube, logq=logaddexp(log(0.8)+logq_LN,log(0.2)); the Gaussian
component has zero density at the cube boundary. Both components enter the
mixture density for every draw. Target weight is exp(logL-logq),
since the normalized-cube prior density is one. Use log-sum-exp arithmetic.
No rejection or silent clipping of saturated/nonfinite transformed draws;
stop with counters if they occur. This is finite adaptive importance
sampling, not an exact posterior or proof of global mode coverage.

Use two independent blocks of 8192 draws each, the same fixed proposal.
Require ESS >=128 and maximum normalized weight <=0.05 in each block,
pooled ESS >=256, absolute block log-normalizer difference <=0.5 nats,
and RMS difference between block mean curves at all plate2 times <=0.0025
OD. These are reference-quality gates, not calibration guarantees. Preserve
all draws and diagnostics. No extra draws, proposal retuning or survivor-only
analysis. If any case fails, finish the other bounded controls, then stop
before ABC and plate2 parsing with an incomplete-pilot result.

## ABC-SMC

Only after all likelihood-control gates pass, use the existing explicit
Gaussian ABC-SMC reference callback; no canonical engine integration.
The prior is uniform on the same normalized cube. A simulator returns
f(t;x) plus a fresh stationary OU path and iid error at training times.
Its discrepancy is sqrt(mean((W*(simulated-y_train))^2)), using Sigma,
not 2*Sigma; two independent noise realizations can therefore contribute
to the discrepancy. Fix epsilon=[16,4,1.5], target 64 particles per
population, attempts=[4096,8192,16384], covariance_scale=2,
lambda_noise=0.01 and nugget=1e-9. No quantile adaptation or extension.

Count every proposal, including support rejection and failed simulation,
discrepancy or weight. Save population callbacks and any partial population.
No failed-particle dropping, previous-population substitution, automatic
retry or epsilon change. The existing runner normally catches and continues
after simulator/discrepancy failures. The adapter must latch any such failure
and pass stop_requested so the runner stops before its next attempt. Persist
all emitted population events. Wrap the runner to preserve outer exception
evidence, since transition-covariance construction can throw before a partial
return. Any nonfinite callback, failed weight or thrown covariance/runtime
error stops the stage with evidence. Any incomplete case stops before the
prediction freeze and plate2 parsing. Finite tolerance changes the target;
do not call agreement a calibrated posterior or weights model probabilities.

Case index j=0..7 follows the well/model order above. Importance block b=0,1
uses seed 30100+10*j+b. ABC uses seed 31100+j. All generation is serialized;
no local LLM request or extra synthetic research campaign is in scope.

## Freeze and score

For each case and method (point, pooled likelihood weighting, final ABC),
freeze plate2 mean predictions and the full parameter/weight representation.
All eight cases and all three methods must freeze before evaluation starts.
A provisional freeze manifest alone is not success evidence. Publish the
final commit manifest only after all artifacts, resource/storage checks and
the successful training receipt have completed. Bind that final receipt's
hash in the commit manifest; evaluation must verify its success status and
hash as well as every manifest/content hash. Bind the actual driver and
used point-search/ABC module hashes from exact review; reject a code change.
Before opening ANY plate2 member, validate ALL eight cases' frozen parameter
and prediction arrays: exact shapes, finite in-support parameters, finite
predictions, nonnegative finite normalized weights and quality/completion
flags. Missing, incomplete or changed evidence stops evaluation. Only then
read the four selected plate2 values from every bound file once.
No refit, offset/gain adjustment, new seed, source change or second score.

Report raw-OD mean-curve RMSE and whole-trajectory negative log predictive
density divided by 271 for each well/model/method. The latter uses the
Gaussian likelihood with plate2's own time covariance and integrates the
fixed weighted cloud with log-sum-exp; point prediction is a delta parameter
distribution. Keep all four well results separate. No joint score, p-value,
independent-culture n, Bayes factor or molecular mechanism claim. A method
or curve can be worse; do not change the protocol to improve a score.

## New finite engineering and run limits

Use one standalone pilot adapter/driver, at most 800 nonblank source lines
(including comments/imports), and at most 200 nonblank lines of focused
checks. Report physical counts too; keep blank lines for readable code.
Primary raised the initial 650-line estimate before any execution:
static review required durable completed-case evidence, a final-success
freeze barrier and validation before held-out reads. The intermediate 725
estimate was too small: the completed repair pass has 763 nonblank lines.
The final 800 ceiling leaves limited repair room and permits no new feature.
This adds no model,
data, inference allowance or product feature. Existing product code and its
994/299 runtime/test counts stay unchanged. NumPy and the checked existing
callbacks only; no install or new inference framework. Numerical checks
must cover innovations/dense parity, OU covariance, curve nesting, transformed
proposal density/weights, fixed seed and incomplete/freeze refusal behavior.
These are code checks, not a new model-ranking or reserved tank campaign.
Use check ID `yeast-diauxic-pilot-check-20261030-v1`, at most 60 seconds
processing, 512 MiB sampled RSS, 8 MiB new artifacts and 10,000 curve calls.
Synthetic fixtures only; no real source member, observation, server or model
request. Exact code/check review precedes even these numerical checks.

Run IDs: `yeast-diauxic-pilot-train-20261030-v1` and
`yeast-diauxic-pilot-eval-20261030-v1`. Use exclusive ignored directories.
One process and one simulation worker, BLAS threads one. Training processing
wall <=600 seconds; evaluation <=120 seconds; sampled RSS <=1 GiB;
combined new artifacts <=64 MiB, compact receipt <=1 MiB. Record excluded
finalization/review time. Preflight and after-run measure project data/
artifacts plus three Downloads originals, retaining a 64 MiB reserve below
50 GiB. No raw archive copies, disk extraction, bulk tracked files or secrets.

At most 32,768 point calls, 88 finite-difference curve calls, 131,072
importance likelihood calls and 229,376 ABC proposal attempts. Prediction
and scoring each require at most 131,592 curve calls; report them separately
from fitting/simulation. Hard combined curve-call cap 660,000, not authority
for extra draws or tuning. Record actual curve, likelihood, simulation,
discrepancy, proposal/support/weight, prediction and score counts, failures,
elapsed stages, RSS and bytes. Tank cumulative costs remain fixed.

Stop on resource, source/hash/schema, numeric or reference-quality failure.
Preserve every partial and completed case; report all eight statuses. Do not
leave an active case labelled running after a stop: save its stopped status,
phase, actual available counters and any unavailable counter scope. Persist
completed likelihood blocks before forecasts, completed ABC clouds before
the next case, and each completed evaluation score before further scoring.
All combined artifact checks include both run directories and final receipts.
Do not
restart an existing ID, silently recompute work or reuse leftover allowance.
A later recovery requires one new justified reviewed scope, not an automatic
budget increase. No user server, external message, purchase or submission.

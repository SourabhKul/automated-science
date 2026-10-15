# Cascaded Tanks week study: frozen design for independent review

2026-10-15 scope note: the first delivery is now the development-only protocol
in `cascaded-tanks-week-development-v1-protocol-2026-10-15.md`. Its independently
triaged 1,000-line/two-file engineering ceiling supersedes this document's
original 600-line estimate. Refit and official access remain deferred.

Protocol: `ct_memory_week_v1_20261014`. New study; separate from ABC6.
Run ID: `ct-memory-week-20261014-v1`; preserve a failed/partial run under this ID.
User authorized implementation on 2026-10-14. This design needs independent
Luna review before real fitting or target access. Preserve ABC6, its partial
B2a diff, and all historical failures. The source interpretation follows the
reviewed `cascaded_tanks_methods_v1_20260926` track. The original generic
external-repeat manifest remains unchanged.

## Question and claim

Does C2's retained hidden state predict recovery better than O2's state loss
at a ceiling? Compare S0, O2 and C2 voltage-equivalent surrogates from
`cascaded_tanks_models.py`. This is a predictive methods study on one rig.
It cannot identify physical overflow or sensor clipping. Seeds describe
compute variation, not physical replication. No calibrated posterior,
coverage, Bayes-factor or new-mechanism claim follows from these fits.

## Source and stage boundaries

Use the existing 7,520,592-byte archive with SHA-256
`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`.
Rehash the same immutable byte snapshot before each stage; do not download a
duplicate. Keep total active research data below 50 GB. Use the reviewed
column-limited training reader. Do not use the general official loader, which
materializes the test output.

Stages are separate CLI invocations with immutable inputs:

1. Development fit sees only `uEst/yEst[0:768]` and future `uEst[768:1024]`.
   Save every arm's parameters, weights, forecasts or terminal failure before
   any suffix output is read. At k, observe y[k], then apply u[k] to reach
   state k+1. The validation forecast carries each fitted latent state across
   the training boundary; it does not use suffix outputs for initialization.
2. Development score exclusively claims its start marker, checks the entire
   declared arm roster is frozen, then reads `yEst[768:1024]` only for complete
   arms. Failed arms retain null error. Save a terminal outcome even on error.
3. Refit may read all `uEst/yEst[0:1024]` only after a valid development-score
   terminal and locked selection. No official test field enters refitting.
4. Official scoring exclusively claims its own start marker before reading
   `uVal` and the first 50 `yVal` values. Fit state only by the rule below;
   dynamics and all settings remain frozen. Attempt each arm's whole forecast
   before reading `yVal[50:]`. Complete predeclared arms may be scored when
   another arm fails; never score a failed arm or drop its failed particles.
   No second official scoring attempt, marker removal or target-driven change.

Readers scan excluded fields only for CSV structure; do not extract, decode,
numeric-convert or log them. A trusted local checkout is assumed. Use source
hashes and one frozen JSON manifest, not the ABC6 authority/grant machinery.
Require stable regular source/receipt files and exclusive stage markers.

## Models, inference and budget

Parameter order `(a,c,p,x1_0)`, independent uniform support:
`a,c=[0.2,1.5]`, `p=[0.05,1.2]`, `x1_0=[0,25]`. Fix training/refit
`x2_0=5.205` and O2/C2 `H=10`. These are training-conditioned screen choices
and restrict the uncertainty represented. Preserve the existing simulator's
dry-state floor and observation-before-update alignment. Use float64,
`max_magnitude=1e6`, and the full requested trajectory length as `max_steps`.

Development has S0/O2/C2 × seeds `[11,29,47]`. Refit has all three families
at seed `71`. This replaces the audit's selected-family/three-seed refit with
an equal-cost three-family/one-seed refit. It permits a predeclared final C2/O2
contrast. It provides only one final inference seed per family; disclose that.
All final arm choices precede the official score. Do not switch a deployed
winner after looking at that score.

For each family/seed, draw 256 prior calibration samples total. Use
`Generator(PCG64(SeedSequence([20261014, stage, family, seed, stream])))`,
where stage is 0 for development and 1 for refit; family is 0/1/2 for S0/O2/C2;
stream is 0 for calibration and 1 for ABC. Recalibrate on the allowed training
record for each stage. Compute full
training RMSE. Failed simulations have +infinity distance and count as calls.
Freeze the two epsilons at q25/q10 of all 256 distances using NumPy's linear
quantile rule; calibration fails if a threshold is nonfinite, nonpositive or
ordered incorrectly (`q25 < q10`; equality is valid). Do not redraw or retune. Use the checked Gaussian ABC-SMC
reference, 64 accepted particles/population, at most 8,192 attempts/population,
the same fixed covariance in proposal and density, support rejection, and
normalized prior-over-mixture weights. Explicit kernel settings are
`covariance_scale=2.0`, `lambda_noise=0.01`, `nugget=1e-9`.
Persist both populations and diagnostics.
No BDSS, proposal clipping, partial-population posterior, or failed-particle
dropping. Incomplete ABC arms are terminal failures for prediction.

Each ABC fit uses at most 256+2×8,192 = 16,640 simulator calls. Fit a paired
bounded coordinate-pattern-search arm for the same family with allowance
equal to that ABC fit's realized calibration plus ABC simulator calls,
including failures. Support rejects are recorded but do not count as simulator
calls. Reuse the existing normalized-cube center/corners multistart algorithm
unchanged, with initial mesh 0.25 and stop threshold `2**-10`; its pinned source
defines poll order, tie handling and restarts. Count actual callback
calls and simulated steps; unused allowance need not be wasted.

Maximum development allowance: 9×16,640×2=299,520 simulator calls. Refit:
3×16,640×2=99,840. Combined: 399,360. Use one CPU worker initially, a 90-minute
hard fitting wall limit across both phases, 4 GiB sampled worker-tree RSS and
1 GiB study artifacts. Stop and retain partial results on a breach. No automatic
restart. Record actual wall/RSS and sampling limitations. Smoke tests get
at most 2,000 short synthetic simulator calls and 120 seconds separately.

## Forecasts, baselines and initialization

For complete ABC arms, forecast every final-population particle. Use the
pointwise weighted mean for squared-error scoring; retain weighted 10/90%
forecast bands using stable value sort and the left empirical inverse CDF
(first cumulative normalized weight >=q, no interpolation). They are
descriptive parameter spread, not calibrated coverage.
Require finite complete forecasts and valid normalized nonnegative weights.
One failed particle fails that arm. Point-search arms forecast their best
finite training parameter. Report every arm, including failures.

Fit one ARX(2,2) baseline with intercept on each allowed training record:
`y[k]=b0+a1*y[k-1]+a2*y[k-2]+b1*u[k-1]+b2*u[k-2]`. Use ordinary least
squares (`numpy.linalg.lstsq`, `rcond=None`), no target tuning or pole repair.
Development free-run starts from the last two measured training outputs.
Official free-run starts from test outputs 48 and 49; exclude the first 50
points from scoring. Mark ARX unstable if any root of `[1,-a1,-a2]` has
magnitude >=1; do not repair poles. Report unstable/nonfinite forecasts as failed. Persistence
uses the last permitted measured output. Report their costs separately;
closed-form ARX and persistence are not simulator-call-matched optimizers.

For each official simulator forecast, hold all fitted dynamics parameters
fixed, including H where defined. Replace the
training-record initial state with a fixed state-only grid fit to the first
50 test outputs: 17 equally spaced `x1_0` values on [0,25], and 17 `x2_0`
values on [0,25] for S0/C2 or [0,10] for O2. Simulate inputs [0:50], minimize
float64 RMSE over outputs [0:50], then carry the selected state using u[49]
to forecast y[50]. Break exact ties by ascending x1 then x2. Count failed
grid trials; fail the arm if no finite initializer exists. This is a nuisance
point initializer, not initialization uncertainty. Cap at 60,000 50-step
initializer calls, 300 seconds and 4 GiB RSS. The fixed grid has 56,355 calls
for 3×64 ABC particles plus 3 point arms. A 50-input initializer simulation
already returns state 50 after u[49]; do not apply u[49] twice. Development
forecasts may replay concatenated permitted training and future inputs in one
1,024-step call and slice outputs [768:1024], preserving carried state.
Charge that full replay's steps. Development forecasts have a cap of
600 1,024-step calls (at most 585 planned); official forecasts have 500. Each forecast phase has
120 seconds and 4 GiB RSS. No dynamics fitting uses the test initializer.

## Decisions and Qwen role

Before opening development targets, freeze this decision rule. Compute RMSE
for all complete arms. For ABC families complete under all three development
seeds, report median RMSE. If fewer than two families are fully eligible,
lock the outcome as unresolved. Otherwise prefer one family only if its median is at least
10% below each other fully eligible family; otherwise declare unresolved.
Do not infer a win over failed families. Report C2/O2 paired seed contrasts
separately. Claim useful predictive performance only if the selected family's
median also improves on both ARX and persistence by 10%. These are engineering
margins, not significance tests. Report point-fit contrasts and cost separately.
Final scoring is a predeclared all-arm check. A reversed C2/O2 direction rejects
the transfer claim; an absent/small contrast stays unresolved. Do not tune from it.

The study makes at most one serialized structured Qwen request before target
access, using exact model `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` at
`http://127.0.0.1:8000/v1/chat/completions`, temperature 0, 768 output tokens,
120-second timeout, `chat_template_kwargs={enable_thinking:false}`, no retry.
Supply mechanism definitions and training-only summaries. Ask for JSON with
family ranking, a recovery prediction, abstention flag and limits. Save raw
request/response, usage, timing and validation status. Do not execute generated
code or change the enumerated families based on this response. If unavailable
or malformed, record it and complete the numerical study. No LLM-benefit claim
follows from ranking three known families.

The user's newer-model comparison is separate and optional: two hours and
12 requests including repairs, fixed development/synthetic tasks, no sealed
outcomes. It is pending exact model availability and must not delay this study.

## Implementation, review and reproducibility

At most two new runtime files and 600 runtime lines, plus one focused test
file up to 300 lines. Reuse numerical and safe-reader code. Reduce scope if
these limits cannot hold; do not grow another gate system. Tests must cover
time alignment, carried-state forecasts, independent target sentinels, fixed
initializer recovery, failure retention, exact weight aggregation and cost
accounting. A different Luna max reviewer must review final bytes and tests
before real fitting and again review saved forecasts before target scoring.

The run manifest freezes archive/adapter/model/reference/pattern-search/new
runner/scorer source hashes, protocol hash, Git head, actual runtime versions,
all numeric settings, seeds, arm roster, resource limits and run ID. Populate
real hashes only after code review; do not invent them. Store raw arrays and
receipts in ignored artifacts. Compact reports include all arm/seed metrics,
costs, ESS, failures, forecast/residual plots, parameter correlations, and one-rig
limits. The sole Luna Git owner syncs only reviewed scoped code and reports,
preserving the deferred B2a edit. No scientific breakthrough is predeclared.

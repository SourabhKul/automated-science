# Cascaded Tanks: first development result

Protocol: `ct_memory_week_dev_v1_20261015`.
Run ID: `ct-memory-week-dev-20261015-v1`.
Primary scope decision, 2026-10-15. Independent design/code review is required
before real fitting or target access. No scientific results exist for this ID.

The first implementation tried to cover fitting, development scoring, full
estimation refit and official test scoring together. Its 477-line partial
scorer still lacked key phases. This version reduces scope to development
fitting and its separate score only.
Preserve existing partial work. Defer full-estimation refit, test initialization
and official scoring to a separately reviewed later scope. The CLI must reject
those phases; no official input or output accessor belongs in this version.
This is a scope cut before any new target access, not a response to outcomes.

Before any outcomes, an independent Luna source triage found no safe 300-line
cut from the 933-line draft. The existing fit wrapper conflicts with the
declared random streams and paired optimizer; the selective suffix scanner
has no real-source replacement. Primary therefore revised the engineering
ceiling once to 1,000 readable runtime lines across the same two files, with
no new framework or file. This supersedes the audit's 600-line estimate.
The numerical, wall, memory, target and test-line limits do not change. Stop
and reassess if this ceiling cannot hold; do not expand it again automatically.

The objective is a complete real-data prediction/cost table, with all failures
visible. The scientific question remains C2 retained memory versus O2 state
loss, with S0 as the no-ceiling alternative. These are surrogates on one rig.
Inference seeds measure compute variation, not independent replication.
The development suffix is model-selection evidence, not independent confirmation.

Use the source interpretation and all development numeric settings in
`cascaded-tanks-week-study-protocol-2026-10-14.md`, with the explicit rules below:

- Archive SHA-256 remains
  `eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`;
  use the existing source and reviewed column-limited training adapter.
- Fit only uEst/yEst[0:768]. Future input is uEst[768:1024]. Never extract
  uVal/yVal, or convert yEst[768:1024] during fitting, prompting or forecasting.
- Enumerate S0/O2/C2, each at seeds 11, 29 and 47. Use the four-parameter
  support, fixed x2_0/H, float64 alignment and magnitude rule from the parent
  design. Development forecasts carry latent state through training.
- Use 256 total calibration draws per family/seed, with the parent's explicit
  PCG64 SeedSequence mapping at stage 0. Save every calibration outcome;
  failures have +infinity distance, encoded as numeric arrays or tagged JSON.
  Use linear q25/q10 over all draws. Fail if either threshold is nonfinite or
  nonpositive, or if q25 < q10; equality is valid. No redraw or retuning.
- Run the checked Gaussian reference with the fixed kernel settings,
  64 particles/population, two populations and 8,192 attempts/population.
  Keep partial populations as failures. Simulator callback calls equal
  `simulated + failed_simulations`; support rejects are not simulator calls.
- Each paired generic pattern-search arm gets the actual calibration plus
  ABC callback allowance for its family/seed, including failed calls. Reuse
  the existing center/corners algorithm unchanged. Count its actual calls,
  cached requests, failed calls, steps and unused allowance honestly.
- Include ARX(2,2) with intercept, least squares rcond=None and the fixed
  characteristic-root failure rule. Include persistence. Both use the last
  permitted training outputs for free-run initialization. They are cheap
  baselines, not simulator-call-matched numerical optimizers.
- ABC prediction is the normalized-weight mean over every final-population
  particle. Retain the parent's weighted 10/90 empirical bands. A failed
  particle fails its arm; do not discard it or renormalize the survivors.

Freeze the exact roster of 18 simulator arms (nine ABC and nine point-fit),
plus ARX and persistence, before execution. Checkpoint each completed or
failed arm. A resource stop terminates the started arm with explicit
`resource_stop=true` and cause, and records never-started arms as not run;
do not label a resource stop as numerical model failure. Do not silently
restart an incomplete fit. Freeze every arm's forecast or explicit failure
before the separate score call. The score stage checks that frozen roster,
manifest and forecast hashes, claims an exclusive start marker and then
converts only yEst[768:1024]. Complete arms may receive errors even when an
ABC arm failed. Failed/not-run arms retain null metrics. If no arm has a
complete forecast, record terminal failure without converting targets.

Retain the parent's median-seed, >=10% engineering margins and unresolved
rules, including unresolved when fewer than two ABC families are eligible.
Report C2/O2 per-seed contrasts, all point-fit and cheap baselines, ESS,
parameter spread/correlations, cost, forecast bands and failures. Do not claim
calibrated probability or interval coverage. Preserve the official test.

Development maximum is 299,520 fitting simulator calls, one worker, 60 minutes
hard fitting wall time, 4 GiB sampled worker-tree RSS and 1 GiB artifacts.
The remaining 99,840 calls and 30 fitting minutes in the parent week ceiling
are reserved for a later frozen refit protocol, not available to this version.
Development forecasts have a separate cap of 600 1,024-step calls and 120
seconds; charge training replay steps. Synthetic smoke stays below 2,000
short calls, 120 seconds and 1 GiB sampled RSS. Record sampled-enforcement
limits. Check workloads and checkpoint evidence before every launch.

The one structured original-Qwen study request can be made operator-side
under the parent's exact model, payload and timeout rules. It cannot change
the enumerated candidates or use suffix targets. Save its request/response;
failure does not block the numerical study. Strata at port 8080 is a separate
optional acceptance check. Its exact served ID is now reported as
`qwen3.8-flash-next-iq3_s`, unloaded at the 09:03Z check. Configure calls for
it; do not start its script or request unload while the user is setting it up.
Any generation/load needs a fresh status check and the bounded acceptance
protocol. It is not a prerequisite for this development result.

Use no more than two new runtime files totalling 1,000 readable lines and one
focused test file of at most 300 lines. No code-golf. Require an independent
Luna max review of exact bytes, sentinel/alignment/weight/accounting tests,
and immutable source/runtime manifest before fitting. A separate reviewer
must inspect saved target-free forecasts before the score call. No ABC6
authority, grant, native attestation or public replay work is required.
The code, tests and results remain unapproved until those checks finish.

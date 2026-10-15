# Project audit and seven-day reset

Date: 2026-10-14. Status: recommendation, not a launch protocol. Research
execution remains paused at the user's request.

The user requested a one-time `gpt-6.1-sol` audit at maximum reasoning. The
auditor read project reports, source, history, the current diff and source-file
metadata. It ran no tests, fits, generation, experiments, commits or pushes.
It did not read protected target values. The primary checked the main source
claims and the Silverbox report, then verified the official tank split through
the pinned loader and benchmark website. Pilot results below are historical
reported evidence, not new runs. Default delegated execution remains
`gpt-6-luna` at max reasoning. The Sol audit is a one-time exception.

## Main finding

The project has useful numerical controls and a real development-run failure.
It has no independently validated scientific finding. Too much recent work
has gone into the ABC6 execution and receipt system. That work has not reached
a launched study. The next week should end with a real-data prediction table,
cost comparison, and decision about the models.

The current unreviewed B2a patch adds 1,105 runtime lines with no new test-file
changes. Its purpose is fake-only lifecycle verification. It still leaves
public v4 replay, production bindings, source/runtime closure and launch
approval open. Preserve this work and defer it. Do not finish it because time
has already been spent on it. Use a new, small protocol for the next study;
do not relabel or bypass the old ABC6 campaign.

## Evidence and its limits

| Area | Evidence in the records | Remaining limit |
| --- | --- | --- |
| Train/validation/sealed boundary | Opt-in canonical path and synthetic sentinels; bounded Qwen smoke | No useful independently confirmed discovery |
| Checked ABC-SMC reference | 18-cell scalar controls and six-cell corrected ODE control | Limited tests of calibration; no broad model-discovery result |
| Observation mismatch | Six-cell grid pilot retained ambiguity and rejected a non-reproduced gain | It used neither Qwen nor ABC-SMC |
| Silverbox | Real Qwen proposal and checked ABC fits; nonlinear validation gain 5.7448% | Final forecast failed for 3/64 particles; no final RMSE |
| Cascaded Tanks | Source adapter smoke, training prior screen, synthetic controls | No real tank ABC fit or development/final score |
| ABC6 | Many reviewed fake-only components | No campaign launch or complete production replay |

Key evidence: [sequential ABC control](abc-reference-sequential-control-pilot-2026-09-13.md),
[corrected ODE control](ode-reference-control-pilot-2026-09-17-corrected-targetcdf.md),
[abstention control](observation-mismatch-abstention-control-pilot-2026-09-18.md),
and [Silverbox v2](silverbox-v2-development-and-locked-final-2026-09-26.md).
The Silverbox development run took 4.334 seconds. Its single validation block
and inference seeds do not supply independent physical replications.

The checked reference is still separate from the canonical evaluator.
`core/sandbox_eval.py:670` uses legacy `SBIEngine`; line 711 takes an
unweighted parameter median. `core/sbi_engine.py:53,105,285` retains the
proposal clipping and BDSS density concerns found in September. Reference
controls did not repair this canonical path. Defer broad integration for this
week; call the checked reference directly through a narrow reviewed runner.

At audit time, the ABC6 modules and two scripts totalled 21,991 lines before
their tests and other tank modules. The watchdog had 5,073 lines and replay
had 4,687 lines. These counts include the unfinished patch. Their complexity
addresses many local file and process failure cases, but does not remove the
trusted-checkout assumption in the case module. It has grown beyond the
integrity needs of a bounded local methods study.

## Best next study

Ranked options:

1. Cascaded Tanks: compare state retention with loss after a ceiling. Existing
   data, simulators, source reader and checked reference make this the shortest
   path to a complete forecast comparison.
2. Silverbox stability diagnosis: useful development work, but its multisine
   confirmation is consumed. A new confirmation needs a separate exposure audit.
3. NASA four-cell capacity study: useful transfer question, but raw MAT files
   are absent in this checkout and prior cell results have been summarized.

The auditor checked only metadata for the tank archive at the backup path in
the prior-screen manifest: 7,520,592 bytes. Rehash it before reuse. Do not
download a duplicate. The official source provides estimation and test data.
The pinned loader maps `uEst/yEst` to development and `uVal/yVal` to test,
with a 50-sample test initializer. See the [official benchmark](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks),
[pinned loader](https://raw.githubusercontent.com/MaartenSchoukens/nonlinear_benchmarks/f9fb3883086870a27b31917ccde1f78c95d53cb2/nonlinear_benchmarks/benchmarks.py),
and [reporting rules](https://www.nonlinearbenchmark.org/results-reporting).
The safe development adapter must avoid loading the test output.

Question: **Does a surrogate that retains hidden state behind an output
ceiling predict recovery better than one that discards excess state, at the
same fitting cost?** Compare existing S0 (no ceiling), O2 (state cap and loss),
and C2 (output cap with retained state). These are voltage-equivalent
surrogates. They cannot establish the physical cause of the ceiling. Overflow
and sensor effects are already known in this benchmark.

Use estimation indices `[0,768)` for training. Forecast `[768,1024)` from the
carried state and freeze forecasts before opening suffix targets. Lock a
selected or unresolved outcome, then refit on all estimation points under
frozen rules. The official test gets one attempt, with fixed coefficients and
a predeclared state-only initializer using the permitted first 50 outputs.
The two records come from one rig. Seeds measure compute variation, not
physical replication. Report record-level errors and all seeds.

Reuse the four-parameter screen box: `a,c=[0.2,1.5]`, `p=[0.05,1.2]`,
`x1_0=[0,25]`. Fixed `x2_0=5.205` and ceiling 10 V are training-conditioned
screen choices. Disclose that these fixed terms limit uncertainty. No prior
draw in the screen reached 1 V RMSE. This warns of poor fit or low acceptance;
it does not prove that a better fit is impossible.

Proposed fitting limits, to freeze and review before execution:

- Three inference seeds per family; 256 training-only calibration draws.
- Two populations of 64 particles; fixed calibration quantiles such as q25/q10.
- At most 8,192 attempts per population; full-trajectory training RMSE.
- Checked Gaussian proposal and correct weights; support rejection and complete
  failure accounting. No BDSS or clipping in this study.
- Compare persistence, fixed low-order ARX, and bounded coordinate pattern
  search for each family. Match point-fit allowance to ABC's actual simulator
  calls, including failures; report support rejects, steps and wall time.
- Maximum 399,360 fitting simulations across development and refit, 90 minutes
  local fitting, 4 GiB sampled worker memory and 1 GiB artifacts. Separate small
  caps are needed for smoke, initialization and forecasting. Caps are stop
  limits, not runtime predictions.

ABC forecast uncertainty here is finite-tolerance parameter exploration. It
is not a calibrated Gaussian-noise posterior or a model probability. Compare
the weighted-mean forecast with the point-fit forecast under squared loss.
The mean need not be a realizable single-model trajectory. A failed arm stays
in the result table; never drop failing particles and renormalize silently.
Complete, predeclared baseline arms can be scored even if ABC fails.

Freeze an engineering preference margin before opening targets; 10% RMSE
improvement is a proposed rule, not statistical significance. Smaller effects
leave C2/O2 unresolved. A reversed official-test direction rejects transfer.

## Minimum engineering and integrity controls

New engineering stops after **two days, two new runtime files, 600 added
runtime lines, and one focused test file of at most 300 lines**. Reduce the
study if it cannot fit. Reuse the existing adapter, simulator and checked
sampler. The real scorer is currently a disabled stub at
`core/real_data/cascaded_tanks_deferred_scorer.py:125`; it needs a small new
protocol-specific path. Do not route real data through a private synthetic
parser. The generic pattern search is reusable, but its cost helper restricts
lengths to synthetic cases; adapt accounting for the 768-point fit explicitly.

Required controls: one frozen manifest; separate fit and score entry points;
sentinels proving hidden outcomes cannot affect fit/prompt/forecast; retained
weights, ESS, parameters, counters and failures; one exclusive scoring marker
and terminal outcome; independent review of alignment, target access, weights
and metrics. Add no authority hierarchy, anonymous grant channel, new replay
version or native-runtime attestation for this study.

## Local model role and optional new-model check

The user is setting up a new Qwen 3.8 Flash Next Q3 model. Its exact served ID
and runtime are not yet known. This is permission for a narrow future
two-model development check, not a broad ranking project. Preserve historical
frozen protocols and consumed holdouts. Use a new protocol and run ID.

Allow two hours and 12 requests total, including repairs: three fixed tasks,
two attempts per task per model. Test a structured mechanism with a falsifying
prediction, abstention on an uninformative synthetic control, and a bounded
proposal after a supplied training-fit failure. Match prompts, development
inputs, token bounds, grammar, fit seeds and budgets. Measure valid proposals,
correct abstention, unsupported claims, cost, time and downstream fit success.
Use no sealed score for model choice. Setup must not delay the week; the
existing verified model is the fallback.

Qwen's study role is one structured prediction/ranking/abstention request.
Do not execute generated Python. Enumerate all three families numerically.
The present evidence does not show that Qwen improves this small search or
that LLM quality caused the delay. Likewise, ABC has not shown an advantage
over point fitting; this study measures that instead of assuming it.

## Seven-day critical path and deliverables

Days start when the user resumes execution. If resumed October 15, day seven
is October 21.

| Day | Required result |
| --- | --- |
| 1 | Freeze the question, split, models, priors, baselines, limits and failure rules; recheck source identity. Optional capped model acceptance check. |
| 2 | Complete the small runner/scorer and meaningful boundary/accounting tests; independent Luna review. |
| 3 | Run all development arms under the cap; retain every complete or failed forecast. |
| 4 | Score development once; lock selected or unresolved outcome; produce the first real comparison table. |
| 5 | Refit the locked family and baselines; make the one official-test attempt if eligible under the frozen protocol. |
| 6 | Independently recompute metrics and costs; write failures and limits. |
| 7 | Publish compact reviewed code/report via one Luna Git owner; choose the next question from evidence. |

Deliver a table of every arm/seed, error, cost, ESS and failure; forecast and
residual plots; parameter spread/correlation diagnostics; a held-out score or
explicit terminal forecast failure; a compact reproducibility bundle; and a
plain statement of what remains unresolved. A negative result qualifies. A
new scientific mechanism does not follow from a benchmark win.

If ABC fails, finish the locked baseline comparison and report its failure at
the matched budget. If Qwen fails, finish the numerical study. If safe scoring
misses the two-day engineering limit, switch to the Silverbox development
stability audit. Do not reopen its consumed multisine condition. Stop broad
domain expansion, meta-learning and ABC6 gate completion for this week.

The prior permission failures were observed under a restricted environment.
The current context reports unrestricted access, but this audit did not retry
sync or process operations. The user pause remains in effect. This report is
local and does not authorize execution.

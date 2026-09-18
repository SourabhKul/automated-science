# Independent review: observation mismatch and abstention control

Review date: 2026-09-17. I read `AGENTS.md`,
`reports/research-state.md`, `reports/observation-mismatch-abstention-control-plan-2026-09-17.md`,
`reports/ode-reference-control-review-2026-09-17.md`, and the corrected ODE
ABC-control report. This review does not call the model endpoint, run a pilot,
modify source or tests, or perform Git operations.

## Decision

**Approve — the completed pilot is accepted as bounded known-simulator control evidence.**
The two prior blockers are resolved: the discriminating-measurement function
now takes the first qualifying predeclared time without optimizing over the
realized disagreements, and incomplete model receipts include declared
observation/solver configuration plus explicit unavailable predictive fields.
The post-pilot receipt audit found no blocker. This approval covers only the
bounded known-simulator controls, the six predeclared cells, and their
information-flow checks. It approves no real-data inference, model
probability, coverage, mechanism-discovery, novelty, or intervention claim.

### Previously blocking findings resolved

1. `discriminating_measurement_rule` now uses the first qualifying candidate
   in protocol order (`selection_order="first_qualifying_predeclared_time"`).
   The adversarial case with disagreements `0`, `0.101`, and `0.201` at
   `[8.5, 9.0, 9.5]` and `sigma=0.05` recommends `9.0`, not the later maximum,
   while retaining `post_hoc_time_search: False`.

2. `_fit_receipt_or_failure` now records the model's `observation_model`, the
   analytic `solver_config`, `posterior_predictive_status`, an explicit empty
   `posterior_predictive_draws` field, and an unavailable predictive summary,
   alongside status, counters, weights/ESS, termination, and seeds. The
   incomplete branch still gates selection and final summaries.

## Required acceptance checks

1. **Explicit simulator and observation scale.** The receipt and tests must
   state the two generators exactly: Experiment A must use
   `x=exp(-0.65*t)`, `y=x/(0.35+x)+Normal(0, 0.025^2)`; Experiment B must use
   `y=exp(-0.65*t)+Normal(0, 0.05^2)`. The fixed initial condition, time grid,
   train/validation/final role boundaries, low-signal future times, parameter
   bounds, and independent seeds must be recorded. The correct A family must
   fit `k,c` through the saturating observation map; its surrogate must fit
   `A,k` directly in observed space. B must compare one-rate exponential with
   the stated two-rate mixture. Residuals and RMSE must be computed on the
   observed values using each model's declared observation map and stated
   sigma; latent-state fitting or a hidden rescaling is not acceptable.

2. **Development and sealed-data boundary.** Fitting may read train
   observations only; validation may drive the frozen selection rule; sealed
   outcomes may be read exactly once by a separate final-evaluation path after
   selection. Final outcomes must be absent from development manifests,
   prompts, diagnostics, normalization, parameter fitting, and selection.
   A deterministic sentinel test must replace sealed outcomes while holding
   development artifacts fixed and show that selection and all pre-final
   receipts are unchanged. Final times may be declared protocol inputs, but
   their outcomes must not influence development. The final receipt must state
   the role hashes and selection/final status.

3. **Frozen practical rules.** Experiment A must record a numeric practical
   validation RMSE margin fixed before data generation and select the correct
   family only when the observed-space validation improvement reaches that
   margin. Ties, insufficient improvement, or incomplete model fits must
   retain both models and report unresolved. Experiment B must freeze the
   `sigma/sqrt(n_validation)` threshold and an explicit sealed reproduction
   rule before generation; two-rate promotion requires both rules. The sealed
   metric must be evaluated only after the rule and selected development
   object are frozen.

4. **Status, weights, and predictive receipts.** Every model receipt must
   retain status and termination reason, epsilon schedule, attempt and
   simulator/failure counters, per-population normalized weights and ESS,
   solver and observation-model configuration, and proposal/data/simulator
   seeds. Incomplete runs must preserve partial evidence but must not expose
   posterior or predictive summaries as complete, or silently substitute a
   previous population. Prediction summaries must identify latent versus
   observed/noise-convolved quantities. If the reviewed Gaussian ABC-SMC
   reference is used, the finite-epsilon target must remain labeled as such;
   ordinary likelihood or model-family quantities must not be presented as
   model probabilities.

5. **Abstention and discriminating measurement.** B must emit the exact
   unresolved/retain-simpler outcome when the two-rate rules fail, and retain
   both models' forecasts. A failed or incomplete control must also abstain
   rather than promote by fallback. The discriminating condition must be a
   declared future low-signal time or a predeclared finite candidate set with
   a deterministic rule; it must not be post-hoc searched for the most
   favorable disagreement. Recommend a condition only when the posterior
   predictive median disagreement exceeds `2*sigma` in observed units. If no
   declared condition qualifies, the receipt must explicitly say
   `no informative measurement under this protocol`. Tests must cover both
   sides of this threshold and the no-search behavior.

6. **Focused falsification coverage and claim limits.** Focused tests must
   cover exact model equations/maps and scales, partition isolation, margin
   boundaries, incomplete-status gating, normalized weights/ESS, both
   abstention branches, the discriminating threshold, and three independent
   generated seeds per experiment. Nearby times are not independent
   replicates. The report and receipts must limit conclusions to this known
   simulator's observation-model validity/abstention behavior. They must not
   claim real kinetic mechanism identification, calibrated ABC model
   probabilities, coverage, novelty, or a real intervention. No Qwen run or
   broad benchmark sweep is in scope.

## Evidence

The focused command

```text
uv run --no-project --with pytest --with numpy pytest -q tests/test_observation_mismatch_controls.py
```

passed **11 tests**. Python compilation of the three scoped files passed.

The repaired adversarial checks passed:

- The no-search probe returned the first qualifying time (`9.0`) despite a
  larger later disagreement and recorded the fixed selection-order rule.
- Forced incomplete A and B fits (`max_grid_points=10`) retained
  `status="incomplete"`, `complete=False`, no selection/final summary, model
  observation configuration, analytic solver configuration, counters, and
  explicit unavailable predictive fields.

Direct, bounded checks used `predictive_draw_count=64` and did not invoke the
pilot driver:

- Experiment A produced `selected_correct_observation_model` for seeds 11 and
  47 and `unresolved_retain_both` for seed 29. The same fixed margin
  `sigma/sqrt(8) = 0.008838834764831844` was used in all three cells; the
  observed validation improvements were approximately `0.04808`, `-0.00045`,
  and `0.03099`. The implementation contains no seed-specific branch, and all
  three predeclared seeds were checked, so the seed-29 outcome is not evidence
  of post-hoc seed selection.
- Experiment B remained `unresolved_retain_simpler` for seeds 11, 29, and 47.
  Seed 47 cleared the validation margin (`0.01826 >= 0.0176777`) but failed
  sealed reproduction (`-0.00993`), exercising the required nested abstention
  gate. Seeds 11 and 29 failed the validation margin.
- Replacing sealed outcomes by a finite sentinel (`100`) or a large finite
  sentinel (`1e90`) left the A development hash and selection unchanged for
  seeds 11, 29, and 47. For B, the development hash and
  `development_receipt.selection` stayed unchanged for all three seeds. Its
  top-level `selection` and final decision changed because B intentionally
  uses sealed outcomes for the second, frozen reproduction gate; this is
  expected only when those fields are clearly treated as the post-sealed final
  promotion decision rather than development selection.
- The changed sealed values altered final scores for evaluated A cells and B
  cells, while unresolved A seed 29 retained `not_run_selection_unresolved`.
  Development receipts contain no sealed outcome hash or sentinel value.
- The baseline is explicitly labeled
`deterministic_likelihood_weighted_grid_reference`, uses observed-scale
Gaussian likelihood weights, sets `posterior_summary: false`, and warns that
it is not a calibrated posterior or model probability. The complete receipts
persist normalized grid weights, ESS, counters, seeds, map, observation
model, analytic solver configuration, and noise-convolved predictive draws.

## Post-pilot artifact and decision audit — 2026-09-18

I independently audited the ignored raw artifacts under
`artifacts/observation_mismatch_control/` and the pilot report and ledger.
All 9 manifest entries matched their recorded byte counts and SHA-256 values;
the partial and aggregate closures contained the same 6/6 completed cells and
the same run ID. The top-level, development, and final receipt hashes
recomputed exactly for all six cells (18/18 hashes).

Representative observed-scale grid recomputations also matched the receipts
exactly. The A/data11 saturating grid had 14,016 points and ESS
`33.281240170350664`; its direct observed-exponential surrogate had 20,586
points and ESS `15.555529957134379`. The A/data29 surrogate ESS was
`16.206876673333348`, and the B/data47 two-rate grid had 37,170 points and
ESS `1280.7031705900313`. In each case the independently reconstructed grid
hash, log weights, normalized weights, and ESS matched the stored values.

The frozen observed-scale margins and decisions recomputed from regenerated
data and stored predictions. A used `0.008838834764831844` and produced
improvements `0.0480834991770983` (seed 11), `-0.0004534722158518814`
(seed 29), and `0.030993334426457703` (seed 47), yielding the recorded
correct-map, unresolved, correct-map outcomes. The A/data29 final evaluation
is `not_run_selection_unresolved` and contains no sealed hash, prediction, or
RMSE.

B used `0.017677669529663688`. Its validation/sealed improvements were,
respectively, `(-0.0012643579066965727, -0.0018020505252094032)`,
`(0.0011174480954099716, 0.000003876281225737022)`, and
`(0.01826007960681251, -0.009925245634271496)`. Thus seed 47 clears the
validation margin but fails the sealed reproduction gate, while seeds 11 and
29 fail validation; all three correctly abstain. Every B development receipt
keeps `sealed_advantage_reproduced: null`. Sealed outcomes occur only in the
single final evaluation path, where the explicit reproduction term is
combined with the validation-margin term; no sealed value enters development
selection. Sentinel reruns replacing the sealed suffix with `100.0` preserved
each development hash and development selection for all A and B seeds. For A
seed 29 the unresolved final status remained unchanged; for B only the
post-validation sealed gate was allowed to respond to the replacement.

The fixed future recommendation was recomputed from the stored predictive
draws at exactly `[8.5, 9.0, 9.5, 10.0]`. The maximum median disagreements
were `0.007904507467127212`, `0.0023062985104547254`, and
`0.026374734029149367`, all below the fixed `2*sigma = 0.1` threshold.
Each cell therefore recorded `no_informative_measurement_under_this_protocol`,
`time: null`, `selection_order: first_qualifying_predeclared_time`, and
`post_hoc_time_search: false`. No post-hoc candidate or seed selection was
used.

The pilot report and ledger retain the same bounded claim language: these
receipts demonstrate only the declared observation-model and abstention
controls for this known generator. They do not support claims about real
kinetic mechanisms, calibrated ABC/model probabilities, coverage, novelty,
or intervention.

## Approval scope and pilot constraints

The completed run was exactly one serial six-cell execution over the
predeclared experiment/seed cells (A and B with data seeds 11, 29, and 47),
using the recorded default grid, split, sigma, practical factor, forecast
times, and predictive-draw settings. Every complete and unresolved receipt
is retained, including the A29 abstention and B47 nested abstention. The
pilot remains labeled as a deterministic numerical reference control; no
Qwen call, broad benchmark sweep, real-data claim, or model-probability
interpretation is in scope.

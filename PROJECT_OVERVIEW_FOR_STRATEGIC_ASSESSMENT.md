# Automated Science / E3: Strategic Project Overview

## Purpose Of This Document

This document is intended as a handoff to another agent or reviewer whose job is
to assess the novelty, strategic importance, and best packaging path for this
work. The key question is whether the project is best understood as:

- a research paper,
- an open-source scientific discovery framework,
- the seed of a startup,
- or a deliberate mix of all three.

The short answer is that the project has a paper draft, a growing framework
implementation, and a possible application path that still depends on strict
real-domain validation.

## Executive Summary

`automated_science` is an autonomous white-box scientific model discovery
system. The current paper-facing name appears to be `$E^3`.

The core premise is narrow and testable: hold observed scientific data fixed,
then automate a loop that proposes interpretable mechanistic model structures,
fits their parameters, diagnoses their failures, and replaces them with better
white-box models.

The canonical loop is:

```text
data -> model -> parameter fit -> diagnostics -> structural hypothesis -> validation -> model replacement -> memory -> meta-learning
```

This is not merely "LLM writes equations." It is closer to an agentic scientific
model-replacement system:

1. A domain starts with an explicit seed model, usually an ODE written in JAX.
2. Parameters are inferred with ABC-SMC style simulation-based inference.
3. The system evaluates fit, posterior behavior, held-out error, and residuals.
4. An LLM proposes a structurally modified white-box model.
5. Generated code is validated, evaluated, repaired if possible, and
   fingerprinted for duplicate suppression.
6. The candidate replaces the incumbent only if it improves the selected
   objective.
7. Accepted and rejected hypotheses are preserved as scientific provenance.

The system currently focuses mainly on continuous-time dynamical systems,
implemented as JAX/Diffrax ODEs. Architecturally, however, the intended invariant
is broader:

```text
candidate model + parameter space + simulator/inference adapter + observable prediction
```

That means the future model class could include ODEs, SDEs, stochastic jump
processes, state-space models, probabilistic graphical models, agent-based
models, MDPs, and hybrid symbolic/neural models, as long as the output remains
inspectable and scorable.

The repository contains source code, tests, a paper draft, result artifacts, real
dataset adapters, and strategic application research. It is not just a sketch.
The main preserved proof-of-life is a Qwen 3.6 27B "hero run" across 25
scientific domains and 10 serial circuits, with 25,351 logged structural
evaluation rows and 1,351 accepted model replacements.

However, the most important caveat is that the hero run mostly optimized
in-sample ABC distance, not strict held-out generalization. The repo now contains
newer held-out evaluation machinery, baseline scripts, diagnostic packets, and
real-data pilots, but the historical headline run should be treated as
proof-of-life rather than final scientific validation.

## Core Thesis

Modern scientific modeling has a bottleneck between data and explanatory
mechanism. Scientists often have noisy time series, cohorts, trajectories,
sensor streams, or assay outputs, but constructing the right mechanistic model
is slow and expert-intensive. Classical symbolic regression searches equation
spaces, but can produce physically meaningless or brittle expressions. Deep
learning can fit trajectories well but often removes interpretability.

This project proposes a third posture:

Use LLMs for semantic, mechanism-aware structural proposals, but use
simulation-based inference and explicit validation as the arbiter.

The LLM is not trusted as a scientist by itself. It is treated as a generator of
candidate white-box hypotheses. The inference/evaluation loop is responsible for
testing those hypotheses against data.

This makes the system strategically interesting because it combines:

- LLM prior knowledge over scientific mechanisms,
- executable white-box model structures,
- Bayesian or approximate Bayesian parameter fitting,
- automatic rejection of bad proposals,
- provenance over accepted and rejected hypotheses,
- and a path toward self-improving scientific search.

## What Exists In The Repository

The project is organized around a canonical implementation, a small
compatibility surface, and locally preserved artifacts.

Primary source-of-truth documents:

- `README.md`
- `ARCHITECTURE.md`
- `CLAIMS.md`
- `CODE_AUDIT.md`
- `docs/RESULTS.md`
- `docs/REPRODUCIBILITY.md`

Canonical implementation files:

- `core/domain_configs.py`
- `scripts/run_domain.py`
- `core/generated_code.py`
- `core/sandbox_eval.py`
- `core/sbi_engine.py`
- `core/evaluation.py`
- `core/run_config.py`
- `scripts/run_gauntlet.py`
- `scripts/run_model_matrix.py`
- `scripts/analyze_model_matrix.py`
- `scripts/legacy/run_baselines.py`

Paper draft:

- `paper/main.tex`
- `paper/sections/`
- `paper/applications/`
- `paper/results_table.tex`
- `paper/main.bib`

Evidence and strategy artifacts:

- `artifacts/evaluations/hero_run_deep_dive_20260703.md`
- `artifacts/evaluations/qwen3_coder_next_5epoch_vs_baselines_20260702.md`
- `artifacts/evaluations/model_matrix_learnings*.md/json`
- `artifacts/research/high_leverage_applications_landscape.md`
- `artifacts/research/application_leverage_rubric.md`
- `artifacts/research/opportunity_cards/`
- `artifacts/research/pilot_integration_specs/`

Real-data adapters and data slots:

- `scripts/legacy/real_warfarin_pkpd_data_loader.py`
- `scripts/legacy/real_battery_nasa_data_loader.py`
- `core/real_data/warfarin_pkpd.py`
- `core/real_data/battery_nasa.py`
- `data/real/pkpd/`
- `data/real/battery_nasa/`

Tests:

- `tests/test_generated_code.py`
- `tests/test_sbi_engine.py`
- `tests/test_sandbox_eval.py`
- `tests/test_run_domain_config.py`
- `tests/test_baselines.py`
- `tests/test_model_matrix_status.py`
- `tests/test_analyze_model_matrix.py`
- `tests/test_domain_quarantine.py`
- `tests/test_real_warfarin_pkpd.py`
- `tests/test_real_battery_nasa.py`
- and several other focused tests.

An inventory script reports:

- active domain configs: 27,
- generated model directories: 408,
- preserved Qwen history files: 250,
- preserved Qwen structural-evaluation rows: 25,351,
- accepted replacements in the preserved Qwen trace: 1,351.

## The Canonical Architecture

The architecture has several nested loops.

### 1. Parameter Inference Loop

This loop asks:

```text
Given fixed model structure M and fixed data D, what parameter regions make M explain D well?
```

The active implementation is in `core/sbi_engine.py`.

The main method is ABC-SMC. The engine samples particles from parameter priors,
simulates the candidate model, scores distances against observed data or summary
statistics, keeps the best particles, and iteratively resamples/perturbs
accepted particles.

Current supported ABC-SMC transition strategies include:

- `gaussian_weighted`
- `gaussian`
- `default`
- `bdss`

`bdss` is implemented as a beta-distributed step-size strategy. It appears to be
available in code, but it still needs careful comparative validation before it
should be treated as a demonstrated scientific improvement.

The parameter-inference loop currently reports:

- accepted parameter samples,
- median distance,
- min distance,
- requested/effective strategy,
- posterior summaries via diagnostic packets,
- parameter bound pressure,
- and warning/failure modes in newer evaluator paths.

### 2. Model Replacement Loop

This loop asks:

```text
Given the incumbent model's fit and failure modes, what model structure should replace it?
```

The active implementation is `scripts/run_domain.py`.

For each domain:

1. Load the domain configuration.
2. Evaluate the seed model.
3. Build a prompt containing current dynamics, score, domain hints, and
   diagnostic context.
4. Call an LM Studio/OpenAI-compatible chat endpoint.
5. Extract a Python code block.
6. Validate generated code.
7. Evaluate the candidate in a subprocess.
8. Optionally attempt repair if validation or evaluation fails.
9. Accept the candidate only if it improves the selection score.
10. Persist the best logic, diagnostics, proposal fingerprints, run logs, and
    held-out metrics.

The LLM is asked to output only a Python block with:

```python
def dynamics(t, y, args):
    ...

metadata = [
    {"name": "param", "range": (low, high)},
]
```

The candidate is then scaffolded into a model class and simulated through
Diffrax.

### 3. Generated-Code Validation And Repair

Generated model code is handled by `core/generated_code.py`.

The validator rejects:

- imports,
- forbidden names such as `open`, `eval`, `exec`, `compile`, `globals`,
  `locals`,
- forbidden modules such as `os`, `sys`, `subprocess`, `socket`, `requests`,
  `pathlib`, and `shutil`,
- top-level code other than `dynamics` and `metadata`,
- wrong function signatures,
- invalid metadata shapes,
- metadata with invalid numeric ranges,
- and string-keyed `args` access.

It also normalizes harmless cases, strips safe imports/docstrings where
possible, and computes stable proposal fingerprints to suppress repeated
candidates.

This is an important safety and reliability layer, but it is not a complete OS
security sandbox. The project is explicit about that.

### 4. Sandbox Evaluation Loop

Candidate evaluation is implemented in `core/sandbox_eval.py`.

Despite the name, this is currently a subprocess robustness boundary rather than
a true security sandbox. It:

- reads candidate code,
- validates the AST,
- scaffolds the candidate model,
- loads domain data,
- performs a finite simulation sanity check,
- runs ABC-SMC fitting,
- computes held-out metrics when enabled,
- emits diagnostics,
- and returns a structured JSON result.

Newer paths support:

- train/test trajectory metrics,
- residual summaries,
- posterior summaries,
- parameter bound-pressure warnings,
- numerical failure modes,
- and subject-holdout smoke evaluation for Warfarin.

### 5. Evaluation And Limitation Loop

The system is moving from "fit score only" toward richer scientific evaluation.

Evaluation now includes or is designed to include:

- train fit,
- held-out prediction,
- posterior concentration,
- identifiability hints,
- numerical stability,
- parameter plausibility,
- model complexity/parsimony,
- residual structure,
- robustness to seeds/splits,
- comparison to classical baselines,
- and scientific interpretability.

The repository explicitly warns that a model can fit in-sample trajectories well
while still being scientifically weak, overfit, non-identifiable, numerically
brittle, or purely descriptive.

### 6. Meta/Self-Improvement Loop

This is the most ambitious but least mature layer.

The desired meta loop asks:

```text
Across many model-search attempts, how should the system improve its search,
prompts, priors, inference kernels, diagnostics, and domain adapters?
```

The preserved Qwen scheduler invoked a meta-agent, but completed-run
`meta_hints.txt` files are empty. So the meta loop is better viewed as an
architectural direction than a fully realized current contribution.

Future meta-loop components would include:

- research ledgers of accepted and rejected hypotheses,
- structured model critiques,
- domain memory,
- strategy memory,
- prompt ablations,
- inference-strategy ablations,
- automated limitation reports,
- model-family comparisons,
- and automated mechanism-quality scoring.

## Domain Coverage

The active domain registry currently includes 27 domains.

The original benchmark-like set includes scientific domains such as:

- ecology,
- oncology,
- economics,
- battery degradation,
- PK/PD,
- synthetic biology,
- climate,
- cardiovascular dynamics,
- Belousov-Zhabotinsky chemistry,
- epidemiology,
- neuroscience,
- fluid dynamics,
- astronomy,
- chemical kinetics,
- immune dynamics,
- population genetics,
- thermodynamics,
- materials,
- agriculture,
- social diffusion,
- real sunspots,
- real Nile flow,
- real macroeconomic time series,
- real theophylline,
- real CO2.

The newer real-data pilots add:

- `real_warfarin_pkpd`,
- `real_battery_nasa_capacity`.

Two domains are currently quarantined from default model-matrix runs:

- `bz_chem`,
- `fluid`.

They are quarantined because seed evaluation currently returns infinite ABC-SMC
metrics under the default Gaussian weighted sandbox path. The project treats
these as domain/evaluator robustness issues rather than deleting them.

## Evidence From The Hero Run

The main proof-of-life artifact is:

```text
artifacts/evaluations/hero_run_deep_dive_20260703.md
```

Summary:

- start: 2026-05-06 17:58:27,
- end: 2026-06-17 01:29:27,
- duration: about 41 days,
- circuits: 10,
- domains: 25,
- domain runs: 250,
- structural-evaluation rows: 25,351,
- accepted mutations: 1,351,
- rejected mutations: 23,433,
- errors: 316,
- meta-hint files: 225,
- nonempty meta-hint files: 0.

The main read from that artifact:

- The core two-loop system works as an engineering mechanism: repeated LLM
  structural replacement plus ABC-SMC fitting can produce accepted white-box
  models across many domains.
- The largest recorded changes occurred where the LLM could name a missing mechanism:
  radiation in thermodynamics, lagged absorption in PK, harmonic forcing in
  sunspots, nonlinear cracking in batteries, Holling responses in ecology.
- In-sample ABC distance was enough to discover plausible structures, but not
  enough to certify scientific validity or forecasting power.
- Accepted mutation count is not itself a quality metric.
- Zero-accept or tiny-improvement domains are informative because they reveal
  where the metric, prompt, seed, model class, or evaluation protocol is weak.

Strong domains in the preserved run included:

- battery: about 89% mean improvement,
- immune: about 85% mean improvement,
- thermo: about 99.8% mean improvement,
- social diffusion: about 63% mean improvement,
- real sunspots: about 70% mean improvement,
- real Nile: about 98.5% mean improvement,
- real macro: about 91.8% mean improvement,
- real theophylline: about 99.2% mean improvement,
- real CO2: about 76.5% mean improvement.

Weak or near-stalled domains included:

- oncology,
- economics,
- PK/PD,
- cardio,
- epidemiology,
- neuro,
- astro,
- population genetics,
- agriculture.

Important caveat:

These improvement percentages come from the preserved historical protocol, which
primarily optimized in-sample ABC distance. They are useful for proof-of-life,
but they are not final scientific validation.

## Examples Of Discovered Or Improved Mechanisms

The hero run surfaced several plausible white-box structures.

### Ecology

The system discovered or promoted predator-prey dynamics with:

- logistic prey growth,
- Holling Type III predation,
- predator growth from consumption,
- natural predator mortality.

This is scientifically recognizable and more mechanistic than arbitrary symbolic
curve fitting.

### Battery Degradation

The system found structures involving:

- SEI growth,
- nonlinear volume-expansion stress,
- crack activation above thresholds,
- active material loss,
- capacity fade amplification.

This is strategically important because battery degradation has expensive
experiments, public run-to-failure data, and high demand for interpretable
remaining-useful-life models.

### Thermodynamics

The system promoted a simple and scientifically interpretable model combining:

- Newtonian cooling,
- Stefan-Boltzmann radiation.

This is probably an "easy domain," but it is useful as a calibration/control
task because the missing mechanism is well known.

### Real Theophylline

The system discovered a one-compartment absorption/elimination model with:

- absorption rate,
- elimination rate,
- lagged source term,
- apparent volume/bioavailability style parameters.

This aligns with the PK/PD startup/paper wedge.

### Real Sunspots

The system found a damped oscillator with harmonic forcing and modulation. This
is descriptively plausible, though sunspots are noisy and nonstationary, so it
needs rolling-origin validation before serious claims.

### Real Nile

The system found a first-order relaxation model toward a sigmoid regime-shift
mean. This fits historical changepoint behavior, but could overfit a particular
regime transition.

### Real CO2

The system found accelerating exponential growth with quadratic time curvature.
This is a strong descriptive fit but not a mechanistic carbon-cycle model.

## Baselines And Held-Out Work

The project includes the compatibility baseline runner
`scripts/legacy/run_baselines.py`, which compares against:

- PySINDy,
- PySR.

The baseline runner has been improved to integrate discovered right-hand-side
dynamics forward and report trajectory-level train/test errors. PySR derivative
fit is retained only as a diagnostic.

An artifact from 2026-07-02 compares a 5-epoch Qwen Coder Next held-out gauntlet
against refreshed baselines:

```text
artifacts/evaluations/qwen3_coder_next_5epoch_vs_baselines_20260702.md
```

That comparison found:

- Qwen better: 11 domains,
- baseline better: 11 domains,
- baseline nonfinite: 1 domain,
- no Qwen metric: 2 domains,
- Qwen accepted updates in 7 domains.

This is useful because it shows the held-out gate is doing real work. Some
proposals improved train fit but were rejected because held-out MSE got worse.

However, the comparison is still preliminary. The historical E3 artifacts were
not generated under the exact same held-out protocol and compute-normalized
budget as the baselines. The repo is explicit that final paper claims require
rerunning selected E3 domains with matched train/test splits, fixed seeds, saved
prompts, and comparable compute budgets.

## Real-Data Pilot Work

The repo is already moving from synthetic/mixed benchmark domains toward
application pilots.

### Warfarin PK/PD

The first recommended real-domain pilot is Warfarin PK/PD.

Relevant files:

- `scripts/legacy/real_warfarin_pkpd_data_loader.py`
- `core/real_data/warfarin_pkpd.py`
- `data/real/pkpd/README.md`
- `artifacts/research/pilot_integration_specs/pk_pd_warfarin_theophylline.md`
- `artifacts/evaluations/real_warfarin_subject_holdout_seed_smoke_20260707.md`

The normalized Warfarin adapter supports:

- sparse subject-level observations,
- concentration and response endpoints,
- endpoint masks,
- subject holdout metadata,
- dense solver-compatible bridge arrays,
- provenance files,
- and a subject-holdout seed smoke path.

The seed smoke result reports:

- status: success,
- effective strategy: `subject_holdout_prior_rejection`,
- train subjects: 26,
- test subjects: 6,
- train observed count: 384,
- test observed count: 95,
- train RMSE: about 12.99,
- test RMSE: about 15.37.

Caveat:

This validates grouped data plumbing and scoring. It is not yet full grouped
ABC-SMC, and it does not evaluate an LLM-discovered Warfarin model.

Strategic importance:

PK/PD is a strong pilot because the domain already uses interpretable
compartmental ODEs, sparse observations, nonlinear mixed effects, and
scientifically meaningful parameter uncertainty. It also has high decision value
but requires careful medical framing: research models only, not dosing advice.

### NASA Battery Aging

The second recommended real-domain pilot is NASA Li-ion battery aging.

Relevant files:

- `scripts/legacy/real_battery_nasa_data_loader.py`
- `core/real_data/battery_nasa.py`
- `data/real/battery_nasa/README.md`
- `artifacts/research/pilot_integration_specs/battery_nasa_aging.md`

The adapter normalizes NASA PCoE battery aging data into cycle-level capacity
fade trajectories.

The desired validation protocol:

- train on early cycles,
- hold out late-life cycles,
- hold out whole cells when multiple cells are available,
- report capacity RMSE,
- report end-of-life cycle error,
- report monotonicity violations,
- report posterior spread and bound pressure,
- and inspect whether discovered terms correspond to plausible degradation
  channels.

Strategic importance:

Battery degradation is a strong second pilot because experiments are expensive,
run-to-failure public data exists, degradation mechanisms matter, and
interpretable equations could be valuable for design, prognostics, and
operations.

## High-Leverage Application Landscape

The project already contains strategy research ranking application domains by
fit to the E3 loop.

Relevant files:

- `artifacts/research/application_leverage_rubric.md`
- `artifacts/research/high_leverage_applications_landscape.md`
- `artifacts/research/opportunity_cards/`

The top-ranked near-term applications are:

1. PK/PD and dose-response model discovery.
2. Battery degradation, electrochemistry, and remaining-useful-life dynamics.
3. Infectious disease dynamics with wastewater observation models.
4. Industrial process control and chemical digital twins.
5. Synthetic biology circuits and gene-regulatory dynamics.
6. Climate subsystem and Earth-system reduced-order modeling.
7. Oncology tumor growth, treatment response, and resistance dynamics.
8. Materials kinetics and reaction-network discovery.
9. Agriculture, soil, crop, and water-use dynamics.
10. Neuroscience population dynamics and neural mass models.

The scoring rubric values:

- mechanistic fit,
- data availability,
- parameter-inference pain,
- white-box value,
- experiment cost,
- decision impact,
- validation feasibility,
- baseline maturity,
- automation fit,
- and risk manageability.

The strategic conclusion is that "all of science" is too broad as a near-term
claim. The best wedge is a domain where:

- mechanistic equations are already the language of the field,
- experiments are expensive or sparse,
- parameter inference is painful,
- public data supports held-out validation,
- and interpretability has real value.

PK/PD and battery degradation fit this especially well.

## What The Project Can Currently Claim

The repo's own claims audit is stricter than the manuscript. Supported or
partially supported claims include:

### Strongly Supported

- LLM-proposed JAX dynamics are evaluated by ABC-SMC.
- The repo contains preserved 25-domain, 10-run Qwen result artifacts.
- Results tables can be generated from run-history CSV files.
- The central runner validates generated code before execution.
- Newer centralized runs can use held-out trajectory selection.
- The repo has reproducibility improvements such as run configs and seed
  support for future runs.

### Partially Supported

- The system can discover useful structures across domains, but this is much
  stronger as a proof-of-life claim than as a final scientific-validity claim.
- Baseline comparisons exist, but final apples-to-apples comparisons require
  matched reruns.
- Generated code is partially sandboxed by AST checks and subprocess execution,
  but not by a true OS/container/Wasm security boundary.
- Real-data pilots exist, but Warfarin and NASA battery pilots still need
  stronger grouped/cell-level validation before LLM-discovered results should be
  promoted.

### Not Supported As Demonstrated Claims

- Full SDE discovery/inference.
- PPP/MMD stochastic scoring.
- GRPO-trained model mutators.
- Fully independent historical runs.
- Fully standardized historical random seeds.
- Full distributed secure execution.
- Fully apples-to-apples superiority over PySR/PySINDy.

These unsupported items may be good future work, but they should not be central
claims in a paper or pitch until implemented and validated.

## Novelty Assessment

The assessing agent should compare this work against at least these categories:

- symbolic regression,
- PySR,
- SINDy,
- AI-Feynman style equation discovery,
- genetic programming over equations,
- neural ODEs and physics-informed neural networks,
- simulation-based inference frameworks,
- agentic coding systems,
- automated machine learning,
- and LLM-based scientific hypothesis generation.

The likely novelty is not any single component. LLMs, ABC-SMC, ODE simulation,
and symbolic regression all exist separately. The novelty is the closed-loop
composition:

```text
LLM structural proposal
+ executable white-box model
+ simulation-based parameter inference
+ diagnostics
+ accept/reject replacement
+ provenance
+ multi-domain repetition
```

This is most interesting if framed as a system for iterative mechanistic model
replacement, not as a generic equation generator.

Potential novelty claims:

1. The project treats the LLM as a mechanism-aware structural mutator rather
   than a final answer generator.
2. Candidate equations are tested by simulation and parameter inference before
   being accepted.
3. The replacement unit is an executable model, not a symbolic expression scored
   only by algebraic residuals.
4. The same loop is applied across many scientific domains with explicit
   provenance.
5. The architecture separates model-structure search from parameter-inference
   strategy search.
6. The project is moving toward diagnostic feedback where posterior health,
   residual structure, and held-out failure modes inform the next hypothesis.
7. The real strategic value is in domains where experts care about mechanisms,
   not just predictive scores.

Potential novelty weaknesses:

1. Without strict held-out validation, the system can look like an expensive
   overfitting loop.
2. If the generated equations are mostly known mechanisms from prompts/hints,
   reviewers may view it as LLM-assisted model selection rather than discovery.
3. ABC-SMC is compute-heavy, which may make the method less attractive unless
   the white-box value is high.
4. Many domains are synthetic or low-dimensional, so broad scientific discovery
   claims should be avoided.
5. The meta-learning loop is currently more vision than proven system.
6. The current sandbox is not strong enough for untrusted distributed execution.

## Strategic Importance

The strategic value comes from applying agentic AI to mechanistic modeling rather
than black-box prediction.

If the project works, it could help scientists and engineers:

- generate candidate mechanisms from sparse data,
- compare structural hypotheses,
- expose parameter uncertainty,
- discover missing terms in known models,
- build interpretable reduced-order models,
- and accelerate model iteration in domains where experiments are costly.

This matters because many high-value domains do not merely need forecasts. They
need explanations, equations, mechanisms, and parameters that experts can
inspect.

High-leverage domains include:

- pharmacokinetics and dose-response,
- battery degradation,
- infectious disease and wastewater surveillance,
- industrial process control,
- synthetic biology,
- oncology treatment response,
- materials kinetics,
- climate reduced-order modeling.

The strategic danger is overclaiming. If framed as "automated science solves
scientific discovery," the project will be easy to attack. If framed as "a
validated loop for proposing and testing interpretable mechanistic model
replacements," it becomes much more credible and strategically sharp.

## Paper Path

A paper draft is present, but it should be treated as a systems-paper starting
point rather than a finished empirical record.

Evidence:

- There is a LaTeX paper draft in `paper/`.
- There is a result table pipeline.
- There are 25-domain historical artifacts.
- There are baseline scripts.
- There is an architecture document.
- There is a claims audit.
- There are explicit limitations and failure-mode discussions.

A defensible paper would not claim definitive scientific discovery
across all domains. It should instead present:

1. A system architecture for LLM-guided white-box model replacement.
2. A parameter-inference-backed accept/reject loop.
3. A multi-domain proof-of-life.
4. A strict discussion of failure modes.
5. A smaller held-out benchmark with matched baselines.
6. One or two real-data pilots as deeper case studies.

Possible paper titles:

- "E3: LLM-Guided White-Box Model Replacement with Simulation-Based Inference"
- "Agentic Mechanistic Model Discovery via LLM Structural Proposals and ABC-SMC"
- "Closing the Loop Between Language Models and Simulation-Based Inference for Scientific Model Discovery"

Best paper contribution framing:

```text
We introduce and evaluate a closed-loop system that uses LLMs to propose
interpretable scientific model structures and simulation-based inference to fit,
diagnose, and select among them.
```

Paper risks:

- Reviewers will demand stronger held-out validation.
- Reviewers will compare against PySR/SINDy.
- Reviewers may object that some "discoveries" are known mechanisms suggested
  in hints.
- Reviewers will notice unsupported claims if the manuscript strays beyond the
  claims audit.
- Security/sandbox language must be precise.

Paper recommendation:

Do a conservative systems paper first. Make the 25-domain hero run a provenance
and exploration artifact, then add a stricter subset benchmark for final
empirical claims.

## Open-Source Repository Path

This project is also strongly open-source-shaped.

Why:

- It has a reusable domain registry.
- It has a canonical runner.
- It has baseline scripts.
- It has generated-code validation.
- It has result artifacts and audits.
- It has tests.
- It has a clear architecture.
- It can become a benchmark harness for LLM-based scientific model discovery.

Open-source positioning:

```text
An open framework for LLM-guided mechanistic model discovery, where candidate
white-box models are fitted, diagnosed, and selected by simulation-based
inference.
```

Potential open-source users:

- scientific ML researchers,
- symbolic regression researchers,
- computational biology modelers,
- battery/prognostics researchers,
- PK/PD modelers,
- agentic AI researchers,
- benchmark builders.

Open-source strengths:

- Transparency is valuable because scientific-discovery claims need provenance.
- The project can attract contributions for new domains, baselines, inference
  strategies, and sandboxing.
- The framework itself could become a benchmark for LLM model-discovery systems.

Open-source weaknesses:

- Historical generated logs and compatibility code still need clear separation
  from the maintained path.
- Some large artifacts are ignored or local-only.
- Setup may not yet be containerized enough for easy reproduction.
- The current execution model is not safe for arbitrary untrusted generated
  code.

Open-source recommendation:

Make a cleaned public GitHub repo after narrowing the canonical path:

- Keep `core/`, `scripts/`, selected `data/` fixtures, tests, docs, and a small
  artifact bundle.
- Keep historical orchestrators in the local archive and document the
  compatibility surface.
- Provide one easy smoke command.
- Provide one held-out demo domain.
- Provide a clear warning that generated code execution is research-only and not
  a secure sandbox.

## Startup Path

The startup path is plausible but should be domain-specific, not "automated all
of science."

The near-term commercial wedges identified here are:

1. PK/PD model discovery and dose-response research tooling.
2. Battery degradation and remaining-useful-life model discovery.
3. Industrial process control / digital twin model refinement.
4. Wastewater infectious disease dynamics and observation-model discovery.

The commercial product should probably not start as a generic model-discovery
platform. It should start as a high-value applied modeling workflow:

```text
Upload trajectories or connect a dataset -> choose mechanistic model family ->
run structural search -> get candidate equations, parameter uncertainty,
residual diagnostics, held-out validation, and expert-review reports.
```

For PK/PD, the product could be:

- research-only model structure exploration for concentration/response data,
- comparison against standard compartment models,
- parameter uncertainty reports,
- covariate/subject holdout diagnostics,
- and exportable model equations.

For battery, the product could be:

- interpretable degradation model discovery,
- early-life to late-life holdout validation,
- cell-level generalization reports,
- RUL and EOL error,
- monotonicity and physical plausibility checks,
- and mechanism-labeled capacity fade terms.

Startup risks:

- Medical use cases require careful regulatory and safety framing.
- Battery models need robust validation across protocols/cells.
- Customers may prefer black-box predictive performance unless white-box
  interpretability is tied to a concrete decision.
- Compute cost from ABC-SMC may be high.
- Expert trust requires excellent reports and uncertainty communication.

Startup recommendation:

Do not pitch "automated scientist" first. Pitch "interpretable mechanistic model
discovery and validation for expensive time-series experiments" in one domain.

Best first startup wedge:

PK/PD if expert partnerships and medical research framing are available.

Best second wedge:

Battery degradation if the goal is lower regulatory friction and clearer
engineering validation.

## Recommended Mixed Path

The best strategy is probably a mix:

1. Clean open-source core.
2. Conservative systems paper.
3. One strict real-data pilot.
4. Then decide which vertical is best supported: PK/PD, battery, or industrial
   process modeling.

Suggested sequence:

### Phase 1: Trustworthy Research Core

- Keep improving held-out evaluation.
- Finish grouped ABC-SMC for Warfarin subject holdout.
- Finish cell-level holdout for NASA battery.
- Add normalized scoring and mechanism-quality reports.
- Keep generated-code validation and duplicate suppression.
- Add stronger environment reproducibility.

### Phase 2: Paper-Grade Benchmark

- Select 8 to 12 stable domains.
- Run E3 with fixed seeds, held-out selection, saved prompts, and compute
  budgets.
- Compare against PySR and PySINDy under matched trajectory-level metrics.
- Include failure cases rather than hiding them.
- Present the hero run as historical scale/provenance, not final benchmark.

### Phase 3: Real-Domain Pilot

- Choose Warfarin PK/PD or NASA battery.
- Run seed, baseline, and LLM-discovered candidates.
- Require held-out improvement and parameter plausibility.
- Produce an expert-readable model card.
- Avoid deployment claims.

### Phase 4: Public Release

- Publish cleaned repo.
- Publish paper preprint.
- Publish artifact bundle.
- Publish demo notebooks or one-command scripts.
- Invite domain contributors.

### Phase 5: Startup Exploration

- Talk to PK/PD modelers, battery RUL engineers, and process-control teams.
- Identify whether the pain is model structure, parameter inference, reporting,
  or integration.
- Package the system around the best-supported workflow, not around the general AI
  story.

## Current Weaknesses And Risks

### Scientific Validation Risk

The biggest risk is that models fit in-sample but do not generalize. The project
already recognizes this and has moved toward held-out selection. Final claims
should be based on the newer held-out machinery, not only the hero run.

### Overclaiming Risk

The paper draft and project vision mention future extensions such as SDEs,
stochastic processes, MMD/PPP scoring, distributed execution, and RL-trained
mutators. These should stay in future work unless code and evidence exist.

### Security Risk

Generated Python/JAX code is executed after AST validation. The centralized path
uses a subprocess, but this is not a true secure sandbox. A public or distributed
system needs stronger isolation, resource limits, and possibly Wasm/gRPC-style
pure math execution.

### Compute Risk

ABC-SMC with large particle counts is expensive. The system needs careful
compute-normalized comparisons and may need better inference strategies for
commercial practicality.

### Domain-Hint Risk

If domain hints already contain the missing scientific mechanism, reviewers may
argue that the LLM is retrieving obvious suggestions rather than discovering
them. The correct framing may be "automated model refinement and hypothesis
testing" rather than pure discovery.

### Reproducibility Risk

New runs can persist seeds and configs, but historical hero-run artifacts do not
fully standardize seeds. A paper should not call the historical 10 runs
independent unless rerun that way.

### Legacy-Code Risk

Historical orchestrators and generated traces are preserved in the local cleanup
backup. The public tree keeps the canonical path and the small compatibility
surface under `scripts/legacy/`.

## Open Questions For The Assessing Agent

The next agent should directly evaluate these questions:

1. What is the most defensible precise novelty claim that survives comparison to PySR,
   SINDy, symbolic regression, neural ODEs, and LLM code-generation baselines?
2. Should the paper emphasize the 25-domain breadth, or a smaller strict
   held-out benchmark?
3. Which real-domain pilot best demonstrates strategic value: Warfarin PK/PD,
   NASA battery aging, wastewater infectious disease, or industrial process
   control?
4. Is ABC-SMC the right inference engine long term, or should the framework
   support multiple inference backends more aggressively?
5. Is the core asset the model-replacement loop, the diagnostic/provenance
   system, the domain registry, or the application-specific adapters?
6. What should be open sourced immediately, and what should be held back until
   safety/reproducibility improves?
7. Does the project need expert domain collaborators before a serious paper
   submission?
8. Would reviewers accept "white-box model replacement" as a distinct framing
   from symbolic regression?
9. What is the minimum real-world result that makes this credible as a startup
   wedge?
10. Should the brand be "E3," "Automated Science," or something more specific
    like "Mechanistic Model Foundry"?

## Suggested One-Sentence Positioning

For a paper:

```text
E3 is a closed-loop system that uses language models to propose interpretable
scientific model structures and simulation-based inference to fit, diagnose,
and select among them.
```

For open source:

```text
An open framework for LLM-guided white-box model discovery with ABC-SMC fitting,
held-out validation, generated-code safety checks, and scientific provenance.
```

For a startup:

```text
Interpretable mechanistic model discovery for expensive scientific and
engineering time-series data, starting with PK/PD and battery degradation.
```

## Bottom-Line Recommendation

This work should not be forced into a single bucket yet. It is best treated as a
research platform with three coordinated outputs:

1. A conservative systems paper.
2. A cleaned open-source framework and benchmark.
3. A focused real-domain pilot that can become the seed of a startup if it
   demonstrates held-out value.

The project's most defensible claim today is:

```text
We have built and tested a closed-loop system where LLMs propose white-box
scientific model replacements and ABC-SMC-style inference evaluates whether
those replacements actually improve fit, with provenance across many domains.
```

The project's most important next claim to earn is:

```text
On a real, high-value scientific dataset, the system discovers an interpretable
model structure that improves held-out prediction and passes expert-readable
plausibility checks against established baselines.
```

If that next claim is earned in Warfarin PK/PD or NASA battery aging, this
becomes not just a paper, but a credible foundation for an open scientific model
discovery framework and possibly a vertical product.

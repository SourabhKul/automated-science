# Canonical Architecture

This project is a framework for automated white-box scientific model search.

The premise is intentionally narrow and testable: hold the observed data fixed, then automate the scientific loop of proposing explanatory model structures, fitting their parameters, diagnosing their failures, and replacing them with better white-box models.

## Scientific Loop

The canonical process is:

1. Choose a scientific domain and observed dataset.
2. Represent a candidate mechanistic model as executable white-box math.
3. Fit the model's parameters to data with a simulation-based inference engine.
4. Inspect posterior behavior, residual gaps, stability, identifiability, and held-out prediction.
5. Ask model-generating agents to propose a structurally different model.
6. Validate, sandbox, and evaluate the candidate.
7. Replace the incumbent model only when the candidate is better under the chosen evaluation protocol.
8. Preserve every accepted and rejected candidate as scientific provenance.
9. Let higher-level agents learn from many runs and improve future search.

In short:

```text
data -> model -> parameter fit -> diagnostics -> structural hypothesis -> validation -> model replacement -> memory -> meta-learning
```

## Model Classes

The system should not be limited to ODEs. ODEs are the current dominant implementation because they are convenient for many physical, biological, and social dynamical systems, but the architecture should support any explicit white-box scientific model that can be simulated and scored:

- ODEs and SDEs
- stochastic jump processes
- probabilistic graphical models and Bayes nets
- Markov decision processes
- state-space models
- agent-based models
- hybrid symbolic/neural models when the symbolic surface remains interpretable

The invariant is not "ODE"; the invariant is:

```text
candidate model + parameter space + simulator/inference adapter + observable prediction
```

## Inner Loop: Parameter Inference

The inner loop answers:

> Given fixed model structure `M` and fixed data `D`, what parameter regions make `M` explain `D` well?

Current implementation:

- `core/sbi_engine.py`
- `core/sandbox_eval.py`
- `core/evaluation.py`

Current canonical default:

- ABC-SMC
- 500 accepted particles
- 15 generations
- 150000 initial particles
- summary-statistic distance unless held-out trajectory metrics are requested

Important research direction:

- Parameter exploration strategy is a first-class research target.
- BDSS / beta-distributed step sizes should be restored or reimplemented as one available ABC-SMC transition strategy.
- The system should compare parameter-search strategies the same way it compares model structures: by fit quality, stability, posterior health, runtime, and held-out prediction.

The architecture is not tied to BDSS, Gaussian kernels, or any one inference method. It needs a clean inference-strategy interface.

## Outer Loop: Model Replacement

The outer loop answers:

> Given the current model's fit and failure modes, what model structure should replace it?

Current implementation:

- `scripts/run_domain.py`
- `core/generated_code.py`

The replacement unit is the model itself. A proposed model is an in-place replacement candidate for the incumbent. It is accepted only if it improves the selected evaluation objective.

The model-generating agent should receive:

- current model structure
- parameter priors and fitted posterior summary
- train/test metrics
- residual and failure diagnostics
- numerical stability issues
- domain hints and scientific context
- prior rejected/accepted candidates
- limitations of the incumbent model

Current gap:

- The hero run mostly optimized in-sample ABC distance.
- The newer hardened runner supports held-out selection, but its diagnostic context is still shallow.

## Evaluation Loop

The system evaluates both model quality and model limitations.

A model is "good" only relative to a domain and evaluation protocol. At minimum, reports should include:

- train fit
- held-out prediction
- posterior concentration and identifiability
- numerical stability
- parameter plausibility
- model complexity and parsimony
- residual structure
- robustness to seeds/splits
- comparison to classical baselines
- scientific interpretability

Known danger:

- A model can be an excellent in-sample trajectory fit and still be scientifically weak, overfit, numerically brittle, or non-identifiable.

## Meta Loop: Self-Improvement

The meta loop answers:

> Across many model-search attempts, how should the system improve its own search, prompts, priors, inference kernels, diagnostics, and domain adapters?

This is the layer intended to improve future search through repeated, recorded runs.

Canonical future components:

- research ledger of accepted and rejected hypotheses
- structured model critiques
- domain memory
- strategy memory
- automated ablations of prompts and inference strategies
- model-family comparison
- inference-kernel comparison
- automated limitation reports

The legacy "Senior Research Director" meta-agent attempted a form of this loop. The preserved qwen36 hero run shows the scheduler invoked it, but qwen36 `meta_hints.txt` outputs are empty in the completed artifact tree. The meta loop therefore needs to be rebuilt as a first-class, structured, testable component.

## Current Canonical Files

| Responsibility | File |
|---|---|
| domain registry | `core/domain_configs.py` |
| single-domain model-replacement loop | `scripts/run_domain.py` |
| generated-code validation | `core/generated_code.py` |
| sandboxed candidate evaluation | `core/sandbox_eval.py` |
| parameter inference | `core/sbi_engine.py` |
| train/test metrics | `core/evaluation.py` |
| domain gauntlet | `scripts/run_gauntlet.py` |
| LM Studio model discovery | `scripts/lmstudio_models.py` |
| multi-model comparison | `scripts/run_model_matrix.py` |
| baselines | `scripts/legacy/run_baselines.py` (compatibility surface) |

## Hero Run Status

The May-June qwen36 run is the main preserved proof-of-life artifact:

- May 6, 2026 17:58:27 to June 17, 2026 01:29:27
- 10 serial circuits
- 25 domains
- 250 domain-run histories
- 25,351 logged structural-evaluation rows
- 1,351 accepted model replacements

What it records:

- LLM-generated white-box model replacement plus ABC-SMC fitting produced accepted model replacements across many scientific domains.

What it does not prove:

- robust held-out generalization
- active preserved meta-agent steering
- globally optimal parameter-search strategy
- identifiability or scientific validity for every accepted model

## Improvement Rule

Every new feature should identify which loop it improves:

- Parameter inference loop
- Model replacement loop
- Evaluation and limitation loop
- Meta/self-improvement loop
- Domain/model-class adapter

If a change cannot be placed in one of these loops, it probably belongs outside the canonical architecture.

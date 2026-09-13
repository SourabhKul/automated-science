# Scientific review and development plan

Initial review and runtime checks: 2026-09-07. Research direction updated: 2026-09-12. Reviewed source: `6ff92f2a4ecc32283cb44a6298e0129770f24c60` from `SourabhKul/automated-science`.

This is a first-principles assessment and proposed research program, not a report of a new scientific discovery. The primary agent owns the reasoning and plan; coding and experiment work is assigned to GPT-5.6 Luna agents at max reasoning. Supporting audits in this directory distinguish inspected implementation, newly executed checks, and historical claims.

## Fixed research direction: 2026-09-12

The sole research LLM is **Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed**, served through oMLX at `http://127.0.0.1:8000`. Phase out research into which LLM works best. Model ranking, multi-model gauntlets, and migration to other research LLMs are outside the active plan. Retain old results as historical records; do not delete them as part of this decision.

The central objective is to discover new scientific knowledge through data exploration, explicit competing hypotheses, and simulation-based inference in the broad ABC-SMC style. Qwen is the fixed instrument; the research variables are the scientific workflow, hypothesis space, use of observations, inference, diagnostics, and choice of informative measurements. All LLM-based comparison arms below use that same Qwen checkpoint. Luna teams remain the engineering and experiment operators, distinct from the model being studied.

Scientific breakthroughs cannot be guaranteed by a workflow. Progress should be measured by reproducible new predictions and explanations that survive independent attempts to falsify them. Methods benchmarks serve this discovery objective; they should not become a replacement objective of accumulating benchmark scores or framework features.

## Work completed locally

The workspace was empty, so the repository was cloned here without overwriting an existing checkout. The local endpoint returned the exact requested model ID, `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`. A short chat request succeeded; a code preflight required disabling thinking via `chat_template_kwargs` and one repair request. Existing single-domain endpoint/model overrides resolve correctly, but the model matrix's LM Studio load/unload API is incompatible with the observed oMLX server. These adaptations are recommendations, not installed changes.

Compilation and the selected dependency-light checks passed. The isolated targeted test run reported six passing tests. This is not a full-suite pass or an end-to-end discovery result. The audit environment is temporary and does not establish reproduction of the original frozen environment. No scientific campaign was launched, and no source implementation was changed. See [runtime audit](runtime-audit.md) for commands, versions, and scope.

The user-specified division of responsibilities is preserved in the repository's root `AGENTS.md`.

## Engineering findings that determine the next work

The [code audit](code-audit.md) records file and line references. Its immediate findings are:

- The canonical runner repeatedly selects using its purported test MSE, and full-trajectory residual diagnostics enter later proposal prompts. Those reported outcomes cannot serve as untouched test evidence.
- The inference implementation needs a mathematical and calibration check, particularly consistency between proposal generation, boundary handling, weight calculation, and downstream particle summaries. The BDSS path is reported to use a Gaussian density despite generating a different proposal distribution.
- Warfarin's configured evaluation mode does not match the canonical evaluator's subject/cell dispatch, making the intended grouped path disconnected.
- Historical hero-run artifacts are absent from the cloned repository. The documents' historical counts and the present registry size describe different things; 25 historical domains versus 27 current configurations is not itself a contradiction.
- The source-gated classification stream largely runs without LLM proposals or ABC-SMC. Its results should remain distinct from discovery-engine evidence.

These are source-review findings, not newly measured scientific performance. The targeted passing tests do not establish inference calibration or absence of test leakage. Repair and verify these boundaries before launching a discovery campaign.

## Recommendation

Concentrate the project on **discovering and falsifying compact dynamical models across independent experiments**. Start by making the evaluator scientifically trustworthy, then pursue one real application. My preferred first application is battery state and degradation dynamics under changing usage, conditional on acquiring sufficiently informative measurements. Enzyme kinetics is the strongest alternative with a laboratory collaborator. Public nonlinear system-identification data provide a practical methods benchmark in the meantime.

The valuable capability is the separation between a model proposer and a numerical evaluator. The missing capability is a reliable transition from “this equation predicts better” to “this new scientific claim survived an informative test.” More domain scripts, accepted mutations, or generation tokens will not by themselves create that transition.

There are two possible publications, with different evidence requirements:

1. A **methods paper** about reliable, resource-bounded scientific model search, including recovery, failure detection, and independent prediction. This can be developed locally.
2. An **application paper** reporting a previously unsupported relationship, mechanism, or validity boundary. This needs a specific novelty search and independent evidence, potentially new physical measurements. A fresh equation on an old dataset is not sufficient by itself.

## First principles

A scientific discovery workflow needs to specify the question, acquire informative evidence, formulate competing explanations, derive distinguishable predictions, test them, and preserve both successes and failures. Fixed-data equation search automates only part of that workflow.

For a dynamical application, the scientific object should include:

\[
\dot x=f_M(x,u,t;\theta),\qquad y=g_M(x;\phi)+\epsilon.
\]

Here `u` is a measured or controlled input, `x` is the physical state, and `y` is the observation. Initial conditions, observation noise, missingness, interventions, units, and population variation belong to the model contract. Editing only the right-hand side of an ODE risks explaining a sensor artifact, an unrecorded input, or between-subject variation as a new mechanism.

| Requirement | What the project offers | What must change for stronger science |
| --- | --- | --- |
| Explicit hypotheses | Executable dynamics and parameter ranges | Add units, state/observation meaning, mechanism assumptions, and falsifying predictions |
| Numerical testing | Simulation, parameter search, validation, diagnostics | Calibrate inference, quantify fit variability, and test solver sensitivity |
| Generalization | A canonical train/held-out path; separate source-gated benchmark protocols | Separate adaptive validation from an untouched final test throughout the canonical loop |
| Mechanistic interpretation | Human-readable equations | Establish what the observations identify; retain equally plausible alternatives |
| Novelty | Model proposals can combine domain knowledge | Compare with literature and algebraically equivalent models; distinguish rediscovery |
| Autonomy | Repeated proposal, evaluation, and replacement | Add informative experiment selection and independently collected evidence |
| Reproducibility | Configuration, seed support, fingerprints, manifests | Publish a replayable evidence bundle and make every relevant source of randomness explicit |

The repository's own strategic overview reports a 25-domain historical run and 1,351 accepted replacements. Those are historical summaries, not results reproduced in this checkout. The supporting run tree is excluded from the public repository. Serial improvements in an adaptive search are also not independent scientific replications. See the accompanying evidence audit before reusing manuscript numbers.

The source-gated classification stream has useful practices: source checks, duplicate accounting, negative controls, and separation of selection from external evaluation. Its ISOLET, digit, and image classification scores do not validate the mechanistic discovery engine. Reuse the evaluation practices; stop treating additional unrelated classification tasks as progress toward mechanism discovery.

### The central evaluation problem

If model proposals are repeatedly accepted using `test_mse`, then that score is a validation score, irrespective of whether its rows were excluded from parameter fitting. The adaptive search has learned from it. Sending its diagnostics back into subsequent prompts increases that dependence, but even an accept/reject bit is feedback.

Use training experiments to fit parameters; validation experiments to choose structures, priors, prompts, and stopping rules; and sealed test experiments to evaluate the locked result. Split at the actual independence unit: cell, subject, experimental replicate, batch, or intervention condition. A chronological suffix answers a forecasting question, not automatically a new-cell or new-condition question. If data are too scarce for three adequate partitions, use a predeclared nested evaluation design and report that limitation.

Do not rewrite the scientific question after observing test outcomes. If a result motivates a new question, that is a new study with new confirmatory evidence. In active experimentation, newly acquired development measurements may enter the training pool after prospective predictions are scored; retain a separate final confirmation campaign.

### Parameter inference is not scientific certification

Keep simulation-based inference, including ABC-SMC, central to the research program. Make its discrepancy, summaries, observation assumptions, population update, and uncertainty interpretation explicit. For deterministic ODE controls with tractable likelihoods, multi-start likelihood fitting or weighted least squares can provide a computational reference and a check on the inference implementation. They need not displace ABC-SMC as the project's main exploration framework.

Small distances between summary statistics need not imply correct trajectories or mechanisms. The literature specifically warns that insufficient summaries can undermine ABC model comparison; the current distance ranking should not be described as a Bayes factor. [Robert et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3174657/).

Treat reported particle distributions as provisional until the algorithm, weights, effective sample size, numerical failures, and uncertainty calibration have been checked. Use analytically tractable examples and simulated parameter recovery; where posterior sampling is claimed, use simulation-based calibration and predictive coverage checks. [Talts et al., 2018](https://arxiv.org/abs/1804.06788).

### Novelty and positioning

LLM-guided equation proposals followed by coefficient fitting already exist in LLM-SR. SINDy already discovers sparse dynamical equations, and recent work also applies LLM agents directly to ODE discovery. Merely combining an LLM and an ODE fitter is an insufficient novelty claim. [SINDy](https://doi.org/10.1073/pnas.1517384113), [LLM-SR](https://arxiv.org/abs/2404.18400), [2026 ODE-discovery preprint](https://arxiv.org/abs/2607.13608).

The promising research question is: **Can constrained model search recognize when a mechanism is unsupported, retain competing explanations, and choose measurements that distinguish them more efficiently than established alternatives?** That is a proposed contribution, not a finding or a claim that the literature has left the entire question open.

Known named equations are useful positive controls, but their recovery can reflect model memory. Include transformed, anonymized, and newly generated mechanisms, as well as real experiment holdouts. Such controls reduce simple recitation; they cannot prove absence of training contamination. LLM-SRBench provides relevant precedent. [LLM-SRBench](https://arxiv.org/abs/2504.10415).

## Applications ranked by practical fit

These rankings are my judgment about this project and a single local machine. They are not estimated probabilities of publication. Sources establish relevant data or prior work; the hypotheses below are proposed research directions whose novelty remains to be checked in depth.

| Priority | Application and precise question | Why it fits | Required evidence and main obstacle |
| --- | --- | --- | --- |
| 1 | Battery dynamics: which compact state representation predicts capacity response under changing usage? | Existing battery adapter; bounded dynamical models; measurable currents, temperature, and impedance | Hold out whole cells and usage regimes; capacity alone cannot identify named degradation mechanisms |
| 2 | Nonlinear physical system identification: when is an input-dependent mechanism distinguishable from a flexible curve fit? | Cheap simulation; controlled inputs; established public benchmarks | Best initial methods study; recovery of known benchmark physics is not new physical science |
| 3 | Enzyme/reaction kinetics: which competing rate law survives concentration or input perturbations? | Small reaction networks, conservation constraints, direct experimental discriminators | Complete time courses and experimental context; new measurements usually need a collaborator |
| 4 | Microbial community dynamics: when are pairwise interactions insufficient to explain recovery or multistability? | Natural extension of ecology ODEs; perturbations and initial-condition changes can discriminate models | Absolute abundance, growth context, and replicate experiments; public papers already establish many familiar mechanisms |
| 5 | Population PK/PD: does a structural model transfer between subjects after accounting for heterogeneity? | Existing subject-aware data work; established compartmental models | Hierarchical inference, observation models, and a sufficiently rich independent cohort; toy fixtures are not discovery evidence |

### 1. Battery state and degradation dynamics

**Target question.** Does a compact, interpretable state model explain how recent history and current operating conditions change future capacity better than scalar state-of-health or calendar/cycle-age models?

**Initial model families.** Compare a scalar irreversible damage state, a model with separate fast reversible and slow irreversible states, and a parsimonious stress-dependent transition model. These are hypotheses, not evidence that the data identify electrochemical SEI, plating, or cracking. In particular, monotonic irreversible damage does not imply monotonic measured discharge capacity under changing protocols.

**Data path.** Use the existing NASA schema as adapter calibration. NASA's source describes cycling, temperature, and impedance measurements. For a stronger application, inspect the variable-usage and impedance data accompanying Jones et al. and the richer battery lifetime literature. Do not assume the necessary fields, license, or split independence until a raw-data audit passes. [NASA dataset](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets), [Jones et al., 2022](https://www.nature.com/articles/s41467-022-32422-w).

**Discriminating prediction.** At similar observed capacity, competing state models should make distinguishable predictions under a specified future load protocol or a rest/recovery condition. Only evaluate conditions actually documented by the source; absent interventions require new data. Joint capacity and impedance measurements are more informative than naming latent processes from a single capacity trace.

**Comparison.** Include persistence/age curves, a published empirical battery predictor appropriate to the dataset, equivalent-circuit or mechanistic baselines where observations support them, and a flexible predictive baseline. Fit all preprocessing on development data. Hold out cells and protocol families; where a source permits it, reserve a separate batch. Compare errors per cell, predictive calibration, model size, and simulation cost. A useful finding could be a reproducible validity boundary for a simpler model, even if a more complex model does not win.

**Publication hurdle.** A lower NASA capacity MSE is a crowded result. Battery forecasting already has rich data-driven and physics-informed work, including Discovery Learning in 2026. The potential advance is an independently tested model relationship, new discriminating measurement, or generalizable failure boundary. [Severson et al., 2019](https://www.nature.com/articles/s41560-019-0356-8), [Discovery Learning, 2026](https://www.nature.com/articles/s41586-025-09951-7).

### 2. Nonlinear physical systems as the first methods application

Use Silverbox or Cascaded Tanks to test recovery and extrapolation with recorded inputs. Silverbox is an electronic realization of Duffing-like dynamics; established benchmark results and data formats make comparison practical. [Official Silverbox source](https://www.nonlinearbenchmark.org/benchmarks/silverbox), [official benchmark evaluation guidance](https://www.nonlinearbenchmark.org/results-reporting).

The repository's pH reactor and DREAM4 contracts are also useful architecture tests, but explicitly describe simulator data. DREAM4's documented recovery task does not license claims about unseen interventions when intervention targets are unknown.

A plausible methods question is whether a mechanism grammar plus independent validation improves accuracy per simulation and reduces false mechanism promotion, relative to unconstrained LLM edits, SINDy, and symbolic regression. New physical-science claims would need unexplained real behavior and new evidence beyond recovering the known benchmark equations. A future instrumented thermal or mechanical rig could supply that evidence; no hardware availability is assumed here.

### 3. Enzyme and reaction kinetics

Start with a small, well-characterized network and compete mass-action, saturation, inhibition, and activation/deactivation hypotheses. Restrict proposals to compatible stoichiometry and positive rate constants. Fit multiple experiments jointly, preserving concentrations, dosing/input histories, assay calibration, and detection limits.

The desired result is a new, reproducible rate-law distinction or a demonstration that a widely used simplification fails in a particular measured regime. Hold out complete concentration or input conditions, and predict a discriminating experiment before measurement. Mechanism classification from kinetic profiles is already established; a 2024 experimental-design study also shows how informative perturbations improve kinetic models. Our extension would need evidence beyond reproducing those results. [Burés and Larrosa, 2023](https://www.nature.com/articles/s41586-022-05639-4), [van Sluijs et al., 2024](https://www.nature.com/articles/s41467-024-45886-9).

That latter study explicitly discusses future experiments designed to distinguish alternative mechanistic assumptions. It is particularly relevant prior art, rather than justification for claiming active model discrimination as wholly new.

### 4. Microbial ecology

Use small experimental communities to compare generalized Lotka–Volterra dynamics with resource-mediated or density-dependent alternatives. Test complete initial-condition regimes, nutrient conditions, or perturbation/recovery experiments. Preserve absolute abundance where available; relative composition alone creates scale and interaction ambiguities.

Cooperative growth and multistability have already been studied experimentally, so rediscovering an Allee term is a control, not a new biological discovery. A candidate paper would need a new regime-dependent interaction, a falsified competing explanation, or a transferable prediction on independent experiments. Public source data can establish feasibility; new confirmation may require a collaborator. [Cooperative growth and multistability, 2024](https://www.nature.com/articles/s41467-024-48521-9).

### 5. PK/PD as a later specialist application

The repository has useful subject/mask plumbing, but its documented subject-holdout smoke is not full hierarchical model inference. Pooled parameters can force the structural proposer to absorb between-subject variation into invented dynamics. Establish a hierarchical baseline and endpoint-specific observation likelihood before structural search. Theophylline/warfarin examples are calibration tasks unless a sufficiently informative new question and cohort are available. This review proposes no clinical-use claim.

### Defer these directions

Do not prioritize sunspots, Nile flow, macroeconomics, aggregate CO2, or more image/digit classification for the first mechanism-discovery publication. Single observational trajectories often admit time-forced descriptive fits without identifying underlying causes. Existing blocked wastewater and Tennessee Eastman sources should remain blocked under their current contracts; changing the question requires a new explicit protocol, not quiet repair of failed gates.

## Evolution plan

### Stage 0: Establish a reproducible local foundation

Use the user-specified oMLX endpoint and fixed Qwen model ID, verifying its availability through `/v1/models`. Build a direct single-model run path. Legacy LM Studio residency APIs should not govern an oMLX session, and adapting the multi-model matrix is no longer an active work item. Record request parameters, response/failure, server/model metadata when available, code revision, and environment. A change of checkpoint, quantization, or chat-template settings requires a new recorded configuration even when the model's displayed name is unchanged.

Preserve the ownership rule: the primary agent plans scientific questions and experiments; Luna max teams implement, test, run, and monitor them. Local Qwen supplies experimental hypothesis proposals, not the authoritative validity or novelty judgment. A successful chat request is only a transport smoke test, not a validation of equation generation or science capability.

Exit condition: an isolated environment, honest test inventory, one bounded model response, and an identified path to a tiny end-to-end run. Missing historical artifacts remain explicitly missing.

### Discovery front end: select questions from informative data

Within a primary-agent-approved application plan, let the scientific system explore development data for unexplained residual patterns, regime changes, inconsistent scaling, failed conservation balances, and disagreements with established models. Preserve measurement metadata, missingness, units, inputs, and independent replicate identities before interpreting any pattern.

Turn a pattern into a structured question: what is unexplained, which mechanisms could explain it, which observation would distinguish those mechanisms, what prior work already says, and whether the available data can answer it. Rank questions by scientific significance, plausibility of novelty, identifiability, available evidence, and achievable computational/experimental cost. Do not rank them solely by how easily the current evaluator can improve a score.

Every explored question enters the research ledger, including abandoned questions and negative findings. Adaptive exploration consumes development evidence; it cannot quietly consume the sealed confirmation set. The primary agent decides campaign scope and scientific priorities, while Qwen performs the hypothesis-generation role inside the specified workflow and Luna teams implement and operate it.

### Stage 1: Repair scientific evaluation before scaling search

1. Introduce an evaluator-owned protocol with train, validation, and final-test partitions; restrict all adaptive feedback to development data. Audit future-aware preprocessing and initial-state construction.
2. Fit and rank models using a declared observation model or justified distance. Compare independent fits of incumbent and challenger before interpreting tiny improvements.
3. Record parameter weights, effective sample size, acceptance/failure rates, and uncertainty method. Fix any sampler correctness problems before calling particles posterior samples.
4. Make data generation, fits, proposals, and baseline randomness independently seeded and recorded. Persist inputs, masks, units, solver settings, and all failed attempts.
5. Replay a result from a compact bundle containing source hashes, splits, model, parameters, prompts, predictions, and metrics. Raw third-party archives need not be embedded, but a versioned acquisition route must exist.

Exit condition: sentinel tests show that changing sealed test outcomes cannot change the candidate search; synthetic inference checks pass; a clean replay reproduces metrics to declared tolerances. Then freeze the evaluation version.

### Stage 2: Make hypotheses easier to test and harder to misinterpret

Replace unrestricted Python as the main scientific representation with a typed expression or mechanism representation. Include states, observations, measured inputs, units, parameter transformations, conservation rules, and event semantics. Compile it to trusted simulation code. Keep an escape hatch only as a separately labeled experimental path.

Track symbolic complexity, dimensional consistency, limiting behavior, and numerically equivalent expressions. Require every candidate to state its mechanism change and a prediction that distinguishes it from the incumbent. Use a small portfolio of plausible structures instead of keeping only the single current winner. Close fits with incompatible mechanisms should produce an “unresolved” result and a measurement recommendation.

Exit condition: invalid-unit and invalid-conservation proposals are rejected; equivalent proposals are deduplicated; known non-identifiable controls remain unresolved; known distinguishable controls are separated under informative data.

### Stage 3: Spend simulation effort where it changes decisions

Use a sequence of checks: syntax/units and prior plausibility; short numerical trajectories; a small initial inference population; progressively larger ABC-SMC populations for viable candidates; validation; then uncertainty and stress tests for finalists. Compare this allocation policy with uniform simulation budgets using the same Qwen model. Tractable fitting methods can serve as cheap reference checks where appropriate. Cache compiled models and results using model/data/solver/configuration hashes. Reuse fits only where parameter semantics agree. Do not compare candidates evaluated at different fidelities as though their scores were interchangeable.

On one local server, begin with a single generation queue and bounded CPU simulation workers. Measure memory pressure and actual throughput before adding parallel requests. Fit time may dominate generation time; measure both. Stop on exhausted budget, recurrent invalid proposals, or failure to improve development performance, retaining those outcomes.

Exit condition: measured cost–quality curves and reproducible budgets, not assumed speedups from a model's name.

### Stage 4: Close the scientific loop

For unresolved alternatives, select the next feasible input, initial condition, or measurement channel based on expected ability to discriminate models, accounting for parameter uncertainty, measurement noise, and cost. Raw prediction disagreement alone may select a numerically unstable or noisy region. Compare the policy with random, space-filling, and parameter-information designs.

First validate the policy on a simulator with hidden ground truth. Next evaluate prospective predictions using a finite pool of real experiments under a frozen acquisition protocol, recognizing that offline replay covers only available conditions. Finally collect new physical experiments with a collaborator or an instrumented system.

Exit condition: better discrimination or lower independent prediction error at a matched measurement budget, including abstention when no feasible experiment is informative.

### Stage 5: Add meta-learning only after an honest baseline exists

Store structured outcomes by mechanism, domain, diagnostics, inference difficulty, and validation result. Learn proposal priorities on development tasks and test that policy on held-out tasks. Never let final-test outcomes become prompt memory for the same evaluation. First test whether structured memory improves search at fixed budget with the fixed Qwen model; postpone fine-tuning, reinforcement learning, and new model classes until simpler ablations justify them. Retire LLM ranking and multi-model matrices from the active research roadmap.

## First campaign specification

This is a proposed campaign, not a launched experiment or a wall-clock estimate.

**Question:** Do mechanism constraints and diagnostic feedback improve model recovery and independent prediction per simulation, while avoiding promotion on uninformative data?

**Development suite:** three small dynamical families with hidden parameters and newly generated input/initial-condition regimes. Include known missing mechanisms, an observation/noise mismatch, and a non-identifiable alternative within the family cases. Known textbook names should not supply the answer to the proposer. Keep a separate final family/condition set for evaluation of the frozen policy.

**Search ablation:** four arms: grammar-based random proposals; Qwen proposals without diagnostic feedback; Qwen proposals with development diagnostics; Qwen proposals with diagnostics plus mechanism constraints. All three LLM arms use the exact same Qwen model. Use the same declared candidate universe where an ablation requires it, and document any unavoidable difference in prior knowledge. Add established SINDy/PySR and expert fixed-structure baselines as separate comparisons; an LLM-SR-style workflow should also use the same Qwen if needed for a methods comparison. No arm compares different LLMs.

**Initial engineering budget:** three families × four search arms × five independent seeds × eight evaluated proposals = 480 candidate evaluations, plus seed-model fits and baseline runs. All repair requests, invalid proposals, retries, and failed fits count toward reported costs. First run a much smaller smoke slice to estimate cost and set hard wall-clock and simulation caps; eight proposals and five seeds are pilot budgets, not a power analysis.

**Metrics:** frozen-test trajectory error, symbolic/mechanism recovery where ground truth exists, false promotion on negative controls, unresolved-case handling, coverage where uncertainty is claimed, candidate validity, solver failures, total simulator calls, LLM tokens, and wall time. Use paired task/seed comparisons; confidence intervals must respect independent experiment/family units, not count adjacent samples or adaptive candidates as independent observations.

**Decision:** progress to one audited real-data pilot only if the system passes integrity checks and the pilot provides a credible cost–quality benefit or an informative failure finding. Choose practical effect thresholds and any confirmatory sample size using development evidence before opening the final test. A null result should narrow the claim or redirect the design, not trigger test-driven threshold changes.

The first application campaign should then be battery dynamics if input/impedance coverage is adequate, or physical system identification if battery source contracts fail. A broad 27-domain campaign should wait until these narrow results are reproducible.

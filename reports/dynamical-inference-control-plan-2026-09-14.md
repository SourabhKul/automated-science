# Dynamical ABC-SMC control: scientific acceptance specification

Primary planning specification, 2026-09-14. This control is not yet implemented or executed. It is the next inference milestone after scalar reference checks and the canonical evaluation boundary; it makes no discovery or calibration claim.

## Question and fixed contract

Does the opt-in Gaussian ABC-SMC reference recover the correct finite-tolerance distribution when its simulator is a numerical ODE solver, and do downstream predictions preserve particle weights and the original time origin?

Use decay dynamics dx/dt = -k x with x(0)=1, k uniform on [0.05, 1.5]. Observe at t=[0.5,1,2,3] with independent Normal(0, 0.05^2) measurement errors. The numerical inference simulator must integrate from t=0, evaluate these four times, and add fresh independently seeded measurement noise for every simulated dataset. Initial condition and noise scale are known in this control. Do not estimate them from future observations.

Use true k values 0.35 and 0.8 and three independent data seeds per value. Separate data-generation, ABC proposal, and simulator-noise streams. Predeclare their seeds before execution. Use the standardized full-data discrepancy D=|| (Y_sim - y_obs)/0.05 ||_2 and an explicit two-population schedule epsilon=[3,2]. Begin with 400 particles and 50,000 proposal attempts per population, plus a measured conservative wall-clock cap. Budget exhaustion is an incomplete result, not a posterior sample. No LLM is needed for this known-model inference control.

## Independent finite-epsilon target

The analytic ODE solution is mu_i(k)=exp(-k t_i). Conditional on k, D^2 has a noncentral chi-square distribution with four degrees of freedom and noncentrality lambda(k)=sum_i[(mu_i(k)-y_i)/0.05]^2. Therefore the ABC target density on the prior interval is proportional to F_noncentral_chi_square(epsilon^2; df=4, lambda(k)). An independent quadrature calculation can normalize this target and compute means, CDFs, and quantiles without using the inference simulator or its summary helper.

Also compute the ordinary Gaussian-likelihood posterior, proportional to exp(-lambda(k)/2), as a separately labeled limit/reference. At finite epsilon the two targets differ; do not require agreement with the exact-likelihood posterior or describe the difference as sampler bias.

The reviewer should independently check the target construction and compare numerical ODE predictions against the analytic solution at boundary and interior parameter values. Set a solver-error tolerance well below the measurement scale before the stochastic pilot. Do not silently replace numerical integration with the analytic solution in the inference arm.

## Required evidence

- Save all populations' normalized weights, ESS, epsilon, attempts, simulator calls, out-of-support retries, solver failures, elapsed time, seeds and solver configuration. Recompute a subset of transition densities and weights independently.
- Compare weighted empirical mean, CDF and quantiles with the corresponding finite-epsilon target for each completed cell. Report descriptive errors across the six independent datasets; do not apply IID confidence formulas to dependent weighted SMC particles or claim coverage from six cases.
- At t=[4,5], compare weighted latent predictions with the quadrature transformation of exp(-k t). Separately include measurement noise for observation-predictive intervals. Score any independently generated future observations only after the inference configuration is frozen; this is a known-model control, not scientific confirmation.
- Use an intentionally unequal-weight fixture to verify downstream prediction summaries actually use weights. A high ESS pilot alone can conceal an unweighted-summary bug.
- Exercise a single future timestamp to catch restarting from x0 at the future horizon. Exercise a forced solver failure and unattainable discrepancy with explicit bounded terminal statuses.

## Decision and scope

Passing supports an ODE-backed reference integration and weighted prediction contract. It does not validate the historical SBIEngine, BDSS, ABC model probabilities, structural identifiability, or general posterior coverage. Canonical integration should use an explicit reference-inference option only after this evidence is reviewed, with weighted outputs propagated end to end and failed/incomplete results excluded from posterior claims.

Keep this control small. If it fails, preserve the failure and repair the specific discrepancy before expanding to multiple mechanisms or real data. If it passes, the next scientific control should introduce an observation-model mismatch and an unresolved mechanism pair, rather than increasing the number of easy decay runs.

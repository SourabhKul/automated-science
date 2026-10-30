# Reference feasibility decision: defer the yeast comparison

Decision at the October 30 13:57Z wake: reject another yeast scientific
comparison within the current engineering and inference scope. Keep plate2
unopened. Defer this pilot until a separately justified reference design is
available. Do not build a new general inference framework to rescue it.
This is a bounded feasibility decision, not a theorem that yeast inference
is impossible or a result favoring either curve.

The existing eight-case run remains closed. Five reference controls failed
its predeclared gates; a separate artifact guard stopped final packaging.
Those facts do not change. The earlier result and analytic diagnosis are in
`yeast-diauxic-pilot-results-2026-10-30.md` and
`yeast-diauxic-analytic-diagnosis-2026-10-30.md`. This decision uses equations,
source and compact receipts only. No new observation, array, fit, simulation,
numerical check, model request or score is used.

## Available reference paths

A bounded source review found these current interfaces:

| Path | What it can establish | Limit for this pilot |
| --- | --- | --- |
| `core/abc_smc_reference.py` and `core/sbi_engine.py` | Finite ABC machinery; the checked reference has explicit simulator/discrepancy callbacks and attempt limits | They approximate a tolerance-dependent target; they are not independent ordinary-likelihood reference integrators |
| `core/abc_reference_controls.py` | Scalar Uniform(0,1) finite-epsilon control with fixed Gauss–Legendre quadrature | Its specified scalar model is not the four- or seven-coordinate yeast likelihood |
| `core/ode_reference_controls.py` | A one-parameter decay reference, including ordinary Gaussian likelihood and finite-epsilon targets | Its target and solver assumptions are specialized; its numerical quadrature is not a general validated multidimensional integrator |
| Frozen yeast driver | Point search and finite importance weighting under the explicit Gaussian error model | Its all-case adequacy gate failed; more draws or new tuning are not authorized |

The search did not find a generic globally validated posterior integrator,
coverage certificate or MCMC interface in the current Python source tree.
That is an inventory result, not a claim that no suitable external method
exists. Source labels such as “exact Gaussian likelihood” describe the
likelihood target; they do not turn numerical quadrature into exact
integration. A Gaussian proposal is also distinct from a Gaussian observation
likelihood. Reusing an API name alone does not transfer its validation.

## Exact accepted law does not give a usable cost bound

Prior rejection gives one simple analytic alternative. Let the prior be
uniform on the cube, L(x)=C0*exp(-Q(x)/2), with fixed Gaussian covariance and
Q>=0. Draw x from that prior and an independent uniform u. Accept when
u<=exp(-Q(x)/2). With Z=integral L(x)dx,

\[
a=P(\mathrm{accept})=Z/C_0,\qquad
p(x\mid\mathrm{accept})=L(x)/Z.
\]

Each accepted point has the ordinary posterior law in ideal arithmetic.
The number accepted in N independent attempts has a Binomial(N,a) law.
The expected attempts per acceptance are 1/a. Finite accepted samples still
have Monte Carlo error; their estimates are not exact integrals. A fixed
attempt cap permits an insufficient-samples outcome. It cannot ensure that
enough points are accepted without a useful lower bound on a. No such
practical bound has been established here, and no rejection run is approved.
The need for a valid envelope and the cost of rejection are discussed in
[Owen, chapter 9, section 9.5](https://artowen.su.domains/mc/Ch-var-is.pdf).
The following example is an algebraic check, not a simulated campaign.

Take x~Uniform(0,1) and y|x~Normal(x,sigma^2), with fixed sigma>0. For y>1,

\[
a_y=\int_0^1\exp[-(y-x)^2/(2\sigma^2)]dx
\leq\exp[-(y-1)^2/(2\sigma^2)]\longrightarrow0.
\]

Thus even a smooth scalar Gaussian model has no uniform positive acceptance
bound over all finite y. A better envelope might help a particular target,
but obtaining and checking it is additional work, not an automatic repair.
This example does not estimate the acceptance rate of any yeast case.

It also illustrates the limit of fixed-budget coverage claims. Write t=1-x.
After removing a constant, the posterior density in t is proportional to
exp[-(y-1)*t/sigma^2-t^2/(2*sigma^2)] on [0,1]. Its characteristic boundary
width tends to zero as sigma^2/(y-1). For any fixed c>0, the interval
0<=t<=c*sigma^2/(y-1) has posterior mass tending to 1-exp(-c). A proposal
with bounded logit-normal center/covariance and a fixed uniform component
puts vanishing mass in this shrinking interval. For fixed N, the union bound gives
P(any proposal in that interval)<=N*q(interval), which tends to zero.
The clipped-center/capped-covariance rule therefore cannot supply a universal
finite-N coverage guarantee for this generic family. No such guarantee was
a gate of the old run. This example does not establish a missed mode,
boundary concentration or rejection cost in the observed yeast data.

## What the other choices would require

A deterministic partition or quadrature approach could address separated
regions only with global error bounds, a bounded domain and a verified
implementation for the actual target. Fixed-node quadrature differences
alone are not such bounds. No checked generic four-/seven-dimensional
implementation or practical error/cost envelope was found in the current
API inventory. This does not prove that one cannot be developed.

MCMC or another adaptive finite sampler could be useful in a separately
planned methods study. A finite trace, independent-looking chains, high ESS
or agreement between runs is not by itself a proof that an unvisited region
has negligible mass. No checked generic MCMC path or quantitative global
mixing bound is available in this source review. This decision does not
rank implementations or select a chain, proposal, tuning value or new seed.

A practical reference need not prove accuracy uniformly for every possible
Gaussian observation. The old gates are not replaced by that stronger
requirement. The analytic example only prevents a universal guarantee from
being claimed on the basis of full support or an exact accepted law.
The actual launch-readiness rejection rests on the failed frozen controls,
absence of a justified replacement within the checked interfaces, and the
ban on unreviewed repair/development runs. Alternative methods remain
unevaluated, not disproved.

## Disposition and next independent work

Keep this assay comparison deferred. Do not run ABC for the three passing
cases, rescore, refit a repair, weaken gates, change the supports/noise,
expand the budget, or reuse unused calls. A storage correction cannot clear
reference failure. Do not rewrite the consumed artifacts; the transaction
requirements in the prior memo still apply to any future design.

The next project step is a small scientific-question screen that starts from
a tractable, independently checkable likelihood reference and the existing
APIs. Assess scientific value, measurement assumptions, raw provenance and
independent experimental units before choosing data or code. A scalar
software control alone is not a new scientific discovery. Do not restart
the closed tank study, official holdout, old model comparison or reserved
synthetic campaign. No new source acquisition, numerical study or model
call is authorized by this decision. Any later proposal needs its own
scientific freeze, review and resource limits.

No new download or other user action is needed. The wider local-model plus
checked ABC-SMC objective remains open. The yeast pilot and its held-out
record are preserved as incomplete evidence, not converted into a success.

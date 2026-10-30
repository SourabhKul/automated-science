# Two-family scientific-question screen

Primary decision at the October 30 19:06Z wake: no new empirical pilot is
selected. Thermal relaxation is not source/measurement ready in this bounded
screen. Radioactive counting has a tractable conditional scalar likelihood
in principle, but the inspected source metadata does not establish its
assumptions and the current reference API does not implement it. Retain the
count measurement contract as a design lead only. Do not download data,
choose a cohort, write a reader, fit or call a model.

This screen covers exactly two new families. It does not reopen the deferred
yeast comparison or any closed tank/model/synthetic campaign. Its purpose is
to match a scientific question to a valid observation model and reference,
rather than select data because an exponential curve is easy to fit.

## Questions and source status

| Family and possible question | Scientific value and reference | Source/measurement limit | Disposition |
| --- | --- | --- | --- |
| Thermal relaxation: does one cooling-rate parameter predict a separately recorded run under fixed apparatus and ambient conditions? | A test of parameter transfer and uncertainty under measured conditions; recovery of Newton's cooling law alone is not novel. A scalar rate is analytically tractable only with a justified baseline, amplitude and error model. | The checked public teaching pages do not establish an independent repeat set with independently known ambient/initial state and common noise scale. A lesson is not an experimental-data deposition. | No source or pilot selected. Do not normalize away unresolved measurement error. |
| Radioactive counting: does a common decay-rate shape remain adequate across separately recorded detection conditions once exposure and detector effects are accounted for? | A measurement/robustness question with a conditional scalar count likelihood under strict assumptions. Reproducing a published half-life is not a new nuclear result. | The author deposition contains spectrometry and processed peak-area analysis. Integer nonoverlapping counts, exposure, background and event-loss assumptions are not established by its landing page. The current Gaussian reference is not a count target. | Keep a count-contract design lead; no empirical source, code or acquisition selected. |

The [Vernier experiment page](https://www.vernier.com/experiment/rwv-16_chill-out-how-hot-objects-cool/)
describes measuring cooling with a temperature probe and a purchase link for
lesson material. It does not establish the required raw repeat provenance or
calibration. No purchase, login, contact or lesson-file request was made.
The bounded search did not prove that no suitable thermal data exist.

For counting, the author [Zenodo record, version 1](https://zenodo.org/records/18310004)
links the 2026 D'Agostino et al. study. It lists an 11.3 MB ZIP with spreadsheet,
gamma spectra, calibration spectra and peak-list material. The description
includes real/live times, peak areas and uncertainties, iterative fit-result
updates, and corrections for pile-up, stability, decay and peak fitting.
These facts support investigating provenance; they do not validate a Poisson
observation contract. The displayed license field did not resolve reuse
terms in this inspection. No archive, workbook, spectrum, peak list or preview
was opened. Listed size is approximate, not a verified local byte count.

The authors' [indexed abstract](https://pubmed.ncbi.nlm.nih.gov/42413453/)
describes two consecutive observation sequences with different detection
conditions and a systematic uncertainty budget. This is not evidence of
independent physical-source replication. Their published numerical outcome
is already known from the abstract and must never become a target or a new
result of this project. The publisher DOI open failed in the web tool; one
bounded title search supplied the author abstract. No access workaround,
external message or further recovery was attempted.

## Exact boundary of the current Gaussian reference

Source review of `core/ode_reference_controls.py` found that
`GaussianLikelihoodODETarget` uses mean exp(-k*t), unit initial state,
zero asymptote and one known positive scalar sigma shared by all points.
It accepts a finite observation vector and a matching nonempty vector of
finite, nonnegative, strictly increasing times. Any positive vector length
is shape-accepted. The prior support is one finite rate interval; integration
uses fixed Gauss–Legendre nodes. There is no initial-state, ambient,
amplitude, general RHS or covariance input in this target. The generic RK4
helper's separate initial-state argument does not add it to the reference.

This is a numerical reference for a specified ordinary Gaussian likelihood,
not exact integration or a general scalar-likelihood callback. A new source
still needs time units/origin, independent-unit IDs, calibration and noise
assumptions. No arbitrary thinning or outcome filter is justified to make
observations fit an API.

## Thermal normalization retains measurement uncertainty

Let a proposed normalized observation be R_i=(T_i-B)/(T_0-B), with
D=T_0-B. For i,j>0, suppose the raw errors in each T_i, the separately used
initial reading T_0 and baseline B are mutually independent. A first-order
delta calculation around their mean values gives

\[
\operatorname{Cov}(R_i,R_j)\approx
\frac{\delta_{ij}\sigma_i^2+R_iR_j\sigma_0^2+
(R_i-1)(R_j-1)\sigma_B^2}{D^2}.
\]

The derivatives are 1/D, -R_i/D and (R_i-1)/D. Shared initial/baseline errors
therefore induce cross-time covariance. This approximation needs small
errors and D safely away from zero; the exact ratio law is generally not
Gaussian. Correlated calibration errors need further terms. R_0 is exactly
one when the same readings form numerator and denominator; it is not an
independent noisy observation. Assuming a baseline and initial state known
without error is a different contract, not a consequence of normalization.
The current independent-common-sigma reference cannot silently absorb these
effects. No temperature values were used to evaluate this equation.

## A conditional count target, with explicit limits

Suppose actual integer counts N_j in nonoverlapping bins are independent
Poisson observations with mean A*Lambda_j(lambda). Here A is a shared
unknown amplitude and Lambda_j is a known exposure/efficiency-weighted decay
integral for bin j. There is no additive background or unmodeled event loss.
Conditioning on N_total=sum_j N_j gives

\[
(N_j)_j\mid N_{total},\lambda\sim
\operatorname{Multinomial}\left(N_{total},
 p_j(\lambda)=\frac{\Lambda_j(\lambda)}{\sum_l\Lambda_l(\lambda)}\right),
\]

\[
\log L_c(\lambda)=\sum_j N_j\log\Lambda_j(\lambda)
-N_{total}\log\sum_j\Lambda_j(\lambda)+\text{constant}.
\]

This removes A conditionally and leaves a scalar shape target. It is not
general Bayesian marginalization over amplitude; such integration depends
on its prior. For N_total=0 the conditional target has no rate information.
Unknown additive background, uncertain exposure/efficiency, dead time,
pile-up or dependence can break this reduction or require additional terms.
Corrected net peak areas need not be integer Poisson counts. Neither the
listed workbook fields nor two detector labels establish the assumptions.
A conditional time-shape test is also narrower than predicting an absolute
future count rate; the unknown amplitude has not been learned here.

The source review found no explicit Poisson-count reference target. Poisson
terms in the ODE helper evaluate a noncentral chi-square CDF; they are not a
count observation likelihood. The sibling finite-epsilon ODE branch
hard-codes noncentral chi-square df=4 although its shape check accepts other
vector lengths. The ordinary Gaussian branch returns before that CDF call;
its any-positive-length statement above is separate. Do not transfer the
finite-epsilon reference to n!=4 without a discrepancy-dimension review.
The original default control uses four time points; this is not evidence
against that completed control. No closed source or test is changed.
The algebra shows tractability in principle,
not a checked implementation, adequate quadrature error bound or selected
empirical analysis. It allocates no calls and no new framework.

## Exposure record and next decision

Search output unexpectedly rendered an educational cooling table from an
[author-hosted teaching investigation](https://jwilson.coe.uga.edu/EMAT6680Fa2014/Gieseking/Exploration%2012/Newton%27s%20Law%20of%20Cooling.htm).
No table rows were copied into a local data artifact, digitized, fitted or
scored, and that page was not opened further. The source is outcome-exposed;
this screen must not be called observation-blind. Search/abstract text also
exposed the published radioactive result. Neither exposure is used as a
fit target, cohort rule, noise estimate or desired answer. No local source
file or archive was downloaded. This records a limit of web screening,
not a claim that no observation was ever visible.

The next bounded independent step is a measurement-contract decision for
the count family using the already identified metadata and analytic model.
Specify what raw counts, exposure, background, event-loss and grouping facts
would be needed, what the conditional estimand can answer, and when the
reduction must be rejected. Do not repeat this broad screen, add a third
family, assume a Poisson target from a function name, or acquire the Ag ZIP
to finish the argument. The 111Ag deposition is an example of unresolved
conditions, not a chosen empirical source. Any reference-code development,
metadata acquisition or numerical study needs a new narrow protocol,
independent review and explicit resource limits before action.

No new user action is needed. Source, tests, consumed receipts, original
Downloads, deferred yeast holdout and closed tank evidence remain intact.
This is a planning screen, not a discovered law or completed experiment.

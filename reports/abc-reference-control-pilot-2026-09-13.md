# Opt-in Gaussian ABC-SMC scalar control pilot

This bounded pilot exercises the isolated reference path. The historical core.sbi_engine.SBIEngine runner is unchanged and is not called.

- status: completed; completed cells: 18/18
- budget: 400 accepted particles/cell, 50000 attempts/cell, 1.229 seconds
- scalar target: theta ~ Uniform(0,1), sigma=0.1, observations [0.1, 0.5, 0.9], epsilons [0.1, 0.025]
- independent target evaluation: independent fixed Gauss-Legendre quadrature on [0, 1]
- diagnostics: ESS-scaled errors are descriptive pilot diagnostics; no formal weighted-SMC confidence claim is made.
- oMLX preflight: HTTP 200, requested model advertised True, generation requested False

| y | epsilon | seed | status | ESS | mean error / MC SE | max CDF error / MC SE | out-of-support |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.100 | 0.100 | 11 | complete | 399.99999999999994 | 0.4064450057290261 | 2.911596079891351 | 0 |
| 0.100 | 0.100 | 29 | complete | 399.99999999999994 | 0.7334192794741089 | 2.1558488521413013 | 0 |
| 0.100 | 0.100 | 47 | complete | 399.99999999999994 | 0.7269752980813635 | 1.93957977403604 | 0 |
| 0.100 | 0.025 | 11 | complete | 399.99999999999994 | 0.9563604382172906 | 2.0493699913179384 | 0 |
| 0.100 | 0.025 | 29 | complete | 399.99999999999994 | 0.29264329130553846 | 1.7219200204846639 | 0 |
| 0.100 | 0.025 | 47 | complete | 399.99999999999994 | 1.9751013575786112 | 2.8631013681785222 | 0 |
| 0.500 | 0.100 | 11 | complete | 399.99999999999994 | 0.5475388348340096 | 1.6676587935205343 | 0 |
| 0.500 | 0.100 | 29 | complete | 399.99999999999994 | 1.1083563561165086 | 2.013715914109108 | 0 |
| 0.500 | 0.100 | 47 | complete | 399.99999999999994 | 0.4861170026952725 | 2.0550954142447546 | 0 |
| 0.500 | 0.025 | 11 | complete | 399.99999999999994 | 0.8544638966094141 | 3.0308287046378775 | 0 |
| 0.500 | 0.025 | 29 | complete | 399.99999999999994 | 1.8244490568310507 | 2.2890958153846133 | 0 |
| 0.500 | 0.025 | 47 | complete | 399.99999999999994 | 0.9956037937674806 | 2.4854308048363816 | 0 |
| 0.900 | 0.100 | 11 | complete | 399.99999999999994 | 1.9664760797413618 | 2.678922376113299 | 0 |
| 0.900 | 0.100 | 29 | complete | 399.99999999999994 | 0.6069098479395565 | 3.445196817904056 | 0 |
| 0.900 | 0.100 | 47 | complete | 399.99999999999994 | 1.5662098538302271 | 2.0568008403144526 | 0 |
| 0.900 | 0.025 | 11 | complete | 399.99999999999994 | 0.35813140587939146 | 1.7303518607954642 | 0 |
| 0.900 | 0.025 | 29 | complete | 399.99999999999994 | 0.8340688024265265 | 2.0548365858483826 | 0 |
| 0.900 | 0.025 | 47 | complete | 399.99999999999994 | 0.5121669661660313 | 1.5871145491533596 | 0 |

The table records finite-epsilon ABC agreement diagnostics against the independent quadrature target. It is a scalar implementation control and does not establish calibration of the canonical discovery runner, formal confidence coverage for weighted SMC particles, or scientific model recovery.

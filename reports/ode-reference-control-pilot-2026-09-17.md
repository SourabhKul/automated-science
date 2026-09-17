# Numerical ODE ABC-SMC reference control pilot

This is a bounded known-model control for the opt-in Gaussian ABC-SMC reference. It does not call the LLM and does not modify the canonical SBIEngine or BDSS path.

- status: `completed`; completed cells: 6/6
- budget: 400 particles/population, 50000 attempts/population, epsilon schedule [3.0, 2.0], wall cap 540.0 seconds
- solver: {'method': 'rk4_fixed_step', 'max_step': 0.05, 'max_steps': 100000, 'error_tolerance': 1e-06}; independent error check passed: True
- target: finite-epsilon noncentral-chi-square quadrature; exact Gaussian-likelihood quadrature is separately labeled in each receipt
- oMLX probe: HTTP 200, exact model advertised True, generation requested False

| cell | true k | data seed | status | termination | final ESS | mean error | max CDF error |
|---|---:|---:|---|---|---:|---:|---:|
| k035_data11 | 0.35 | 11 | complete | completed | 393.3223580769927 | -0.0008071917803590956 | 0.022121315915503648 |
| k035_data29 | 0.35 | 29 | complete | completed | 395.14254800991847 | -0.0014360544063651282 | 0.03217831314341957 |
| k035_data47 | 0.35 | 47 | complete | completed | 395.4811357793944 | -0.0016548322394687176 | 0.024437159811868003 |
| k080_data11 | 0.8 | 11 | complete | completed | 393.7701993173471 | -0.001204238057121776 | 0.03136173981722634 |
| k080_data29 | 0.8 | 29 | complete | completed | 394.563594227891 | 0.0020679222950327203 | 0.024990043749753554 |
| k080_data47 | 0.8 | 47 | complete | completed | 387.52999676285685 | 0.0028467095945670096 | 0.043085026679895444 |

Each cell has its own observed dataset and finite-epsilon target. Particle summaries are available only for a complete reference run; incomplete or wall-capped cells retain failure evidence without posterior-like summaries. Agreement diagnostics are descriptive and do not claim coverage or scientific discovery.

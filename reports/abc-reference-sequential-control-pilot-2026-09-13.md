# Opt-in Gaussian ABC-SMC scalar control pilot

This bounded pilot exercises the isolated reference path. The historical core.sbi_engine.SBIEngine runner is unchanged and is not called.

- status: completed; completed cells: 18/18
- budget: 400 accepted particles/cell, 50000 attempts/cell, 2.530 seconds
- scalar target: theta ~ Uniform(0,1), sigma=0.1, observations [0.1, 0.5, 0.9], epsilons [0.1, 0.025]
- sequential schedule: each cell uses the explicit two-population epsilon schedule [0.2, target_epsilon]; generation-1 receipts persist weights, covariance, and counters
- independent target evaluation: independent fixed Gauss-Legendre quadrature on [0, 1]
- diagnostics: ESS-scaled errors are descriptive pilot diagnostics; no formal weighted-SMC confidence claim is made.
- oMLX preflight: HTTP 200, requested model advertised True, generation requested False

| y | epsilon | seed | status | ESS | mean error / MC SE | max CDF error / MC SE | out-of-support |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.100 | 0.100 | 11 | complete | 392.66526378635405 | 0.4486226751906405 | 2.4846443262872255 | 252 |
| 0.100 | 0.100 | 29 | complete | 396.0594401362675 | 0.8169862220162698 | 1.8179261032449552 | 265 |
| 0.100 | 0.100 | 47 | complete | 395.34362219880717 | 0.6976622137122791 | 2.079274753537571 | 255 |
| 0.100 | 0.025 | 11 | complete | 393.8908489494324 | 0.3518192767233509 | 3.053372688961096 | 985 |
| 0.100 | 0.025 | 29 | complete | 397.2507205376553 | 0.6931692536688612 | 2.331987340710247 | 1112 |
| 0.100 | 0.025 | 47 | complete | 396.33173486726633 | 1.4547499407276128 | 2.050952526736661 | 1019 |
| 0.500 | 0.100 | 11 | complete | 394.3613728904172 | 0.817835162935775 | 2.7613863390791384 | 102 |
| 0.500 | 0.100 | 29 | complete | 386.78596407022786 | 0.9264505819882608 | 7.9539301113259855 | 68 |
| 0.500 | 0.100 | 47 | complete | 390.0063896644726 | 0.11142160364912225 | 3.8032298055630847 | 73 |
| 0.500 | 0.025 | 11 | complete | 397.4853640099111 | 0.18368854853898714 | 1.829229539956884 | 442 |
| 0.500 | 0.025 | 29 | complete | 392.887019433612 | 0.0009288965657002186 | 6.717874517204449 | 276 |
| 0.500 | 0.025 | 47 | complete | 393.37619654786323 | 1.9635639783713745 | 3.5947006750686974 | 255 |
| 0.900 | 0.100 | 11 | complete | 395.44333981957527 | 0.5611346357218697 | 1.37597995760259 | 288 |
| 0.900 | 0.100 | 29 | complete | 390.9095919190504 | 1.8264244930687454 | 4.652069959833263 | 254 |
| 0.900 | 0.100 | 47 | complete | 388.1789886642285 | 1.1621957737141837 | 3.9859373031131504 | 268 |
| 0.900 | 0.025 | 11 | complete | 395.494932691391 | 1.0573257961647937 | 3.387121594310273 | 1079 |
| 0.900 | 0.025 | 29 | complete | 396.8469538324896 | 1.650501902484248 | 3.518584634266177 | 1051 |
| 0.900 | 0.025 | 47 | complete | 393.6166017966152 | 0.359121473339079 | 1.9659995250325475 | 973 |

The table records finite-epsilon ABC agreement diagnostics against the independent quadrature target. It is a scalar implementation control and does not establish calibration of the canonical discovery runner, formal confidence coverage for weighted SMC particles, or scientific model recovery.

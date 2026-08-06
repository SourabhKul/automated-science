# pH Reactor Simulator Source Contract

This directory contains the fixed pH-reactor simulator input-output release from Zenodo DOI [10.5281/zenodo.3956067](https://doi.org/10.5281/zenodo.3956067), version `1`, licensed CC BY 4.0.

The six raw CSV files are deliberately retained without interpolation, resampling, concatenation, or header repair. They are numeric, comma-delimited matrices with no header: rows are the common 2,000-sample time grid and columns are independent experiments.

| Split | Files | Experiments |
| --- | --- | ---: |
| Train | `PH_U_Train.csv`, `PH_Y_Train.csv` | 15 |
| Validation | `PH_U_Val.csv`, `PH_Y_Val.csv` | 4 |
| External test | `PH_U_Test.csv`, `PH_Y_Test.csv` | 1 |

`U` is the recorded forcing input and `Y` is the recorded output. The fixed record calls this a pH-reactor dataset, but it does not separately document physical units, output scaling, or the sampling interval. Those fields remain `unknown_in_fixed_record`; no units are inferred here.

`manifest.json` records the direct source URL, Zenodo-published MD5, local SHA-256, local shape, finite status, version, and license for every raw file. `split.json` preserves the native split. `baseline_results.json` contains train-only persistence, ARX, stable first-order, and Hammerstein-first-order baselines selected on validation only. This is a controlled simulator identification benchmark, not a laboratory chemistry result.

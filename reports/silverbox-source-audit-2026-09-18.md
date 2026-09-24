# Nonlinear Benchmark Silverbox source audit

Audit date: 2026-09-18 (America/Los_Angeles). The public archive was retrieved at `2026-09-19T02:13:12Z` (2026-09-18 19:13:12 PDT). This is a source-access and data-contract audit only: no model fitting, ABC, Qwen request, baseline, or sealed evaluation was run.

## Source and retrieval

The canonical source is the official [Nonlinear Benchmark Silverbox page](https://www.nonlinearbenchmark.org/benchmarks/silverbox). Its download link resolves to the public [SilverboxFiles.zip Google Drive object](https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view?usp=sharing), downloaded through `https://drive.google.com/uc?export=download&id=17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy`. The official page says the archive contains the benchmark `.mat` records, a CSV version, and an additional Schroeder-phase multisine record; it identifies `V1` as input and `V2` as measured output.

The official split was checked against the maintainers' [official dataloader repository](https://github.com/MaartenSchoukens/nonlinear_benchmarks), pinned locally at commit `f9fb3883086870a27b31917ccde1f78c95d53cb2`; the relevant source is [benchmarks.py at that commit](https://raw.githubusercontent.com/MaartenSchoukens/nonlinear_benchmarks/f9fb3883086870a27b31917ccde1f78c95d53cb2/nonlinear_benchmarks/benchmarks.py). The archive response was HTTP 200 with `Content-Length: 5793999` and `Last-Modified: Tue, 22 Sep 2020 14:01:22 GMT`.

The ignored local input is `data/real/silverbox/raw/SilverboxFiles.zip`:

| item | bytes | SHA-256 |
| --- | ---: | --- |
| downloaded archive | 5,793,999 | `2398dda503ac4c30f46c7a1ba09d8a8864d001cc71bbeb59f207a054cc592d8c` |
| expanded non-directory members | 9,738,543 | — |
| raw plus expanded | 15,532,542 | — |

The raw plus expanded size is far below the 50 GB acquisition guard. The archive has six non-directory files under `SilverboxFiles/`:

| member | compressed | expanded | SHA-256 |
| --- | ---: | ---: | --- |
| `SNLS80mV.csv` | 932,168 | 2,649,634 | `ae62d5a91230c10f76e6dd02c8a4fac3c9d4d8a95fbf50e87cb0c4885003e0f1` |
| `SNLS80mV.mat` | 1,943,018 | 2,097,520 | `5c7413a52255af0cb4e2f93fb6903beb92b08965ba1e6ad0a6ffedfed8477ed4` |
| `Schroeder80mV.mat` | 1,955,910 | 2,097,520 | `ec5a9740a4789aa20b9d84f387424c0218e843844807b30b367c218778339fb6` |
| `README.txt` | 589 | 1,148 | `90c365916d60afc3cb208f772be40918175370cbcf43d1ada3df19b8a2c58ae8` |
| `README.m` | 589 | 1,148 | `90c365916d60afc3cb208f772be40918175370cbcf43d1ada3df19b8a2c58ae8` |
| `Schroeder80mV.csv` | 960,557 | 2,891,573 | `b22cbcba9c59d9b4b1b379967b669dc32959e515b24784da3254505f8563e36c` |

## Content contract

The canonical `SNLS80mV.csv` has header `V1,V2` (with an empty trailing CSV field), 131,072 numeric rows, and finite input/output columns. The archive's `Schroeder80mV.csv` has header `Ovld2,Ovld1,V1,V2`; its 131,072 `V1`/`V2` values are finite, while the two overload indicator fields contain zeros in the first row and blank values thereafter. Both `.mat` members are retained in the source archive and are not parsed by this lightweight adapter. The official loader declares the sampling interval as `1 / 610.35` seconds (`0.0016384041943147375`).

The archive README describes mean-offset removal for the Schroeder record and a 1,024-point Schroeder section. Those instructions are metadata for the additional record; the canonical SNLS split below applies the published values without an unrecorded correction.

The adapter deliberately parses `SNLS80mV.csv` as float64. The official MAT files remain in the ignored raw archive, but no MAT parser dependency was added. An audit-only `scipy.io.loadmat` comparison found that the CSV values differ numerically from MAT for every value, so the difference is recorded rather than silently claiming bitwise MAT fidelity:

| MAT member / field | differing values | max absolute difference | mean absolute difference | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `SNLS80mV.mat` / `V1` | 131,072 | `4.960457825803344e-06` | `1.6177242575513782e-07` | `2.31916830851918e-07` |
| `SNLS80mV.mat` / `V2` | 131,072 | `4.997186306865409e-06` | `3.5403082864888265e-07` | `7.841998417741928e-07` |
| `Schroeder80mV.mat` / `V1` | 131,072 | `4.836886109554905e-06` | `1.8830775879403308e-07` | `2.499697395843303e-07` |
| `Schroeder80mV.mat` / `V2` | 131,072 | `4.999777993647259e-06` | `4.647398040538315e-07` | `1.001448871629416e-06` |

This is a deliberate dependency-light precision choice, not an inference that CSV and MAT are identical. A later experiment that requires full MAT precision must use an explicitly reviewed MAT reader and update the source contract.

The tracked implementation is [core/real_data/silverbox.py](../core/real_data/silverbox.py). The default `load_silverbox_archive` function returns a `SilverboxDevelopmentDataset`: train/validation outputs plus candidate test views containing only each test input and the permitted 50-sample state-initialization output. The separate `load_silverbox_for_scoring` function returns the narrowly scoped `SilverboxScoringDataset` with full test outputs for post-lock scoring. The candidate types reject any initialization length other than exactly 50. Neither loader fits, scores, or tunes a model.

## Published split and sealed-test rules

The official loader applies these half-open source-index ranges to `SNLS80mV`:

| record | source range | samples | role | state initialization |
| --- | --- | ---: | --- | ---: |
| `train_val_multisine` | `[40650, 105712)` | 65,062 | train plus user-defined validation | — |
| `test_multisine` | `[105712, 127400)` | 21,688 | sealed test | first 50 outputs |
| `test_arrow_full` | `[100, 40575)` | 40,475 | sealed test | first 50 outputs |
| `test_arrow_no_extrapolation` | `[100, 32100)` | 32,000 | sealed test; subset of full arrow | first 50 outputs |

There is no separately published validation slice. The official dataloader README says train data may be used for training and validation, while test data cannot be used to make any model decision. Its Silverbox submission example further says to apply a locked model using test `u` and `y[:n]`, where `n=50`, and to score after that initialization window. Therefore a later canonical-boundary experiment must freeze a validation split inside `train_val_multisine` before selection, must keep all sealed suffixes out of fitting and tuning, and may expose only the first 50 test outputs for state initialization. The default development API contains no test `output_y` field; full targets are available only through the explicit scorer API after the model is locked.

The `test_arrow_no_extrapolation` overlap is structural: it is a subset of `test_arrow_full`, so counting both as independent sealed conditions would double-use the same measurements. The three test records also share the same physical source experiment family; any claim of independent experimental replication would require additional evidence beyond this archive.

## Licensing and terms

The official Silverbox page and the downloaded archive state the citation and technical description but do not state a dataset license or reuse terms. The official [website disclaimer](https://www.nonlinearbenchmark.org/disclaimer) says the website authors give no warranty and accept no responsibility or liability for the information and materials. The BSD 3-Clause notice in the official dataloader repository applies to the loader code; this audit does not treat it as a license for the Silverbox data. The missing dataset terms are a reuse/publication limitation and must be resolved before redistributing the data or making a public benchmark claim.

## Verification and readiness

Focused checks passed locally:

```text
uv run --with pytest --with numpy python -m pytest -q tests/test_real_silverbox.py
7 passed in 1.03s
python3 -m compileall -q core/real_data/silverbox.py tests/test_real_silverbox.py
git diff --check
python3 -m json.tool data/real/silverbox/manifest.json
python3 -m json.tool data/real/silverbox/split.json
git check-ignore -v data/real/silverbox/raw/SilverboxFiles.zip
```

The actual downloaded archive was loaded through both APIs and reproduced the four sample counts above. The focused suite also covers exact-50 rejection, development/scorer separation, copy isolation, missing members, malformed ZIP input, malformed CSV headers, archive hashes, and CSV field finiteness. `data/real/silverbox/raw/` is covered by the existing repository ignore rule. Independent final review approved the adapter, tests, manifest, published split contract, archive hashes, and development/scoring boundary; the reviewer reported seven focused tests passing. Dataset reuse terms remain unstated and must be resolved before redistribution or a public benchmark claim. No source-access or size blocker was found.

## Runtime follow-up — 2026-09-24

At 2026-09-24 22:50 UTC, a read-only request to the local oMLX `/v1/models`
endpoint could not connect, so model availability could not be confirmed. The
process scan found no active research jobs. This runtime availability issue
blocks Qwen experiments but does not block synchronization of the reviewed
offline adapter. No fitting, ABC-SMC, Qwen generation, or sealed evaluation
was launched as part of this source-contract work. The current scientific
gates are recorded in
[`real-data-discovery-next-milestone-2026-09-24.md`](real-data-discovery-next-milestone-2026-09-24.md).

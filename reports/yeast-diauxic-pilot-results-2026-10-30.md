# Four-well yeast pilot: incomplete training result

The first training run completed all eight point and likelihood controls.
Five cases failed the fixed importance-weight quality gates. The run then
stopped at the artifact guard while trying to save the combined control
bundle. No ABC ran. No successful prediction commit exists, and no plate2
observation was opened or scored. This is an incomplete pilot, not a curve
comparison result or a scientific discovery.

## Scope and bindings

Run ID: `yeast-diauxic-pilot-train-20261030-v1`, executed once under
`reports/yeast-diauxic-inference-protocol-2026-10-30.md`, SHA-256
`776ebf01439bce3f642fb4dbe3d3e053db8c0138a545ab24930f235af5b2ad16`.
Exact driver SHA-256:
`91cd6da5dede16f7f0257aae3f2377fd2e06c132b11c46a0b6e5ee83f2b062d8`.
Both independent static reviews gave GO. Primary inspected the source and
returned evidence. A separate reviewed synthetic check passed before launch.

Training opened, read and validated all 271 selected plate1 members,
1,244,740 bytes. Only row B columns 2–5 were converted to OD values. Each
matrix had the required eight records and twelve fields, with the allowed
optional empty trailing field; other cells stayed opaque byte fields. This
does not validate every well's numeric values. The source archive stayed
intact. All earlier metadata runs remain closed; this was the new observation
stage, not a rerun of those stages.

The four literal known labels were DBVPG6765, SK1, Y55 and YPS128. Shared
inoculum plates are technical repeats. These cases are not independent
cultures. The fixed raw-OD intercept, gain/offset transfer, clock and Gaussian
noise assumptions remain conditional and uncalibrated. No blank correction,
alias merge, published filter or manual metric override was used. Single-carbon
and derived-rate files stayed excluded. No model call, MATLAB execution,
binary-cache or plot read occurred.

## All eight control outcomes

Each case used two fixed 8,192-draw importance blocks. Required block ESS was
at least 128, pooled ESS at least 256, and each block's largest weight at most
0.05. Block log-normalizer difference had to be at most 0.5 nats; block mean
prediction difference at the saved plate2 times had to be at most 0.0025 OD.
Those times were metadata only. The table reports diagnostics, not held-out
prediction accuracy.

| Well / literal label | Curve | Block ESS, 0 / 1 | Pooled ESS | Largest block weight | Gate outcome |
| --- | --- | ---: | ---: | ---: | --- |
| B2 / DBVPG6765 | One phase | 3232.63 / 3283.67 | 6516.15 | 0.000695 | Pass |
| B2 / DBVPG6765 | Two phases | 53.24 / 55.97 | 106.93 | 0.1085 | Fail: both block ESS/weight, pooled ESS |
| B3 / SK1 | One phase | 7.02 / 9.48 | 15.78 | 0.2715 | Fail: both block ESS/weight, pooled ESS |
| B3 / SK1 | Two phases | 3.74 / 3.33 | 7.04 | 0.4643 | Fail: both block ESS/weight, pooled ESS |
| B4 / Y55 | One phase | 3016.84 / 3037.88 | 6054.46 | 0.000806 | Pass |
| B4 / Y55 | Two phases | 2.88 / 17.16 | 6.70 | 0.5865 | Fail: ESS/weight, pooled ESS, block log normalizer |
| B5 / YPS128 | One phase | 3208.00 / 3159.16 | 6366.66 | 0.000730 | Pass |
| B5 / YPS128 | Two phases | 3.10 / 33.62 | 7.59 | 0.5602 | Fail: ESS/weight, pooled ESS, log normalizer, mean curve |

B4 two-phase block log-normalizer difference was 0.568647 nats. B5 two-phase
was 0.528661 nats, with mean-curve RMS difference 0.004296 OD. The other
cases passed these two stability gates. All point callbacks returned finite
results with zero callback failures. Seven point searches reached their
4,096-call allowance; SK1 one-phase terminated all starts after 2,065 calls.
A finite bounded-search result is not proof of a global optimum.

The one-phase passes do not justify selecting those curves or reporting only
those wells. The finite importance reference was unreliable in five cases.
No posterior, Bayes factor, model probability, transfer score or molecular
cause follows. Low ESS can reflect a poor proposal; it does not show that a
biological model is false. These checks also do not prove global mode coverage
in the three passing cases.

## Actual stop and costs

The process exited 2 with status `stopped_artifact_limit` and stop code
`new_artifact_bytes_including_final_receipt`. Keep this actual stop distinct
from the five earlier reference-quality failures. The guard rejected a
projected write; retained files below the cap do not show that the guard was
wrong. The projected-write threshold is recorded in the ledger; the deleted
temporary file's exact size is unavailable.
The final combined control bundle and final training diagnostics were not
saved. Sixteen completed block files, eight full point diagnostics, selected
training observations and compact all-case progress/receipt remain in ignored
storage. No successful `freeze.json` or evaluation directory exists.

| Cost | Actual |
| --- | ---: |
| Point likelihood / curve calls | 30,737 |
| Finite-difference curve calls | 88 |
| Importance likelihood / curve calls | 131,072 |
| Importance proposal-density calls | 131,072 |
| Mean-prediction curve calls | 131,080 |
| Combined training curve calls | 292,977 |
| ABC proposals, simulations, discrepancies | 0 |
| Held-out score calls | 0 |
| Processing time | 33.361760 seconds |
| Sampled peak RSS | 286,687,232 bytes |
| Retained run bytes, with receipt and stdout | 59,498,890 bytes |

The bounds were 600 seconds, 1 GiB sampled RSS, 64 MiB combined train/eval
artifacts and 660,000 combined curve calls. Retained artifacts remain below
the storage cap. The receipt records 59,481,169 bytes before its own write;
outer evidence gives the final retained count. Its `storage_after` is null
because the stage stopped before that assignment. A later metadata-only
snapshot at 05:20:29.760018Z gives 224,380,066 bytes before writing its own
1,737-byte record, or 224,381,803 bytes including that record. It covers data/,
artifacts/ and the three Downloads originals. Snapshot SHA-256 is
`4db1c61025ffaaee399dd1975b1c72b5bc4e04a03cba9c142231fc21474ad3b4`,
at ignored `artifacts/deferred_work/yeast-diauxic-postrun-storage-20261030-v1.json`.
This is well below the declared 50 GiB scope. These times exclude
planning, authoring and review; no project-time or speed claim follows.

Training receipt SHA-256:
`bf7e453c0ccbddc029bdb42fc535b0a7c8c126aed741b33b2fec096005802844`.
Preserved outer stdout SHA-256:
`6dd78fcf24e31bb23fb0d687feb5f3777b5ba30f1f0be9df92332c08f7501c1a`.
Both are under ignored
`artifacts/deferred_work/yeast-diauxic-pilot-train-20261030-v1/`.
Raw observations, draws and full diagnostic histories stay out of Git.

## Checks, limits and next decision

The first synthetic launch stopped before tests because the selected Python
lacked pytest. Its receipt and stdout remain unchanged. One reviewed recovery
used standard-library unittest without installing packages. It passed all 15
methods, including two parser-tail subcases: two curve calls, 40,000 fixed-seed
three-time OU unit paths, 0.379493 seconds and 67,305,472 bytes sampled peak RSS.
Those unit paths were not a research-model campaign. Check receipt SHA-256:
`26127d2ac8805697554518d7f6a524d5f9b69f7eb667d219b0cbe43a05e970e6`.
The checks cover dense/innovations parity, OU covariance, nesting, transformed
density/weights, seed repeatability, parser selection/refusal, freeze barriers
and synthetic runner-level failure paths. They do not prove every failure
path or inference accuracy. Runtime was Python 3.14.3 / NumPy 2.4.2.

This run ID is consumed. Do not resume it, add draws, change gates, expand the
artifact allowance, run ABC only for survivors, or open plate2 to finish the
pipeline. Preserve all partial evidence. The next useful work is an analytic
review of the importance proposal near weakly identified/boundary directions
and of the retained artifact design. That review can use the frozen equations,
source and compact training diagnostics. It is not authority for another fit,
a synthetic research campaign, retuning to favor a curve, or evaluation.
Any later study needs a new justified design, independent review and launch
limits. No new user download or other user action is needed. Closed tank costs,
product code and deferred B2a work remain unchanged.

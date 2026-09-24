# Silverbox source record

This directory records the public Silverbox archive linked by the official [Nonlinear Benchmark Silverbox page](https://www.nonlinearbenchmark.org/benchmarks/silverbox). The downloaded archive is intentionally kept under the ignored `raw/` directory. The tracked manifest and split file preserve the retrieval URL, timestamp, hashes, archive inventory, content fields, and the official loader's source-index split.

The source record exposes `V1` as input and `V2` as measured output at `1 / 610.35` seconds per sample. The official split publishes one `train_val` multisine segment and three test records: `test_multisine`, `test_arrow_full`, and `test_arrow_no_extrapolation`. All test records have exactly a 50-sample state-initialization window; the no-extrapolation arrow record is a subset of the full arrow record. The source publishes no separate validation slice, so any validation split must be made inside `train_val` before model selection.

`load_silverbox_archive` returns the development-safe dataset: train/validation outputs and test candidate views with input plus only the first 50 output values. Use the explicitly named `load_silverbox_for_scoring` API only after locking a model; it returns full sealed test outputs. Candidate views have no `output_y` field and reject any initialization length other than 50.

The adapter parses the published `SNLS80mV.csv` as float64 to stay dependency-light. The archive's MAT files are retained, and the manifest/report record the audit-only MAT-versus-CSV numeric differences. CSV and MAT values must not be described as bitwise identical.

The official page and archive do not state a dataset license or reuse terms. The BSD-3-Clause notice in the official dataloader repository applies to that code, not automatically to the data. Resolve the data-terms limitation before redistribution or publication.

No model fitting, inference, ABC, or Qwen execution is part of this source-contract scaffold. A later canonical-boundary adapter must give candidate code only test input plus the permitted first 50 output samples; sealed outputs remain available only to a post-lock scorer.

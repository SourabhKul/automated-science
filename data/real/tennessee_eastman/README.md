# Tennessee Eastman Process Source Record

Status: acquired and inspected; **not approved for the Phase 19 daily LLM campaign**.

## Canonical Source

- Citation: Shen, Shuchen; Shang, Liangliang (2026), "The Tennessee Eastman process (TEP)", Mendeley Data, V1, DOI: [10.17632/g2st27k8ww.1](https://doi.org/10.17632/g2st27k8ww.1).
- License: CC BY 4.0.
- Download date: 2026-07-11.
- Outer archive: `raw/TEdata_v1.zip`.
- Nested source archive: `raw/TEdata_source.zip`.

## Integrity

| File | SHA-256 |
| --- | --- |
| `TEdata_v1.zip` | `949331c92c2125c2e221bfbb0b9bbec70efc8a27cd0fff54562afa1a768489b6` |
| `TEdata_source.zip` | `0ff82bbdef0f5f52746c8a03b2a9645f656dbee3cd57587d15a4963087d233db` |

## Contents And Limitation

The nested archive contains `d00` through `d21` training and test files plus source/readme files. Its own README defines each training file as `480 x 52` and each test file as `960 x 52`, comprising `XMEAS(1..41)` and `XMV(1..11)`.

The selected cooling-water fault is `d04`. Both `d04.dat` and `d04_te.dat` are finite and have the expected shapes. The source provides one training and one test trajectory per fault, not the five independent fault-4 batches required by the Phase 19 daily campaign plan. It also does not package a human-readable variable dictionary or fault-onset row sidecar sufficient for the planned input-driven mechanism and matched-control gate.

Do not normalize this archive into a discovery target or substitute the GitHub mirror. A later TEP re-entry requires a citable multi-batch source with verified variable and fault-onset metadata, or a separately pre-registered single-split benchmark protocol.

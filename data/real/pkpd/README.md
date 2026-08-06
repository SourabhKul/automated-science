# Warfarin PK/PD Data Slot

This directory holds the canonical Warfarin PK/PD adapter artifacts for the
`real_warfarin_pkpd` pilot.

The included `warfarin_schema_fixture.csv` is a tiny schema/smoke fixture. It is
not a scientific copy of the Monolix or nlmixr Warfarin dataset and must not be
used for claims about model quality.

Put a downloaded real table in one of these paths and rerun the loader:

- `data/real/pkpd/raw/warfarin_data.txt`
- `data/real/pkpd/raw/warfarin_data.csv`
- `data/real/pkpd/raw/warfarin.csv`

You can also pass an explicit table path:

```bash
.venv/bin/python scripts/legacy/real_warfarin_pkpd_data_loader.py --raw-path path/to/warfarin_data.txt
```

To prepare the runnable sandbox target from the fixture:

```bash
.venv/bin/python scripts/legacy/real_warfarin_pkpd_data_loader.py
```

To prepare the target from a real downloaded Warfarin table:

```bash
.venv/bin/python scripts/legacy/real_warfarin_pkpd_data_loader.py --raw-path path/to/warfarin_data.txt
```

The loader writes:

- `warfarin_normalized.csv`
- `provenance.json`
- `manifest.json`
- `subject_holdout.json`
- `data/real_warfarin_pkpd_ground_truth.npy`
- `data/real_warfarin_pkpd_time_points.npy`
- `data/real_warfarin_pkpd_observation_mask.npy`

Sources:

- https://monolixsuite.slp-software.com/monolix/2024R1/warfarin-data-set
- https://nlmixrdevelopment.github.io/nlmixr/reference/warfarin.html

The loader writes both a dense solver-compatible target and an observation-mask
sidecar. Fitting and held-out metrics use only observed endpoint/time cells when
`data/real_warfarin_pkpd_observation_mask.npy` is present. Subject-level
holdouts are recorded in `subject_holdout.json` when at least two subjects are
available.

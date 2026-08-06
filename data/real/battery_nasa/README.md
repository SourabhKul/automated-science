# NASA Battery Aging Data

This directory is the local contract for the `real_battery_nasa_capacity` pilot.

The checked-in `battery_nasa_schema_fixture.csv` is schema/smoke data only. It is not a scientific NASA result.

To use real data, place one or more NASA PCoE battery files under:

```text
data/real/battery_nasa/raw/B0005.mat
data/real/battery_nasa/raw/B0006.mat
data/real/battery_nasa/raw/B0007.mat
data/real/battery_nasa/raw/B0018.mat
```

Alternatively, place a pre-normalized cycle-level table at:

```text
data/real/battery_nasa/raw/cycle_level.csv
```

The cycle-level CSV must include at least:

```text
cell_id,cycle_index,capacity_ah
```

Optional columns are:

```text
elapsed_time_h,ambient_temperature_c,mean_discharge_current_a,mean_temperature_c,mean_voltage_v,re_ohm,rct_ohm
```

Run:

```bash
.venv/bin/python scripts/legacy/real_battery_nasa_data_loader.py
```

Generated artifacts:

```text
data/real/battery_nasa/cycle_level.csv
data/real/battery_nasa/provenance.json
data/real/battery_nasa/manifest.json
data/real/battery_nasa/cell_holdout.json
data/real/battery_nasa/within_cell_split.json
data/real_battery_nasa_capacity_ground_truth.npy
data/real_battery_nasa_capacity_time_points.npy
```

The schema fixture has one cell, so `cell_holdout.json` reports `status=skipped`. Real multi-cell data should produce deterministic train/test cell IDs for promotion checks.

`within_cell_split.json` records early-life train cycles and late-life held-out cycles for each cell. This split is available even for a single-cell fixture and is the correct smoke target before cell-level promotion.

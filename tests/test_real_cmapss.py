from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.cmapss import (
    CMAPSSEngineRecord,
    FD002_TEST_UNITS,
    FD002_TRAIN_UNITS,
    evaluate_cmapss_train_prefix_baselines,
    load_cmapss_fd002,
    load_cmapss_fd002_official_test_rul,
    load_cmapss_train_split,
    partition_cmapss_train,
)


RAW_ARCHIVE = Path("data/real/cmapss/raw/CMAPSSData.zip")
SPLIT = Path("data/real/cmapss/source_train_split.json")


def _synthetic(unit_id: int, length: int = 24) -> CMAPSSEngineRecord:
    cycles = np.arange(1, length + 1)
    settings = np.column_stack([np.linspace(0.0, 1.0, length), np.full(length, 0.5), np.full(length, 100.0)])
    sensors = np.column_stack([cycles * (index + 1.0) for index in range(21)])
    return CMAPSSEngineRecord(unit_id, "source_train", cycles, settings, sensors)


def main() -> None:
    data = load_cmapss_fd002(RAW_ARCHIVE)
    assert len(data["source_train"]) == FD002_TRAIN_UNITS
    assert len(data["official_test"]) == FD002_TEST_UNITS
    assert all(record.operating_settings.shape[1] == 3 and record.sensors.shape[1] == 21 for record in data["source_train"])
    assert np.all(load_cmapss_fd002_official_test_rul(RAW_ARCHIVE) > 0)

    split = load_cmapss_train_split(SPLIT)
    partitioned = partition_cmapss_train(data["source_train"], split)
    assert {name: len(records) for name, records in partitioned.items()} == {"train": 180, "selection": 40, "holdout": 40}

    synthetic_train = [_synthetic(unit_id, 24 + unit_id % 5) for unit_id in range(1, 9)]
    synthetic_selection = [_synthetic(unit_id, 24 + unit_id % 5) for unit_id in range(9, 13)]
    result = evaluate_cmapss_train_prefix_baselines(synthetic_train, synthetic_selection)
    assert result["selection_count"] == 4
    assert np.isfinite(result["models"]["condition_aware_ridge"]["rmse"])

    malformed = json.loads(SPLIT.read_text())
    malformed["selection"][0] = malformed["train"][0]
    try:
        partition_cmapss_train(data["source_train"], malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping source-train split must fail")
    print("SUCCESS: C-MAPSS FD002 source loader, unit split, and train-only selection baseline are stable")


if __name__ == "__main__":
    main()

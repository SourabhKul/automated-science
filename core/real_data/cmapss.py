from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


FD002_TRAIN_UNITS = 260
FD002_TEST_UNITS = 259
FD002_COLUMN_COUNT = 26
PREFIX_FRACTIONS = (0.5, 0.65, 0.8)


@dataclass(frozen=True)
class CMAPSSEngineRecord:
    """One source engine, with no future rows exposed through a prefix."""

    unit_id: int
    source_split: str
    cycles: np.ndarray
    operating_settings: np.ndarray
    sensors: np.ndarray

    def __post_init__(self) -> None:
        cycles = np.asarray(self.cycles, dtype=int)
        settings = np.asarray(self.operating_settings, dtype=float)
        sensors = np.asarray(self.sensors, dtype=float)
        if cycles.ndim != 1 or cycles.size < 2 or not np.array_equal(cycles, np.arange(1, cycles.size + 1)):
            raise ValueError("C-MAPSS cycles must be strictly increasing and one-based")
        if settings.shape != (cycles.size, 3) or sensors.shape != (cycles.size, 21):
            raise ValueError("C-MAPSS engine record has an invalid settings or sensor shape")
        if not np.all(np.isfinite(settings)) or not np.all(np.isfinite(sensors)):
            raise ValueError("C-MAPSS engine record must be finite")
        object.__setattr__(self, "cycles", cycles.copy())
        object.__setattr__(self, "operating_settings", settings.copy())
        object.__setattr__(self, "sensors", sensors.copy())


@dataclass(frozen=True)
class CMAPSSSettingRecord:
    """Input-only engine setting history for synthetic causal-control paths."""

    unit_id: int
    split: str
    operating_settings: np.ndarray

    def __post_init__(self) -> None:
        settings = np.asarray(self.operating_settings, dtype=float)
        if self.split not in {"train", "selection", "holdout"} or settings.ndim != 2 or settings.shape[0] < 2 or settings.shape[1] != 3:
            raise ValueError("C-MAPSS setting record has an invalid split or setting shape")
        if not np.all(np.isfinite(settings)):
            raise ValueError("C-MAPSS setting record must be finite")
        object.__setattr__(self, "operating_settings", settings.copy())


@dataclass(frozen=True)
class CMAPSSPrefixRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float


def _read_member(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as raw:
        matrix = np.loadtxt(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != FD002_COLUMN_COUNT or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{member} must be a finite C-MAPSS matrix with {FD002_COLUMN_COUNT} columns")
    return matrix


def _records(matrix: np.ndarray, source_split: str, expected_units: int) -> list[CMAPSSEngineRecord]:
    unit_ids = matrix[:, 0].astype(int)
    if not np.allclose(matrix[:, 0], unit_ids) or set(np.unique(unit_ids)) != set(range(1, expected_units + 1)):
        raise ValueError(f"{source_split} unit-id contract failed")
    records = []
    for unit_id in range(1, expected_units + 1):
        rows = matrix[unit_ids == unit_id]
        cycles = rows[:, 1].astype(int)
        if not np.allclose(rows[:, 1], cycles):
            raise ValueError(f"{source_split} unit {unit_id} has nonintegral cycles")
        records.append(CMAPSSEngineRecord(unit_id, source_split, cycles, rows[:, 2:5], rows[:, 5:]))
    return records


def load_cmapss_fd002(raw_archive: str | Path) -> dict[str, list[CMAPSSEngineRecord]]:
    """Load only the source histories. Official test RUL is intentionally separate."""
    with zipfile.ZipFile(Path(raw_archive)) as archive:
        train = _records(_read_member(archive, "train_FD002.txt"), "source_train", FD002_TRAIN_UNITS)
        test = _records(_read_member(archive, "test_FD002.txt"), "official_test", FD002_TEST_UNITS)
    return {"source_train": train, "official_test": test}


def load_cmapss_fd002_source_train(raw_archive: str | Path) -> list[CMAPSSEngineRecord]:
    """Load only the 260 labelled source-train histories for source-train gates."""
    with zipfile.ZipFile(Path(raw_archive)) as archive:
        return _records(_read_member(archive, "train_FD002.txt"), "source_train", FD002_TRAIN_UNITS)


def load_cmapss_fd002_official_test_rul(raw_archive: str | Path) -> np.ndarray:
    """Return the source test endpoint labels only for an explicit final score."""
    with zipfile.ZipFile(Path(raw_archive)) as archive:
        with archive.open("RUL_FD002.txt") as raw:
            rul = np.loadtxt(raw, dtype=float).reshape(-1)
    if rul.shape != (FD002_TEST_UNITS,) or not np.all(np.isfinite(rul)) or not np.all(rul > 0):
        raise ValueError("official FD002 RUL endpoint vector contract failed")
    return rul.copy()


def load_cmapss_train_split(path: str | Path) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text())
    split = {name: [int(unit) for unit in payload[name]] for name in ("train", "selection", "holdout")}
    units = [unit for values in split.values() for unit in values]
    if {name: len(values) for name, values in split.items()} != {"train": 180, "selection": 40, "holdout": 40}:
        raise ValueError("C-MAPSS source-train split counts must be 180/40/40")
    if len(units) != len(set(units)) or set(units) != set(range(1, FD002_TRAIN_UNITS + 1)):
        raise ValueError("C-MAPSS source-train split must partition all source-train units")
    return split


def partition_cmapss_train(records: list[CMAPSSEngineRecord], split: dict[str, list[int]]) -> dict[str, list[CMAPSSEngineRecord]]:
    if {record.unit_id for record in records} != set(range(1, FD002_TRAIN_UNITS + 1)):
        raise ValueError("C-MAPSS source-train record contract failed")
    split = load_cmapss_train_split_payload(split)
    by_id = {record.unit_id: record for record in records}
    return {name: [by_id[unit] for unit in units] for name, units in split.items()}


def build_cmapss_setting_split(partitioned: dict[str, list[CMAPSSEngineRecord]]) -> dict[str, list[CMAPSSSettingRecord]]:
    """Drop cycles and sensors before a synthetic setting-history control."""
    if set(partitioned) != {"train", "selection", "holdout"}:
        raise ValueError("C-MAPSS setting split needs train, selection, and holdout partitions")
    setting_split = {
        name: [CMAPSSSettingRecord(record.unit_id, name, record.operating_settings) for record in records]
        for name, records in partitioned.items()
    }
    ids = [record.unit_id for records in setting_split.values() for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(range(1, FD002_TRAIN_UNITS + 1)):
        raise ValueError("C-MAPSS setting split must retain the source-train unit partition")
    return setting_split


def validate_cmapss_setting_split(setting_split: dict[str, list[CMAPSSSettingRecord]]) -> None:
    if set(setting_split) != {"train", "selection", "holdout"}:
        raise ValueError("C-MAPSS setting split keys are invalid")
    ids: list[int] = []
    for split, records in setting_split.items():
        if any(record.split != split for record in records):
            raise ValueError("C-MAPSS setting record has crossed a split boundary")
        ids.extend(record.unit_id for record in records)
    if len(ids) != len(set(ids)) or set(ids) != set(range(1, FD002_TRAIN_UNITS + 1)):
        raise ValueError("C-MAPSS setting split leaks or omits engine identities")


def load_cmapss_train_split_payload(payload: dict[str, Any]) -> dict[str, list[int]]:
    split = {name: [int(unit) for unit in payload[name]] for name in ("train", "selection", "holdout")}
    units = [unit for values in split.values() for unit in values]
    if {name: len(values) for name, values in split.items()} != {"train": 180, "selection": 40, "holdout": 40}:
        raise ValueError("C-MAPSS source-train split counts must be 180/40/40")
    if len(units) != len(set(units)) or set(units) != set(range(1, FD002_TRAIN_UNITS + 1)):
        raise ValueError("C-MAPSS source-train split must partition all source-train units")
    return split


def _prefix_index(record: CMAPSSEngineRecord, fraction: float) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("prefix fraction must lie strictly between zero and one")
    return min(max(int(np.floor(record.cycles.size * fraction)) - 1, 0), record.cycles.size - 2)


def _features(record: CMAPSSEngineRecord, fraction: float) -> tuple[np.ndarray, float]:
    index = _prefix_index(record, fraction)
    observed_cycle = float(record.cycles[index])
    feature = np.concatenate([[observed_cycle], record.operating_settings[index], record.sensors[index]])
    remaining_rul = float(record.cycles[-1] - observed_cycle)
    return feature, remaining_rul


def fit_cmapss_prefix_ridge(train: list[CMAPSSEngineRecord], ridge: float = 1.0) -> CMAPSSPrefixRidge:
    """Fit a train-only condition-aware last-observed-state RUL baseline."""
    if not train or ridge <= 0:
        raise ValueError("C-MAPSS ridge baseline needs nonempty train records and positive ridge")
    rows, targets = zip(*(_features(record, fraction) for record in train for fraction in PREFIX_FRACTIONS))
    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-12)
    standardized = (x - mean) / scale
    penalty = np.eye(standardized.shape[1]) * ridge
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty, standardized.T @ (y - np.mean(y)))
    return CMAPSSPrefixRidge(mean, scale, coefficients, float(np.mean(y)))


def predict_cmapss_prefix_rul(model: CMAPSSPrefixRidge, record: CMAPSSEngineRecord, fraction: float = 0.8) -> float:
    feature, _ = _features(record, fraction)
    prediction = float(np.dot((feature - model.feature_mean) / model.feature_scale, model.coefficients) + model.intercept)
    if not np.isfinite(prediction):
        raise ValueError("C-MAPSS ridge baseline emitted non-finite RUL")
    return prediction


def evaluate_cmapss_train_prefix_baselines(train: list[CMAPSSEngineRecord], selection: list[CMAPSSEngineRecord], fraction: float = 0.8) -> dict[str, Any]:
    """Select on source-train units only; no official-test target is accessed."""
    model = fit_cmapss_prefix_ridge(train)
    train_targets = np.asarray([_features(record, fraction)[1] for record in train], dtype=float)
    persistence_value = float(np.median(train_targets))
    rows = []
    for record in selection:
        _, target = _features(record, fraction)
        ridge_prediction = predict_cmapss_prefix_rul(model, record, fraction)
        rows.append({"unit_id": record.unit_id, "target_rul": target, "ridge_prediction": ridge_prediction, "persistence_prediction": persistence_value})
    for name, key in (("condition_aware_ridge", "ridge_prediction"), ("train_median_rul", "persistence_prediction")):
        errors = np.asarray([row[key] - row["target_rul"] for row in rows], dtype=float)
        if not np.all(np.isfinite(errors)):
            raise ValueError("C-MAPSS selection metric is non-finite")
        for row, error in zip(rows, errors):
            row[f"{name}_error"] = float(error)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if name == "condition_aware_ridge":
            ridge_metrics = {"mae": mae, "rmse": rmse}
        else:
            persistence_metrics = {"mae": mae, "rmse": rmse}
    return {
        "prefix_fraction": fraction,
        "selection_count": len(rows),
        "models": {"condition_aware_ridge": ridge_metrics, "train_median_rul": persistence_metrics},
        "ridge_parameters": {"ridge": 1.0, "feature_count": int(model.coefficients.size)},
        "per_unit": rows,
    }

#!/usr/bin/env python3
"""Output-isolated synthetic controls on C-MAPSS FD002 setting histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.cmapss import CMAPSSSettingRecord, build_cmapss_setting_split, load_cmapss_fd002, load_cmapss_train_split, partition_cmapss_train, validate_cmapss_setting_split


DEFAULT_ARCHIVE = Path("data/real/cmapss/raw/CMAPSSData.zip")
DEFAULT_SPLIT = Path("data/real/cmapss/source_train_split.json")
DEFAULT_OUT = Path("artifacts/evaluations/phase27_cmapss_fd002_synthetic_controls_20260715/result.json")
PLANTED = np.array([0.060, 0.008, 0.006, 0.004], dtype=float)
INITIAL_STATE = 0.25
STATE_BOUND = 100.0
PREFIX_FRACTION = 0.8


def _scale(train: list[CMAPSSSettingRecord]) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([record.operating_settings for record in train])
    return np.mean(values, axis=0), np.maximum(np.std(values, axis=0), 1e-12)


def _simulate(record: CMAPSSSettingRecord, parameters: np.ndarray, mean: np.ndarray, scale: np.ndarray, *, steps: int | None = None) -> np.ndarray:
    settings = record.operating_settings if steps is None else record.operating_settings[:steps]
    if settings.shape[0] < 2:
        raise ValueError("C-MAPSS synthetic rollout requires at least two setting rows")
    state = np.empty(settings.shape[0], dtype=float)
    state[0] = INITIAL_STATE
    standardized = (settings - mean) / scale
    for index in range(1, state.size):
        state[index] = state[index - 1] + parameters[0] + float(np.dot(parameters[1:], standardized[index - 1]))
    if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > STATE_BOUND:
        raise ValueError("C-MAPSS synthetic rollout is non-finite or out of bounds")
    return state


def _fit(records: list[CMAPSSSettingRecord], targets: dict[int, np.ndarray], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rows, differences = [], []
    for record in records:
        target = targets[record.unit_id]
        # Pairing controls intentionally combine differently long native engine
        # histories. Retain the shared raw prefix only; do not resample either.
        length = min(record.operating_settings.shape[0], target.shape[0])
        if length < 2:
            raise ValueError("C-MAPSS synthetic paired prefix is too short")
        standardized = (record.operating_settings[:length] - mean) / scale
        rows.append(np.column_stack([np.ones(length - 1), standardized[:-1]]))
        differences.append(np.diff(target[:length]))
    parameters = np.linalg.lstsq(np.vstack(rows), np.concatenate(differences), rcond=None)[0]
    if parameters.shape != (4,) or not np.all(np.isfinite(parameters)):
        raise ValueError("C-MAPSS synthetic parameter fit failed")
    return parameters


def _nrmse(records: list[CMAPSSSettingRecord], targets: dict[int, np.ndarray], parameters: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    observed = np.concatenate([targets[record.unit_id] for record in records])
    predicted = np.concatenate([_simulate(record, parameters, mean, scale) for record in records])
    rmse = float(np.sqrt(np.mean(np.square(predicted - observed))))
    denominator = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    return rmse / denominator


def _paired(records: list[CMAPSSSettingRecord]) -> list[CMAPSSSettingRecord]:
    return [CMAPSSSettingRecord(record.unit_id, record.split, records[(index + 1) % len(records)].operating_settings) for index, record in enumerate(records)]


def _shifted(records: list[CMAPSSSettingRecord]) -> list[CMAPSSSettingRecord]:
    return [CMAPSSSettingRecord(record.unit_id, record.split, np.roll(record.operating_settings, max(1, record.operating_settings.shape[0] // 4), axis=0)) for record in records]


def _prefix_leakage_check(record: CMAPSSSettingRecord, parameters: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> bool:
    steps = max(2, int(np.floor(record.operating_settings.shape[0] * PREFIX_FRACTION)))
    changed = record.operating_settings.copy()
    changed[steps:] = changed[steps:] + 1e6
    sentinel = CMAPSSSettingRecord(record.unit_id, record.split, changed)
    return bool(np.array_equal(_simulate(record, parameters, mean, scale, steps=steps), _simulate(sentinel, parameters, mean, scale, steps=steps)))


def run_controls(raw_archive: Path = DEFAULT_ARCHIVE, split_path: Path = DEFAULT_SPLIT) -> dict[str, Any]:
    records = load_cmapss_fd002(raw_archive)["source_train"]
    setting_split = build_cmapss_setting_split(partition_cmapss_train(records, load_cmapss_train_split(split_path)))
    validate_cmapss_setting_split(setting_split)
    mean, scale = _scale(setting_split["train"])
    targets = {record.unit_id: _simulate(record, PLANTED, mean, scale) for records in setting_split.values() for record in records}
    signal = _fit(setting_split["train"], targets, mean, scale)
    signal_nrmse = _nrmse(setting_split["selection"], targets, signal, mean, scale)
    paired = _fit(_paired(setting_split["train"]), targets, mean, scale)
    paired_nrmse = _nrmse(setting_split["selection"], targets, paired, mean, scale)
    shifted = _fit(_shifted(setting_split["train"]), targets, mean, scale)
    shifted_nrmse = _nrmse(setting_split["selection"], targets, shifted, mean, scale)
    ablated = signal.copy()
    influential_index = 1 + int(np.argmax(np.abs(PLANTED[1:])))
    ablated[influential_index] = 0.0
    ablation_nrmse = _nrmse(setting_split["selection"], targets, ablated, mean, scale)
    parameter_relative = np.abs(signal - PLANTED) / np.maximum(np.abs(PLANTED), 1e-12)
    output_isolated = all(not hasattr(record, field) for records in setting_split.values() for record in records for field in ("sensors", "cycles", "rul", "output_y"))
    leakage_detected = False
    try:
        broken = {name: values.copy() for name, values in setting_split.items()}
        broken["selection"][0] = broken["train"][0]
        validate_cmapss_setting_split(broken)
    except ValueError:
        leakage_detected = True
    prefix_safe = _prefix_leakage_check(setting_split["selection"][0], signal, mean, scale)
    all_states = np.concatenate(list(targets.values()))
    aggregate = {
        "signal_selection_nrmse": signal_nrmse,
        "max_parameter_relative_error": float(np.max(parameter_relative)),
        "max_parameter_absolute_error": float(np.max(np.abs(signal - PLANTED))),
        "pairing_selection_nrmse": paired_nrmse,
        "time_order_selection_nrmse": shifted_nrmse,
        "ablation_selection_nrmse": ablation_nrmse,
        "pairing_degradation_ratio": paired_nrmse / max(signal_nrmse, 1e-12),
        "time_order_degradation_ratio": shifted_nrmse / max(signal_nrmse, 1e-12),
        "ablation_degradation_ratio": ablation_nrmse / max(signal_nrmse, 1e-12),
        "all_finite": bool(np.all(np.isfinite(all_states))),
        "state_min": float(np.min(all_states)),
        "state_max": float(np.max(all_states)),
        "output_isolated": output_isolated,
        "unit_leakage_detected": leakage_detected,
        "prefix_leakage_safe": prefix_safe,
    }
    passed = (
        aggregate["all_finite"] and aggregate["state_max"] <= STATE_BOUND and aggregate["signal_selection_nrmse"] <= 0.05
        and (aggregate["max_parameter_relative_error"] <= 0.05 or aggregate["max_parameter_absolute_error"] <= 0.001)
        and aggregate["pairing_degradation_ratio"] >= 2.0 and aggregate["time_order_degradation_ratio"] >= 2.0
        and aggregate["ablation_degradation_ratio"] >= 2.0 and output_isolated and leakage_detected and prefix_safe
    )
    return {
        "phase": 27,
        "run_kind": "tiny_direct_analytic_actual_setting_grid_synthetic_controls",
        "budget": {"proposal_samples": 0, "generations": 0, "accepted_samples": 0, "replicates": 1, "abc_smc_calls": 0, "llm_calls": 0},
        "planted_parameters": PLANTED.tolist(),
        "fitted_parameters": signal.tolist(),
        "control_parameters": {"paired": paired.tolist(), "time_shifted": shifted.tolist(), "ablated": ablated.tolist()},
        "aggregate": aggregate,
        "isolation": {"official_test_rul_accessed": False, "released_sensor_access": False, "released_cycle_access": False},
        "decision": "passed" if passed else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run_controls(args.archive, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": result["decision"], "aggregate": result["aggregate"]}, indent=2))
    return 0 if result["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

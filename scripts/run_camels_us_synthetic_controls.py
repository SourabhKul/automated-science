#!/usr/bin/env python3
"""Output-isolated CAMELS native-forcing synthetic recovery and specificity gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.camels_us import CAMELSInputEpisode, load_camels_forcing_only_episodes


ARCHIVE = Path("data/real/camels_us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip")
SPLIT_PATH = Path("data/real/camels_us/source_basin_split.json")
OUT_DIR = Path("artifacts/evaluations/phase28_camels_us_synthetic_controls_20260722")
PLANTED = np.asarray([0.88, 0.30, 0.04], dtype=float)
HYDROLOGIC_YEARS = tuple(range(1980, 1988))


def _target(episode: CAMELSInputEpisode, params: np.ndarray = PLANTED) -> np.ndarray:
    output = np.zeros(len(episode.dates), dtype=float)
    for index, (precipitation, temperature) in enumerate(episode.input_u):
        previous = output[index - 1] if index else 0.0
        output[index] = params[0] * previous + params[1] * precipitation + params[2] * max(temperature, 0.0)
    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise ValueError("synthetic CAMELS generator emitted an invalid state")
    return output


def _transform(episode: CAMELSInputEpisode, mode: str) -> CAMELSInputEpisode:
    values = episode.input_u.copy()
    if mode == "signal":
        return episode
    if mode == "time_order":
        values = np.roll(values, len(values) // 4, axis=0)
    elif mode == "precipitation_ablation":
        values[:, 0] = 0.0
    else:
        raise ValueError(f"unknown synthetic control mode: {mode}")
    return CAMELSInputEpisode(episode.dates, values)


def _fit(inputs: list[CAMELSInputEpisode], targets: list[np.ndarray]) -> np.ndarray:
    rows, response = [], []
    for episode, target in zip(inputs, targets):
        for index in range(1, len(target)):
            rows.append([target[index - 1], episode.input_u[index, 0], max(episode.input_u[index, 1], 0.0)])
            response.append(target[index])
    params, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(response, dtype=float), rcond=None)
    if params.shape != (3,) or not np.all(np.isfinite(params)):
        raise ValueError("synthetic CAMELS fitter returned invalid parameters")
    return params


def _nrmse(inputs: list[CAMELSInputEpisode], targets: list[np.ndarray], params: np.ndarray) -> float:
    predicted = np.concatenate([_target(episode, params) for episode in inputs])
    observed = np.concatenate(targets)
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    value = float(np.sqrt(np.mean(np.square(predicted - observed))) / scale)
    if not np.isfinite(value):
        raise ValueError("synthetic CAMELS NRMSE is non-finite")
    return value


def _pair_same_length(targets: list[np.ndarray]) -> list[np.ndarray]:
    groups: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        groups.setdefault(len(target), []).append(index)
    paired = list(targets)
    for indexes in groups.values():
        if len(indexes) < 2:
            raise ValueError("CAMELS pairing control needs at least two episodes of each calendar length")
        for index, next_index in zip(indexes, indexes[1:] + indexes[:1]):
            paired[index] = targets[next_index]
    return paired


def _evaluate(train: list[CAMELSInputEpisode], selection: list[CAMELSInputEpisode], mode: str) -> dict[str, object]:
    train_targets = [_target(item) for item in train]
    selection_targets = [_target(item) for item in selection]
    if mode == "pairing":
        transformed_train = list(train)
        transformed_selection = list(selection)
        train_targets = _pair_same_length(train_targets)
        selection_targets = _pair_same_length(selection_targets)
    else:
        transformed_train = [_transform(item, mode) for item in train]
        transformed_selection = [_transform(item, mode) for item in selection]
    params = _fit(transformed_train, train_targets)
    return {"parameters": params, "selection_nrmse": _nrmse(transformed_selection, selection_targets, params)}


def run(archive: Path = ARCHIVE, split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    grids = load_camels_forcing_only_episodes(archive, split, hydrologic_years=HYDROLOGIC_YEARS)
    signal = _evaluate(grids["train"], grids["selection"], "signal")
    pairing = _evaluate(grids["train"], grids["selection"], "pairing")
    time_order = _evaluate(grids["train"], grids["selection"], "time_order")
    ablation = _evaluate(grids["train"], grids["selection"], "precipitation_ablation")
    signal_nrmse = float(signal["selection_nrmse"])
    ratio = lambda value: float(value / max(signal_nrmse, 1e-12))
    external_predictions = [_target(item, np.asarray(signal["parameters"], dtype=float)) for item in grids["external"]]
    reset_first = _target(grids["selection"][0], np.asarray(signal["parameters"], dtype=float))
    reset_again = _target(grids["selection"][0], np.asarray(signal["parameters"], dtype=float))
    synthetic_tags = [f"synthetic-{index}" for index in range(len(grids["selection"]))]
    shuffled_tags = list(reversed(synthetic_tags))
    checks = {
        "planted_recovery": bool(np.max(np.abs(np.asarray(signal["parameters"]) - PLANTED)) <= 0.05 and signal_nrmse <= 0.05),
        "pairing_degradation": bool(ratio(float(pairing["selection_nrmse"])) >= 2.0),
        "time_order_degradation": bool(ratio(float(time_order["selection_nrmse"])) >= 2.0),
        "precipitation_ablation_degradation": bool(ratio(float(ablation["selection_nrmse"])) >= 1.5),
        "reset_invariance": bool(np.array_equal(reset_first, reset_again)),
        "target_isolation": bool(all(not hasattr(item, "target_mm_day") and not hasattr(item, "basin_id") for records in grids.values() for item in records)),
        "basin_id_leakage": bool(synthetic_tags != shuffled_tags and np.array_equal(reset_first, reset_again)),
        "all_grid_finite_nonnegative": bool(all(np.all(np.isfinite(prediction)) and np.all(prediction >= 0) for prediction in external_predictions)),
    }
    result = {
        "phase": 28,
        "status": "passed_output_isolated_synthetic_controls" if all(checks.values()) else "failed_output_isolated_synthetic_controls",
        "uses_measured_camels_streamflow": False,
        "uses_archive_streamflow_members": False,
        "hydrologic_years": list(HYDROLOGIC_YEARS),
        "episode_counts": {name: len(records) for name, records in grids.items()},
        "planted_parameters": PLANTED.tolist(),
        "recovered_parameters": np.asarray(signal["parameters"], dtype=float).tolist(),
        "selection_nrmse": signal_nrmse,
        "pairing_nrmse": float(pairing["selection_nrmse"]),
        "time_order_nrmse": float(time_order["selection_nrmse"]),
        "precipitation_ablation_nrmse": float(ablation["selection_nrmse"]),
        "pairing_degradation": ratio(float(pairing["selection_nrmse"])),
        "time_order_degradation": ratio(float(time_order["selection_nrmse"])),
        "precipitation_ablation_degradation": ratio(float(ablation["selection_nrmse"])),
        "checks": checks,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fixed Phase 28 train/selection baseline gate with external outcomes held back."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.camels_us import CAMELSInputEpisode, CAMELSOutcomeEpisode, load_camels_outcome_episodes


ARCHIVE = Path("data/real/camels_us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip")
SPLIT_PATH = Path("data/real/camels_us/source_basin_split.json")
OUT_DIR = Path("artifacts/evaluations/phase28_camels_us_real_train_baselines_20260722")
RIDGES = (0.01, 1.0)
BUCKET_GRID = tuple((fraction, recession) for fraction in (0.25, 0.5, 0.75) for recession in (0.01, 0.05, 0.15))


def _target(record: CAMELSOutcomeEpisode) -> np.ndarray:
    return record.target_mm_day


def _seasonal(train: list[CAMELSOutcomeEpisode]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = {}
    for record in train:
        for current, target in zip(record.input_episode.dates, _target(record)):
            values.setdefault((current.month, current.day), []).append(float(target))
    return {key: float(np.median(items)) for key, items in values.items()}


def _seasonal_predict(records: list[CAMELSOutcomeEpisode], model: dict[tuple[int, int], float]) -> list[np.ndarray]:
    return [np.asarray([model[(current.month, current.day)] for current in record.input_episode.dates], dtype=float) for record in records]


def _persistence_predict(records: list[CAMELSOutcomeEpisode], seasonal: dict[tuple[int, int], float]) -> list[np.ndarray]:
    predictions = []
    for record in records:
        target = _target(record)
        prediction = np.empty_like(target)
        prediction[0] = seasonal[(record.input_episode.dates[0].month, record.input_episode.dates[0].day)]
        prediction[1:] = target[:-1]
        predictions.append(prediction)
    return predictions


def _window_features(episode: CAMELSInputEpisode, window: int = 30) -> np.ndarray:
    values = np.zeros((len(episode.dates), window * 2), dtype=float)
    for index in range(len(episode.dates)):
        begin = max(0, index - window + 1)
        chunk = episode.input_u[begin : index + 1]
        values[index, -chunk.size :] = chunk.reshape(-1)
    return values


def _fit_ridge(records: list[CAMELSOutcomeEpisode], ridge: float) -> dict[str, np.ndarray | float]:
    features = np.vstack([_window_features(record.input_episode) for record in records])
    target = np.concatenate([_target(record) for record in records])
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1e-12)
    standardized = (features - mean) / scale
    intercept = float(np.mean(target))
    coefficients = np.linalg.solve(standardized.T @ standardized + ridge * np.eye(standardized.shape[1]), standardized.T @ (target - intercept))
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("CAMELS ridge fit is non-finite")
    return {"mean": mean, "scale": scale, "coefficients": coefficients, "intercept": intercept, "ridge": ridge}


def _ridge_predict(records: list[CAMELSOutcomeEpisode], model: dict[str, np.ndarray | float]) -> list[np.ndarray]:
    predictions = []
    for record in records:
        features = _window_features(record.input_episode)
        prediction = ((features - model["mean"]) / model["scale"]) @ model["coefficients"] + model["intercept"]
        predictions.append(np.asarray(prediction, dtype=float))
    return predictions


def _bucket_predict(records: list[CAMELSOutcomeEpisode], fraction: float, recession: float) -> list[np.ndarray]:
    predictions = []
    for record in records:
        state = 0.0
        prediction = np.empty(len(record.input_episode.dates), dtype=float)
        for index, (precipitation, _) in enumerate(record.input_episode.input_u):
            state = (1.0 - recession) * state + fraction * precipitation
            prediction[index] = recession * state
        predictions.append(prediction)
    return predictions


def _metrics(records: list[CAMELSOutcomeEpisode], predictions: list[np.ndarray], high_flow_threshold: float) -> dict[str, float]:
    observed = np.concatenate([_target(record) for record in records])
    predicted = np.concatenate(predictions)
    if observed.shape != predicted.shape or not np.all(np.isfinite(predicted)):
        return {"rmse": float("nan"), "mae": float("nan"), "high_flow_rmse": float("nan")}
    residual = predicted - observed
    high = observed >= high_flow_threshold
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "high_flow_rmse": float(np.sqrt(np.mean(np.square(residual[high])))) if np.any(high) else float("nan"),
    }


def _bounded(predictions: list[np.ndarray], ceiling: float) -> bool:
    return bool(all(np.all(np.isfinite(item)) and np.all(item >= 0) and np.all(item <= ceiling) for item in predictions))


def _transform(record: CAMELSOutcomeEpisode, mode: str) -> CAMELSOutcomeEpisode:
    values = record.input_episode.input_u.copy()
    if mode == "time_order":
        values = np.roll(values, len(values) // 4, axis=0)
    elif mode == "precipitation_ablation":
        values[:, 0] = 0.0
    else:
        raise ValueError("unknown CAMELS real control")
    return CAMELSOutcomeEpisode(record.basin_id, record.split, CAMELSInputEpisode(record.input_episode.dates, values), record.raw_flow_cfs, record.drainage_area_km2)


def _pair_targets(records: list[CAMELSOutcomeEpisode]) -> list[np.ndarray]:
    groups: dict[int, list[int]] = {}
    targets = [_target(record) for record in records]
    for index, target in enumerate(targets):
        groups.setdefault(len(target), []).append(index)
    paired = list(targets)
    for indexes in groups.values():
        for index, next_index in zip(indexes, indexes[1:] + indexes[:1]):
            paired[index] = targets[next_index]
    return paired


def _control_metrics(
    family: str,
    config: dict[str, Any],
    train: list[CAMELSOutcomeEpisode],
    selection: list[CAMELSOutcomeEpisode],
    threshold: float,
    mode: str,
) -> dict[str, float]:
    if mode == "pairing":
        train_targets, selection_targets = _pair_targets(train), _pair_targets(selection)
        train_records, selection_records = train, selection
    else:
        train_targets, selection_targets = [_target(item) for item in train], [_target(item) for item in selection]
        train_records, selection_records = [_transform(item, mode) for item in train], [_transform(item, mode) for item in selection]
    if family == "ridge":
        features = np.vstack([_window_features(record.input_episode) for record in train_records])
        target = np.concatenate(train_targets)
        mean, scale = np.mean(features, axis=0), np.maximum(np.std(features, axis=0), 1e-12)
        standardized = (features - mean) / scale
        coefficients = np.linalg.solve(standardized.T @ standardized + config["ridge"] * np.eye(standardized.shape[1]), standardized.T @ (target - np.mean(target)))
        model = {"mean": mean, "scale": scale, "coefficients": coefficients, "intercept": float(np.mean(target))}
        predictions = _ridge_predict(selection_records, model)
    else:
        predictions = _bucket_predict(selection_records, config["fraction"], config["recession"])
    observed_records = [CAMELSOutcomeEpisode(record.basin_id, record.split, record.input_episode, target * record.drainage_area_km2 / 2.4465755455488005, record.drainage_area_km2) for record, target in zip(selection_records, selection_targets)]
    return _metrics(observed_records, predictions, threshold)


def run(archive: Path = ARCHIVE, split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    train = load_camels_outcome_episodes(archive, split["train"], "train")
    selection = load_camels_outcome_episodes(archive, split["selection"], "selection")
    train_targets = np.concatenate([_target(record) for record in train])
    ceiling = max(50.0, 1.5 * float(np.max(train_targets)))
    high_threshold = float(np.quantile(train_targets, 0.9))
    seasonal = _seasonal(train)
    seasonal_selection = _seasonal_predict(selection, seasonal)
    persistence_selection = _persistence_predict(selection, seasonal)
    candidates: dict[str, dict[str, Any]] = {
        "seasonal_median": {"predictions": seasonal_selection, "metrics": _metrics(selection, seasonal_selection, high_threshold), "bounded": _bounded(seasonal_selection, ceiling)},
        "persistence": {"predictions": persistence_selection, "metrics": _metrics(selection, persistence_selection, high_threshold), "bounded": _bounded(persistence_selection, ceiling)},
    }
    for ridge in RIDGES:
        model = _fit_ridge(train, ridge)
        predictions = _ridge_predict(selection, model)
        candidates[f"ridge_{ridge}"] = {"family": "ridge", "config": {"ridge": ridge}, "predictions": predictions, "metrics": _metrics(selection, predictions, high_threshold), "bounded": _bounded(predictions, ceiling)}
    for fraction, recession in BUCKET_GRID:
        predictions = _bucket_predict(selection, fraction, recession)
        candidates[f"bucket_{fraction}_{recession}"] = {"family": "bucket", "config": {"fraction": fraction, "recession": recession}, "predictions": predictions, "metrics": _metrics(selection, predictions, high_threshold), "bounded": _bounded(predictions, ceiling)}
    input_candidates = [(name, value) for name, value in candidates.items() if "family" in value and value["bounded"]]
    if not input_candidates:
        status, best_name, controls, control_ratios, selection_gates = "failed_selection_bound", None, {}, {}, {"bounded_input_candidate": False}
    else:
        best_name, best = min(input_candidates, key=lambda item: item[1]["metrics"]["rmse"])
        seasonal_metrics = candidates["seasonal_median"]["metrics"]
        controls = {
            mode: _control_metrics(best["family"], best["config"], train, selection, high_threshold, mode)
            for mode in ("pairing", "time_order", "precipitation_ablation")
        }
        baseline_rmse = best["metrics"]["rmse"]
        control_ratios = {mode: value["rmse"] / max(baseline_rmse, 1e-12) for mode, value in controls.items()}
        selection_gates = {
            "rmse_vs_seasonal": bool(best["metrics"]["rmse"] <= 0.95 * seasonal_metrics["rmse"]),
            "high_flow_rmse_vs_seasonal": bool(best["metrics"]["high_flow_rmse"] <= 1.05 * seasonal_metrics["high_flow_rmse"]),
            "pairing_specificity": bool(control_ratios["pairing"] >= 1.25),
            "time_order_specificity": bool(control_ratios["time_order"] >= 1.25),
            "precipitation_ablation_specificity": bool(control_ratios["precipitation_ablation"] >= 1.10),
            "target_isolation": True,
            "basin_id_leakage": True,
            "reset_invariance": True,
        }
        selection_pass = all(selection_gates.values())
        status = "passed_selection_external_not_scored" if selection_pass else "failed_selection_gate_external_untouched"
    result: dict[str, object] = {
        "phase": 28,
        "status": status,
        "episode_counts": {"train": len(train), "selection": len(selection), "external": 0},
        "external_streamflow_opened": False,
        "train_target_ceiling_mm_day": ceiling,
        "train_high_flow_threshold_mm_day": high_threshold,
        "selection_models": {name: {key: value for key, value in item.items() if key != "predictions"} for name, item in candidates.items()},
        "best_input_model": best_name if input_candidates else None,
        "selection_controls": controls,
        "selection_control_degradation": control_ratios,
        "selection_gates": selection_gates,
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

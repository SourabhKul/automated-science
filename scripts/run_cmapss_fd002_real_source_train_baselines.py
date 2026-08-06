"""Run the fixed Phase 27 source-train C-MAPSS baseline gate only.

The runner deliberately opens ``train_FD002.txt`` only.  It has no route to
the official test histories or their RUL endpoint labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.cmapss import (
    CMAPSSEngineRecord,
    PREFIX_FRACTIONS,
    load_cmapss_fd002_source_train,
    load_cmapss_train_split,
    partition_cmapss_train,
)


DEFAULT_ARCHIVE = ROOT / "data/real/cmapss/raw/CMAPSSData.zip"
DEFAULT_SPLIT = ROOT / "data/real/cmapss/source_train_split.json"
DEFAULT_OUT = ROOT / "artifacts/evaluations/phase27_cmapss_fd002_real_source_train_baselines_20260721"
RIDGE_GRID = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class PrefixExample:
    unit_id: int
    fraction: float
    observed_cycle: float
    target_rul: float
    settings: np.ndarray
    sensors: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    ridge: float


def _prefix_index(record: CMAPSSEngineRecord, fraction: float) -> int:
    return min(max(int(np.floor(record.cycles.size * fraction)) - 1, 0), record.cycles.size - 2)


def _prefix_examples(
    records: list[CMAPSSEngineRecord],
    setting_transform: Callable[[CMAPSSEngineRecord, float], np.ndarray] | None = None,
) -> list[PrefixExample]:
    examples = []
    for record in records:
        for fraction in PREFIX_FRACTIONS:
            index = _prefix_index(record, fraction)
            settings = record.operating_settings[index] if setting_transform is None else setting_transform(record, fraction)
            settings = np.asarray(settings, dtype=float)
            if settings.shape != (3,) or not np.all(np.isfinite(settings)):
                raise ValueError("C-MAPSS setting control emitted invalid features")
            examples.append(
                PrefixExample(
                    unit_id=record.unit_id,
                    fraction=float(fraction),
                    observed_cycle=float(record.cycles[index]),
                    target_rul=float(record.cycles[-1] - record.cycles[index]),
                    settings=settings.copy(),
                    sensors=record.sensors[index].copy(),
                )
            )
    return examples


def _matrix(examples: list[PrefixExample], feature_fn: Callable[[PrefixExample], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([feature_fn(example) for example in examples], dtype=float)
    y = np.asarray([example.target_rul for example in examples], dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("C-MAPSS baseline feature matrix is non-finite")
    return x, y


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> RidgeModel:
    if ridge <= 0:
        raise ValueError("C-MAPSS ridge penalty must be positive")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-12)
    standardized = (x - mean) / scale
    centered_y = y - np.mean(y)
    coefficients = np.linalg.solve(
        standardized.T @ standardized + ridge * np.eye(standardized.shape[1]),
        standardized.T @ centered_y,
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("C-MAPSS ridge fit is non-finite")
    return RidgeModel(mean, scale, coefficients, float(np.mean(y)), float(ridge))


def _predict_ridge(model: RidgeModel, x: np.ndarray) -> np.ndarray:
    predicted = (x - model.feature_mean) / model.feature_scale @ model.coefficients + model.intercept
    if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
        raise ValueError("C-MAPSS ridge emitted a non-finite or negative RUL")
    return predicted


def _metrics(examples: list[PrefixExample], predictions: np.ndarray) -> dict[str, Any]:
    target = np.asarray([example.target_rul for example in examples], dtype=float)
    if predictions.shape != target.shape or not np.all(np.isfinite(predictions)) or np.any(predictions < 0.0):
        raise ValueError("C-MAPSS score has non-finite or negative predictions")
    errors = predictions - target
    per_fraction: dict[str, Any] = {}
    per_unit = []
    for fraction in PREFIX_FRACTIONS:
        mask = np.asarray([example.fraction == fraction for example in examples])
        fraction_errors = errors[mask]
        per_fraction[str(fraction)] = {
            "mae": float(np.mean(np.abs(fraction_errors))),
            "rmse": float(np.sqrt(np.mean(np.square(fraction_errors)))),
            "worst_absolute_error": float(np.max(np.abs(fraction_errors))),
        }
    for example, prediction, error in zip(examples, predictions, errors):
        per_unit.append(
            {
                "unit_id": example.unit_id,
                "prefix_fraction": example.fraction,
                "target_rul": example.target_rul,
                "prediction": float(prediction),
                "absolute_error": float(abs(error)),
                "setting_stratum": [float(value) for value in example.settings],
            }
        )
    return {
        "aggregate": {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "worst_absolute_error": float(np.max(np.abs(errors))),
        },
        "per_fraction": per_fraction,
        "per_unit": per_unit,
    }


def _age_features(example: PrefixExample) -> np.ndarray:
    return np.asarray([example.observed_cycle, example.fraction], dtype=float)


def _settings_sensor_features(example: PrefixExample, *, include_settings: bool = True) -> np.ndarray:
    prefix = [example.observed_cycle, example.fraction]
    if include_settings:
        prefix.extend(example.settings.tolist())
    return np.asarray(prefix + example.sensors.tolist(), dtype=float)


def _fit_condition_health(train: list[PrefixExample]) -> dict[str, Any]:
    settings = np.asarray([example.settings for example in train], dtype=float)
    sensors = np.asarray([example.sensors for example in train], dtype=float)
    target = np.asarray([example.target_rul for example in train], dtype=float)
    cycle = np.asarray([example.observed_cycle for example in train], dtype=float)
    setting_mean = np.mean(settings, axis=0)
    setting_scale = np.maximum(np.std(settings, axis=0), 1e-12)
    standardized_settings = (settings - setting_mean) / setting_scale
    design = np.column_stack([np.ones(standardized_settings.shape[0]), standardized_settings])
    sensor_coefficients = np.linalg.solve(
        design.T @ design + np.diag([0.0, 1.0, 1.0, 1.0]),
        design.T @ sensors,
    )
    residual = sensors - design @ sensor_coefficients
    residual_scale = np.maximum(np.std(residual, axis=0), 1e-12)
    standardized_residual = residual / residual_scale
    centered_cycle = cycle - np.mean(cycle)
    weights = standardized_residual.T @ centered_cycle
    if np.linalg.norm(weights) <= 1e-12:
        raise ValueError("C-MAPSS condition-health score is unidentified")
    weights = weights / np.linalg.norm(weights)
    damage = standardized_residual @ weights
    centered_damage = damage - np.mean(damage)
    unconstrained = float(np.dot(centered_damage, target - np.mean(target)) / np.dot(centered_damage, centered_damage))
    slope = max(0.0, -unconstrained)
    return {
        "setting_mean": setting_mean,
        "setting_scale": setting_scale,
        "sensor_coefficients": sensor_coefficients,
        "residual_scale": residual_scale,
        "damage_weights": weights,
        "damage_mean": float(np.mean(damage)),
        "target_mean": float(np.mean(target)),
        "nonnegative_degradation_slope": slope,
    }


def _predict_condition_health(model: dict[str, Any], examples: list[PrefixExample]) -> np.ndarray:
    settings = np.asarray([example.settings for example in examples], dtype=float)
    sensors = np.asarray([example.sensors for example in examples], dtype=float)
    standardized_settings = (settings - model["setting_mean"]) / model["setting_scale"]
    design = np.column_stack([np.ones(standardized_settings.shape[0]), standardized_settings])
    residual = sensors - design @ model["sensor_coefficients"]
    damage = residual / model["residual_scale"] @ model["damage_weights"]
    prediction = model["target_mean"] - model["nonnegative_degradation_slope"] * (damage - model["damage_mean"])
    if not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
        raise ValueError("C-MAPSS condition-health baseline emitted invalid RUL")
    return prediction


def _median_predictions(train: list[PrefixExample], scored: list[PrefixExample]) -> np.ndarray:
    values = {
        fraction: float(np.median([example.target_rul for example in train if example.fraction == fraction]))
        for fraction in PREFIX_FRACTIONS
    }
    return np.asarray([values[example.fraction] for example in scored], dtype=float)


def _fit_score_ridge(
    train: list[PrefixExample],
    scored: list[PrefixExample],
    ridge: float,
    feature_fn: Callable[[PrefixExample], np.ndarray],
) -> tuple[RidgeModel, dict[str, Any]]:
    train_x, train_y = _matrix(train, feature_fn)
    scored_x, _ = _matrix(scored, feature_fn)
    model = _fit_ridge(train_x, train_y, ridge)
    return model, _metrics(scored, _predict_ridge(model, scored_x))


def _cyclic_pairing_transform(records: list[CMAPSSEngineRecord]) -> Callable[[CMAPSSEngineRecord, float], np.ndarray]:
    by_id = {record.unit_id: record for record in records}
    ordered_ids = sorted(by_id)
    donor = {unit_id: by_id[ordered_ids[(index + 1) % len(ordered_ids)]] for index, unit_id in enumerate(ordered_ids)}

    def transform(record: CMAPSSEngineRecord, fraction: float) -> np.ndarray:
        paired = donor[record.unit_id]
        return paired.operating_settings[_prefix_index(paired, fraction)]

    return transform


def _quarter_shift_transform(record: CMAPSSEngineRecord, fraction: float) -> np.ndarray:
    shift = max(1, record.operating_settings.shape[0] // 4)
    shifted = np.roll(record.operating_settings, shift=shift, axis=0)
    return shifted[_prefix_index(record, fraction)]


def _control_metrics(
    train_records: list[CMAPSSEngineRecord],
    selection: list[PrefixExample],
    ridge: float,
    mode: str,
) -> dict[str, Any]:
    if mode == "pairing":
        transformed = _prefix_examples(train_records, _cyclic_pairing_transform(train_records))
        feature_fn = _settings_sensor_features
    elif mode == "time_order":
        transformed = _prefix_examples(train_records, _quarter_shift_transform)
        feature_fn = _settings_sensor_features
    elif mode == "setting_ablation":
        transformed = _prefix_examples(train_records)
        feature_fn = lambda example: _settings_sensor_features(example, include_settings=False)
    else:
        raise ValueError(f"unknown C-MAPSS control: {mode}")
    _, score = _fit_score_ridge(transformed, selection, ridge, feature_fn)
    return score


def _correlation_checks(score: dict[str, Any]) -> dict[str, Any]:
    rows = score["per_unit"]
    errors = np.asarray([row["absolute_error"] for row in rows], dtype=float)
    fractions = np.asarray([row["prefix_fraction"] for row in rows], dtype=float)
    prefix_correlation = float(np.corrcoef(errors, fractions)[0, 1]) if np.std(errors) > 0 else 0.0
    strata = [tuple(row["setting_stratum"]) for row in rows]
    stratum_correlations = {}
    for stratum in sorted(set(strata)):
        indicator = np.asarray([value == stratum for value in strata], dtype=float)
        correlation = float(np.corrcoef(errors, indicator)[0, 1]) if np.std(errors) > 0 and np.std(indicator) > 0 else 0.0
        stratum_correlations["|".join(str(value) for value in stratum)] = correlation
    max_stratum = max((abs(value) for value in stratum_correlations.values()), default=0.0)
    return {
        "absolute_error_prefix_fraction_correlation": prefix_correlation,
        "max_absolute_error_setting_stratum_indicator_correlation": max_stratum,
        "near_perfect_threshold": 0.999,
        "passed": abs(prefix_correlation) < 0.999 and max_stratum < 0.999,
    }


def _selection_gate(
    normal: dict[str, Any],
    median: dict[str, Any],
    controls: dict[str, dict[str, Any]],
) -> tuple[bool, list[str], dict[str, float], dict[str, Any]]:
    failures = []
    improvement = {}
    for fraction in PREFIX_FRACTIONS:
        key = str(fraction)
        ratio = normal["per_fraction"][key]["rmse"] / median["per_fraction"][key]["rmse"]
        improvement[key] = 1.0 - ratio
        if ratio > 0.9:
            failures.append(f"no_ten_percent_rmse_improvement_at_prefix_{fraction}")
        if normal["per_fraction"][key]["worst_absolute_error"] > 1.05 * median["per_fraction"][key]["worst_absolute_error"]:
            failures.append(f"worst_engine_error_degraded_over_five_percent_at_prefix_{fraction}")
    degradation = {
        name: score["aggregate"]["rmse"] / normal["aggregate"]["rmse"]
        for name, score in controls.items()
    }
    if degradation["pairing"] < 1.5:
        failures.append("pairing_control_did_not_degrade_one_point_five_fold")
    if degradation["time_order"] < 1.5:
        failures.append("time_order_control_did_not_degrade_one_point_five_fold")
    if degradation["setting_ablation"] < 1.25:
        failures.append("setting_ablation_did_not_degrade_one_point_two_five_fold")
    correlations = _correlation_checks(normal)
    if not correlations["passed"]:
        failures.append("near_perfect_error_correlation_with_prefix_or_setting_stratum")
    return not failures, failures, degradation, correlations


def _serializable_health(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 27 C-MAPSS FD002 Real Source-Train Baselines",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "This is a simulator-only source-train forecast gate. It does not read or score official-test RUL endpoints and makes no turbofan mechanism, maintenance, or causal operating-setting claim.",
        "",
        "## Fixed Selection",
        f"- units: `{result['counts']}`",
        f"- selected settings-plus-sensors ridge: `{result['selected_ridge']}`",
    ]
    for name, score in result.get("selection", {}).items():
        lines.append(f"- `{name}` aggregate RMSE: `{score['aggregate']['rmse']:.4f}`")
    if result.get("control_degradation"):
        lines.extend(["", "## Controls"])
        for name, ratio in result["control_degradation"].items():
            lines.append(f"- `{name}` RMSE degradation: `{ratio:.4f}x`")
    lines.extend(["", "## Decision"])
    if result.get("selection_failures"):
        lines.append("- selection failures: `" + ", ".join(result["selection_failures"]) + "`")
    else:
        lines.append("- selection passed; the source-train holdout was scored exactly once.")
    if result.get("holdout"):
        lines.append(f"- source-train holdout settings-plus-sensors ridge RMSE: `{result['holdout']['settings_sensor_ridge']['aggregate']['rmse']:.4f}`")
    lines.extend(["", "No official-test history or RUL endpoint was accessed by this runner.", ""])
    return "\n".join(lines)


def run_gate(
    raw_archive: Path = DEFAULT_ARCHIVE,
    split_path: Path = DEFAULT_SPLIT,
    output_dir: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    records = load_cmapss_fd002_source_train(raw_archive)
    partitioned = partition_cmapss_train(records, load_cmapss_train_split(split_path))
    if {name: len(value) for name, value in partitioned.items()} != {"train": 180, "selection": 40, "holdout": 40}:
        raise ValueError("C-MAPSS fixed source-train split revalidation failed")
    train = _prefix_examples(partitioned["train"])
    selection = _prefix_examples(partitioned["selection"])
    holdout = _prefix_examples(partitioned["holdout"])
    age_model, age_score = _fit_score_ridge(train, selection, 1.0, _age_features)
    ridge_candidates = {str(ridge): _fit_score_ridge(train, selection, ridge, _settings_sensor_features) for ridge in RIDGE_GRID}
    selected_ridge = min(RIDGE_GRID, key=lambda ridge: ridge_candidates[str(ridge)][1]["aggregate"]["rmse"])
    settings_model, settings_score = ridge_candidates[str(selected_ridge)]
    health_model = _fit_condition_health(train)
    health_score = _metrics(selection, _predict_condition_health(health_model, selection))
    median_score = _metrics(selection, _median_predictions(train, selection))
    controls = {
        name: _control_metrics(partitioned["train"], selection, selected_ridge, name)
        for name in ("pairing", "time_order", "setting_ablation")
    }
    passed, failures, degradation, correlations = _selection_gate(settings_score, median_score, controls)
    result: dict[str, Any] = {
        "phase": 27,
        "official_test_rul_accessed": False,
        "official_test_history_accessed": False,
        "abc_smc_executed": False,
        "llm_called": False,
        "campaign_launched": False,
        "counts": {name: len(value) for name, value in partitioned.items()},
        "selection_prefix_fractions": list(PREFIX_FRACTIONS),
        "selected_ridge": selected_ridge,
        "selection": {
            "train_median_rul": median_score,
            "cycle_age_ridge": age_score,
            "settings_sensor_ridge": settings_score,
            "condition_normalized_sensor_health_linear": health_score,
        },
        "settings_sensor_ridge_grid": {ridge: score for ridge, (_, score) in ridge_candidates.items()},
        "condition_health_model": _serializable_health(health_model),
        "control_degradation": degradation,
        "controls": controls,
        "correlation_checks": correlations,
        "selection_passed": passed,
        "selection_failures": failures,
    }
    if passed:
        holdout_scores = {
            "train_median_rul": _metrics(holdout, _median_predictions(train, holdout)),
            "cycle_age_ridge": _metrics(holdout, _predict_ridge(age_model, _matrix(holdout, _age_features)[0])),
            "settings_sensor_ridge": _metrics(holdout, _predict_ridge(settings_model, _matrix(holdout, _settings_sensor_features)[0])),
            "condition_normalized_sensor_health_linear": _metrics(holdout, _predict_condition_health(health_model, holdout)),
        }
        result["holdout"] = holdout_scores
        result["decision"] = "passed_source_train_selection_gate_holdout_scored_once"
    else:
        result["decision"] = "negative_closed_selection_gate_failed_no_holdout_or_official_test_score"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(_markdown(result))
    return result


def main() -> None:
    result = run_gate()
    print(f"C-MAPSS source-train baseline gate: {result['decision']}")


if __name__ == "__main__":
    main()

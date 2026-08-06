from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_EXPERIMENTS = {"train": 15, "validation": 4, "test": 1}
EXPECTED_SAMPLES = 2000


@dataclass(frozen=True)
class PHExperiment:
    experiment_id: str
    split: str
    input_u: np.ndarray
    output_y: np.ndarray


@dataclass(frozen=True)
class CausalInputSignal:
    """Index-time zero-order hold that never reads a future recorded input."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("causal input signal requires a non-empty finite one-dimensional array")
        object.__setattr__(self, "values", values.copy())

    def at(self, time_index: float) -> float:
        if not np.isfinite(time_index):
            raise ValueError("time index must be finite")
        index = min(max(int(np.floor(time_index)), 0), len(self.values) - 1)
        return float(self.values[index])


def _read_matrix(path: Path, expected_columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",")
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape != (EXPECTED_SAMPLES, expected_columns) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} must be finite with shape {(EXPECTED_SAMPLES, expected_columns)}, found {values.shape}")
    return values


def load_ph_reactor(raw_root: str | Path) -> dict[str, list[PHExperiment]]:
    """Load fixed Zenodo pH-reactor simulator U/Y files without resampling."""
    root = Path(raw_root)
    source_names = {"train": "Train", "validation": "Val", "test": "Test"}
    result: dict[str, list[PHExperiment]] = {}
    for split, source_name in source_names.items():
        count = EXPECTED_EXPERIMENTS[split]
        u = _read_matrix(root / f"PH_U_{source_name}.csv", count)
        y = _read_matrix(root / f"PH_Y_{source_name}.csv", count)
        result[split] = [
            PHExperiment(
                experiment_id=f"{split}_{index + 1:02d}",
                split=split,
                input_u=u[:, index].copy(),
                output_y=y[:, index].copy(),
            )
            for index in range(count)
        ]
    return result


def _metrics(observed: np.ndarray, predicted: np.ndarray, *, warmup: int = 1) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)[warmup:]
    predicted = np.asarray(predicted, dtype=float)[warmup:]
    if observed.shape != predicted.shape or observed.size == 0 or not np.all(np.isfinite(predicted)):
        return {"mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "late_nrmse": float("nan")}
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    late_start = int(np.floor(len(observed) * 0.75))
    late_rmse = float(np.sqrt(np.mean(np.square(residual[late_start:]))))
    return {"mse": float(np.mean(np.square(residual))), "rmse": rmse, "nrmse": rmse / scale, "late_nrmse": late_rmse / scale}


def _aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.median([item[key] for item in metrics])) for key in ("mse", "rmse", "nrmse", "late_nrmse")}


def _persistence(experiment: PHExperiment) -> np.ndarray:
    return np.full_like(experiment.output_y, experiment.output_y[0], dtype=float)


def _arx_design(experiments: list[PHExperiment], order: int, ridge: float) -> np.ndarray:
    features = []
    targets = []
    for experiment in experiments:
        for index in range(order, len(experiment.output_y)):
            features.append(
                np.concatenate(
                    [experiment.output_y[index - order:index][::-1], experiment.input_u[index - order:index + 1][::-1], [1.0]]
                )
            )
            targets.append(experiment.output_y[index])
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _arx_predict(experiment: PHExperiment, coefficients: np.ndarray, order: int) -> np.ndarray:
    prediction = experiment.output_y[:order].astype(float, copy=True).tolist()
    for index in range(order, len(experiment.output_y)):
        y_history = np.asarray(prediction[index - order:index][::-1])
        u_history = experiment.input_u[index - order:index + 1][::-1]
        value = float(np.dot(coefficients, np.concatenate([y_history, u_history, [1.0]])))
        prediction.append(value)
    return np.asarray(prediction, dtype=float)


def _fit_arx(train: list[PHExperiment], validation: list[PHExperiment]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for order in (1, 2, 3, 5, 10):
        for ridge in (1e-8, 1e-6, 1e-4, 1e-2):
            coefficients = _arx_design(train, order, ridge)
            metrics = [_metrics(item.output_y, _arx_predict(item, coefficients, order), warmup=order) for item in validation]
            score = _aggregate(metrics)["nrmse"]
            candidate = {"order": order, "ridge": ridge, "coefficients": coefficients, "validation_nrmse": score}
            if best is None or score < best["validation_nrmse"]:
                best = candidate
    assert best is not None
    return best


def _first_order_features(experiments: list[PHExperiment], alpha: float, transform: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for experiment in experiments:
        transformed = transform(experiment.input_u)
        rows.append(np.column_stack([transformed[1:], np.ones(len(transformed) - 1)]))
        targets.append(experiment.output_y[1:] - alpha * experiment.output_y[:-1])
    return np.vstack(rows), np.concatenate(targets)


def _first_order_predict(experiment: PHExperiment, alpha: float, beta: float, intercept: float, transform: Any) -> np.ndarray:
    transformed = transform(experiment.input_u)
    prediction = np.empty_like(experiment.output_y, dtype=float)
    prediction[0] = experiment.output_y[0]
    for index in range(1, len(prediction)):
        prediction[index] = alpha * prediction[index - 1] + beta * transformed[index] + intercept
    return prediction


def _fit_stable_first_order(train: list[PHExperiment], validation: list[PHExperiment], *, hammerstein: bool) -> dict[str, Any]:
    train_u = np.concatenate([item.input_u for item in train])
    mean = float(np.mean(train_u))
    scale = max(float(np.std(train_u)), 1e-12)
    slopes = (0.25, 0.5, 1.0, 2.0, 4.0) if hammerstein else (1.0,)
    best: dict[str, Any] | None = None
    for alpha in np.linspace(0.70, 0.999, 31):
        for slope in slopes:
            transform = (lambda values, slope=slope: np.tanh(slope * (values - mean) / scale)) if hammerstein else (lambda values: values)
            x, y = _first_order_features(train, float(alpha), transform)
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            beta, intercept = float(coefficients[0]), float(coefficients[1])
            metrics = [_metrics(item.output_y, _first_order_predict(item, float(alpha), beta, intercept, transform)) for item in validation]
            score = _aggregate(metrics)["nrmse"]
            candidate = {"alpha": float(alpha), "beta": beta, "intercept": intercept, "input_mean": mean, "input_scale": scale, "input_slope": float(slope), "validation_nrmse": score}
            if best is None or score < best["validation_nrmse"]:
                best = candidate
    assert best is not None
    return best


def _first_order_transform(model: dict[str, Any], *, hammerstein: bool) -> Any:
    if not hammerstein:
        return lambda values: values
    return lambda values: np.tanh(model["input_slope"] * (values - model["input_mean"]) / model["input_scale"])


def _score_predictions(experiments: list[PHExperiment], predictor: Any, *, warmup: int = 1) -> dict[str, Any]:
    rows = []
    for experiment in experiments:
        metrics = _metrics(experiment.output_y, predictor(experiment), warmup=warmup)
        rows.append({"experiment_id": experiment.experiment_id, **metrics})
    return {"per_experiment": rows, "aggregate": _aggregate([{key: row[key] for key in ("mse", "rmse", "nrmse", "late_nrmse")} for row in rows])}


def evaluate_ph_reactor_baselines(data: dict[str, list[PHExperiment]]) -> dict[str, Any]:
    """Fit all parameters on published train data, choose on validation, score once on test."""
    train, validation, test = data["train"], data["validation"], data["test"]
    arx = _fit_arx(train, validation)
    linear = _fit_stable_first_order(train, validation, hammerstein=False)
    hammerstein = _fit_stable_first_order(train, validation, hammerstein=True)
    baselines = {
        "persistence": {"parameters": {}, "validation": _score_predictions(validation, _persistence), "test": _score_predictions(test, _persistence)},
        "regularized_arx": {
            "parameters": {"order": arx["order"], "ridge": arx["ridge"]},
            "validation": _score_predictions(validation, lambda item: _arx_predict(item, arx["coefficients"], arx["order"]), warmup=arx["order"]),
            "test": _score_predictions(test, lambda item: _arx_predict(item, arx["coefficients"], arx["order"]), warmup=arx["order"]),
        },
        "stable_first_order": {
            "parameters": {key: linear[key] for key in ("alpha", "beta", "intercept", "validation_nrmse")},
            "validation": _score_predictions(validation, lambda item: _first_order_predict(item, linear["alpha"], linear["beta"], linear["intercept"], _first_order_transform(linear, hammerstein=False))),
            "test": _score_predictions(test, lambda item: _first_order_predict(item, linear["alpha"], linear["beta"], linear["intercept"], _first_order_transform(linear, hammerstein=False))),
        },
        "hammerstein_first_order": {
            "parameters": {key: hammerstein[key] for key in ("alpha", "beta", "intercept", "input_slope", "validation_nrmse")},
            "validation": _score_predictions(validation, lambda item: _first_order_predict(item, hammerstein["alpha"], hammerstein["beta"], hammerstein["intercept"], _first_order_transform(hammerstein, hammerstein=True))),
            "test": _score_predictions(test, lambda item: _first_order_predict(item, hammerstein["alpha"], hammerstein["beta"], hammerstein["intercept"], _first_order_transform(hammerstein, hammerstein=True))),
        },
    }
    for baseline in baselines.values():
        for split in ("validation", "test"):
            if not np.isfinite(baseline[split]["aggregate"]["nrmse"]):
                raise ValueError("non-finite pH-reactor baseline metric")
    return baselines


def prescreen_ph_reactor_rollouts(
    data: dict[str, list[PHExperiment]],
    rollout: Any,
    *,
    output_range_multiple: float = 10.0,
) -> dict[str, Any]:
    """Check a candidate rollout on every native U grid before expensive inference.

    ``rollout`` receives only a one-dimensional U vector and may return either a
    prediction array or ``(prediction, diagnostics)``. Diagnostics can set
    ``state_clipped`` or ``derivative_clipped`` when the candidate simulator
    explicitly applies a numerical clip.
    """
    train_y = np.concatenate([item.output_y for item in data["train"]])
    train_range = max(float(np.ptp(train_y)), float(np.std(train_y)), 1e-12)
    max_abs_output = float(np.max(np.abs(train_y))) + output_range_multiple * train_range
    rows = []
    for split, experiments in data.items():
        for item in experiments:
            try:
                response = rollout(item.input_u.copy())
                diagnostics: dict[str, Any] = {}
                if isinstance(response, tuple):
                    prediction, diagnostics = response
                    diagnostics = dict(diagnostics or {})
                else:
                    prediction = response
                prediction = np.asarray(prediction, dtype=float)
                if prediction.shape != item.input_u.shape:
                    reason = "shape_mismatch"
                elif not np.all(np.isfinite(prediction)):
                    reason = "nonfinite_output"
                elif diagnostics.get("state_clipped") or diagnostics.get("derivative_clipped"):
                    reason = "clipping_active"
                elif float(np.max(np.abs(prediction))) > max_abs_output:
                    reason = "output_bound_exceeded"
                else:
                    reason = None
                rows.append(
                    {
                        "experiment_id": item.experiment_id,
                        "split": split,
                        "passed": reason is None,
                        "reason": reason,
                        "max_abs_output": float(np.max(np.abs(prediction))) if prediction.size and np.all(np.isfinite(prediction)) else None,
                        "diagnostics": diagnostics,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "experiment_id": item.experiment_id,
                        "split": split,
                        "passed": False,
                        "reason": "rollout_exception",
                        "max_abs_output": None,
                        "diagnostics": {"exception": f"{type(exc).__name__}: {exc}"},
                    }
                )
    failures = [row for row in rows if not row["passed"]]
    return {
        "native_input_grid_count": len(rows),
        "train_output_range": train_range,
        "max_abs_output_limit": max_abs_output,
        "output_range_multiple": output_range_multiple,
        "passed": not failures,
        "failure_count": len(failures),
        "rows": rows,
    }

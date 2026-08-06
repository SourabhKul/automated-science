#!/usr/bin/env python3
"""Tiny non-LLM causal-input controls for the fixed pH-reactor U grids."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.ph_reactor import PHExperiment, load_ph_reactor
from scripts.legacy.real_ph_reactor_data_loader import DEFAULT_RAW_ROOT


PLANTED = {"alpha": 0.92, "beta": 0.30, "slope": 0.80, "intercept": 0.02}
MAX_SIGNAL_NRMSE = 0.05
MAX_RELATIVE_PARAMETER_ERROR = 0.25
MAX_INTERCEPT_ERROR = 0.05
MIN_CONTROL_DEGRADATION = 2.0


@dataclass(frozen=True)
class SyntheticExperiment:
    experiment_id: str
    input_u: np.ndarray
    output_y: np.ndarray


class OutputAccessSentinel:
    """Raises if a validation/test target is accidentally read during prediction."""

    def __getitem__(self, _: Any) -> float:
        raise AssertionError("validation/test output leakage into predictor")

    def __array__(self, *_: Any, **__: Any) -> np.ndarray:
        raise AssertionError("validation/test output leakage into predictor")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _simulate(input_u: np.ndarray, *, parameters: dict[str, float], u_reference: float) -> np.ndarray:
    output = np.zeros_like(input_u, dtype=float)
    for index in range(1, len(output)):
        forcing = np.tanh(parameters["slope"] * (input_u[index - 1] - u_reference))
        output[index] = parameters["alpha"] * output[index - 1] + parameters["beta"] * forcing + parameters["intercept"]
    if not np.all(np.isfinite(output)):
        raise ValueError("planted trajectory is non-finite")
    return output


def _make_synthetic(experiments: list[PHExperiment], *, u_reference: float) -> list[SyntheticExperiment]:
    return [
        SyntheticExperiment(item.experiment_id, item.input_u.copy(), _simulate(item.input_u, parameters=PLANTED, u_reference=u_reference))
        for item in experiments
    ]


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if observed.shape != predicted.shape or not np.all(np.isfinite(predicted)):
        return {"mse": float("nan"), "nrmse": float("nan")}
    residual = predicted - observed
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    return {"mse": float(np.mean(np.square(residual))), "nrmse": float(np.sqrt(np.mean(np.square(residual))) / scale)}


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.median([row[key] for row in rows])) for key in ("mse", "nrmse")}


def _fit_for_slope(train: list[SyntheticExperiment], *, slope: float, u_reference: float) -> dict[str, float] | None:
    features = []
    targets = []
    for item in train:
        forcing = np.tanh(slope * (item.input_u[:-1] - u_reference))
        features.append(np.column_stack([item.output_y[:-1], forcing, np.ones(len(forcing))]))
        targets.append(item.output_y[1:])
    coefficients = np.linalg.lstsq(np.vstack(features), np.concatenate(targets), rcond=None)[0]
    alpha, beta, intercept = (float(value) for value in coefficients)
    if not 0.0 < alpha < 1.0 or not np.all(np.isfinite(coefficients)):
        return None
    return {"alpha": alpha, "beta": beta, "slope": float(slope), "intercept": intercept}


def _predict(model: dict[str, float], input_u: np.ndarray, *, u_reference: float, output_forbidden: Any = None) -> np.ndarray:
    """Roll out from a fixed synthetic initial state; no output target is read."""
    del output_forbidden
    output = np.zeros_like(input_u, dtype=float)
    for index in range(1, len(output)):
        forcing = np.tanh(model["slope"] * (input_u[index - 1] - u_reference))
        output[index] = model["alpha"] * output[index - 1] + model["beta"] * forcing + model["intercept"]
    return output


def _score(experiments: list[SyntheticExperiment], model: dict[str, float], *, u_reference: float) -> dict[str, Any]:
    rows = []
    for item in experiments:
        rows.append({"experiment_id": item.experiment_id, **_metrics(item.output_y, _predict(model, item.input_u, u_reference=u_reference))})
    return {"per_experiment": rows, "aggregate": _aggregate(rows)}


def _fit_select(train: list[SyntheticExperiment], validation: list[SyntheticExperiment], *, u_reference: float) -> dict[str, float]:
    best: dict[str, float] | None = None
    for slope in np.linspace(0.20, 1.40, 61):
        model = _fit_for_slope(train, slope=float(slope), u_reference=u_reference)
        if model is None:
            continue
        score = _score(validation, model, u_reference=u_reference)["aggregate"]["nrmse"]
        if best is None or score < best["validation_nrmse"]:
            best = {**model, "validation_nrmse": float(score)}
    if best is None:
        raise ValueError("no stable fitted synthetic model")
    return best


def _pairing_permutation(train: list[SyntheticExperiment]) -> list[SyntheticExperiment]:
    inputs = [item.input_u for item in train]
    return [SyntheticExperiment(item.experiment_id, inputs[(index + 1) % len(inputs)], item.output_y) for index, item in enumerate(train)]


def _time_shift(train: list[SyntheticExperiment], offset: int = 317) -> list[SyntheticExperiment]:
    return [SyntheticExperiment(item.experiment_id, np.roll(item.input_u, offset), item.output_y) for item in train]


def _parameter_errors(model: dict[str, float]) -> dict[str, float]:
    errors = {}
    for key, planted in PLANTED.items():
        if key == "intercept":
            errors[key] = abs(model[key] - planted)
        else:
            errors[key] = abs(model[key] - planted) / abs(planted)
    return errors


def _run_condition(train: list[SyntheticExperiment], validation: list[SyntheticExperiment], test: list[SyntheticExperiment], *, u_reference: float) -> dict[str, Any]:
    model = _fit_select(train, validation, u_reference=u_reference)
    return {
        "model": model,
        "train": _score(train, model, u_reference=u_reference),
        "validation": _score(validation, model, u_reference=u_reference),
        "test": _score(test, model, u_reference=u_reference),
    }


def run_smoke(raw_root: Path = DEFAULT_RAW_ROOT) -> dict[str, Any]:
    data = load_ph_reactor(raw_root)
    u_reference = float(np.mean(np.concatenate([item.input_u for item in data["train"]])))
    synthetic = {split: _make_synthetic(items, u_reference=u_reference) for split, items in data.items()}
    signal = _run_condition(synthetic["train"], synthetic["validation"], synthetic["test"], u_reference=u_reference)
    pairing = _run_condition(_pairing_permutation(synthetic["train"]), synthetic["validation"], synthetic["test"], u_reference=u_reference)
    time_order = _run_condition(_time_shift(synthetic["train"]), synthetic["validation"], synthetic["test"], u_reference=u_reference)

    sentinel = OutputAccessSentinel()
    isolation_prediction = _predict(signal["model"], synthetic["test"][0].input_u, u_reference=u_reference, output_forbidden=sentinel)
    isolation_metrics = _metrics(synthetic["test"][0].output_y, isolation_prediction)
    parameter_errors = _parameter_errors(signal["model"])
    signal_nrmse = signal["test"]["aggregate"]["nrmse"]
    pairing_nrmse = pairing["test"]["aggregate"]["nrmse"]
    time_order_nrmse = time_order["test"]["aggregate"]["nrmse"]
    all_metrics = [
        signal[split]["aggregate"][key]
        for split in ("train", "validation", "test")
        for key in ("mse", "nrmse")
    ] + [
        pairing[split]["aggregate"][key]
        for split in ("train", "validation", "test")
        for key in ("mse", "nrmse")
    ] + [
        time_order[split]["aggregate"][key]
        for split in ("train", "validation", "test")
        for key in ("mse", "nrmse")
    ] + list(isolation_metrics.values())
    passed = (
        all(np.isfinite(all_metrics))
        and signal_nrmse <= MAX_SIGNAL_NRMSE
        and all(parameter_errors[key] <= (MAX_INTERCEPT_ERROR if key == "intercept" else MAX_RELATIVE_PARAMETER_ERROR) for key in parameter_errors)
        and pairing_nrmse / max(signal_nrmse, 1e-12) >= MIN_CONTROL_DEGRADATION
        and time_order_nrmse / max(signal_nrmse, 1e-12) >= MIN_CONTROL_DEGRADATION
    )
    return {
        "phase": 22,
        "run_kind": "tiny_non_llm_ph_reactor_input_specificity_smoke",
        "completed_at": _utc_now(),
        "source_root": str(raw_root),
        "spec": {"train_validation_test": "15/4/1 native U grids", "samples_per_experiment": 2000, "fit": "direct constrained causal least-squares grid search", "abc_smc_calls": 0, "llm_calls": 0},
        "planted_parameters": {**PLANTED, "u_reference_train_mean": u_reference},
        "thresholds": {"signal_test_nrmse_max": MAX_SIGNAL_NRMSE, "relative_parameter_error_max": MAX_RELATIVE_PARAMETER_ERROR, "intercept_error_max": MAX_INTERCEPT_ERROR, "control_degradation_min_ratio": MIN_CONTROL_DEGRADATION},
        "signal": signal,
        "pairing_control": pairing,
        "time_order_control": time_order,
        "output_isolation": {"sentinel_passed": bool(np.all(np.isfinite(isolation_prediction))), "test_metrics": isolation_metrics, "predictor_contract": "predictor receives input_u and an unreadable output sentinel; rollout starts from fixed synthetic zero state"},
        "aggregate": {"all_metrics_finite": bool(all(np.isfinite(all_metrics))), "signal_test_nrmse": signal_nrmse, "pairing_test_nrmse": pairing_nrmse, "time_order_test_nrmse": time_order_nrmse, "pairing_degradation_ratio": pairing_nrmse / max(signal_nrmse, 1e-12), "time_order_degradation_ratio": time_order_nrmse / max(signal_nrmse, 1e-12), "signal_parameter_errors": parameter_errors},
        "decision": "passed" if passed else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/phase22_ph_reactor_synthetic_controls_20260712/summary.json"))
    args = parser.parse_args()
    summary = run_smoke(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": summary["decision"], "aggregate": summary["aggregate"]}, indent=2))
    return 0 if summary["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

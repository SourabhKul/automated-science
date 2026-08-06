#!/usr/bin/env python3
"""Authorized fixed-family, non-LLM pH-reactor real-simulator inference smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.input_driven_eval import compile_input_driven_candidate, evaluate_input_driven_candidate
from core.real_data.ph_reactor import PHExperiment, load_ph_reactor
from scripts.legacy.real_ph_reactor_data_loader import DEFAULT_RAW_ROOT


ALPHA_GRID = np.linspace(0.70, 0.99, 291)
BETA_BOUNDS = (-0.50, 0.50)
OFFSET_BOUNDS = (-0.20, 0.20)
TEST_NRMSE_MAX = 0.07645
VALIDATION_MEDIAN_NRMSE_MAX = 0.3503701540555456
VALIDATION_WORST_NRMSE_MAX = 0.4287824250526461
CONTROL_DEGRADATION_MIN = 1.5

CODE_TEMPLATE = """def dynamics(t, y, args, u):
    alpha, beta, offset = args
    return jnp.array([(alpha - 1.0) * y[0] + beta * jnp.tanh(u - {u_reference:.12g}) + offset])

metadata = [
    {{'name': 'alpha', 'range': (0.70, 0.99)}},
    {{'name': 'beta', 'range': (-0.50, 0.50)}},
    {{'name': 'offset', 'range': (-0.20, 0.20)}}
]"""


def _basis(input_u: np.ndarray, alpha: float, u_reference: float) -> tuple[np.ndarray, np.ndarray]:
    forcing = np.tanh(input_u - u_reference)
    beta_basis = np.zeros_like(input_u, dtype=float)
    offset_basis = np.zeros_like(input_u, dtype=float)
    for index in range(1, len(input_u)):
        beta_basis[index] = alpha * beta_basis[index - 1] + forcing[index - 1]
        offset_basis[index] = alpha * offset_basis[index - 1] + 1.0
    return beta_basis, offset_basis


def fit_fixed_family(train: list[PHExperiment], u_reference: float) -> dict[str, float]:
    """Fit alpha on train data only; solve beta/offset under fixed declared bounds."""
    best: dict[str, float] | None = None
    for alpha in ALPHA_GRID:
        rows = []
        targets = []
        for item in train:
            beta_basis, offset_basis = _basis(item.input_u, float(alpha), u_reference)
            rows.append(np.column_stack([beta_basis[1:], offset_basis[1:]]))
            targets.append(item.output_y[1:])
        result = lsq_linear(
            np.vstack(rows),
            np.concatenate(targets),
            bounds=([BETA_BOUNDS[0], OFFSET_BOUNDS[0]], [BETA_BOUNDS[1], OFFSET_BOUNDS[1]]),
            method="trf",
        )
        prediction_error = np.vstack(rows) @ result.x - np.concatenate(targets)
        mse = float(np.mean(np.square(prediction_error)))
        candidate = {"alpha": float(alpha), "beta": float(result.x[0]), "offset": float(result.x[1]), "train_one_step_mse": mse}
        if best is None or candidate["train_one_step_mse"] < best["train_one_step_mse"]:
            best = candidate
    assert best is not None
    return best


def _paired_train(train: list[PHExperiment]) -> list[PHExperiment]:
    return [
        PHExperiment(item.experiment_id, item.split, train[(index + 1) % len(train)].input_u, item.output_y)
        for index, item in enumerate(train)
    ]


def _time_shifted_train(train: list[PHExperiment], offset: int = 317) -> list[PHExperiment]:
    return [PHExperiment(item.experiment_id, item.split, np.roll(item.input_u, offset), item.output_y) for item in train]


def _candidate_and_result(data: dict[str, list[PHExperiment]], model: dict[str, float], u_reference: float) -> dict[str, Any]:
    candidate = compile_input_driven_candidate(CODE_TEMPLATE.format(u_reference=u_reference))
    parameters = np.array([model["alpha"], model["beta"], model["offset"]], dtype=float)
    return evaluate_input_driven_candidate(data, candidate, parameters)


def run_smoke(raw_root: Path = DEFAULT_RAW_ROOT) -> dict[str, Any]:
    data = load_ph_reactor(raw_root)
    u_reference = float(np.mean(np.concatenate([item.input_u for item in data["train"]])))
    model = fit_fixed_family(data["train"], u_reference)
    signal = _candidate_and_result(data, model, u_reference)

    paired_model = fit_fixed_family(_paired_train(data["train"]), u_reference)
    paired = _candidate_and_result(data, paired_model, u_reference)
    shifted_model = fit_fixed_family(_time_shifted_train(data["train"]), u_reference)
    shifted = _candidate_and_result(data, shifted_model, u_reference)

    signal_validation = signal["selection_metric"]["validation_median_nrmse"] if signal["status"] == "success" else float("nan")
    paired_validation = paired["selection_metric"]["validation_median_nrmse"] if paired["status"] == "success" else float("nan")
    shifted_validation = shifted["selection_metric"]["validation_median_nrmse"] if shifted["status"] == "success" else float("nan")
    signal_test = signal["untouched_test_metric"]["test_median_nrmse"] if signal["status"] == "success" else float("nan")
    validation_worst = signal["selection_metric"]["validation_worst_nrmse"] if signal["status"] == "success" else float("nan")
    aggregate = {
        "all_metrics_finite": bool(signal["status"] == "success" and paired["status"] == "success" and shifted["status"] == "success"),
        "signal_validation_median_nrmse": signal_validation,
        "signal_validation_worst_nrmse": validation_worst,
        "signal_test_nrmse": signal_test,
        "pairing_validation_nrmse": paired_validation,
        "time_order_validation_nrmse": shifted_validation,
        "pairing_degradation_ratio": paired_validation / max(signal_validation, 1e-12),
        "time_order_degradation_ratio": shifted_validation / max(signal_validation, 1e-12),
    }
    passed = (
        aggregate["all_metrics_finite"]
        and signal_validation < VALIDATION_MEDIAN_NRMSE_MAX
        and validation_worst < VALIDATION_WORST_NRMSE_MAX
        and signal_test <= TEST_NRMSE_MAX
        and aggregate["pairing_degradation_ratio"] >= CONTROL_DEGRADATION_MIN
        and aggregate["time_order_degradation_ratio"] >= CONTROL_DEGRADATION_MIN
    )
    return {
        "phase": 22,
        "run_kind": "authorized_fixed_family_non_llm_real_simulator_smoke",
        "spec": {"fit": "train-only bounded alpha grid plus beta/offset least squares", "selection": "validation scored only; no alternate family", "test": "single untouched score after fitting", "initial_state_policy": "fixed_zero", "abc_smc_calls": 0, "llm_calls": 0},
        "candidate": {"code": CODE_TEMPLATE.format(u_reference=u_reference), "u_reference_train_mean": u_reference, "parameters": model, "bounds": {"alpha": [0.70, 0.99], "beta": list(BETA_BOUNDS), "offset": list(OFFSET_BOUNDS)}},
        "signal": signal,
        "pairing_control": {"fitted_parameters": paired_model, "result": paired},
        "time_order_control": {"fitted_parameters": shifted_model, "result": shifted},
        "thresholds": {"validation_median_nrmse_max": VALIDATION_MEDIAN_NRMSE_MAX, "validation_worst_nrmse_max": VALIDATION_WORST_NRMSE_MAX, "test_nrmse_max": TEST_NRMSE_MAX, "control_degradation_min_ratio": CONTROL_DEGRADATION_MIN},
        "aggregate": aggregate,
        "decision": "passed" if passed else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/phase22_ph_reactor_non_llm_inference_20260712/summary.json"))
    args = parser.parse_args()
    result = run_smoke(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": result["decision"], "aggregate": result["aggregate"]}, indent=2))
    return 0 if result["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

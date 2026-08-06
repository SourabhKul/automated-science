"""Isolated causal input-driven evaluator primitives for pH-reactor gates."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.generated_code import GeneratedCodeError, validate_generated_math_code
from core.real_data.ph_reactor import CausalInputSignal, PHExperiment, prescreen_ph_reactor_rollouts


INPUT_DRIVEN_SIGNATURE = ("t", "y", "args", "u")


@dataclass(frozen=True)
class InputDrivenCandidate:
    dynamics: Callable[[float, np.ndarray, np.ndarray, float], Any]
    metadata: list[dict[str, Any]]
    code: str


def compile_input_driven_candidate(code: str) -> InputDrivenCandidate:
    """Validate the pH-only four-argument surface and compile with math-only globals."""
    validate_generated_math_code(code, dynamics_signature=INPUT_DRIVEN_SIGNATURE)
    tree = ast.parse(code)
    if any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "u"
        for node in ast.walk(tree)
    ):
        raise GeneratedCodeError("input-driven dynamics receives scalar causal u; indexing u is not allowed")
    namespace: dict[str, Any] = {"jnp": np, "__builtins__": {}}
    try:
        exec(compile(code, "<input_driven_candidate>", "exec"), namespace, namespace)
    except Exception as exc:
        raise GeneratedCodeError(f"input-driven candidate execution failed: {type(exc).__name__}: {exc}") from exc
    dynamics = namespace.get("dynamics")
    metadata = namespace.get("metadata")
    if not callable(dynamics) or not isinstance(metadata, list):
        raise GeneratedCodeError("validated input-driven candidate did not define callable dynamics and metadata")
    return InputDrivenCandidate(dynamics=dynamics, metadata=metadata, code=code)


def rollout_input_driven(
    candidate: InputDrivenCandidate,
    input_u: np.ndarray,
    params: np.ndarray,
    *,
    initial_state: float = 0.0,
    step_size: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Explicit-Euler causal rollout with a fresh declared state for one experiment."""
    signal = CausalInputSignal(input_u)
    output = np.empty(len(signal.values), dtype=float)
    output[0] = float(initial_state)
    max_abs_state = abs(float(initial_state))
    for index in range(1, len(output)):
        state = np.array([output[index - 1]], dtype=float)
        derivative = np.asarray(
            candidate.dynamics(float(index - 1), state, np.asarray(params, dtype=float), signal.at(float(index - 1))),
            dtype=float,
        ).reshape(-1)
        if derivative.shape != (1,) or not np.all(np.isfinite(derivative)):
            return np.full_like(output, np.nan), {"state_clipped": False, "derivative_clipped": False, "invalid_derivative": True}
        output[index] = output[index - 1] + step_size * derivative[0]
        max_abs_state = max(max_abs_state, abs(float(output[index])))
    return output, {"state_clipped": False, "derivative_clipped": False, "max_abs_state": max_abs_state, "input_hold": "U[floor(t)]"}


def _experiment_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or not np.all(np.isfinite(predicted)):
        return {"mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "late_nrmse": float("nan")}
    residual = predicted - observed
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    late = residual[int(0.75 * len(residual)):]
    mse = float(np.mean(np.square(residual)))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "nrmse": float(np.sqrt(mse) / scale), "late_nrmse": float(np.sqrt(np.mean(np.square(late))) / scale)}


def score_input_driven_experiments(
    experiments: list[PHExperiment],
    candidate: InputDrivenCandidate,
    params: np.ndarray,
    *,
    initial_state: float = 0.0,
) -> dict[str, Any]:
    """Score independent experiments without exposing their target Y to rollout."""
    rows = []
    for item in experiments:
        predicted, diagnostics = rollout_input_driven(candidate, item.input_u, params, initial_state=initial_state)
        metrics = _experiment_metrics(item.output_y, predicted)
        rows.append({"experiment_id": item.experiment_id, "initial_state_policy": "fixed_zero", "finite": bool(all(np.isfinite(value) for value in metrics.values())), "max_abs_output": diagnostics.get("max_abs_state"), "input_hold": diagnostics.get("input_hold"), **metrics})
    metric_names = ("mse", "rmse", "nrmse", "late_nrmse")
    aggregate = {f"median_{key}": float(np.median([row[key] for row in rows])) for key in metric_names}
    aggregate.update({f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in metric_names})
    aggregate.update({f"worst_{key}": float(np.max([row[key] for row in rows])) for key in metric_names})
    return {"per_experiment": rows, "aggregate": aggregate}


def evaluate_input_driven_candidate(
    data: dict[str, list[PHExperiment]],
    candidate: InputDrivenCandidate,
    params: np.ndarray,
    *,
    initial_state: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one candidate under the pH causal contract without fitting or ABC-SMC."""
    prescreen = prescreen_ph_reactor_rollouts(
        data,
        lambda u: rollout_input_driven(candidate, u, params, initial_state=initial_state),
    )
    if not prescreen["passed"]:
        failure_categories = sorted(
            {
                {
                    "nonfinite_output": "prescreen_nonfinite",
                    "output_bound_exceeded": "prescreen_bound_exceeded",
                    "clipping_active": "prescreen_clipping_active",
                }.get(row["reason"], "prescreen_rollout_failure")
                for row in prescreen["rows"]
                if not row["passed"]
            }
        )
        return {"status": "failed", "failure_modes": failure_categories, "prescreen": prescreen}
    scores = {split: score_input_driven_experiments(experiments, candidate, params, initial_state=initial_state) for split, experiments in data.items()}
    finite = all(row["finite"] for split in scores.values() for row in split["per_experiment"])
    return {
        "status": "success" if finite else "failed",
        "failure_modes": [] if finite else ["nonfinite_experiment_metrics"],
        "initial_state_policy": "fixed_zero",
        "prescreen": prescreen,
        "scores": scores,
        "train_equal_experiment_distance": scores["train"]["aggregate"]["mean_nrmse"],
        "selection_metric": {"validation_median_nrmse": scores["validation"]["aggregate"]["median_nrmse"], "validation_worst_nrmse": scores["validation"]["aggregate"]["worst_nrmse"]},
        "untouched_test_metric": {"test_median_nrmse": scores["test"]["aggregate"]["median_nrmse"], "test_late_nrmse": scores["test"]["aggregate"]["median_late_nrmse"]},
    }

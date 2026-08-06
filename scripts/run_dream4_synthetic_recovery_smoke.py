#!/usr/bin/env python3
"""Tiny non-LLM synthetic identifiability and negative-control smoke for DREAM4 recovery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.dream4 import (
    Dream4Trajectory,
    build_recovery_split,
    directed_edge_metrics,
    fit_sparse_linear_recovery,
    load_dream4_size10,
    predict_sparse_linear_recovery,
    recovery_metrics,
)
from scripts.legacy.real_dream4_data_loader import DEFAULT_SOURCE_ROOT


NOISE_STD = 0.001
SIGNAL_MAX_MEDIAN_NRMSE = 0.05
SIGNAL_MIN_MEDIAN_AUPR = 0.70
CONTROL_MIN_RELATIVE_DEGRADATION = 2.0
CONTROL_MIN_ABSOLUTE_NRMSE = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _simulate(a: np.ndarray, b: np.ndarray, time_points: np.ndarray, initial: np.ndarray) -> np.ndarray:
    dimension = len(initial)
    augmented = np.zeros((dimension + 1, dimension + 1), dtype=float)
    augmented[:dimension, :dimension] = a
    augmented[:dimension, dimension] = b
    values = []
    for elapsed in time_points:
        transition = expm(augmented * float(elapsed))
        values.append(transition[:dimension, :dimension] @ initial + transition[:dimension, dimension])
    output = np.asarray(values)
    if not np.all(np.isfinite(output)):
        raise ValueError("synthetic trajectory is non-finite")
    return output


def _planted_system(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    a = -np.diag(np.linspace(0.006, 0.015, dimension))
    for target, source, weight in ((1, 0, 0.0015), (3, 1, -0.0012), (5, 2, 0.0010), (7, 4, -0.0014), (9, 6, 0.0011), (2, 8, -0.0009), (6, 3, 0.0008), (8, 5, -0.0007)):
        a[target, source] = weight
    equilibrium = np.linspace(0.25, 0.70, dimension)
    return a, -a @ equilibrium


def _synthetic_trajectory(base: Dream4Trajectory, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Dream4Trajectory:
    time_points, observed = base.recovery_window()
    values = _simulate(a, b, time_points, observed[0])
    values = values + rng.normal(0.0, NOISE_STD, size=values.shape)
    values[0] = observed[0]
    expression = np.repeat(observed[:1], len(base.time_points), axis=0)
    recovery_index = int(np.flatnonzero(np.isclose(base.time_points, 500.0))[0])
    expression[recovery_index:] = values
    return Dream4Trajectory(base.network_id, base.trajectory_id, base.time_points, expression, base.gene_names)


def _wrong_recovery_start(external: Dream4Trajectory, donor: Dream4Trajectory) -> Dream4Trajectory:
    expression = external.expression.copy()
    recovery_index = int(np.flatnonzero(np.isclose(external.time_points, 500.0))[0])
    donor_values = donor.recovery_window()[1]
    expression[recovery_index] = donor_values[0]
    return Dream4Trajectory(external.network_id, external.trajectory_id, external.time_points, expression, external.gene_names)


def _time_scramble(trajectory: Dream4Trajectory, rng: np.random.Generator) -> Dream4Trajectory:
    expression = trajectory.expression.copy()
    recovery_index = int(np.flatnonzero(np.isclose(trajectory.time_points, 500.0))[0])
    recovery = expression[recovery_index:].copy()
    permutation = np.concatenate(([0], rng.permutation(np.arange(1, len(recovery)))))
    expression[recovery_index:] = recovery[permutation]
    return Dream4Trajectory(trajectory.network_id, trajectory.trajectory_id, trajectory.time_points, expression, trajectory.gene_names)


def _relative_error(recovered: np.ndarray, planted: np.ndarray) -> float:
    mask = ~np.eye(planted.shape[0], dtype=bool)
    return float(np.linalg.norm((recovered - planted)[mask]) / max(np.linalg.norm(planted[mask]), 1e-12))


def _finite_metrics(metrics: dict[str, float]) -> bool:
    return all(np.isfinite(float(value)) for value in metrics.values())


def run_smoke(source_root: Path = DEFAULT_SOURCE_ROOT) -> dict[str, Any]:
    rng = np.random.default_rng(20262101)
    networks = load_dream4_size10(source_root)
    a, b = _planted_system(10)
    planted_edges = (np.abs(a.T) > 0.0).astype(int)
    np.fill_diagonal(planted_edges, 0)
    rows = []
    for network_id, network in networks.items():
        split = build_recovery_split(network)
        base = {trajectory.trajectory_id: trajectory for trajectory in network.trajectories}
        synthetic = {trajectory_id: _synthetic_trajectory(trajectory, a, b, rng) for trajectory_id, trajectory in base.items()}
        train = [synthetic[index] for index in split["train_trajectory_ids"]]
        validation = synthetic[split["validation_trajectory_id"]]
        external = synthetic[split["external_trajectory_id"]]
        model = fit_sparse_linear_recovery(train, validation)
        prediction = predict_sparse_linear_recovery(external, model)
        observed = external.recovery_window()[1]
        signal_metrics = recovery_metrics(observed, prediction)
        topology = directed_edge_metrics(np.abs(model["a"].T), planted_edges)

        wrong_start = _wrong_recovery_start(external, synthetic[split["validation_trajectory_id"]])
        paired_prediction = predict_sparse_linear_recovery(wrong_start, model)
        pairing_metrics = recovery_metrics(observed, paired_prediction)

        scrambled_train = [_time_scramble(trajectory, rng) for trajectory in train]
        scrambled_validation = _time_scramble(validation, rng)
        scrambled_model = fit_sparse_linear_recovery(scrambled_train, scrambled_validation)
        scrambled_prediction = predict_sparse_linear_recovery(external, scrambled_model)
        time_order_metrics = recovery_metrics(observed, scrambled_prediction)

        rows.append(
            {
                "network_id": network_id,
                "finite": {
                    "signal": _finite_metrics(signal_metrics),
                    "recovery_state_pairing": _finite_metrics(pairing_metrics),
                    "time_order": _finite_metrics(time_order_metrics),
                },
                "signal": {
                    **signal_metrics,
                    "topology": topology,
                    "off_diagonal_a_relative_error": _relative_error(model["a"], a),
                    "nonzero_edges": int(np.count_nonzero(model["a"] * (~np.eye(10, dtype=bool)))),
                },
                "recovery_state_pairing_control": pairing_metrics,
                "time_order_control": time_order_metrics,
            }
        )

    signal_nrmse = np.asarray([row["signal"]["nrmse"] for row in rows], dtype=float)
    signal_aupr = np.asarray([row["signal"]["topology"]["aupr"] for row in rows], dtype=float)
    pairing_nrmse = np.asarray([row["recovery_state_pairing_control"]["nrmse"] for row in rows], dtype=float)
    order_nrmse = np.asarray([row["time_order_control"]["nrmse"] for row in rows], dtype=float)
    finite = all(all(check.values()) for row in rows for check in [row["finite"]])
    summary = {
        "phase": 21,
        "run_kind": "tiny_non_llm_synthetic_sparse_linear_recovery_smoke",
        "started_and_completed_at": utc_now(),
        "source_root": str(source_root),
        "spec": {
            "networks": 5,
            "recovery_grid": "actual per-network t=500..1000 grid",
            "train_validation_external": "3/1/1 trajectory blocks",
            "noise_std": NOISE_STD,
            "abc_smc_calls": 0,
            "llm_calls": 0,
        },
        "thresholds": {
            "finite_all_conditions": True,
            "signal_max_median_nrmse": SIGNAL_MAX_MEDIAN_NRMSE,
            "signal_min_median_aupr": SIGNAL_MIN_MEDIAN_AUPR,
            "control_min_relative_degradation": CONTROL_MIN_RELATIVE_DEGRADATION,
            "control_min_absolute_nrmse": CONTROL_MIN_ABSOLUTE_NRMSE,
        },
        "rows": rows,
        "aggregate": {
            "all_finite": finite,
            "signal_median_nrmse": float(np.median(signal_nrmse)),
            "signal_median_aupr": float(np.median(signal_aupr)),
            "pairing_median_nrmse": float(np.median(pairing_nrmse)),
            "time_order_median_nrmse": float(np.median(order_nrmse)),
            "pairing_relative_to_signal": float(np.median(pairing_nrmse) / max(np.median(signal_nrmse), 1e-12)),
            "time_order_relative_to_signal": float(np.median(order_nrmse) / max(np.median(signal_nrmse), 1e-12)),
        },
    }
    aggregate = summary["aggregate"]
    passed = (
        aggregate["all_finite"]
        and aggregate["signal_median_nrmse"] <= SIGNAL_MAX_MEDIAN_NRMSE
        and aggregate["signal_median_aupr"] >= SIGNAL_MIN_MEDIAN_AUPR
        and aggregate["pairing_relative_to_signal"] >= CONTROL_MIN_RELATIVE_DEGRADATION
        and aggregate["time_order_relative_to_signal"] >= CONTROL_MIN_RELATIVE_DEGRADATION
        and aggregate["pairing_median_nrmse"] >= CONTROL_MIN_ABSOLUTE_NRMSE
        and aggregate["time_order_median_nrmse"] >= CONTROL_MIN_ABSOLUTE_NRMSE
    )
    summary["decision"] = "passed" if passed else "failed"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/phase21_dream4_synthetic_recovery_smoke_20260712/summary.json"),
    )
    args = parser.parse_args()
    summary = run_smoke(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": summary["decision"], "aggregate": summary["aggregate"]}, indent=2))
    return 0 if summary["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

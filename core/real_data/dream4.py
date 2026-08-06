from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm


RECOVERY_TIME = 500.0


@dataclass(frozen=True)
class Dream4Trajectory:
    network_id: str
    trajectory_id: int
    time_points: np.ndarray
    expression: np.ndarray
    gene_names: tuple[str, ...]

    def recovery_window(self) -> tuple[np.ndarray, np.ndarray]:
        start = np.flatnonzero(np.isclose(self.time_points, RECOVERY_TIME))
        if len(start) != 1:
            raise ValueError(f"{self.network_id}/{self.trajectory_id} must contain exactly one t={RECOVERY_TIME:g}")
        index = int(start[0])
        return self.time_points[index:] - self.time_points[index], self.expression[index:]


@dataclass(frozen=True)
class Dream4Network:
    network_id: str
    gene_names: tuple[str, ...]
    trajectories: tuple[Dream4Trajectory, ...]
    gold_edges: np.ndarray | None


def _read_time_series(path: Path, network_id: str) -> tuple[tuple[str, ...], tuple[Dream4Trajectory, ...]]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    if not lines:
        raise ValueError(f"empty DREAM4 time-series file: {path}")
    header = tuple(value.strip('"') for value in lines[0].split('\t'))
    if header[0] != "Time" or len(header) != 11 or len(set(header[1:])) != 10:
        raise ValueError(f"unexpected DREAM4 Size-10 header in {path}: {header}")

    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in lines[1:]:
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        values = [float(value) for value in line.split('\t')]
        if len(values) != len(header) or not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite or malformed DREAM4 row in {path}: {line}")
        current.append(values)
    if current:
        blocks.append(current)
    if len(blocks) != 5:
        raise ValueError(f"{path} must contain five trajectory blocks, found {len(blocks)}")

    trajectories: list[Dream4Trajectory] = []
    for trajectory_id, rows in enumerate(blocks, start=1):
        array = np.asarray(rows, dtype=float)
        if array.shape != (21, 11):
            raise ValueError(f"{path} trajectory {trajectory_id} must have shape (21, 11), found {array.shape}")
        if not np.all(np.diff(array[:, 0]) > 0.0) or array[0, 0] != 0.0 or array[-1, 0] != 1000.0:
            raise ValueError(f"{path} trajectory {trajectory_id} has an invalid time grid")
        trajectories.append(
            Dream4Trajectory(
                network_id=network_id,
                trajectory_id=trajectory_id,
                time_points=array[:, 0],
                expression=array[:, 1:],
                gene_names=header[1:],
            )
        )
    return header[1:], tuple(trajectories)


def _read_gold_edges(path: Path, gene_names: tuple[str, ...]) -> np.ndarray:
    index = {gene: position for position, gene in enumerate(gene_names)}
    matrix = np.zeros((len(gene_names), len(gene_names)), dtype=int)
    rows = [line.strip().split('\t') for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(gene_names) * (len(gene_names) - 1):
        raise ValueError(f"{path} must contain every directed non-self edge exactly once")
    for row in rows:
        if len(row) != 3 or row[0] not in index or row[1] not in index or row[0] == row[1]:
            raise ValueError(f"invalid DREAM4 gold edge row in {path}: {row}")
        value = int(row[2])
        if value not in {0, 1}:
            raise ValueError(f"gold edge label must be binary in {path}: {row}")
        source, target = index[row[0]], index[row[1]]
        matrix[source, target] = value
    return matrix


def load_dream4_size10(
    source_root: str | Path,
    *,
    include_gold_edges: bool = False,
) -> dict[str, Dream4Network]:
    """Load fixed DREAM4 Size-10 source tables without fitting or normalization.

    Gold edges are opt-in to make it straightforward to keep them out of all
    train/validation fitting paths. The matrix orientation is [source, target].
    """
    root = Path(source_root)
    networks: dict[str, Dream4Network] = {}
    for index in range(1, 6):
        network_id = f"insilico_size10_{index}"
        directory = root / network_id
        genes, trajectories = _read_time_series(directory / "timeseries.tsv", network_id)
        gold = _read_gold_edges(directory / "goldStandard.tsv", genes) if include_gold_edges else None
        networks[network_id] = Dream4Network(network_id, genes, trajectories, gold)
    return networks


def build_recovery_split(network: Dream4Network) -> dict[str, Any]:
    """Return the fixed per-network 3/1/1 trajectory split for recovery scoring."""
    ids = [trajectory.trajectory_id for trajectory in network.trajectories]
    if ids != [1, 2, 3, 4, 5]:
        raise ValueError(f"unexpected trajectory IDs for {network.network_id}: {ids}")
    return {
        "network_id": network.network_id,
        "train_trajectory_ids": [1, 2, 3],
        "validation_trajectory_id": 4,
        "external_trajectory_id": 5,
        "recovery_start_time": RECOVERY_TIME,
        "metric_time_start": 550.0,
    }


def _selected(network: Dream4Network, ids: list[int]) -> list[Dream4Trajectory]:
    by_id = {trajectory.trajectory_id: trajectory for trajectory in network.trajectories}
    return [by_id[trajectory_id] for trajectory_id in ids]


def _recovery_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)[1:]
    predicted = np.asarray(predicted, dtype=float)[1:]
    if observed.shape != predicted.shape or observed.size == 0 or not np.all(np.isfinite(predicted)):
        return {"mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "late_nrmse": float("nan")}
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    late_start = max(0, int(np.floor(len(observed) * 0.75)))
    late_rmse = float(np.sqrt(np.mean(np.square(residual[late_start:]))))
    return {
        "mse": float(np.mean(np.square(residual))),
        "rmse": rmse,
        "nrmse": rmse / scale,
        "late_nrmse": late_rmse / scale,
    }


def _persistence_prediction(trajectory: Dream4Trajectory) -> np.ndarray:
    _, observed = trajectory.recovery_window()
    return np.repeat(observed[:1], len(observed), axis=0)


def _fit_exponential_relaxation(train: list[Dream4Trajectory]) -> dict[str, np.ndarray]:
    recovery = [trajectory.recovery_window() for trajectory in train]
    gene_count = recovery[0][1].shape[1]
    rates = np.empty(gene_count, dtype=float)
    equilibria = np.empty(gene_count, dtype=float)
    candidate_rates = np.geomspace(1e-6, 0.1, 200)
    for gene in range(gene_count):
        best: tuple[float, float, float] | None = None
        for rate in candidate_rates:
            numer = 0.0
            denom = 0.0
            error = 0.0
            cached: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for time_points, values in recovery:
                decay = np.exp(-rate * time_points)
                coefficient = 1.0 - decay
                target = values[:, gene] - values[0, gene] * decay
                numer += float(np.dot(coefficient, target))
                denom += float(np.dot(coefficient, coefficient))
                cached.append((coefficient, decay, values[:, gene]))
            equilibrium = numer / max(denom, 1e-12)
            for coefficient, decay, values in cached:
                prediction = equilibrium * coefficient + values[0] * decay
                error += float(np.sum(np.square(prediction - values)))
            candidate = (error, rate, equilibrium)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        _, rates[gene], equilibria[gene] = best
    return {"rates": rates, "equilibria": equilibria}


def _exponential_prediction(trajectory: Dream4Trajectory, model: dict[str, np.ndarray]) -> np.ndarray:
    time_points, observed = trajectory.recovery_window()
    decay = np.exp(-time_points[:, None] * model["rates"][None, :])
    return model["equilibria"][None, :] + (observed[:1] - model["equilibria"][None, :]) * decay


def _fit_sparse_linear_recovery(
    train: list[Dream4Trajectory],
    validation: Dream4Trajectory,
) -> dict[str, Any]:
    features: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for trajectory in train:
        time_points, values = trajectory.recovery_window()
        delta_t = np.diff(time_points)
        features.append(np.column_stack([values[:-1], np.ones(len(values) - 1)]))
        derivatives.append(np.diff(values, axis=0) / delta_t[:, None])
    x = np.vstack(features)
    y = np.vstack(derivatives)
    regularizations = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0.0
    best: dict[str, Any] | None = None
    for regularization in regularizations:
        coefficients = np.linalg.solve(x.T @ x + regularization * penalty, x.T @ y)
        a = coefficients[:-1].T
        b = coefficients[-1]
        threshold = max(float(np.max(np.abs(a))) * 0.05, 1e-12)
        sparse_a = np.where(np.abs(a) >= threshold, a, 0.0)
        candidate = {"a": sparse_a, "b": b, "regularization": regularization, "threshold": threshold}
        prediction, observed = _linear_prediction_and_observed(validation, candidate)
        metrics = _recovery_metrics(observed, prediction)
        score = metrics["nrmse"] if np.isfinite(metrics["nrmse"]) else float("inf")
        if best is None or score < best["validation_nrmse"]:
            best = {**candidate, "validation_nrmse": score}
    assert best is not None
    return best


def _linear_prediction_and_observed(trajectory: Dream4Trajectory, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    time_points, observed = trajectory.recovery_window()
    dimension = observed.shape[1]
    augmented = np.zeros((dimension + 1, dimension + 1), dtype=float)
    augmented[:dimension, :dimension] = model["a"]
    augmented[:dimension, dimension] = model["b"]
    predictions = []
    for elapsed in time_points:
        transition = expm(augmented * float(elapsed))
        predictions.append(transition[:dimension, :dimension] @ observed[0] + transition[:dimension, dimension])
    return np.asarray(predictions), observed


def _edge_metrics(scores: np.ndarray, gold_edges: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(scores.shape[0], dtype=bool)
    values = np.asarray(scores, dtype=float)[mask]
    labels = np.asarray(gold_edges, dtype=int)[mask]
    order = np.argsort(-values, kind="stable")
    ordered_labels = labels[order]
    positives = int(np.sum(labels))
    if positives <= 0 or positives >= len(labels):
        return {"aupr": float("nan"), "auroc": float("nan"), "edge_prevalence": float(np.mean(labels))}
    precision = np.cumsum(ordered_labels) / np.arange(1, len(ordered_labels) + 1)
    aupr = float(np.sum(precision * ordered_labels) / positives)
    positive_scores = values[labels == 1]
    negative_scores = values[labels == 0]
    comparisons = (positive_scores[:, None] > negative_scores[None, :]).mean()
    ties = (positive_scores[:, None] == negative_scores[None, :]).mean()
    return {
        "aupr": aupr,
        "auroc": float(comparisons + 0.5 * ties),
        "edge_prevalence": float(np.mean(labels)),
    }


def fit_sparse_linear_recovery(
    train: list[Dream4Trajectory],
    validation: Dream4Trajectory,
) -> dict[str, Any]:
    """Fit the train-only sparse linear recovery baseline."""
    return _fit_sparse_linear_recovery(train, validation)


def predict_sparse_linear_recovery(
    trajectory: Dream4Trajectory,
    model: dict[str, Any],
) -> np.ndarray:
    """Predict a trajectory's recovery window from its observed t=500 state."""
    return _linear_prediction_and_observed(trajectory, model)[0]


def recovery_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score only t=550..1000 recovery observations."""
    return _recovery_metrics(observed, predicted)


def directed_edge_metrics(scores: np.ndarray, gold_edges: np.ndarray) -> dict[str, float]:
    """Evaluate directed non-self edge rankings against an evaluation-only matrix."""
    return _edge_metrics(scores, gold_edges)


def evaluate_recovery_baselines(
    network: Dream4Network,
    split: dict[str, Any] | None = None,
    *,
    gold_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit baselines on train recovery windows and score the untouched external one.

    `gold_edges` is optional and passed only at evaluation time. It is never used
    by persistence, exponential, or linear-state-space fitting.
    """
    split = split or build_recovery_split(network)
    train = _selected(network, list(split["train_trajectory_ids"]))
    validation = _selected(network, [int(split["validation_trajectory_id"])])[0]
    external = _selected(network, [int(split["external_trajectory_id"])])[0]
    exponential = _fit_exponential_relaxation(train)
    linear = fit_sparse_linear_recovery(train, validation)
    persistence_prediction, observed = _persistence_prediction(external), external.recovery_window()[1]
    exponential_prediction = _exponential_prediction(external, exponential)
    linear_prediction = predict_sparse_linear_recovery(external, linear)
    result: dict[str, Any] = {
        "prediction_contract": "condition_on_observed_t500_state_predict_t550_to_t1000",
        "persistence": _recovery_metrics(observed, persistence_prediction),
        "exponential_relaxation": {**_recovery_metrics(observed, exponential_prediction), "rates": exponential["rates"].tolist()},
        "sparse_linear_state_space": {
            **_recovery_metrics(observed, linear_prediction),
            "regularization": float(linear["regularization"]),
            "threshold": float(linear["threshold"]),
            "nonzero_edges": int(np.count_nonzero(linear["a"])),
        },
    }
    if gold_edges is not None:
        result["sparse_linear_state_space"]["topology"] = directed_edge_metrics(np.abs(linear["a"].T), gold_edges)
    return result

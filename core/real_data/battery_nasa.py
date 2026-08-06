from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import jax
from scipy.io import loadmat

from core.evaluation import MaskedMetrics, masked_metrics
from core.sbi_engine import resolve_abc_smc_strategy


CANONICAL_COLUMNS = [
    "cell_id",
    "cycle_index",
    "elapsed_time_h",
    "ambient_temperature_c",
    "capacity_ah",
    "soh",
    "rul_cycles",
    "mean_discharge_current_a",
    "mean_temperature_c",
    "mean_voltage_v",
    "re_ohm",
    "rct_ohm",
]


@dataclass(frozen=True)
class CellTargetBundle:
    """One cell's SOH trajectory for opt-in grouped-cell evaluation."""

    cell_id: str
    time_points: np.ndarray
    observed: np.ndarray
    y0: np.ndarray
    late_cycle_mask: np.ndarray


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        return obj[name]
    except Exception:
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    try:
        scalar = float(np.ravel(arr)[0])
    except (TypeError, ValueError):
        return default
    return scalar if np.isfinite(scalar) else default


def _nanmean(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return None
    return float(np.nanmean(arr))


def _iter_cycles(mat_payload: dict[str, Any]) -> list[Any]:
    if "cycle" in mat_payload:
        cycles = mat_payload["cycle"]
    else:
        candidates = [value for key, value in mat_payload.items() if not key.startswith("__")]
        if not candidates:
            raise ValueError("MATLAB file does not contain a cycle structure")
        root = candidates[0]
        cycles = _get_field(root, "cycle", root)
    return list(np.ravel(cycles))


def read_nasa_mat_cycles(path: str | Path, *, cell_id: str | None = None) -> pd.DataFrame:
    """Read NASA PCoE MATLAB cycle structures into discharge-cycle rows."""
    path = Path(path)
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    inferred_cell = cell_id or path.stem
    rows: list[dict[str, Any]] = []
    last_re = None
    last_rct = None
    discharge_index = 0
    elapsed_time_h = 0.0

    for cycle in _iter_cycles(payload):
        cycle_type = str(_get_field(cycle, "type", "")).lower()
        data = _get_field(cycle, "data")
        if cycle_type == "impedance":
            last_re = _as_float(_get_field(data, "Re"), last_re)
            last_rct = _as_float(_get_field(data, "Rct"), last_rct)
            continue
        if cycle_type != "discharge":
            continue

        capacity = _as_float(_get_field(data, "Capacity"))
        if capacity is None:
            continue
        time_s = np.asarray(_get_field(data, "Time", []), dtype=float)
        duration_h = float(np.nanmax(time_s) / 3600.0) if time_s.size else 0.0
        rows.append(
            {
                "cell_id": inferred_cell,
                "cycle_index": discharge_index,
                "elapsed_time_h": elapsed_time_h,
                "ambient_temperature_c": _as_float(_get_field(cycle, "ambient_temperature")),
                "capacity_ah": capacity,
                "mean_discharge_current_a": _nanmean(_get_field(data, "Current_measured")),
                "mean_temperature_c": _nanmean(_get_field(data, "Temperature_measured")),
                "mean_voltage_v": _nanmean(_get_field(data, "Voltage_measured")),
                "re_ohm": last_re,
                "rct_ohm": last_rct,
            }
        )
        discharge_index += 1
        elapsed_time_h += duration_h

    if not rows:
        raise ValueError(f"no discharge capacity rows found in {path}")
    return pd.DataFrame(rows)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return read_nasa_mat_cycles(path)
    return pd.read_csv(path, sep=None, engine="python", comment="#")


def normalize_battery_table(raw_path: str | Path, *, eol_soh: float = 0.7) -> pd.DataFrame:
    """Normalize NASA battery cycle-level data for capacity-fade modeling."""
    raw = _read_table(raw_path)
    required = {"cell_id", "cycle_index", "capacity_ah"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"missing required NASA battery columns: {missing}")

    normalized = pd.DataFrame()
    normalized["cell_id"] = raw["cell_id"].astype(str)
    normalized["cycle_index"] = pd.to_numeric(raw["cycle_index"], errors="coerce").astype("Int64")
    if "elapsed_time_h" in raw:
        normalized["elapsed_time_h"] = pd.to_numeric(raw["elapsed_time_h"], errors="coerce")
    else:
        normalized["elapsed_time_h"] = normalized["cycle_index"].astype(float)
    normalized["ambient_temperature_c"] = pd.to_numeric(raw.get("ambient_temperature_c"), errors="coerce")
    normalized["capacity_ah"] = pd.to_numeric(raw["capacity_ah"], errors="coerce")
    normalized["mean_discharge_current_a"] = pd.to_numeric(raw.get("mean_discharge_current_a"), errors="coerce")
    normalized["mean_temperature_c"] = pd.to_numeric(raw.get("mean_temperature_c"), errors="coerce")
    normalized["mean_voltage_v"] = pd.to_numeric(raw.get("mean_voltage_v"), errors="coerce")
    normalized["re_ohm"] = pd.to_numeric(raw.get("re_ohm"), errors="coerce")
    normalized["rct_ohm"] = pd.to_numeric(raw.get("rct_ohm"), errors="coerce")
    normalized = normalized.dropna(subset=["cell_id", "cycle_index", "capacity_ah"])
    if normalized.empty:
        raise ValueError("NASA battery table contains no finite capacity observations")

    normalized = normalized.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    soh_parts = []
    rul_parts = []
    for _, cell_frame in normalized.groupby("cell_id", sort=False):
        initial_capacity = float(cell_frame.iloc[0]["capacity_ah"])
        if initial_capacity <= 0:
            raise ValueError("initial capacity must be positive")
        soh = cell_frame["capacity_ah"].astype(float) / initial_capacity
        below_eol = cell_frame.loc[soh <= eol_soh, "cycle_index"].astype(int)
        eol_cycle = int(below_eol.iloc[0]) if not below_eol.empty else int(cell_frame["cycle_index"].max())
        soh_parts.append(soh)
        rul_parts.append((eol_cycle - cell_frame["cycle_index"].astype(int)).clip(lower=0))
    normalized["soh"] = pd.concat(soh_parts).sort_index()
    normalized["rul_cycles"] = pd.concat(rul_parts).sort_index().astype(int)
    return normalized[CANONICAL_COLUMNS]


def build_capacity_target(
    normalized: pd.DataFrame,
    *,
    cell_id: str | None = None,
    target_column: str = "soh",
) -> tuple[np.ndarray, np.ndarray]:
    if target_column not in {"soh", "capacity_ah"}:
        raise ValueError("target_column must be `soh` or `capacity_ah`")
    selected_cell = str(cell_id) if cell_id is not None else sorted(normalized["cell_id"].astype(str).unique())[0]
    cell = normalized[normalized["cell_id"].astype(str) == selected_cell].sort_values("cycle_index")
    if cell.empty:
        raise ValueError(f"unknown NASA battery cell_id: {selected_cell}")
    if len(cell) < 3:
        raise ValueError("capacity target requires at least three discharge cycles")
    time_points = cell["cycle_index"].to_numpy(dtype=float)
    target = cell[target_column].to_numpy(dtype=float).reshape(-1, 1)
    return time_points, target


def build_cell_target_bundles(
    normalized: pd.DataFrame,
    cell_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
    *,
    target_column: str = "soh",
    late_cycle_train_fraction: float = 0.8,
) -> list[CellTargetBundle]:
    """Build per-cell trajectories without mixing external-cell observations.

    The initial state is specific to each cell and late-cycle masks retain the
    existing chronological within-cell evaluation convention. This is opt-in;
    the dense one-cell target path remains unchanged.
    """
    if not 0.0 < late_cycle_train_fraction < 1.0:
        raise ValueError("late_cycle_train_fraction must be between 0 and 1")
    available = normalized["cell_id"].astype(str)
    selected_cells = (
        [str(value) for value in cell_ids]
        if cell_ids is not None
        else sorted(available.unique().tolist())
    )
    if not selected_cells:
        raise ValueError("at least one cell_id is required")

    bundles: list[CellTargetBundle] = []
    for cell_id in selected_cells:
        time_points, observed = build_capacity_target(
            normalized,
            cell_id=cell_id,
            target_column=target_column,
        )
        split_index = int(len(time_points) * late_cycle_train_fraction)
        split_index = min(max(split_index, 1), len(time_points) - 1)
        late_cycle_mask = np.zeros_like(observed, dtype=bool)
        late_cycle_mask[split_index:] = True
        bundles.append(
            CellTargetBundle(
                cell_id=cell_id,
                time_points=time_points,
                observed=observed,
                y0=observed[0].astype(float, copy=True),
                late_cycle_mask=late_cycle_mask,
            )
        )
    return bundles


def build_cell_holdout_bundles(
    normalized: pd.DataFrame,
    holdout_split: dict[str, Any],
    *,
    target_column: str = "soh",
    late_cycle_train_fraction: float = 0.8,
) -> dict[str, list[CellTargetBundle]]:
    """Build disjoint train and external-cell bundles from a recorded split."""
    if holdout_split.get("status") != "ok":
        raise ValueError(f"cell holdout split not available: {holdout_split.get('reason', 'unknown reason')}")
    train_cells = [str(value) for value in holdout_split.get("train_cells", [])]
    test_cells = [str(value) for value in holdout_split.get("test_cells", [])]
    if not train_cells or not test_cells:
        raise ValueError("cell holdout split requires non-empty train_cells and test_cells")
    overlap = sorted(set(train_cells) & set(test_cells))
    if overlap:
        raise ValueError(f"cell holdout train/test overlap: {overlap}")
    return {
        "train": build_cell_target_bundles(
            normalized,
            train_cells,
            target_column=target_column,
            late_cycle_train_fraction=late_cycle_train_fraction,
        ),
        "test": build_cell_target_bundles(
            normalized,
            test_cells,
            target_column=target_column,
            late_cycle_train_fraction=late_cycle_train_fraction,
        ),
    }


def build_leave_one_cell_out_splits(
    normalized: pd.DataFrame,
    cell_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic folds for conditional external-cell robustness checks.

    Each held-out trajectory retains its own observed initial SOH through
    ``CellTargetBundle.y0``. The folds therefore test conditional trajectory
    forecasting, consistent with the existing external-cell contract, without
    leaking later test-cell observations into parameter fitting.
    """
    available = sorted(normalized["cell_id"].astype(str).unique().tolist())
    selected = sorted({str(value) for value in cell_ids}) if cell_ids is not None else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown cell_ids for leave-one-cell-out: {unknown}")
    if len(selected) < 2:
        raise ValueError("leave-one-cell-out requires at least two cells")

    folds = []
    for test_cell in selected:
        train_cells = [cell_id for cell_id in selected if cell_id != test_cell]
        folds.append(
            {
                "schema_version": 1,
                "status": "ok",
                "fold_id": f"test_{test_cell}",
                "train_cells": train_cells,
                "test_cells": [test_cell],
                "prediction_contract": "condition_on_observed_test_cell_initial_soh",
            }
        )
    return folds


def _metrics_dict(metrics: MaskedMetrics) -> dict[str, Any]:
    return {
        "mse": float(metrics.mse),
        "rmse": float(metrics.rmse),
        "observed_count": int(metrics.observed_count),
    }


def cell_prediction_metrics(bundle: CellTargetBundle, predicted: np.ndarray) -> dict[str, Any]:
    """Score a prediction for one cell, including its late-cycle tail."""
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != bundle.observed.shape:
        raise ValueError(f"prediction shape mismatch for cell {bundle.cell_id}: {predicted.shape} != {bundle.observed.shape}")
    return {
        "cell_id": bundle.cell_id,
        "aggregate": _metrics_dict(masked_metrics(bundle.observed, predicted)),
        "late_cycle": _metrics_dict(masked_metrics(bundle.observed, predicted, bundle.late_cycle_mask)),
    }


def _prediction_for_cell(
    predictions_by_cell: dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
    bundle_index: int,
    cell_id: str,
) -> np.ndarray:
    if isinstance(predictions_by_cell, dict):
        if cell_id not in predictions_by_cell:
            raise ValueError(f"missing prediction for cell {cell_id}")
        return predictions_by_cell[cell_id]
    if bundle_index >= len(predictions_by_cell):
        raise ValueError(f"missing prediction at bundle index {bundle_index}")
    return predictions_by_cell[bundle_index]


def evaluate_cell_predictions(
    bundles: list[CellTargetBundle],
    predictions_by_cell: dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """Score grouped cell predictions with observation-count weighted metrics."""
    if not bundles:
        raise ValueError("at least one cell bundle is required")

    per_cell = []
    observed_values = []
    predicted_values = []
    late_observed_values = []
    late_predicted_values = []
    for index, bundle in enumerate(bundles):
        predicted = np.asarray(_prediction_for_cell(predictions_by_cell, index, bundle.cell_id), dtype=float)
        if predicted.shape != bundle.observed.shape:
            raise ValueError(f"prediction shape mismatch for cell {bundle.cell_id}: {predicted.shape} != {bundle.observed.shape}")
        per_cell.append(cell_prediction_metrics(bundle, predicted))
        observed_values.append(bundle.observed.ravel())
        predicted_values.append(predicted.ravel())
        late_observed_values.append(bundle.observed[bundle.late_cycle_mask])
        late_predicted_values.append(predicted[bundle.late_cycle_mask])

    aggregate = masked_metrics(np.concatenate(observed_values), np.concatenate(predicted_values))
    late_cycle = masked_metrics(np.concatenate(late_observed_values), np.concatenate(late_predicted_values))
    return {
        "cell_count": len(bundles),
        "aggregate": _metrics_dict(aggregate),
        "late_cycle": _metrics_dict(late_cycle),
        "per_cell": per_cell,
    }


def predict_cell_bundles(
    model: Any,
    params: np.ndarray,
    bundles: list[CellTargetBundle],
) -> dict[str, np.ndarray]:
    """Simulate one global parameter vector against each independent cell."""
    if not bundles:
        raise ValueError("at least one cell bundle is required")
    params = np.asarray(params, dtype=float)
    if params.ndim != 1:
        raise ValueError(f"params must be one-dimensional, got shape {params.shape}")

    predictions: dict[str, np.ndarray] = {}
    for bundle in bundles:
        predicted = np.asarray(model.simulate(params, bundle.time_points, bundle.y0), dtype=float)
        if predicted.shape != bundle.observed.shape:
            raise ValueError(f"prediction shape mismatch for cell {bundle.cell_id}: {predicted.shape} != {bundle.observed.shape}")
        predictions[bundle.cell_id] = predicted
    return predictions


def grouped_cell_distances(
    model: Any,
    params_batch: np.ndarray,
    bundles: list[CellTargetBundle],
) -> np.ndarray:
    """Compute observation-weighted train-cell RMSE for a parameter batch."""
    if not bundles:
        raise ValueError("at least one cell bundle is required")
    params_batch = np.asarray(params_batch, dtype=float)
    if params_batch.ndim == 1:
        params_batch = params_batch[np.newaxis, :]
    if params_batch.ndim != 2:
        raise ValueError(f"params_batch must be one- or two-dimensional, got shape {params_batch.shape}")

    distances = []
    for params in params_batch:
        squared_error_sum = 0.0
        observed_count = 0
        for bundle in bundles:
            predicted = np.asarray(model.simulate(params, bundle.time_points, bundle.y0), dtype=float)
            if predicted.shape != bundle.observed.shape:
                raise ValueError(f"prediction shape mismatch for cell {bundle.cell_id}: {predicted.shape} != {bundle.observed.shape}")
            observed_count += int(bundle.observed.size)
            if not np.all(np.isfinite(predicted)):
                squared_error_sum = float("inf")
                break
            residuals = predicted - bundle.observed
            squared_error_sum += float(np.sum(np.square(residuals)))
        if observed_count == 0:
            raise ValueError("cell bundles must include at least one observation")
        distances.append(float(np.sqrt(squared_error_sum / observed_count)))
    return np.asarray(distances, dtype=float)


def grouped_cell_distance(model: Any, params: np.ndarray, bundles: list[CellTargetBundle]) -> float:
    """Compute grouped train-cell RMSE for one global parameter vector."""
    return float(grouped_cell_distances(model, params, bundles)[0])


def persistence_cell_predictions(bundles: list[CellTargetBundle]) -> dict[str, np.ndarray]:
    """Predict each cell's SOH as its observed initial state at every cycle."""
    if not bundles:
        raise ValueError("at least one cell bundle is required")
    return {
        bundle.cell_id: np.repeat(bundle.y0.reshape(1, -1), len(bundle.time_points), axis=0)
        for bundle in bundles
    }


def fit_global_linear_decay(train_bundles: list[CellTargetBundle]) -> float:
    """Fit one no-intercept SOH degradation slope using train cells only."""
    if not train_bundles:
        raise ValueError("at least one train cell bundle is required")
    time_values = np.concatenate([bundle.time_points for bundle in train_bundles])
    residual_values = np.concatenate(
        [(bundle.observed[:, 0] - bundle.y0[0]) for bundle in train_bundles]
    )
    denominator = float(np.dot(time_values, time_values))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("train-cell time points must have positive finite variation")
    slope = float(np.dot(time_values, residual_values) / denominator)
    if not np.isfinite(slope):
        raise ValueError("global linear-decay slope is non-finite")
    return slope


def linear_decay_cell_predictions(
    bundles: list[CellTargetBundle],
    slope_per_cycle: float,
) -> dict[str, np.ndarray]:
    """Predict independent cell trajectories from a shared linear fade slope."""
    if not np.isfinite(slope_per_cycle):
        raise ValueError("slope_per_cycle must be finite")
    return {
        bundle.cell_id: bundle.y0.reshape(1, -1) + float(slope_per_cycle) * bundle.time_points.reshape(-1, 1)
        for bundle in bundles
    }


def evaluate_cell_holdout_baselines(
    train_bundles: list[CellTargetBundle],
    test_bundles: list[CellTargetBundle],
) -> dict[str, Any]:
    """Score honest persistence and train-fitted linear-decay cell baselines."""
    persistence_train = evaluate_cell_predictions(train_bundles, persistence_cell_predictions(train_bundles))
    persistence_test = evaluate_cell_predictions(test_bundles, persistence_cell_predictions(test_bundles))
    slope_per_cycle = fit_global_linear_decay(train_bundles)
    linear_train = evaluate_cell_predictions(
        train_bundles,
        linear_decay_cell_predictions(train_bundles, slope_per_cycle),
    )
    linear_test = evaluate_cell_predictions(
        test_bundles,
        linear_decay_cell_predictions(test_bundles, slope_per_cycle),
    )
    return {
        "persistence": {"train": persistence_train, "test": persistence_test},
        "linear_decay": {
            "slope_per_cycle": slope_per_cycle,
            "train": linear_train,
            "test": linear_test,
        },
    }


def _validate_parameter_metadata(metadata: list[dict[str, Any]]) -> None:
    if not metadata:
        raise ValueError("model must define at least one parameter prior")
    for index, prior in enumerate(metadata):
        bounds = prior.get("range")
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"prior {index} must include a two-value range")
        low, high = (float(bounds[0]), float(bounds[1]))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"prior {index} has invalid range: {bounds}")


def _sample_cell_priors(metadata: list[dict[str, Any]], n_particles: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack(
        [rng.uniform(float(prior["range"][0]), float(prior["range"][1]), n_particles) for prior in metadata]
    )


def run_grouped_cell_abc_smc(
    model: Any,
    train_bundles: list[CellTargetBundle],
    *,
    target_samples: int,
    generations: int,
    initial_particles: int,
    strategy: str = "gaussian_weighted",
    lambda_noise: float = 0.01,
    nugget: float = 1e-9,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run ABC-SMC over shared parameters and independent train-cell trajectories.

    This intentionally mirrors the repository's sequential transition strategies
    while keeping the distance evaluation in the battery adapter, where cells
    may have distinct time grids and initial states.
    """
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if generations <= 0:
        raise ValueError("generations must be positive")
    if initial_particles < target_samples:
        raise ValueError("initial_particles must be >= target_samples")
    if not train_bundles:
        raise ValueError("at least one train cell bundle is required")

    metadata = model.get_parameter_metadata()
    _validate_parameter_metadata(metadata)
    rng = np.random.default_rng(seed)
    smc_strategy = resolve_abc_smc_strategy(strategy)
    params_batch = _sample_cell_priors(metadata, initial_particles, rng)
    accepted_params = None
    accepted_weights = None
    previous_params = None
    previous_weights = None
    kernel_cov = None
    generation_history = []

    for generation in range(generations):
        if generation > 0:
            params_batch, kernel_cov = smc_strategy.transition(
                rng=rng,
                accepted_params=accepted_params,
                accepted_weights=accepted_weights,
                priors=metadata,
                initial_particles=initial_particles,
                lambda_noise=lambda_noise,
                nugget=nugget,
            )
            params_batch = np.asarray(params_batch, dtype=float)

        distances = grouped_cell_distances(model, params_batch, train_bundles)
        distances = np.asarray(distances, dtype=float)
        distances[~np.isfinite(distances)] = np.inf
        finite_idx = np.where(np.isfinite(distances))[0]
        if len(finite_idx) < target_samples:
            return {
                "accepted_params": np.empty((0, len(metadata))),
                "median_distance": float("inf"),
                "min_distance": float("inf"),
                "requested_strategy": strategy,
                "effective_strategy": smc_strategy.name,
                "generation_history": generation_history,
                "failure": f"generation {generation} produced only {len(finite_idx)} finite particles; need {target_samples}",
            }

        keep_idx = finite_idx[np.argsort(distances[finite_idx])[:target_samples]]
        accepted_params = np.asarray(params_batch, dtype=float)[keep_idx]
        accepted_distances = distances[keep_idx]
        if generation == 0:
            accepted_weights = np.full(target_samples, 1.0 / target_samples)
        else:
            if kernel_cov is None:
                accepted_weights = np.full(target_samples, 1.0 / target_samples)
            else:
                densities = []
                for candidate in accepted_params:
                    try:
                        density = np.asarray(
                            jax.scipy.stats.multivariate_normal.pdf(
                                candidate,
                                mean=np.atleast_2d(previous_params),
                                cov=kernel_cov,
                            )
                        )
                        densities.append(float(np.sum(density * previous_weights)))
                    except Exception:
                        densities.append(0.0)
                inverse_density = 1.0 / (np.asarray(densities, dtype=float) + 1e-12)
                if not np.all(np.isfinite(inverse_density)) or float(np.sum(inverse_density)) <= 0.0:
                    accepted_weights = np.full(target_samples, 1.0 / target_samples)
                else:
                    accepted_weights = inverse_density / np.sum(inverse_density)

        previous_params = accepted_params.copy()
        previous_weights = accepted_weights.copy()
        generation_history.append(
            {
                "generation": generation,
                "finite_particle_count": int(len(finite_idx)),
                "epsilon": float(accepted_distances[-1]),
                "median_distance": float(np.median(accepted_distances)),
                "min_distance": float(np.min(accepted_distances)),
            }
        )

    return {
        "accepted_params": accepted_params,
        "accepted_weights": accepted_weights,
        "median_distance": float(np.median(accepted_distances)),
        "min_distance": float(np.min(accepted_distances)),
        "requested_strategy": strategy,
        "effective_strategy": smc_strategy.name,
        "generation_history": generation_history,
    }


def cell_summary(normalized: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    for cell_id, cell_frame in normalized.groupby("cell_id", sort=True):
        ordered = cell_frame.sort_values("cycle_index")
        summaries.append(
            {
                "cell_id": str(cell_id),
                "cycle_count": int(len(ordered)),
                "min_cycle_index": int(ordered["cycle_index"].min()),
                "max_cycle_index": int(ordered["cycle_index"].max()),
                "initial_capacity_ah": float(ordered.iloc[0]["capacity_ah"]),
                "final_capacity_ah": float(ordered.iloc[-1]["capacity_ah"]),
                "initial_soh": float(ordered.iloc[0]["soh"]),
                "final_soh": float(ordered.iloc[-1]["soh"]),
                "eol_reached": bool(np.any(ordered["soh"].to_numpy(dtype=float) <= 0.7)),
            }
        )
    return summaries


def build_cell_holdout_split(
    normalized: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    cells = np.array(sorted(normalized["cell_id"].astype(str).unique().tolist()), dtype=object)
    if len(cells) < 2:
        raise ValueError("cell holdout requires at least two cells")
    rng = np.random.default_rng(seed)
    shuffled = cells.copy()
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_fraction))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_cells = sorted(str(value) for value in shuffled[:train_count])
    test_cells = sorted(str(value) for value in shuffled[train_count:])
    return {
        "schema_version": 1,
        "status": "ok",
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "train_cells": train_cells,
        "test_cells": test_cells,
        "train_cycle_count": int(normalized[normalized["cell_id"].astype(str).isin(train_cells)].shape[0]),
        "test_cycle_count": int(normalized[normalized["cell_id"].astype(str).isin(test_cells)].shape[0]),
    }


def write_cell_holdout_split(
    normalized: pd.DataFrame,
    *,
    path: str | Path,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_cell_holdout_split(normalized, train_fraction=train_fraction, seed=seed)
    except ValueError as exc:
        payload = {
            "schema_version": 1,
            "status": "skipped",
            "reason": str(exc),
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "cell_count": int(normalized["cell_id"].astype(str).nunique()),
        }
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_within_cell_time_splits(
    normalized: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    cells = []
    for cell_id, cell_frame in normalized.groupby("cell_id", sort=True):
        ordered = cell_frame.sort_values("cycle_index")
        if len(ordered) < 3:
            cells.append(
                {
                    "cell_id": str(cell_id),
                    "status": "skipped",
                    "reason": "within-cell split requires at least three cycles",
                    "cycle_count": int(len(ordered)),
                }
            )
            continue
        split_index = int(len(ordered) * train_fraction)
        split_index = min(max(split_index, 1), len(ordered) - 1)
        train_cycles = ordered.iloc[:split_index]["cycle_index"].astype(int).tolist()
        test_cycles = ordered.iloc[split_index:]["cycle_index"].astype(int).tolist()
        cells.append(
            {
                "cell_id": str(cell_id),
                "status": "ok",
                "train_fraction": float(train_fraction),
                "split_index": int(split_index),
                "train_cycles": train_cycles,
                "test_cycles": test_cycles,
                "train_cycle_count": int(len(train_cycles)),
                "test_cycle_count": int(len(test_cycles)),
            }
        )
    return {
        "schema_version": 1,
        "status": "ok" if any(cell["status"] == "ok" for cell in cells) else "skipped",
        "train_fraction": float(train_fraction),
        "cells": cells,
    }


def write_within_cell_time_splits(
    normalized: pd.DataFrame,
    *,
    path: str | Path,
    train_fraction: float = 0.8,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_within_cell_time_splits(normalized, train_fraction=train_fraction)
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_normalized_artifacts(
    raw_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    normalized_csv: str | Path,
    provenance_json: str | Path,
    source_url: str,
) -> pd.DataFrame:
    if isinstance(raw_path, (list, tuple)):
        if not raw_path:
            raise ValueError("at least one NASA battery raw path is required")
        normalized_parts = [normalize_battery_table(path) for path in raw_path]
        normalized = pd.concat(normalized_parts, ignore_index=True).sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
        raw_path_value: str | list[str] = [str(path) for path in raw_path]
    else:
        normalized = normalize_battery_table(raw_path)
        raw_path_value = str(raw_path)
    normalized_path = Path(normalized_csv)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(normalized_path, index=False)

    provenance = {
        "source_url": source_url,
        "raw_path": raw_path_value,
        "normalized_csv": str(normalized_path),
        "schema_version": 1,
        "row_count": int(len(normalized)),
        "cell_count": int(normalized["cell_id"].nunique()),
        "cells": sorted(normalized["cell_id"].astype(str).unique().tolist()),
        "cell_summary": cell_summary(normalized),
        "target": "cycle-level discharge capacity and state of health",
        "notes": [
            "Fixture rows are schema/smoke data only, not a scientific NASA battery result.",
            "First runnable target uses one-cell SOH capacity fade; promotion should require cell-level holdout.",
        ],
    }
    Path(provenance_json).write_text(__import__("json").dumps(provenance, indent=2, sort_keys=True) + "\n")
    return normalized


def write_capacity_target(
    normalized: pd.DataFrame,
    *,
    data_dir: str | Path = "data",
    domain: str = "real_battery_nasa_capacity",
    cell_id: str | None = None,
    target_column: str = "soh",
) -> dict[str, Any]:
    time_points, target = build_capacity_target(normalized, cell_id=cell_id, target_column=target_column)
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = out_dir / f"{domain}_ground_truth.npy"
    time_points_path = out_dir / f"{domain}_time_points.npy"
    np.save(ground_truth_path, target)
    np.save(time_points_path, time_points)
    return {
        "domain": domain,
        "cell_id": cell_id or "first_sorted_cell",
        "ground_truth_path": str(ground_truth_path),
        "time_points_path": str(time_points_path),
        "target_column": target_column,
        "time_points": int(len(time_points)),
        "state_count": int(target.shape[1]),
    }

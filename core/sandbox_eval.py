from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

# Ensure path includes workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.domain_configs import DOMAIN_CONFIGS
from core.evaluation_boundary import (
    CANONICAL_PROTOCOL_VERSION,
    DEVELOPMENT_ROLES,
    load_partition_manifest,
)
from core.evaluation import split_time_series, train_test_metrics
from core.generated_code import GeneratedCodeError, validate_generated_math_code
from core.real_data.warfarin_pkpd import (
    build_subject_target_bundles,
    evaluate_subject_predictions,
    grouped_subject_distances,
    predict_subject_bundles,
)
from core.real_data.battery_nasa import (
    build_cell_holdout_bundles,
    evaluate_cell_predictions,
    grouped_cell_distances,
    predict_cell_bundles,
    run_grouped_cell_abc_smc,
)
from core.sbi_engine import SBIEngine


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def held_out_metric_failure(
    train_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    *,
    held_out_label: str = "test",
) -> dict[str, str] | None:
    invalid_fields: list[str] = []
    for split, metrics in (("train", train_metrics), (held_out_label, test_metrics)):
        for metric in ("mse", "rmse"):
            value = None if metrics is None else metrics.get(metric)
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                scalar = float("nan")
            if not np.isfinite(scalar):
                invalid_fields.append(f"{split}.{metric}")
    if not invalid_fields:
        return None
    fields = ", ".join(invalid_fields)
    return {
        "category": "nonfinite_held_out_metrics",
        "detail": f"Posterior-median full trajectory produced missing or non-finite held-out metrics: {fields}.",
    }


def finite_metric_mapping(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    sanitized = dict(metrics)
    for metric in ("mse", "rmse"):
        sanitized[metric] = _safe_float(metrics.get(metric))
    return sanitized


def _segment_bounds(length: int) -> list[tuple[str, int, int]]:
    if length <= 0:
        return []
    if length < 3:
        return [("full", 0, length)]
    edges = np.linspace(0, length, 4, dtype=int)
    labels = ("early", "mid", "late")
    segments: list[tuple[str, int, int]] = []
    for index, label in enumerate(labels):
        start = int(edges[index])
        end = int(edges[index + 1])
        if end > start:
            segments.append((label, start, end))
    return segments or [("full", 0, length)]


def summarize_residuals(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {observed.shape} != {predicted.shape}")
    if observed.ndim != 2:
        raise ValueError(f"expected 2D trajectories, received shape {observed.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {observed.shape}")

    residuals = predicted - observed
    segments = []
    for label, start, end in _segment_bounds(len(observed)):
        segment_residuals = residuals[start:end]
        segment_mask = mask[start:end] if mask is not None else None
        state_summaries = []
        for state_index in range(segment_residuals.shape[1]):
            state_values = segment_residuals[:, state_index]
            if segment_mask is not None:
                state_values = state_values[segment_mask[:, state_index]]
            if state_values.size == 0:
                state_summaries.append(
                    {
                        "state": f"state_{state_index}",
                        "observed_count": 0,
                        "mean_error": None,
                        "mae": None,
                        "rmse": None,
                        "max_abs_error": None,
                    }
                )
                continue
            state_summaries.append(
                {
                    "state": f"state_{state_index}",
                    "observed_count": int(state_values.size),
                    "mean_error": _safe_float(np.mean(state_values)),
                    "mae": _safe_float(np.mean(np.abs(state_values))),
                    "rmse": _safe_float(np.sqrt(np.mean(np.square(state_values)))),
                    "max_abs_error": _safe_float(np.max(np.abs(state_values))),
                }
            )
        segments.append(
            {
                "label": label,
                "start_index": start,
                "end_index": end - 1,
                "states": state_summaries,
            }
        )

    global_values = residuals[mask] if mask is not None else residuals
    return {
        "global": {
            "observed_count": int(global_values.size),
            "mae": _safe_float(np.mean(np.abs(global_values))) if global_values.size else None,
            "rmse": _safe_float(np.sqrt(np.mean(np.square(global_values)))) if global_values.size else None,
            "max_abs_error": _safe_float(np.max(np.abs(global_values))) if global_values.size else None,
        },
        "segments": segments,
    }


def _segment_mean_rmse(segment: dict[str, Any]) -> float | None:
    states = segment.get("states") or []
    values = [float(state["rmse"]) for state in states if state.get("rmse") is not None]
    if not values:
        return None
    return float(np.mean(values))


def build_diagnostic_packet(
    *,
    res: dict[str, Any] | None,
    metadata: list[dict[str, Any]] | None,
    status: str,
    observed: np.ndarray | None = None,
    predicted: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
    split_idx: int | None = None,
    train_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
    validation_metrics: dict[str, Any] | None = None,
    evaluation_protocol: str = "legacy_held_out",
    data_receipt: dict[str, Any] | None = None,
    error: str | None = None,
    extra_failure_modes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    accepted_params = None
    if res and res.get("accepted_params") is not None:
        accepted_params = np.atleast_2d(np.asarray(res["accepted_params"], dtype=float))
        if accepted_params.size == 0:
            accepted_params = None

    best_parameters = None
    posterior_summary = None
    bound_pressure = None
    warnings: list[str] = []
    failure_modes: list[dict[str, str]] = [dict(mode) for mode in (extra_failure_modes or [])]

    if accepted_params is not None and metadata:
        parameter_summaries = []
        bound_pressure = []
        medians = np.median(accepted_params, axis=0)
        best_parameters = {}
        for index, prior in enumerate(metadata):
            name = prior.get("name", f"param_{index}")
            low, high = prior["range"]
            span = max(float(high) - float(low), 1e-12)
            tolerance = span * 0.05
            values = accepted_params[:, index]
            median_value = float(medians[index])
            best_parameters[name] = median_value
            parameter_summaries.append(
                {
                    "name": name,
                    "median": median_value,
                    "p10": _safe_float(np.percentile(values, 10)),
                    "p90": _safe_float(np.percentile(values, 90)),
                    "std": _safe_float(np.std(values)),
                }
            )
            lower_fraction = float(np.mean(values <= float(low) + tolerance))
            upper_fraction = float(np.mean(values >= float(high) - tolerance))
            bound_pressure.append(
                {
                    "name": name,
                    "lower_fraction": lower_fraction,
                    "upper_fraction": upper_fraction,
                    "median_relative_position": _safe_float((median_value - float(low)) / span),
                }
            )
            if lower_fraction >= 0.5:
                warnings.append(f"Posterior mass for `{name}` is concentrated near the lower prior bound.")
            if upper_fraction >= 0.5:
                warnings.append(f"Posterior mass for `{name}` is concentrated near the upper prior bound.")
        posterior_summary = {
            "accepted_particle_count": int(accepted_params.shape[0]),
            "parameters": parameter_summaries,
        }
    elif status != "success":
        failure_modes.append(
            {
                "category": "no_accepted_particles",
                "detail": "ABC-SMC did not return a usable accepted parameter population.",
            }
        )

    residual_summary = None
    if observed is not None and predicted is not None:
        residual_summary = summarize_residuals(observed, predicted, observed_mask)

    development_metrics = validation_metrics if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else test_metrics
    development_label = "Validation" if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else "Held-out"
    if train_metrics and development_metrics:
        train_rmse = train_metrics.get("rmse")
        development_rmse = development_metrics.get("rmse")
        if train_rmse and development_rmse and train_rmse > 0 and development_rmse >= train_rmse * 1.25:
            warnings.append(
                f"{development_label} RMSE is {development_rmse / train_rmse:.2f}x train RMSE; candidate may generalize poorly."
            )

    if residual_summary:
        segments = residual_summary.get("segments") or []
        if len(segments) >= 2:
            early_rmse = _segment_mean_rmse(segments[0])
            late_rmse = _segment_mean_rmse(segments[-1])
            if early_rmse and late_rmse and early_rmse > 0 and late_rmse >= early_rmse * 1.5:
                warnings.append(
                    f"Residual RMSE grows from {segments[0]['label']} to {segments[-1]['label']} "
                    f"({late_rmse / early_rmse:.2f}x); candidate may drift over longer horizons."
                )

    if error:
        failure_modes.append(
            {
                "category": "exception" if status == "error" else "evaluation_failed",
                "detail": error,
            }
        )

    packet = {
        "schema_version": 1,
        "status": status,
        "fit": {
            "median_distance": _safe_float(res.get("median_distance")) if res else None,
            "min_distance": _safe_float(res.get("min_distance")) if res else None,
            "accepted_particle_count": int(accepted_params.shape[0]) if accepted_params is not None else 0,
            "train_split_index": split_idx,
            "requested_strategy": res.get("requested_strategy") if res else None,
            "effective_strategy": res.get("effective_strategy") if res else None,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "evaluation_protocol": evaluation_protocol,
        "data_receipt": data_receipt,
        "best_parameters": best_parameters,
        "posterior_summary": posterior_summary,
        "parameter_bound_pressure": bound_pressure,
        "residual_summary": residual_summary,
        "warnings": warnings,
        "failure_modes": failure_modes,
    }
    return packet


def build_candidate_model(raw_code: str, config: dict[str, Any]):
    y0_list = config["y0"]
    dt0 = config["dt0"]
    max_steps = config["max_steps"]
    domain_name = config["name"]
    full_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{raw_code}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        sol = diffeqsolve(
            ODETerm(dynamics), Tsit5(),
            t0=float(time_points[0]), t1=float(time_points[-1]),
            dt0={dt0}, y0=y0, args=params, saveat=saveat, max_steps={max_steps}
        )
        return sol.ys

    def get_parameter_metadata(self):
        return metadata

    def get_initial_conditions(self):
        return jnp.array({y0_list})

    def get_latex(self):
        return "{domain_name} generated model"
"""
    module_name = f"dynamic_candidate_{np.random.randint(100000)}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(full_module, mod.__dict__)
    return mod.CandidateModel()


def _sample_prior_batch(metadata: list[dict[str, Any]], n_particles: int, rng: np.random.Generator) -> np.ndarray:
    samples = []
    for index, prior in enumerate(metadata):
        if "range" not in prior or len(prior["range"]) != 2:
            raise ValueError(f"prior {index} must include a two-value range")
        low, high = prior["range"]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"prior {index} has invalid range: {prior['range']}")
        samples.append(rng.uniform(float(low), float(high), n_particles))
    return np.column_stack(samples)


def evaluate_subject_holdout_candidate(raw_code: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if config.get("real_data_adapter") != "warfarin_pkpd":
        raise ValueError("subject_holdout evaluation currently supports only the warfarin_pkpd adapter")
    if args.generations != 1:
        raise ValueError("subject_holdout sandbox branch currently supports generations=1 grouped prior-rejection smoke")
    if args.target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if args.initial_particles < args.target_samples:
        raise ValueError("initial_particles must be >= target_samples")

    normalized_path = config.get("normalized_data_path")
    holdout_path = config.get("subject_holdout_path")
    if not normalized_path or not holdout_path:
        raise ValueError("subject_holdout evaluation requires normalized_data_path and subject_holdout_path")

    normalized = pd.read_csv(normalized_path)
    with open(holdout_path, "r") as f:
        holdout = json.load(f)
    if holdout.get("status") != "ok":
        reason = holdout.get("reason", "subject holdout split is unavailable")
        raise ValueError(f"subject_holdout split not available: {reason}")

    train_bundles = build_subject_target_bundles(normalized, holdout["train_subjects"])
    test_bundles = build_subject_target_bundles(normalized, holdout["test_subjects"])
    model = build_candidate_model(raw_code, config)
    metadata = model.get_parameter_metadata()

    test_params = np.array([float(prior["range"][0]) for prior in metadata], dtype=float)
    for bundle in train_bundles[:1] + test_bundles[:1]:
        check_ys = np.asarray(model.simulate(test_params, bundle.time_points, bundle.y0), dtype=float)
        if check_ys.shape != bundle.observed.shape:
            raise ValueError(f"Model produced shape {check_ys.shape}, expected {bundle.observed.shape}.")
        if not np.all(np.isfinite(check_ys)):
            raise ValueError("Model produced NaN or Inf in initial subject-holdout simulation.")

    rng = np.random.default_rng(args.seed)
    params_batch = _sample_prior_batch(metadata, args.initial_particles, rng)
    distances = grouped_subject_distances(model, params_batch, train_bundles)
    distances[~np.isfinite(distances)] = np.inf
    finite_idx = np.where(np.isfinite(distances))[0]
    if len(finite_idx) == 0:
        res = {
            "median_distance": float("inf"),
            "min_distance": float("inf"),
            "requested_strategy": args.inference_strategy,
            "effective_strategy": "subject_holdout_prior_rejection",
        }
        return {
            "status": "failed",
            "median_distance": float("inf"),
            "min_distance": float("inf"),
            "diagnostics": build_diagnostic_packet(res=res, metadata=metadata, status="failed"),
            "effective_strategy": "subject_holdout_prior_rejection",
        }

    sorted_idx = finite_idx[np.argsort(distances[finite_idx])]
    keep_idx = sorted_idx[: args.target_samples]
    accepted_params = params_batch[keep_idx]
    accepted_distances = distances[keep_idx]
    best_params = np.median(accepted_params, axis=0)

    train_predictions = predict_subject_bundles(model, best_params, train_bundles)
    test_predictions = predict_subject_bundles(model, best_params, test_bundles)
    train_subject_metrics = evaluate_subject_predictions(train_bundles, train_predictions)
    test_subject_metrics = evaluate_subject_predictions(test_bundles, test_predictions)
    train_aggregate = finite_metric_mapping(train_subject_metrics["aggregate"])
    test_aggregate = finite_metric_mapping(test_subject_metrics["aggregate"])
    metric_failure = held_out_metric_failure(train_aggregate, test_aggregate)
    status = "failed" if metric_failure else "success"

    res = {
        "accepted_params": accepted_params,
        "median_distance": float(np.median(accepted_distances)),
        "min_distance": float(np.min(accepted_distances)),
        "requested_strategy": args.inference_strategy,
        "effective_strategy": "subject_holdout_prior_rejection",
    }
    diagnostics = build_diagnostic_packet(
        res=res,
        metadata=metadata,
        status=status,
        train_metrics=train_aggregate,
        test_metrics=test_aggregate,
        extra_failure_modes=[metric_failure] if metric_failure else None,
    )
    diagnostics["subject_holdout"] = {
        "schema_version": 1,
        "train_subjects": holdout["train_subjects"],
        "test_subjects": holdout["test_subjects"],
        "train": train_subject_metrics,
        "test": test_subject_metrics,
        "warning": "Grouped subject-holdout branch is a prior-rejection smoke path, not final ABC-SMC.",
    }
    return {
        "status": status,
        "median_distance": res["median_distance"],
        "min_distance": res["min_distance"],
        "error": metric_failure["detail"] if metric_failure else None,
        "train_metrics": train_aggregate,
        "test_metrics": test_aggregate,
        "diagnostics": diagnostics,
        "requested_strategy": args.inference_strategy,
        "effective_strategy": "subject_holdout_prior_rejection",
    }


def evaluate_cell_holdout_candidate(raw_code: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded grouped-cell seed gate for NASA battery capacity fade."""
    if config.get("real_data_adapter") != "battery_nasa":
        raise ValueError("cell_holdout evaluation currently supports only the battery_nasa adapter")
    if args.target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if args.initial_particles < args.target_samples:
        raise ValueError("initial_particles must be >= target_samples")

    normalized_path = config.get("normalized_data_path")
    holdout_path = args.cell_holdout_path or config.get("cell_holdout_path")
    if not normalized_path or not holdout_path:
        raise ValueError("cell_holdout evaluation requires normalized_data_path and cell_holdout_path")

    normalized = pd.read_csv(normalized_path)
    with open(holdout_path, "r") as f:
        holdout = json.load(f)
    bundles = build_cell_holdout_bundles(normalized, holdout)
    train_bundles = bundles["train"]
    test_bundles = bundles["test"]
    model = build_candidate_model(raw_code, config)
    metadata = model.get_parameter_metadata()

    test_params = np.array([float(prior["range"][0]) for prior in metadata], dtype=float)
    for bundle in train_bundles[:1] + test_bundles[:1]:
        check_ys = np.asarray(model.simulate(test_params, bundle.time_points, bundle.y0), dtype=float)
        if check_ys.shape != bundle.observed.shape:
            raise ValueError(f"Model produced shape {check_ys.shape}, expected {bundle.observed.shape}.")
        if not np.all(np.isfinite(check_ys)):
            raise ValueError("Model produced NaN or Inf in initial cell-holdout simulation.")

    if args.generations == 1:
        rng = np.random.default_rng(args.seed)
        params_batch = _sample_prior_batch(metadata, args.initial_particles, rng)
        distances = grouped_cell_distances(model, params_batch, train_bundles)
        distances[~np.isfinite(distances)] = np.inf
        finite_idx = np.where(np.isfinite(distances))[0]
        if len(finite_idx) < args.target_samples:
            res = {
                "median_distance": float("inf"),
                "min_distance": float("inf"),
                "requested_strategy": args.inference_strategy,
                "effective_strategy": "cell_holdout_prior_rejection",
            }
            return {
                "status": "failed",
                "median_distance": float("inf"),
                "min_distance": float("inf"),
                "diagnostics": build_diagnostic_packet(res=res, metadata=metadata, status="failed"),
                "effective_strategy": "cell_holdout_prior_rejection",
            }
        keep_idx = finite_idx[np.argsort(distances[finite_idx])[: args.target_samples]]
        accepted_params = params_batch[keep_idx]
        accepted_distances = distances[keep_idx]
        res = {
            "accepted_params": accepted_params,
            "median_distance": float(np.median(accepted_distances)),
            "min_distance": float(np.min(accepted_distances)),
            "requested_strategy": args.inference_strategy,
            "effective_strategy": "cell_holdout_prior_rejection",
            "generation_history": [
                {
                    "generation": 0,
                    "finite_particle_count": int(len(finite_idx)),
                    "epsilon": float(accepted_distances[-1]),
                    "median_distance": float(np.median(accepted_distances)),
                    "min_distance": float(np.min(accepted_distances)),
                }
            ],
        }
        evaluator_warning = "Grouped cell-holdout branch is a one-generation prior-rejection smoke path, not final ABC-SMC."
    else:
        res = run_grouped_cell_abc_smc(
            model,
            train_bundles,
            target_samples=args.target_samples,
            generations=args.generations,
            initial_particles=args.initial_particles,
            strategy=args.inference_strategy,
            seed=args.seed,
        )
        if not np.isfinite(res["median_distance"]):
            return {
                "status": "failed",
                "median_distance": float("inf"),
                "min_distance": float("inf"),
                "diagnostics": build_diagnostic_packet(res=res, metadata=metadata, status="failed"),
                "effective_strategy": res["effective_strategy"],
            }
        accepted_params = np.asarray(res["accepted_params"], dtype=float)
        evaluator_warning = "Grouped cell-holdout branch used multi-generation ABC-SMC over shared parameters and independent train-cell trajectories."

    best_params = np.median(accepted_params, axis=0)
    train_predictions = predict_cell_bundles(model, best_params, train_bundles)
    test_predictions = predict_cell_bundles(model, best_params, test_bundles)
    train_cell_metrics = evaluate_cell_predictions(train_bundles, train_predictions)
    test_cell_metrics = evaluate_cell_predictions(test_bundles, test_predictions)
    train_aggregate = finite_metric_mapping(train_cell_metrics["aggregate"])
    test_aggregate = finite_metric_mapping(test_cell_metrics["aggregate"])
    metric_failure = held_out_metric_failure(train_aggregate, test_aggregate)
    status = "failed" if metric_failure else "success"

    diagnostics = build_diagnostic_packet(
        res=res,
        metadata=metadata,
        status=status,
        train_metrics=train_aggregate,
        test_metrics=test_aggregate,
        extra_failure_modes=[metric_failure] if metric_failure else None,
    )
    diagnostics["cell_holdout"] = {
        "schema_version": 1,
        "train_cells": holdout["train_cells"],
        "test_cells": holdout["test_cells"],
        "train": train_cell_metrics,
        "test": test_cell_metrics,
        "generation_history": res.get("generation_history", []),
        "warning": evaluator_warning,
    }
    return {
        "status": status,
        "median_distance": res["median_distance"],
        "min_distance": res["min_distance"],
        "error": metric_failure["detail"] if metric_failure else None,
        "train_metrics": train_aggregate,
        "test_metrics": test_aggregate,
        "diagnostics": diagnostics,
        "requested_strategy": args.inference_strategy,
        "effective_strategy": res["effective_strategy"],
    }


def _trajectory_metrics(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Return finite development metrics for one named trajectory partition."""

    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    if observed.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {observed.shape} != {predicted.shape}")
    if mask is None:
        values = observed - predicted
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {observed.shape}")
        values = (observed - predicted)[mask]
    if values.size == 0:
        raise ValueError("trajectory partition has no observed values")
    mse = float(np.mean(np.square(values)))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "observed_count": int(values.size)}


def _simulate_canonical_development(model, train_part, validation_part, params):
    """Simulate train and validation using declared times only.

    Chronological partitions are simulated in one rollout, preserving the
    train-to-validation state. If the validation times restart, the protocol
    is interpreted as an independent trajectory and starts from the declared
    initial condition; no validation outcome is used to initialize the state.
    """

    train_times = np.asarray(train_part.time_points)
    validation_times = np.asarray(validation_part.time_points)
    joined = np.concatenate([train_times, validation_times])
    monotonic = len(joined) == 1 or bool(np.all(np.diff(joined) > 0))
    if monotonic:
        joined_prediction = np.asarray(
            model.simulate(params, joined, model.get_initial_conditions()), dtype=float
        )
        return (
            joined_prediction[: len(train_times)],
            joined_prediction[len(train_times) :],
        )
    train_prediction = np.asarray(
        model.simulate(params, train_times, model.get_initial_conditions()), dtype=float
    )
    validation_prediction = np.asarray(
        model.simulate(params, validation_times, model.get_initial_conditions()), dtype=float
    )
    return train_prediction, validation_prediction


def evaluate_canonical_trajectory_candidate(
    raw_code: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate one candidate using train fitting and validation selection.

    The development worker accepts a manifest containing exactly train and
    validation roles. It rejects a manifest that also contains a final role,
    so a sealed outcome cannot become an accidental fitting or prompt input.
    """

    manifest_path = getattr(args, "protocol_manifest", None)
    if not manifest_path:
        raise ValueError("canonical trajectory evaluation requires --protocol-manifest")
    manifest = load_partition_manifest(
        manifest_path,
        required_roles=DEVELOPMENT_ROLES,
        forbidden_roles=("final",),
    )
    train_part = manifest["roles"]["train"]
    validation_part = manifest["roles"]["validation"]
    model = build_candidate_model(raw_code, config)
    metadata = model.get_parameter_metadata()

    # The only observations supplied to SBIEngine are the declared training
    # outcomes. Validation outcomes are reserved for structural selection.
    engine = SBIEngine(
        train_part.observations,
        train_part.time_points,
        use_summary_stats=True,
        seed=args.seed,
        observation_mask=train_part.observation_mask,
    )
    test_params = jnp.array([prior["range"][0] for prior in metadata])
    check_ys = np.asarray(model.simulate(test_params, train_part.time_points[: min(5, len(train_part.time_points))], model.get_initial_conditions()), dtype=float)
    expected_shape = (min(5, len(train_part.time_points)), len(config["y0"]))
    if check_ys.shape != expected_shape:
        raise ValueError(f"Model produced shape {check_ys.shape}, expected {expected_shape}.")
    if not np.all(np.isfinite(check_ys)):
        raise ValueError("Model produced NaN or Inf in canonical development sanity simulation.")

    res = engine.run_abc_smc(
        model,
        target_samples=args.target_samples,
        generations=args.generations,
        initial_particles=args.initial_particles,
        seed=args.seed,
        strategy=args.inference_strategy,
    )
    results: dict[str, Any] = {
        "status": "failed",
        "median_distance": res["median_distance"],
        "min_distance": res.get("min_distance", float("inf")),
        "error": None,
        "train_metrics": None,
        "validation_metrics": None,
        "test_metrics": None,
        "requested_strategy": res.get("requested_strategy", args.inference_strategy),
        "effective_strategy": res.get("effective_strategy"),
        "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
        "development_data_receipt": manifest["receipt"],
    }
    fit_succeeded = np.isfinite(float(res["median_distance"]))
    predictions = None
    metric_failure = None
    if fit_succeeded:
        best_params = jnp.array(np.median(res["accepted_params"], axis=0))
        train_prediction, validation_prediction = _simulate_canonical_development(
            model, train_part, validation_part, best_params
        )
        train_metrics = _trajectory_metrics(train_part.observations, train_prediction, train_part.observation_mask)
        validation_metrics = _trajectory_metrics(
            validation_part.observations,
            validation_prediction,
            validation_part.observation_mask,
        )
        results["train_metrics"] = finite_metric_mapping(train_metrics)
        results["validation_metrics"] = finite_metric_mapping(validation_metrics)
        predictions = np.concatenate([train_prediction, validation_prediction], axis=0)
        metric_failure = held_out_metric_failure(
            results["train_metrics"], results["validation_metrics"], held_out_label="validation"
        )
        results["status"] = "success" if metric_failure is None else "failed"
        if metric_failure:
            results["error"] = metric_failure["detail"]
    else:
        results["status"] = "failed"
        results["error"] = "canonical development fit produced non-finite distance"

    results["diagnostics"] = build_diagnostic_packet(
        res=res,
        metadata=metadata,
        status=results["status"],
        observed=np.concatenate([train_part.observations, validation_part.observations], axis=0) if predictions is not None else None,
        predicted=predictions,
        observed_mask=np.concatenate([train_part.observation_mask, validation_part.observation_mask], axis=0)
        if predictions is not None and train_part.observation_mask is not None and validation_part.observation_mask is not None
        else None,
        split_idx=len(train_part.observations),
        train_metrics=results["train_metrics"],
        validation_metrics=results["validation_metrics"],
        evaluation_protocol=CANONICAL_PROTOCOL_VERSION,
        data_receipt=manifest["receipt"],
        extra_failure_modes=[metric_failure] if metric_failure else None,
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate math model in an isolated process.")
    parser.add_argument("--code-file", required=True, help="Path to file containing raw dynamics + metadata code.")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. ecology).")
    parser.add_argument("--held-out", action="store_true", help="Enable 80/20 train/test trajectory split protocol.")
    parser.add_argument(
        "--evaluation-protocol",
        default="legacy",
        choices=("legacy", CANONICAL_PROTOCOL_VERSION),
        help="Opt-in canonical train/validation development protocol; legacy keeps historical semantics.",
    )
    parser.add_argument(
        "--protocol-manifest",
        help="Development manifest for the canonical trajectory protocol (train and validation roles only).",
    )
    parser.add_argument("--target-samples", type=int, default=500, help="ABC-SMC target samples.")
    parser.add_argument("--generations", type=int, default=15, help="ABC-SMC generations.")
    parser.add_argument("--initial-particles", type=int, default=150000, help="ABC-SMC initial particles.")
    parser.add_argument(
        "--inference-strategy",
        default="gaussian_weighted",
        help="ABC-SMC transition strategy, e.g. gaussian_weighted or bdss.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--cell-holdout-path",
        help="Optional recorded battery cell-holdout split JSON. Defaults to the domain configuration.",
    )
    parser.add_argument("--output", required=True, help="Path to write the evaluation output JSON.")
    args = parser.parse_args()

    results = {
        "status": "failed",
        "median_distance": float("inf"),
        "min_distance": float("inf"),
        "error": None,
        "train_metrics": None,
        "test_metrics": None,
        "validation_metrics": None,
        "diagnostics": None,
        "requested_strategy": args.inference_strategy,
        "effective_strategy": None,
    }

    try:
        # 1. Load domain configuration
        if args.domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {args.domain}")
        config = DOMAIN_CONFIGS[args.domain]

        # 2. Read code
        with open(args.code_file, "r") as f:
            raw_code = f.read()

        # 3. AST validate code
        validate_generated_math_code(raw_code)

        if args.evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
            if args.held_out:
                raise ValueError("--held-out cannot be combined with the canonical trajectory protocol")
            results.update(evaluate_canonical_trajectory_candidate(raw_code, config, args))
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            return 0 if results["status"] == "success" else 1

        evaluation_mode = config.get("evaluation_mode")
        if evaluation_mode in {"subject_holdout", "cell_holdout"}:
            evaluator = (
                evaluate_subject_holdout_candidate
                if evaluation_mode == "subject_holdout"
                else evaluate_cell_holdout_candidate
            )
            results.update(evaluator(raw_code, config, args))
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            return 0 if results["status"] == "success" else 1

        # 4. Load ground truth datasets
        gt_data = np.load(f"data/{args.domain}_ground_truth.npy")
        t_points = np.load(f"data/{args.domain}_time_points.npy")
        mask_path = f"data/{args.domain}_observation_mask.npy"
        observation_mask = np.load(mask_path).astype(bool) if os.path.exists(mask_path) else None
        if observation_mask is not None and observation_mask.shape != gt_data.shape:
            raise ValueError(f"observation mask shape {observation_mask.shape} does not match ground truth {gt_data.shape}")

        # 5. Split data if held-out is enabled
        split_idx = None
        if args.held_out:
            t_train, gt_train, t_test, gt_test, split_idx = split_time_series(t_points, gt_data, train_fraction=0.8)
            train_mask = observation_mask[:split_idx] if observation_mask is not None else None
            # Initialize engine only on train set
            engine = SBIEngine(gt_train, t_train, use_summary_stats=True, seed=args.seed, observation_mask=train_mask)
        else:
            engine = SBIEngine(gt_data, t_points, use_summary_stats=True, seed=args.seed, observation_mask=observation_mask)

        # 6. Scaffold the module code
        y0_list = config["y0"]
        dt0 = config["dt0"]
        max_steps = config["max_steps"]
        domain_name = config["name"]

        full_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{raw_code}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        sol = diffeqsolve(
            ODETerm(dynamics), Tsit5(),
            t0=float(time_points[0]), t1=float(time_points[-1]),
            dt0={dt0}, y0=y0, args=params, saveat=saveat, max_steps={max_steps}
        )
        return sol.ys

    def get_parameter_metadata(self):
        return metadata

    def get_initial_conditions(self):
        return jnp.array({y0_list})

    def get_latex(self):
        return "{domain_name} generated model"
"""

        # 7. Dynamically import and compile module
        module_name = f"dynamic_candidate_{np.random.randint(100000)}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        exec(full_module, mod.__dict__)
        model = mod.CandidateModel()
        candidate_metadata = model.get_parameter_metadata()

        # 8. Sanity check: verify simulation works without crashing/NaNs
        test_params = jnp.array([p["range"][0] for p in candidate_metadata])
        check_t = t_points[:5]
        check_ys = model.simulate(test_params, check_t, model.get_initial_conditions())
        expected_shape = (len(check_t), len(config["y0"]))
        if tuple(check_ys.shape) != expected_shape:
            raise ValueError(f"Model produced shape {tuple(check_ys.shape)}, expected {expected_shape}.")
        if jnp.any(jnp.isnan(check_ys)) or jnp.any(jnp.isinf(check_ys)):
            raise ValueError("Model produced NaN or Inf in initial test simulation.")

        # 9. Run parameter fitting
        res = engine.run_abc_smc(
            model,
            target_samples=args.target_samples,
            generations=args.generations,
            initial_particles=args.initial_particles,
            seed=args.seed,
            strategy=args.inference_strategy,
        )

        results["median_distance"] = res["median_distance"]
        results["min_distance"] = res.get("min_distance", float("inf"))
        results["requested_strategy"] = res.get("requested_strategy", args.inference_strategy)
        results["effective_strategy"] = res.get("effective_strategy")

        full_y_pred = None
        best_params = None
        fit_succeeded = np.isfinite(float(res["median_distance"]))
        metric_failure = None

        # 10. Perform held-out trajectory evaluation if requested
        if args.held_out and fit_succeeded:
            # Compute median parameters from final accepted population
            best_params = jnp.array(np.median(res["accepted_params"], axis=0))
            # Simulate over FULL time course
            full_y_pred = model.simulate(best_params, t_points, model.get_initial_conditions())
            metrics = train_test_metrics(gt_data, full_y_pred, train_fraction=0.8, mask=observation_mask)
            raw_train_metrics = {
                "mse": float(metrics.train_mse),
                "rmse": float(metrics.train_rmse),
            }
            raw_test_metrics = {
                "mse": float(metrics.test_mse),
                "rmse": float(metrics.test_rmse),
            }
            metric_failure = held_out_metric_failure(raw_train_metrics, raw_test_metrics)
            results["train_metrics"] = finite_metric_mapping(raw_train_metrics)
            results["test_metrics"] = finite_metric_mapping(raw_test_metrics)

        results["status"] = "success" if fit_succeeded and metric_failure is None else "failed"
        if metric_failure:
            results["error"] = metric_failure["detail"]
        results["diagnostics"] = build_diagnostic_packet(
            res=res,
            metadata=candidate_metadata,
            status=results["status"],
            observed=gt_data if full_y_pred is not None else None,
            predicted=np.asarray(full_y_pred) if full_y_pred is not None else None,
            observed_mask=observation_mask if full_y_pred is not None else None,
            split_idx=split_idx,
            train_metrics=results["train_metrics"],
            test_metrics=results["test_metrics"],
            extra_failure_modes=[metric_failure] if metric_failure else None,
        )

    except Exception as exc:
        results["status"] = "error"
        results["error"] = f"{type(exc).__name__}: {str(exc)}"
        results["traceback"] = traceback.format_exc()
        results["diagnostics"] = build_diagnostic_packet(
            res=None,
            metadata=None,
            status=results["status"],
            error=results["error"],
        )

    # Write output to JSON
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    return 0 if results["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

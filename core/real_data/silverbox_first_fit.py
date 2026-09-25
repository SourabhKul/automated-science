"""Bounded, train-only ABC-SMC fit and separate Silverbox development choice.

The fitting API accepts exactly the four predeclared training windows. It has
no access to the broader development container or validation targets. A
separate selection API reads a completed, hash-checked fit receipt and one
fixed validation series. This module intentionally has no Qwen client and no
sealed-test scorer.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Literal

import numpy as np

from core.abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference
from core.real_data.silverbox_controlled import (
    STATE_INITIALIZATION_LENGTH,
    TRAIN_SOURCE_START,
    TRAIN_WINDOW_LENGTH,
    TRAIN_WINDOW_PREDICTION_LENGTH,
    TRAIN_WINDOW_RELATIVE_STARTS,
    VALIDATION_SOURCE_START,
    VALIDATION_SOURCE_STOP,
    SilverboxControlledSeries,
)


PROTOCOL_ID = "silverbox-first-fit-2026-09-25"
TERM_IDS = ("y_cubed", "u_cubed", "u_y_product")
TermId = Literal["y_cubed", "u_cubed", "u_y_product"]
HYPOTHESES = ("linear", "nonlinear")
LINEAR_PARAMETER_NAMES = ("a1", "a2", "b", "c")
NONLINEAR_PARAMETER_NAMES = (*LINEAR_PARAMETER_NAMES, "d")
LINEAR_BOUNDS = ((1.35, 1.57), (-1.05, -0.82), (0.30, 0.54), (-0.005, 0.001))
NONLINEAR_BOUNDS = (*LINEAR_BOUNDS, (-0.10, 0.10))
CALIBRATION_DRAWS = 256
CALIBRATION_MIN_FINITE = 128
CALIBRATION_SEEDS = {"linear": 26092501, "nonlinear": 26092502}
ABC_SEEDS = {"linear": 26092511, "nonlinear": 26092512}
CALIBRATION_QUANTILES = (0.50, 0.20)
ABC_PARTICLES = 64
ABC_POPULATIONS = 2
ABC_ATTEMPTS_PER_POPULATION = 512
ABC_COVARIANCE_SCALE = 2.0
ABC_LAMBDA_NOISE = 0.01
ABC_NUGGET = 1e-9
PILOT_WALL_SECONDS = 600.0
PILOT_ADDITIONAL_MEMORY_BYTES = 2 * 1024**3
NONLINEAR_PROMOTION_MARGIN = 0.05
DEFAULT_RECEIPT_DIR = Path("artifacts/silverbox_first_fit")


class SimulationFailure(RuntimeError):
    """An explicit controlled-simulator failure with a stable reason code."""

    def __init__(self, failure_mode: str, detail: str):
        self.failure_mode = failure_mode
        self.detail = detail
        super().__init__(f"{failure_mode}: {detail}")


@dataclass(frozen=True)
class FrozenSilverboxFit:
    """Reference to an atomically persisted and hash-checked fit receipt."""

    run_id: str
    receipt_path: Path
    receipt_sha256: str
    status: Literal["complete", "incomplete"]
    term_id: TermId
    source_sha256: str
    proposal_receipt_sha256: str


@dataclass(frozen=True)
class SilverboxSelection:
    """Selection outcome and its persisted receipt."""

    status: Literal["selected", "unresolved"]
    selected_hypothesis: Literal["linear", "nonlinear"] | None
    receipt_path: Path
    receipt_sha256: str
    reason: str


class _PilotBudget:
    def __init__(self, started_monotonic: float, started_unix: float, rss_baseline: int | None):
        self.started_monotonic = started_monotonic
        self.started_unix = started_unix
        self.rss_baseline = rss_baseline
        self.stop_reason: str | None = None
        self.peak_additional_rss_bytes = 0

    def check(self) -> bool:
        if self.stop_reason is not None:
            return True
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed >= PILOT_WALL_SECONDS:
            self.stop_reason = "wall_time_limit"
            return True
        current = _current_rss_bytes()
        if current is None or self.rss_baseline is None:
            self.stop_reason = "memory_monitor_unavailable"
            return True
        additional = max(0, current - self.rss_baseline)
        self.peak_additional_rss_bytes = max(self.peak_additional_rss_bytes, additional)
        if additional > PILOT_ADDITIONAL_MEMORY_BYTES:
            self.stop_reason = "additional_memory_limit"
            return True
        return False

    def receipt(self) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        return {
            "wall_limit_seconds": PILOT_WALL_SECONDS,
            "wall_elapsed_seconds": elapsed,
            "wall_started_unix": self.started_unix,
            "wall_started_monotonic": self.started_monotonic,
            "memory_limit_bytes": PILOT_ADDITIONAL_MEMORY_BYTES,
            "memory_baseline_rss_bytes": self.rss_baseline,
            "peak_additional_rss_bytes": self.peak_additional_rss_bytes,
            "process_id": os.getpid(),
            "stop_reason": self.stop_reason,
        }


def train_only_scalers(
    train_windows: tuple[SilverboxControlledSeries, ...], term_id: TermId
) -> dict[str, float | str]:
    """Compute the frozen RMS scalers only from the four training windows."""

    _validate_train_windows(train_windows)
    _validate_term_id(term_id)
    train_targets = np.concatenate([window.target_y for window in train_windows])
    s_y = _rms(train_targets)
    observed_features: list[np.ndarray] = []
    for window in train_windows:
        observed_y = np.concatenate((window.initialization_y, window.target_y))
        # For predicted steps k=50..255, the observed lag k-1 is 49..254.
        lag_y = observed_y[STATE_INITIALIZATION_LENGTH - 1 : TRAIN_WINDOW_LENGTH - 1]
        lag_u = window.input_u[STATE_INITIALIZATION_LENGTH - 1 : TRAIN_WINDOW_LENGTH - 1]
        observed_features.append(_feature(term_id, lag_y, lag_u))
    s_feature = _rms(np.concatenate(observed_features))
    if not math.isfinite(s_y) or s_y <= 0:
        raise ValueError("training output RMS scaler must be finite and positive")
    if not math.isfinite(s_feature) or s_feature <= 0:
        raise ValueError("selected training feature RMS scaler must be finite and positive")
    training_max_abs_y = float(max(np.max(np.abs(window.initialization_y)) for window in train_windows))
    training_max_abs_y = max(
        training_max_abs_y,
        *(float(np.max(np.abs(window.target_y))) for window in train_windows),
    )
    return {
        "s_y": s_y,
        "s_feature": s_feature,
        "feature_term_id": term_id,
        "divergence_bound": max(1.0, 10.0 * training_max_abs_y),
        "training_target_count": int(sum(window.target_y.size for window in train_windows)),
        "training_max_abs_y": training_max_abs_y,
    }


def simulate_controlled_ar2(
    input_u: Any,
    initialization_y: Any,
    parameters: Any,
    *,
    hypothesis: Literal["linear", "nonlinear"],
    term_id: TermId,
    s_y: float,
    s_feature: float,
    divergence_bound: float,
) -> np.ndarray:
    """Return one deterministic free-run suffix under the declared AR(2).

    The first prediction is at ``k=50`` and uses ``yhat[49]``,
    ``yhat[48]``, and ``u[49]``. Later steps use the simulator's own output
    history. Failure is an exception carrying a stable ``failure_mode``;
    predictions are never clipped.
    """

    _validate_term_id(term_id)
    if hypothesis not in HYPOTHESES:
        raise SimulationFailure("invalid_hypothesis", f"unsupported hypothesis {hypothesis!r}")
    try:
        u = np.asarray(input_u, dtype=float)
        initializer = np.asarray(initialization_y, dtype=float)
        theta = np.asarray(parameters, dtype=float)
    except Exception as error:
        raise SimulationFailure("invalid_input", f"cannot convert simulator input: {error}") from error
    dimension = 4 if hypothesis == "linear" else 5
    if u.ndim != 1 or u.size <= STATE_INITIALIZATION_LENGTH:
        raise SimulationFailure("wrong_input_shape", "input must be 1-D and extend beyond the 50-sample initializer")
    if initializer.shape != (STATE_INITIALIZATION_LENGTH,):
        raise SimulationFailure("wrong_initializer_shape", "initializer must contain exactly 50 outputs")
    if theta.shape != (dimension,):
        raise SimulationFailure("wrong_parameter_shape", f"expected {dimension} parameters, received {theta.shape}")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(initializer)) or not np.all(np.isfinite(theta)):
        raise SimulationFailure("nonfinite_input", "input, initializer, and parameters must be finite")
    if hypothesis == "nonlinear" and (not np.isfinite(s_y) or not np.isfinite(s_feature) or s_y <= 0 or s_feature <= 0):
        raise SimulationFailure("invalid_scaler", "nonlinear model scalers must be finite and positive")
    if not np.isfinite(divergence_bound) or divergence_bound <= 0:
        raise SimulationFailure("invalid_divergence_bound", "divergence bound must be finite and positive")

    a1, a2, b, c = theta[:4]
    d = float(theta[4]) if hypothesis == "nonlinear" else 0.0
    trajectory = np.empty(u.size, dtype=float)
    trajectory[:STATE_INITIALIZATION_LENGTH] = initializer
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(STATE_INITIALIZATION_LENGTH, u.size):
            lag_y = trajectory[k - 1]
            lag_u = u[k - 1]
            value = a1 * lag_y + a2 * trajectory[k - 2] + b * lag_u + c
            if hypothesis == "nonlinear":
                value += d * s_y * float(_feature(term_id, np.asarray([lag_y]), np.asarray([lag_u]))[0]) / s_feature
            trajectory[k] = value
    if not np.all(np.isfinite(trajectory)):
        raise SimulationFailure("nonfinite_trajectory", "recurrence produced a nonfinite value")
    maximum = float(np.max(np.abs(trajectory)))
    if maximum > divergence_bound:
        raise SimulationFailure(
            "divergence_bound_exceeded",
            f"trajectory maximum {maximum:.17g} exceeds fixed bound {divergence_bound:.17g}",
        )
    return np.array(trajectory[STATE_INITIALIZATION_LENGTH:], copy=True)


def fit_silverbox_development(
    train_windows: tuple[SilverboxControlledSeries, ...],
    term_id: TermId,
    *,
    source_sha256: str,
    proposal_receipt_sha256: str,
    run_id: str,
    receipt_dir: str | Path = DEFAULT_RECEIPT_DIR,
) -> FrozenSilverboxFit:
    """Run one exclusive, train-only pilot fit with the upstream term choice."""

    _validate_train_windows(train_windows)
    _validate_term_id(term_id)
    _validate_sha256(source_sha256, "source_sha256")
    _validate_proposal_receipt_sha256(proposal_receipt_sha256)
    normalized_source_hash = source_sha256.lower()
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run_id must be a filesystem-safe 1–128 character identifier")
    with _exclusive_fit_process(receipt_dir):
        return _fit_silverbox_development_locked(
            train_windows,
            term_id,
            source_sha256=normalized_source_hash,
            proposal_receipt_sha256=proposal_receipt_sha256,
            run_id=run_id,
            receipt_dir=receipt_dir,
        )


def _fit_silverbox_development_locked(
    train_windows: tuple[SilverboxControlledSeries, ...],
    term_id: TermId,
    *,
    source_sha256: str,
    proposal_receipt_sha256: str,
    run_id: str,
    receipt_dir: str | Path,
) -> FrozenSilverboxFit:
    """Calibrate and fit both fixed hypotheses from training windows only.

    ``term_id`` must come from the upstream frozen proposal parser. There is
    deliberately no default or fallback value. This function accepts neither
    a development container nor validation data.
    """

    _validate_train_windows(train_windows)
    _validate_term_id(term_id)
    _validate_sha256(source_sha256, "source_sha256")
    source_sha256 = source_sha256.lower()
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run_id must be a filesystem-safe 1–128 character identifier")
    started_monotonic = time.monotonic()
    started_unix = time.time()
    rss_baseline = _current_rss_bytes()
    budget = _PilotBudget(started_monotonic, started_unix, rss_baseline)
    run_dir = Path(receipt_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = _fit_metadata(
        train_windows,
        term_id,
        source_sha256,
        proposal_receipt_sha256,
        run_id,
        started_unix,
        started_monotonic,
        rss_baseline,
    )
    _write_signed_json(run_dir / "fit_started.json", {**metadata, "status": "started"})

    try:
        scalers = train_only_scalers(train_windows, term_id)
    except Exception as error:
        receipt = {
            **metadata,
            "status": "incomplete",
            "reason": f"training_scaler_failure: {type(error).__name__}: {error}",
            "training_scalers": None,
            "hypotheses": {},
            "budget": budget.receipt(),
        }
        path, digest = _write_signed_json(run_dir / "fit.json", receipt)
        return FrozenSilverboxFit(
            run_id, path, digest, "incomplete", term_id, source_sha256, proposal_receipt_sha256
        )

    family_results: dict[str, dict[str, Any]] = {}
    for hypothesis in HYPOTHESES:
        if budget.check():
            family_results[hypothesis] = _not_started_family(hypothesis, budget.stop_reason)
            _write_signed_json(run_dir / f"{hypothesis}_calibration.json", family_results[hypothesis]["calibration"])
            continue
        family_results[hypothesis] = _fit_hypothesis(
            hypothesis,
            train_windows,
            term_id,
            scalers,
            budget,
            run_dir,
        )

    budget.check()
    all_complete = (
        budget.stop_reason is None
        and all(family_results[name]["status"] == "complete" for name in HYPOTHESES)
    )
    reason = None if all_complete else budget.stop_reason or "one_or_more_hypotheses_incomplete"
    fit_receipt = {
        **metadata,
        "status": "complete" if all_complete else "incomplete",
        "reason": reason,
        "training_scalers": scalers,
        "hypotheses": family_results,
        "budget": budget.receipt(),
        "selection": {
            "rule": "pointwise weighted median; unresolved if either fit/forecast fails or neither beats persistence; nonlinear only if at least 5% below linear",
            "nonlinear_promotion_margin": NONLINEAR_PROMOTION_MARGIN,
            "validation_accessed": False,
        },
    }
    path, digest = _write_signed_json(run_dir / "fit.json", fit_receipt)
    return FrozenSilverboxFit(
        run_id,
        path,
        digest,
        fit_receipt["status"],
        term_id,
        source_sha256,
        proposal_receipt_sha256,
    )


def select_silverbox_development(
    fit: FrozenSilverboxFit,
    validation_series: SilverboxControlledSeries,
) -> SilverboxSelection:
    """Freeze the pointwise ensemble forecast and apply the development rule.

    Validation targets are read only after the fit receipt is integrity checked,
    both families are complete, and the pilot's wall/memory limits remain valid.
    This API cannot access sealed records.
    """

    fit_receipt = _load_signed_json(fit.receipt_path, fit.receipt_sha256)
    if fit_receipt.get("run_id") != fit.run_id or fit_receipt.get("source_sha256") != fit.source_sha256:
        raise ValueError("fit handle metadata does not match its frozen receipt")
    if fit_receipt.get("term_id") != fit.term_id or fit_receipt.get("status") != fit.status:
        raise ValueError("fit handle does not match its frozen receipt")
    if fit_receipt.get("proposal_receipt_sha256") != fit.proposal_receipt_sha256:
        raise ValueError("fit handle does not match its upstream proposal receipt binding")
    run_dir = fit.receipt_path.parent
    selection_path = run_dir / "selection.json"
    if selection_path.exists() or (run_dir / "selection_started.json").exists():
        raise ValueError("this frozen fit already has a development selection attempt")
    started_receipt = {
        "protocol_id": PROTOCOL_ID,
        "run_id": fit.run_id,
        "fit_receipt_sha256": fit.receipt_sha256,
        "status": "started",
        "validation_accessed": False,
    }
    _write_signed_json(run_dir / "selection_started.json", started_receipt)

    reason: str | None = None
    if fit.status != "complete" or fit_receipt.get("status") != "complete":
        reason = "fit_incomplete"
    if reason is None and not _fit_code_hashes_match(fit_receipt):
        reason = "fit_code_hash_mismatch"
    budget_info = fit_receipt.get("budget", {})
    elapsed = time.monotonic() - float(budget_info.get("wall_started_monotonic", time.monotonic()))
    if reason is None and elapsed >= PILOT_WALL_SECONDS:
        reason = "wall_time_limit"
    if reason is None:
        baseline = budget_info.get("memory_baseline_rss_bytes")
        current = _current_rss_bytes()
        if baseline is None or current is None:
            reason = "memory_monitor_unavailable"
        elif max(0, current - int(baseline)) > PILOT_ADDITIONAL_MEMORY_BYTES:
            reason = "additional_memory_limit"
    if reason is not None:
        receipt = _unresolved_selection_receipt(fit, reason)
        path, digest = _write_signed_json(selection_path, receipt)
        return SilverboxSelection("unresolved", None, path, digest, reason)

    try:
        _validate_validation_series(validation_series)
    except Exception as error:
        reason = f"invalid_validation_series: {type(error).__name__}: {error}"
        receipt = _unresolved_selection_receipt(fit, reason)
        receipt["validation_accessed"] = True
        path, digest = _write_signed_json(selection_path, receipt)
        return SilverboxSelection("unresolved", None, path, digest, reason)
    scalers = fit_receipt["training_scalers"]
    bound = float(scalers["divergence_bound"])
    hypotheses_receipt = fit_receipt["hypotheses"]
    forecasts: dict[str, dict[str, Any]] = {}
    for hypothesis in HYPOTHESES:
        family = hypotheses_receipt[hypothesis]
        population = family["abc_result"]["populations"][-1]
        parameters = np.asarray(population["accepted_params"], dtype=float)
        weights = np.asarray(population["weights"], dtype=float)
        if parameters.shape != (ABC_PARTICLES, len(family["parameter_names"])):
            forecasts[hypothesis] = _failed_forecast("invalid_particle_shape", parameters.shape)
            continue
        if weights.shape != (ABC_PARTICLES,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            forecasts[hypothesis] = _failed_forecast("invalid_particle_weights", weights.shape)
            continue
        if not np.isclose(float(np.sum(weights)), 1.0, rtol=1e-10, atol=1e-12) or float(np.sum(weights)) <= 0:
            forecasts[hypothesis] = _failed_forecast("unnormalized_particle_weights", float(np.sum(weights)))
            continue
        particle_predictions: list[np.ndarray] = []
        failures: list[dict[str, Any]] = []
        for index, theta in enumerate(parameters):
            try:
                prediction = simulate_controlled_ar2(
                    validation_series.input_u,
                    validation_series.initialization_y,
                    theta,
                    hypothesis=hypothesis,
                    term_id=fit.term_id,
                    s_y=float(scalers["s_y"]),
                    s_feature=float(scalers["s_feature"]),
                    divergence_bound=bound,
                )
                if prediction.shape != validation_series.target_y.shape:
                    raise SimulationFailure(
                        "wrong_output_shape",
                        f"expected {validation_series.target_y.shape}, received {prediction.shape}",
                    )
                particle_predictions.append(prediction)
            except Exception as error:
                mode = error.failure_mode if isinstance(error, SimulationFailure) else "simulator_exception"
                failures.append({"particle_index": index, "failure_mode": mode, "detail": str(error)})
                # Continue through all particles so the receipt exposes every failure.
                particle_predictions.append(np.full(validation_series.target_y.shape, np.nan, dtype=float))
        if failures:
            forecasts[hypothesis] = {
                "status": "failed",
                "failure_mode": "particle_simulation_failure",
                "failures": failures,
                "particle_trajectories": _json_safe(np.asarray(particle_predictions)),
                "pointwise_weighted_median": None,
                "forecast_sha256": None,
            }
        else:
            prediction_matrix = np.asarray(particle_predictions, dtype=float)
            median = pointwise_weighted_median(prediction_matrix, weights)
            forecasts[hypothesis] = {
                "status": "success",
                "failures": [],
                "particle_trajectories": _json_safe(prediction_matrix),
                "pointwise_weighted_median": _json_safe(median),
                "forecast_sha256": _sha256_array(median),
            }

    if any(forecasts[name]["status"] != "success" for name in HYPOTHESES):
        receipt = _unresolved_selection_receipt(fit, "validation_forecast_failure")
        receipt.update(
            {
                "validation_accessed": True,
                "validation_source_range": [validation_series.source_start, validation_series.source_stop],
                "validation_input_sha256": _sha256_array(validation_series.input_u),
                "validation_initializer_sha256": _sha256_array(validation_series.initialization_y),
                "forecasts": forecasts,
                "scores": None,
            }
        )
        path, digest = _write_signed_json(selection_path, receipt)
        return SilverboxSelection("unresolved", None, path, digest, "validation_forecast_failure")

    # Targets are first accessed here, after every particle of both models passed.
    target = np.asarray(validation_series.target_y, dtype=float)
    forecast_arrays = {
        name: np.asarray(forecasts[name]["pointwise_weighted_median"], dtype=float)
        for name in HYPOTHESES
    }
    validation_scores = {
        name: _rmse(forecast_arrays[name], target)
        for name in HYPOTHESES
    }
    persistence_prediction = np.full(target.shape, float(validation_series.initialization_y[-1]), dtype=float)
    persistence_rmse = _rmse(persistence_prediction, target)
    finite_scores = all(math.isfinite(value) for value in (*validation_scores.values(), persistence_rmse))
    neither_beats_persistence = all(value >= persistence_rmse for value in validation_scores.values())
    if not finite_scores:
        outcome = "unresolved"
        selected: str | None = None
        reason = "nonfinite_validation_rmse"
    elif neither_beats_persistence:
        outcome = "unresolved"
        selected = None
        reason = "neither_model_beats_persistence"
    elif validation_scores["nonlinear"] <= (1.0 - NONLINEAR_PROMOTION_MARGIN) * validation_scores["linear"]:
        outcome = "selected"
        selected = "nonlinear"
        reason = "nonlinear_at_least_five_percent_below_linear"
    else:
        outcome = "selected"
        selected = "linear"
        reason = "linear_retained_by_five_percent_rule"

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "run_id": fit.run_id,
        "source_sha256": fit.source_sha256,
        "proposal_receipt_sha256": fit.proposal_receipt_sha256,
        "fit_receipt_sha256": fit.receipt_sha256,
        "status": outcome,
        "reason": reason,
        "selected_hypothesis": selected,
        "validation_accessed": True,
        "validation_source_range": [validation_series.source_start, validation_series.source_stop],
        "validation_input_sha256": _sha256_array(validation_series.input_u),
        "validation_initializer_sha256": _sha256_array(validation_series.initialization_y),
        "validation_target_sha256": _sha256_array(target),
        "validation_target_count": int(target.size),
        "forecasts": forecasts,
        "scores": {
            "rmse": validation_scores,
            "persistence_rmse": persistence_rmse,
            "persistence_forecast": _json_safe(persistence_prediction),
            "persistence_forecast_sha256": _sha256_array(persistence_prediction),
        },
        "selection_rule": {
            "minimum_margin": NONLINEAR_PROMOTION_MARGIN,
            "both_models_must_beat_persistence": True,
            "summary": "per-timepoint weighted median of all particle trajectories; stable value sort; first cumulative normalized weight >= 0.5",
        },
    }
    path, digest = _write_signed_json(selection_path, receipt)
    return SilverboxSelection(outcome, selected, path, digest, reason)


def pointwise_weighted_median(trajectories: Any, weights: Any) -> np.ndarray:
    """Take the stable, lower weighted median independently at each time."""

    paths = np.asarray(trajectories, dtype=float)
    mass = np.asarray(weights, dtype=float)
    if paths.ndim != 2 or paths.shape[0] == 0:
        raise ValueError("trajectories must have shape (particles, time) with at least one particle")
    if mass.shape != (paths.shape[0],) or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("weights must be finite, non-negative, and match the particle count")
    total = float(math.fsum(float(value) for value in mass))
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("weights must be normalized")
    if not np.all(np.isfinite(paths)):
        raise ValueError("particle trajectories must be finite")
    result = np.empty(paths.shape[1], dtype=float)
    normalized = mass / total
    for time_index in range(paths.shape[1]):
        order = np.argsort(paths[:, time_index], kind="stable")
        sorted_values = paths[order, time_index]
        sorted_weights = normalized[order]
        cumulative = 0.0
        for particle_index, weight in enumerate(sorted_weights):
            cumulative = math.fsum((cumulative, float(weight)))
            if cumulative >= 0.5:
                result[time_index] = sorted_values[particle_index]
                break
        else:  # tolerate only a final rounding residue after normalized weights
            result[time_index] = sorted_values[-1]
    return result


def _fit_hypothesis(
    hypothesis: str,
    train_windows: tuple[SilverboxControlledSeries, ...],
    term_id: TermId,
    scalers: dict[str, Any],
    budget: _PilotBudget,
    run_dir: Path,
) -> dict[str, Any]:
    parameter_names = LINEAR_PARAMETER_NAMES if hypothesis == "linear" else NONLINEAR_PARAMETER_NAMES
    bounds = np.asarray(LINEAR_BOUNDS if hypothesis == "linear" else NONLINEAR_BOUNDS, dtype=float)
    calibration_rng = np.random.default_rng(CALIBRATION_SEEDS[hypothesis])
    calibration_draws = calibration_rng.uniform(bounds[:, 0], bounds[:, 1], size=(CALIBRATION_DRAWS, len(bounds)))
    calibration_records: list[dict[str, Any]] = []
    calibration_failures: Counter[str] = Counter()
    finite_discrepancies: list[float] = []
    training_truth = tuple(np.asarray(window.target_y, dtype=float) for window in train_windows)

    simulation_attempts = 0
    for draw_index, theta in enumerate(calibration_draws):
        if budget.check():
            break
        simulation_attempts += 1
        try:
            simulated = _simulate_training(theta, hypothesis, term_id, scalers, train_windows)
            distance = _pooled_rmse(simulated, training_truth)
            if not math.isfinite(distance):
                raise SimulationFailure("nonfinite_discrepancy", "pooled training RMSE was not finite")
            finite_discrepancies.append(distance)
            calibration_records.append(
                {"draw_index": draw_index, "parameters": theta, "status": "finite", "discrepancy": distance}
            )
        except Exception as error:
            mode = error.failure_mode if isinstance(error, SimulationFailure) else "simulator_exception"
            calibration_failures[mode] += 1
            calibration_records.append(
                {
                    "draw_index": draw_index,
                    "parameters": theta,
                    "status": "failed",
                    "discrepancy": float("inf"),
                    "failure_mode": mode,
                    "failure_detail": str(error),
                }
            )

    for draw_index in range(len(calibration_records), CALIBRATION_DRAWS):
        calibration_records.append(
            {
                "draw_index": draw_index,
                "parameters": calibration_draws[draw_index],
                "status": "not_simulated",
                "failure_mode": budget.stop_reason or "calibration_interrupted",
            }
        )
    calibration_complete_budget = simulation_attempts == CALIBRATION_DRAWS and budget.stop_reason is None
    calibration_ok = calibration_complete_budget and len(finite_discrepancies) >= CALIBRATION_MIN_FINITE
    calibration: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "hypothesis": hypothesis,
        "seed": CALIBRATION_SEEDS[hypothesis],
        "prior_draw_count": CALIBRATION_DRAWS,
        "draws_generated": int(len(calibration_draws)),
        "draws_completed": simulation_attempts,
        "parameter_names": parameter_names,
        "bounds": bounds,
        "finite_count": len(finite_discrepancies),
        "failure_count": int(sum(record["status"] == "failed" for record in calibration_records)),
        "not_simulated_count": int(sum(record["status"] == "not_simulated" for record in calibration_records)),
        "failure_modes": dict(sorted(calibration_failures.items())),
        "calibration_draws": calibration_records,
        "quantile_probabilities": CALIBRATION_QUANTILES,
        "quantile_method": "linear",
        "status": "complete" if calibration_ok else "incomplete",
        "reason": None
        if calibration_ok
        else budget.stop_reason or ("fewer_than_128_finite_draws" if len(finite_discrepancies) < CALIBRATION_MIN_FINITE else "calibration_budget_incomplete"),
        "epsilon_schedule": None,
    }
    if calibration_ok:
        q50, q20 = np.quantile(np.asarray(finite_discrepancies), CALIBRATION_QUANTILES, method="linear")
        calibration["epsilon_schedule"] = [float(q50), float(q20)]
        calibration["epsilon_quantiles"] = {"q50": float(q50), "q20": float(q20)}
    _write_signed_json(run_dir / f"{hypothesis}_calibration.json", calibration)
    if not calibration_ok:
        return {
            "status": "incomplete",
            "reason": calibration["reason"],
            "parameter_names": parameter_names,
            "bounds": bounds,
            "calibration": calibration,
            "abc_result": None,
            "abc_failure_modes": {},
        }

    prior_sampler, prior_logpdf = make_uniform_prior(bounds)
    abc_failure_modes: Counter[str] = Counter()

    def simulator(theta: np.ndarray, _rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        try:
            return _simulate_training(theta, hypothesis, term_id, scalers, train_windows)
        except Exception as error:
            mode = error.failure_mode if isinstance(error, SimulationFailure) else "simulator_exception"
            abc_failure_modes[mode] += 1
            raise

    def discrepancy(simulated: tuple[np.ndarray, ...]) -> float:
        return _pooled_rmse(simulated, training_truth)

    def population_event(event: str, generation: int, record: dict[str, Any] | None) -> None:
        path = run_dir / f"{hypothesis}_population_{generation:02d}.json"
        if event == "start":
            _write_signed_json(
                path,
                {
                    "protocol_id": PROTOCOL_ID,
                    "run_id": run_dir.name,
                    "hypothesis": hypothesis,
                    "generation": generation,
                    "status": "started",
                    "epsilon": calibration["epsilon_schedule"][generation],
                    "seed": ABC_SEEDS[hypothesis],
                },
            )
        elif event == "end":
            _write_signed_json(
                path,
                {
                    "protocol_id": PROTOCOL_ID,
                    "run_id": run_dir.name,
                    "hypothesis": hypothesis,
                    "generation": generation,
                    "status": "complete" if record and record["diagnostics"]["complete"] else "incomplete",
                    "population": record,
                    "failure_modes_through_generation": dict(sorted(abc_failure_modes.items())),
                },
            )

    abc_error: str | None = None
    try:
        abc_result = run_gaussian_abc_smc_reference(
            prior_sampler,
            simulator,
            discrepancy,
            target_samples=ABC_PARTICLES,
            epsilon_schedule=calibration["epsilon_schedule"],
            max_attempts_per_population=ABC_ATTEMPTS_PER_POPULATION,
            bounds=bounds,
            prior_logpdf=prior_logpdf,
            covariance_scale=ABC_COVARIANCE_SCALE,
            lambda_noise=ABC_LAMBDA_NOISE,
            nugget=ABC_NUGGET,
            seed=ABC_SEEDS[hypothesis],
            population_event=population_event,
            stop_requested=budget.check,
        )
    except Exception as error:
        abc_result = None
        abc_error = f"{type(error).__name__}: {error}"
    complete = (
        abc_result is not None
        and abc_result.get("status") == "complete"
        and abc_result.get("complete") is True
        and len(abc_result.get("populations", [])) == ABC_POPULATIONS
        and all(bool(population.get("diagnostics", {}).get("complete")) for population in abc_result["populations"])
    )
    return {
        "status": "complete" if complete else "incomplete",
        "reason": None if complete else abc_error or budget.stop_reason or (abc_result or {}).get("termination_reason", "abc_incomplete"),
        "parameter_names": parameter_names,
        "bounds": bounds,
        "calibration": calibration,
        "abc_result": abc_result,
        "abc_failure_modes": dict(sorted(abc_failure_modes.items())),
    }


def _simulate_training(
    parameters: np.ndarray,
    hypothesis: str,
    term_id: TermId,
    scalers: dict[str, Any],
    train_windows: tuple[SilverboxControlledSeries, ...],
) -> tuple[np.ndarray, ...]:
    outputs: list[np.ndarray] = []
    for index, window in enumerate(train_windows):
        try:
            outputs.append(
                simulate_controlled_ar2(
                    window.input_u,
                    window.initialization_y,
                    parameters,
                    hypothesis=hypothesis,  # type: ignore[arg-type]
                    term_id=term_id,
                    s_y=float(scalers["s_y"]),
                    s_feature=float(scalers["s_feature"]),
                    divergence_bound=float(scalers["divergence_bound"]),
                )
            )
        except SimulationFailure as error:
            raise SimulationFailure(error.failure_mode, f"training window {index}: {error.detail}") from error
    return tuple(outputs)


def _pooled_rmse(predictions: Any, targets: Any) -> float:
    predicted = tuple(np.asarray(value, dtype=float) for value in predictions)
    observed = tuple(np.asarray(value, dtype=float) for value in targets)
    if len(predicted) != len(observed) or not predicted:
        return float("inf")
    if any(left.shape != right.shape or left.ndim != 1 for left, right in zip(predicted, observed)):
        return float("inf")
    if any(not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) for left, right in zip(predicted, observed)):
        return float("inf")
    squared_error_sum = math.fsum(
        float(value) * float(value)
        for left, right in zip(predicted, observed)
        for value in (left - right)
    )
    count = sum(value.size for value in observed)
    return math.sqrt(squared_error_sum / count) if count else float("inf")


def _feature(term_id: str, lag_y: np.ndarray, lag_u: np.ndarray) -> np.ndarray:
    if term_id == "y_cubed":
        return np.asarray(lag_y, dtype=float) ** 3
    if term_id == "u_cubed":
        return np.asarray(lag_u, dtype=float) ** 3
    if term_id == "u_y_product":
        return np.asarray(lag_u, dtype=float) * np.asarray(lag_y, dtype=float)
    raise ValueError(f"unsupported term_id {term_id!r}")


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return math.sqrt(math.fsum(float(value) * float(value) for value in values) / values.size)


def _validate_train_windows(train_windows: Any) -> None:
    if not isinstance(train_windows, tuple) or len(train_windows) != len(TRAIN_WINDOW_RELATIVE_STARTS):
        raise ValueError("fit requires the tuple of exactly four fixed training windows")
    expected_starts = tuple(TRAIN_SOURCE_START + offset for offset in TRAIN_WINDOW_RELATIVE_STARTS)
    for index, (window, expected_start) in enumerate(zip(train_windows, expected_starts)):
        if type(window) is not SilverboxControlledSeries:
            raise ValueError(f"training window {index} is not a SilverboxControlledSeries")
        if window.role != "train" or window.source_start != expected_start or window.source_stop != expected_start + TRAIN_WINDOW_LENGTH:
            raise ValueError(f"training window {index} does not match its fixed role and source interval")
        if window.input_u.shape != (TRAIN_WINDOW_LENGTH,) or window.initialization_y.shape != (STATE_INITIALIZATION_LENGTH,):
            raise ValueError(f"training window {index} has an unexpected input or initializer length")
        if window.target_y.shape != (TRAIN_WINDOW_PREDICTION_LENGTH,):
            raise ValueError(f"training window {index} has an unexpected target length")
        if not all(np.all(np.isfinite(array)) for array in (window.input_u, window.initialization_y, window.target_y)):
            raise ValueError(f"training window {index} contains nonfinite values")


def _validate_validation_series(series: Any) -> None:
    if type(series) is not SilverboxControlledSeries:
        raise ValueError("selection requires exactly one SilverboxControlledSeries validation input")
    if (
        series.role != "validation"
        or series.source_start != VALIDATION_SOURCE_START
        or series.source_stop != VALIDATION_SOURCE_STOP
        or series.input_u.size != VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START
        or series.initialization_y.shape != (STATE_INITIALIZATION_LENGTH,)
        or series.target_y.size != series.input_u.size - STATE_INITIALIZATION_LENGTH
    ):
        raise ValueError("selection requires the fixed validation source interval and 50-sample initializer")
    if not all(np.all(np.isfinite(array)) for array in (series.input_u, series.initialization_y, series.target_y)):
        raise ValueError("validation series contains nonfinite values")


def _validate_term_id(term_id: Any) -> None:
    if not isinstance(term_id, str) or term_id not in TERM_IDS:
        raise ValueError(f"term_id must be one of {TERM_IDS}; no fallback term is defined")


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")


def _validate_proposal_receipt_sha256(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("proposal_receipt_sha256 must be a 64-character lowercase hexadecimal SHA-256")


def _fit_metadata(
    train_windows: tuple[SilverboxControlledSeries, ...],
    term_id: TermId,
    source_sha256: str,
    proposal_receipt_sha256: str,
    run_id: str,
    started_unix: float,
    started_monotonic: float,
    rss_baseline: int | None,
) -> dict[str, Any]:
    controlled_hash, abc_hash = _controlled_and_abc_hashes()
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "source_sha256": source_sha256.lower(),
        "proposal_receipt_sha256": proposal_receipt_sha256,
        "term_id": term_id,
        "source_indices": [[window.source_start, window.source_stop] for window in train_windows],
        "sampling_time": float(train_windows[0].sampling_time),
        "protocol_constants": {
            "calibration_draws": CALIBRATION_DRAWS,
            "calibration_min_finite": CALIBRATION_MIN_FINITE,
            "calibration_quantiles": CALIBRATION_QUANTILES,
            "calibration_quantile_method": "linear",
            "calibration_seeds": CALIBRATION_SEEDS,
            "abc_seeds": ABC_SEEDS,
            "abc_particles": ABC_PARTICLES,
            "abc_populations": ABC_POPULATIONS,
            "abc_attempts_per_population": ABC_ATTEMPTS_PER_POPULATION,
            "covariance_scale": ABC_COVARIANCE_SCALE,
            "lambda_noise": ABC_LAMBDA_NOISE,
            "nugget": ABC_NUGGET,
            "linear_parameter_names": LINEAR_PARAMETER_NAMES,
            "linear_bounds": LINEAR_BOUNDS,
            "nonlinear_parameter_names": NONLINEAR_PARAMETER_NAMES,
            "nonlinear_bounds": NONLINEAR_BOUNDS,
            "divergence_rule": "max(abs(full trajectory)) <= max(1.0, 10 * max(abs(training observed y)))",
        },
        "implementation_sha256": _sha256_file(Path(__file__)),
        "controlled_contract_sha256": controlled_hash,
        "abc_reference_sha256": abc_hash,
        "fit_started_unix": started_unix,
        "fit_started_monotonic": started_monotonic,
        "memory_baseline_rss_bytes": rss_baseline,
        "process_id": os.getpid(),
    }


def _fit_code_hashes_match(receipt: dict[str, Any]) -> bool:
    controlled_hash, abc_hash = _controlled_and_abc_hashes()
    return (
        receipt.get("implementation_sha256") == _sha256_file(Path(__file__))
        and receipt.get("controlled_contract_sha256") == controlled_hash
        and receipt.get("abc_reference_sha256") == abc_hash
    )


def _controlled_and_abc_hashes() -> tuple[str, str]:
    controlled_file = Path(__import__("core.real_data.silverbox_controlled", fromlist=["__file__"]).__file__)
    abc_file = Path(__import__("core.abc_smc_reference", fromlist=["__file__"]).__file__)
    return _sha256_file(controlled_file), _sha256_file(abc_file)


def _not_started_family(hypothesis: str, reason: str | None) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "reason": reason or "not_started",
        "parameter_names": LINEAR_PARAMETER_NAMES if hypothesis == "linear" else NONLINEAR_PARAMETER_NAMES,
        "bounds": LINEAR_BOUNDS if hypothesis == "linear" else NONLINEAR_BOUNDS,
        "calibration": {
            "protocol_id": PROTOCOL_ID,
            "hypothesis": hypothesis,
            "seed": CALIBRATION_SEEDS[hypothesis],
            "prior_draw_count": CALIBRATION_DRAWS,
            "draws_completed": 0,
            "finite_count": 0,
            "failure_count": 0,
            "calibration_draws": [],
            "status": "incomplete",
            "reason": reason or "not_started",
        },
        "abc_result": None,
        "abc_failure_modes": {},
    }


def _failed_forecast(failure_mode: str, detail: Any) -> dict[str, Any]:
    return {
        "status": "failed",
        "failure_mode": failure_mode,
        "failure_detail": _json_safe(detail),
        "failures": [],
        "particle_trajectories": None,
        "pointwise_weighted_median": None,
        "forecast_sha256": None,
    }


def _unresolved_selection_receipt(fit: FrozenSilverboxFit, reason: str) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": fit.run_id,
        "source_sha256": fit.source_sha256,
        "proposal_receipt_sha256": fit.proposal_receipt_sha256,
        "fit_receipt_sha256": fit.receipt_sha256,
        "status": "unresolved",
        "reason": reason,
        "selected_hypothesis": None,
        "validation_accessed": False,
        "forecasts": None,
        "scores": None,
    }


def _write_signed_json(path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    canonical = json.dumps(safe_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    envelope = {**safe_payload, "receipt_sha256": digest}
    serialized = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return path, digest


def _load_signed_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    digest = receipt.pop("receipt_sha256", None)
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    observed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != expected_sha256 or observed != expected_sha256:
        raise ValueError("frozen fit receipt hash mismatch")
    receipt["receipt_sha256"] = digest
    return receipt


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    if isinstance(value, Path):
        return str(value)
    return value


def _current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


@contextmanager
def _exclusive_fit_process(receipt_dir: str | Path):
    """Permit at most one fit using a given local receipt root at a time."""

    root = Path(receipt_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".silverbox_first_fit.lock"
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Silverbox first-fit process is active for this receipt directory") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(target, dtype=float)
    if predicted.shape != observed.shape or predicted.size == 0:
        return float("nan")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        return float("nan")
    squared = math.fsum(float(value) * float(value) for value in (predicted - observed))
    return math.sqrt(squared / predicted.size)


__all__ = [
    "PROTOCOL_ID",
    "TERM_IDS",
    "LINEAR_PARAMETER_NAMES",
    "NONLINEAR_PARAMETER_NAMES",
    "LINEAR_BOUNDS",
    "NONLINEAR_BOUNDS",
    "FrozenSilverboxFit",
    "SilverboxSelection",
    "SimulationFailure",
    "train_only_scalers",
    "simulate_controlled_ar2",
    "pointwise_weighted_median",
    "fit_silverbox_development",
    "select_silverbox_development",
]

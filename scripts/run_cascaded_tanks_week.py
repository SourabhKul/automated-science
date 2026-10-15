"""Development fit and score commands for the frozen Cascaded Tanks study."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference
from core.real_data import cascaded_tanks_controlled as source
from core.real_data.cascaded_tanks_models import (
    TankModel, TankParameters, TankSimulationFailure, TankSimulationLimits,
    TankState, simulate_cascaded_tanks,
)
from core.real_data.cascaded_tanks_pattern_search import (
    ObjectiveFailure, ObjectiveSuccess, run_normalized_cube_pattern_search,
)
from core.real_data import cascaded_tanks_week_score as scoring

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "cascaded-tanks-week" / scoring.RUN_ID
MANIFEST_PATH = RUN_DIR / "manifest.json"
FORECAST_PATH = RUN_DIR / "development-forecasts.json"
BOUNDS = np.asarray(scoring.SETTINGS["bounds"], dtype=np.float64)
LOWER, UPPER = BOUNDS[:, 0], BOUNDS[:, 1]
FAMILY_INDEX = {name: index for index, name in enumerate(scoring.MODELS)}
FIT_LIMIT = 3600.0
FORECAST_LIMIT = 120.0
MAX_FIT_CALLS = 299_520
MAX_FORECAST_CALLS = 600
MAX_ARTIFACT_BYTES = 1_073_741_824
MAX_RSS_BYTES = 4_294_967_296


class ResourceStop(RuntimeError):
    """A frozen wall, memory, call, or artifact limit ended this run."""


def _rng(model: str, seed: int, stream: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([20261014, 0, FAMILY_INDEX[model], seed, stream])
    return np.random.Generator(np.random.PCG64(sequence))


def _parameters(theta: Any) -> TankParameters:
    return TankParameters(*(float(value) for value in theta[:3]))


def _simulate(model: str, theta: Any, inputs: tuple[float, ...] | np.ndarray):
    length = len(inputs)
    state = TankState(float(theta[3]), float(scoring.SETTINGS["fixed_x2_0"]))
    return simulate_cascaded_tanks(
        inputs, _parameters(theta), state, model=TankModel(model),
        ceiling=None if model == "S0" else scoring.SETTINGS["fixed_ceiling"],
        limits=TankSimulationLimits(length, scoring.SETTINGS["max_magnitude"]),
    )


def _rmse(generated: Any, observed: Any) -> float:
    pred, target = np.asarray(generated, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if pred.shape != target.shape or pred.size == 0 or not np.isfinite(pred).all():
        raise ValueError("simulation did not return a complete finite trajectory")
    value = float(np.sqrt(np.mean(np.square(pred - target), dtype=np.float64)))
    if not math.isfinite(value):
        raise ValueError("RMSE is non-finite")
    return value


def weighted_mean_bands(particle_forecasts: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, mass = np.asarray(particle_forecasts, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != mass.size or not mass.size:
        raise ValueError("particle forecasts and weights have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("particle forecasts and weights must be finite and nonnegative")
    if not math.isclose(math.fsum(map(float, mass)), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ABC weights must already sum to one")
    mean = np.sum(values * mass[:, None], axis=0, dtype=np.float64)
    bands = []
    for q in (0.1, 0.9):
        band = np.empty(values.shape[1], dtype=np.float64)
        for col in range(values.shape[1]):
            order = np.argsort(values[:, col], kind="stable")
            total = 0.0
            for index in order:
                total = math.fsum((total, float(mass[index])))
                if total >= q:
                    band[col] = values[index, col]
                    break
        bands.append(band)
    return mean, bands[0], bands[1]


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _guard(start: float, seconds: float, run_dir: Path,
           total_calls: list[int] | None = None) -> None:
    if time.monotonic() - start >= seconds:
        raise ResourceStop("wall limit reached")
    if _rss_bytes() >= MAX_RSS_BYTES:
        raise ResourceStop("sampled process RSS limit reached")
    if _directory_bytes(run_dir) >= MAX_ARTIFACT_BYTES:
        raise ResourceStop("artifact byte limit reached")
    if total_calls is not None and total_calls[0] >= MAX_FIT_CALLS:
        raise ResourceStop("development simulator-call cap reached")


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _save_arm(key: tuple[str, str, int], record: dict[str, Any], arrays: dict[str, Any],
              phase: str) -> None:
    if phase not in ("fit", "forecast"):
        raise ValueError("arm checkpoint phase must be fit or forecast")
    kind, model, seed = key
    stem = f"{kind}.{model}.{seed}"
    target = RUN_DIR / "arms" / phase / stem
    buf = io.BytesIO()
    np.savez_compressed(buf, **{name: np.asarray(value) for name, value in arrays.items()})
    _write_exclusive(Path(f"{target}.npz"), buf.getvalue())
    _write_exclusive(Path(f"{target}.json"), scoring.canonical_bytes(record))


def _manifest_source(data: Any) -> dict[str, Any]:
    return {
        "archive_bytes": source.SOURCE_ARCHIVE_BYTES,
        "archive_sha256": data.archive_sha256,
        "track_id": data.track_id,
        "contract_sha256": data.contract_sha256,
        "source_visible_sha256": data.source_visible_sha256,
        "training_sha256": data.stage_receipts.training_sha256,
        "forecast_inputs_sha256": data.stage_receipts.forecast_inputs_sha256,
    }


def freeze_manifest(archive_path: str | Path) -> Path:
    """Freeze source receipts and runtime hashes without fitting or target access."""

    data = source.load_development_data(archive_path)
    if data.archive_sha256 != source.SOURCE_ARCHIVE_SHA256:
        raise scoring.WeekStageError("training reader did not return the pinned archive")
    manifest = {
        "schema": scoring.MANIFEST_SCHEMA, "protocol_id": scoring.PROTOCOL_ID,
        "run_id": scoring.RUN_ID, "git_head": scoring.git_head(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "code_sha256": scoring.code_hashes(), "settings": scoring.SETTINGS,
        "source": _manifest_source(data),
        "arms": [dict(kind=k, model=m, seed=s) for k, m, s in sorted(scoring.expected_roster())],
    }
    _write_exclusive(MANIFEST_PATH, scoring.canonical_bytes(manifest))
    return MANIFEST_PATH


def _load_fit_inputs(archive_path: str | Path, manifest: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    snapshot = scoring.archive_snapshot(archive_path)
    data = source.load_development_data(archive_path)
    scoring.verify_source(manifest, data)
    if len(snapshot) != source.SOURCE_ARCHIVE_BYTES:
        raise scoring.WeekStageError("archive length changed after source validation")
    train_u = np.asarray(data.training_u_est, dtype=np.float64)
    future_u = np.asarray(data.development_u_est, dtype=np.float64)
    train_y = np.asarray(data.training_y_est, dtype=np.float64)
    return data, np.concatenate((train_u, future_u)), train_y, train_u


def _fit_abc(model: str, seed: int, u_train: np.ndarray, y_train: np.ndarray,
             fit_start: float, total_calls: list[int]) -> tuple[dict[str, Any], dict[str, Any], int]:
    calibration_theta = _rng(model, seed, 0).uniform(LOWER, UPPER, size=(256, 4))
    distances = np.full(256, np.nan, dtype=np.float64)
    outcomes = np.full(256, 4, dtype=np.uint8)
    calibration_calls = 0
    resource_stop = None
    for index, theta in enumerate(calibration_theta):
        try:
            _guard(fit_start, FIT_LIMIT, RUN_DIR, total_calls)
        except ResourceStop as error:
            resource_stop = error
            break
        calibration_calls += 1
        total_calls[0] += 1
        try:
            outcome = _simulate(model, theta, u_train)
        except ResourceStop as error:
            distances[index], outcomes[index], resource_stop = np.inf, 3, error
            break
        if isinstance(outcome, TankSimulationFailure):
            distances[index], outcomes[index] = np.inf, 1
        else:
            try:
                distances[index] = _rmse(outcome.observations, y_train)
                outcomes[index] = 0
            except Exception:
                distances[index], outcomes[index] = np.inf, 2
        try:
            _guard(fit_start, FIT_LIMIT, RUN_DIR, total_calls)
        except ResourceStop as error:
            resource_stop = error
            break
    completed_theta = calibration_theta[:calibration_calls]
    distances, outcomes = distances[:calibration_calls], outcomes[:calibration_calls]
    arrays = {"calibration_theta": completed_theta, "calibration_distance": distances,
              "calibration_outcome": outcomes}
    base = {"kind": "abc", "model": model, "seed": seed, "calibration_calls": calibration_calls,
            "calibration_complete": calibration_calls == 256 and resource_stop is None,
            "calibration_failed": int(np.count_nonzero(np.isin(outcomes, (1, 2)))), "epsilon": None,
            "calibration_outcome_codes": ["finite_distance", "simulation_failure", "invalid_distance",
                                           "resource_stop_during_simulator"],
            "abc_calls": 0, "status": "failed"}
    if resource_stop is not None:
        base.update({"resource_stop": True, "failure": str(resource_stop)})
        return base, arrays, calibration_calls
    with np.errstate(invalid="ignore"):
        eps = np.quantile(distances, [0.25, 0.1], method="linear")
    if not np.isfinite(eps).all() or np.any(eps <= 0) or eps[0] < eps[1]:
        base["failure"] = "calibration epsilon is nonfinite, nonpositive, or unordered"
        return base, arrays, calibration_calls
    base["epsilon"] = [float(eps[0]), float(eps[1])]
    sampler, prior_logpdf = make_uniform_prior(BOUNDS)
    counters = {"calls": 0}

    def simulator(theta: np.ndarray, rng: np.random.Generator):
        counters["calls"] += 1
        outcome = _simulate(model, theta, u_train)
        if isinstance(outcome, TankSimulationFailure):
            raise RuntimeError(outcome.category.value)
        return np.asarray(outcome.observations, dtype=np.float64)

    def discrepancy(generated: Any) -> float:
        return _rmse(generated, y_train)

    reference = None
    try:
        reference = run_gaussian_abc_smc_reference(
            sampler, simulator, discrepancy, target_samples=64,
            epsilon_schedule=eps.tolist(), max_attempts_per_population=8192,
            bounds=BOUNDS, prior_logpdf=prior_logpdf, rng=_rng(model, seed, 1),
            covariance_scale=2.0, lambda_noise=0.01, nugget=1e-9,
            stop_requested=lambda: _stop_requested(fit_start, FIT_LIMIT, total_calls),
        )
    except Exception as error:
        base["failure"] = f"ABC reference error: {type(error).__name__}: {error}"[:240]
    abc_calls = counters["calls"]
    base["abc_calls"] = abc_calls
    total_calls[0] += abc_calls
    if reference is None:
        return base, arrays, calibration_calls + abc_calls
    populations = reference["populations"]
    base["abc_status"] = reference["status"]
    base["populations"] = [
        {"generation": int(item["generation"]), "epsilon": float(item["epsilon"]),
         "diagnostics": item["diagnostics"],
         "effective_sample_size": float(item["effective_sample_size"]),
         "proposal_covariance": None if item["proposal_covariance"] is None else
             np.asarray(item["proposal_covariance"], dtype=np.float64).tolist()}
        for item in populations
    ]
    final = populations[-1]
    parameters = np.asarray(final["accepted_params"], dtype=np.float64)
    weights = np.asarray(final["weights"], dtype=np.float64)
    arrays.update({"population_params": parameters, "population_weights": weights,
                   "population_distances": np.asarray(final["distances"], dtype=np.float64)})
    for index, population in enumerate(populations):
        arrays[f"population{index}_params"] = np.asarray(population["accepted_params"], dtype=np.float64)
        arrays[f"population{index}_weights"] = np.asarray(population["weights"], dtype=np.float64)
        arrays[f"population{index}_distances"] = np.asarray(population["distances"], dtype=np.float64)
        arrays[f"population{index}_log_weights"] = np.asarray(population["log_weights"], dtype=np.float64)
        arrays[f"population{index}_log_prior_density"] = np.asarray(population["log_prior_density"], dtype=np.float64)
        arrays[f"population{index}_log_proposal_mixture_density"] = np.asarray(population["log_proposal_mixture_density"], dtype=np.float64)
        covariance = population["proposal_covariance"]
        arrays[f"population{index}_proposal_covariance"] = (
            np.empty((0, 0), dtype=np.float64) if covariance is None else np.asarray(covariance, dtype=np.float64)
        )
    actual = paired_allowance(0, populations)
    if actual != abc_calls:
        base["failure"] = "ABC callback and reference simulator-call counts differ"
        return base, arrays, calibration_calls + actual
    if reference["status"] != "complete" or parameters.shape != (64, 4) or weights.shape != (64,):
        base["failure"] = base.get("failure", "ABC population is incomplete")
        if reference["termination_reason"] == "wall_budget_exhausted":
            base.update({"resource_stop": True, "failure": "resource limit reached during ABC"})
        return base, arrays, calibration_calls + actual
    if not np.isfinite(parameters).all() or not np.isfinite(weights).all() or np.any(weights < 0) or not math.isclose(math.fsum(map(float, weights)), 1.0, rel_tol=0, abs_tol=1e-12):
        base["failure"] = "ABC final population parameters or weights are invalid"
        return base, arrays, calibration_calls + actual
    base.update({"status": "fit_complete", "effective_sample_size": float(reference["effective_sample_size"])})
    return base, arrays, calibration_calls + actual


def _stop_requested(start: float, seconds: float, total_calls: list[int]) -> bool:
    try:
        _guard(start, seconds, RUN_DIR, total_calls)
        return False
    except ResourceStop:
        return True


def _fit_pattern(model: str, seed: int, u_train: np.ndarray, y_train: np.ndarray,
                 allowance: int, fit_start: float, total_calls: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = {"calls": 0, "steps": 0, "failures": 0}

    def objective(cube: tuple[float, ...]):
        _guard(fit_start, FIT_LIMIT, RUN_DIR, total_calls)
        theta = LOWER + np.asarray(cube, dtype=np.float64) * (UPPER - LOWER)
        counts["calls"] += 1
        total_calls[0] += 1
        outcome = _simulate(model, theta, u_train)
        if isinstance(outcome, TankSimulationFailure):
            counts["failures"] += 1
            counts["steps"] += len(outcome.trace)
            return ObjectiveFailure(outcome.category.value, completed_steps=len(outcome.trace))
        counts["steps"] += len(outcome.trace)
        return ObjectiveSuccess(_rmse(outcome.observations, y_train), completed_steps=len(outcome.trace))

    try:
        result = run_normalized_cube_pattern_search(objective, dimension=4, max_calls=allowance)
    except ResourceStop as error:
        error.calls, error.steps = counts["calls"], counts["steps"]
        raise
    best = result.best_evaluation
    record = {"kind": "pattern", "model": model, "seed": seed,
              "status": "fit_complete" if best is not None else "failed",
              "call_allowance": allowance, "callback_calls": result.callback_calls,
              "candidate_attempts": result.candidate_attempts, "cache_hits": result.cache_hits,
              "failures": result.failure_count, "support_rejects": result.support_reject_count,
              "completed_steps": result.completed_steps_total,
              "unused_allowance": result.unused_call_budget, "stop_reason": result.stop_reason}
    arrays = {"evaluations": np.asarray([item.coordinates for item in result.evaluations], dtype=np.float64),
              "objectives": np.asarray([item.objective for item in result.evaluations], dtype=np.float64)}
    if best is not None and math.isfinite(best.objective):
        cube = np.asarray(best.coordinates, dtype=np.float64)
        theta = LOWER + cube * (UPPER - LOWER)
        record["parameter"] = theta.tolist()
        record["training_rmse"] = float(best.objective)
        arrays["best_parameter"] = theta
    else:
        record.update({"status": "failed", "failure": "pattern search has no finite evaluation"})
    return record, arrays


def _fit_arx(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    rows = np.column_stack((np.ones(len(y) - 2), y[1:-1], y[:-2], u[1:len(y) - 1], u[:len(y) - 2]))
    return np.linalg.lstsq(rows, y[2:], rcond=None)[0]


def _forecast_arx(coef: np.ndarray, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not np.isfinite(coef).all() or np.any(np.abs(np.roots([1.0, -coef[1], -coef[2]])) >= 1.0):
        raise ValueError("ARX is nonfinite or has a root with magnitude >= 1")
    history = list(map(float, y[-2:]))
    for k in range(768, 1024):
        value = coef[0] + coef[1] * history[-1] + coef[2] * history[-2] + coef[3] * u[k - 1] + coef[4] * u[k - 2]
        if not math.isfinite(float(value)):
            raise ValueError("ARX free-run forecast is nonfinite")
        history.append(float(value))
    return np.asarray(history[-256:], dtype=np.float64)


def _forecast_arm(record: dict[str, Any], arrays: dict[str, Any], model: str,
                  inputs: np.ndarray, forecast_clock: list[float | int]) -> tuple[dict[str, Any], dict[str, Any]]:
    if record["status"] != "fit_complete":
        return {**record, "forecast": None,
                "failure": record.get("failure", "fit did not complete")}, arrays
    kind = record["kind"]
    if kind == "abc":
        particles, weights = arrays["population_params"], arrays["population_weights"]
    else:
        particles = np.asarray([record["parameter"]], dtype=np.float64)
        weights = np.asarray([1.0], dtype=np.float64)
    trajectories = []
    steps = 0
    calls = 0
    try:
        for theta in particles:
            if forecast_clock[1] >= MAX_FORECAST_CALLS or time.monotonic() - float(forecast_clock[0]) >= FORECAST_LIMIT:
                raise ResourceStop("development forecast call or wall cap reached")
            forecast_clock[1] += 1
            calls += 1
            outcome = _simulate(model, theta, inputs)
            steps += len(outcome.trace)
            forecast_clock[2] += len(outcome.trace)
            if isinstance(outcome, TankSimulationFailure) or len(outcome.observations) != 1024:
                raise ValueError("one particle failed its full 1024-step replay")
            trajectories.append(np.asarray(outcome.observations[768:1024], dtype=np.float64))
        values = np.asarray(trajectories, dtype=np.float64)
        mean, lower, upper = weighted_mean_bands(values, weights)
        record.update({"status": "complete", "forecast": mean.tolist(), "band10": lower.tolist(), "band90": upper.tolist(),
                       "forecast_calls": calls, "forecast_steps": steps})
        arrays["forecast_particles"] = values
        return record, arrays
    except ResourceStop:
        raise
    except Exception as error:
        record.update({"status": "failed", "forecast": None,
                       "failure": f"{type(error).__name__}: {error}"[:240],
                       "forecast_calls": calls, "forecast_steps": steps})
        return record, arrays


def paired_allowance(calibration_calls: int, populations: list[dict[str, Any]]) -> int:
    """Charge calibration plus actual simulator callbacks, including failures."""
    return calibration_calls + sum(
        int(item["diagnostics"]["simulated"]) + int(item["diagnostics"]["failed_simulations"])
        for item in populations
    )


def fit_development(archive_path: str | Path, manifest_path: str | Path = MANIFEST_PATH) -> Path:
    manifest, manifest_hash = scoring.read_manifest(manifest_path)
    data, inputs, y_train, u_train = _load_fit_inputs(archive_path, manifest)
    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_hash = scoring.claim_marker(run_dir, "development-fit", {
        "manifest_sha256": manifest_hash, "archive_sha256": data.archive_sha256})
    started = time.monotonic()
    total_calls = [0]
    fit_records: dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    try:
        baseline_records = []
        try:
            coef = _fit_arx(u_train, y_train)
            forecast = _forecast_arx(coef, inputs, y_train)
            baseline_records.append({"kind": "arx", "model": "ARX(2,2)", "seed": 0,
                                    "status": "complete", "forecast": forecast.tolist(), "coefficients": coef.tolist()})
        except Exception as error:
            baseline_records.append({"kind": "arx", "model": "ARX(2,2)", "seed": 0,
                                    "status": "failed", "failure": f"{type(error).__name__}: {error}"[:240], "forecast": None})
        baseline_records.append({"kind": "persistence", "model": "persistence", "seed": 0,
                                 "status": "complete", "forecast": [float(y_train[-1])] * 256})
        for base in baseline_records:
            key = (base["kind"], base["model"], base["seed"])
            _save_arm(key, base, {"coefficients": np.asarray(base.get("coefficients", []), dtype=np.float64)}, "fit")
            fit_records[key] = (base, {})
        stopped = None
        for model in scoring.MODELS:
            for seed in scoring.SEEDS:
                key = ("abc", model, seed)
                if stopped:
                    abc = {"kind": "abc", "model": model, "seed": seed, "status": "not_run", "failure": stopped}
                    abc_arrays = {}
                    allowance = 0
                else:
                    try:
                        abc, abc_arrays, abc_total = _fit_abc(model, seed, u_train, y_train, started, total_calls)
                        allowance = 256 + abc_total - 256
                    except ResourceStop as error:
                        stopped = str(error)
                        abc = {"kind": "abc", "model": model, "seed": seed, "status": "failed",
                               "failure": stopped, "resource_stop": True}
                        abc_arrays, allowance = {}, 0
                    if abc.get("resource_stop"):
                        stopped = abc["failure"]
                if stopped:
                    pattern = {"kind": "pattern", "model": model, "seed": seed,
                               "status": "not_run", "failure": stopped, "call_allowance": 0}
                    pattern_arrays = {}
                else:
                    try:
                        pattern, pattern_arrays = _fit_pattern(model, seed, u_train, y_train,
                                                               allowance, started, total_calls)
                    except ResourceStop as error:
                        stopped = str(error)
                        pattern = {"kind": "pattern", "model": model, "seed": seed,
                                   "status": "failed", "failure": stopped, "resource_stop": True,
                                   "call_allowance": allowance, "callback_calls": getattr(error, "calls", 0),
                                   "completed_steps": getattr(error, "steps", 0)}
                        pattern_arrays = {}
                for record, arrays in ((abc, abc_arrays), (pattern, pattern_arrays)):
                    arm_key = (record["kind"], model, seed)
                    record.setdefault("status", "failed")
                    _save_arm(arm_key, record, arrays, "fit")
                    fit_records[arm_key] = (record, arrays)
                if stopped is None:
                    try:
                        _guard(started, FIT_LIMIT, run_dir, total_calls)
                    except ResourceStop as error:
                        stopped = str(error)
        forecast_clock: list[float | int] = [time.monotonic(), 0, 0]
        final_arms = []
        forecast_stopped = None
        for key in sorted(scoring.expected_roster()):
            record, arrays = fit_records[key]
            if key[0] in ("arx", "persistence"):
                arm = dict(record)
            elif forecast_stopped:
                arm = {**record, "status": "not_run", "forecast": None,
                       "failure": forecast_stopped}
            else:
                try:
                    arm, f_arrays = _forecast_arm(dict(record), dict(arrays), key[1], inputs, forecast_clock)
                except ResourceStop as error:
                    forecast_stopped = str(error)
                    arm = {**record, "status": "failed", "forecast": None,
                           "failure": forecast_stopped, "resource_stop": True}
                    f_arrays = {}
                arm["forecast_calls_total"] = int(forecast_clock[1])
                arrays = {**arrays, **f_arrays}
            final_arms.append(arm)
            _save_arm(key, arm, arrays, "forecast")
        bundle = {"schema": scoring.FORECAST_SCHEMA, "run_id": scoring.RUN_ID,
                  "manifest_sha256": manifest_hash, "archive_sha256": data.archive_sha256,
                  "fit_marker_sha256": marker_hash, "fit_simulator_calls": total_calls[0],
                  "fit_wall_seconds": time.monotonic() - started,
                  "forecast_calls": int(forecast_clock[1]), "arms": final_arms}
        _write_exclusive(FORECAST_PATH, scoring.canonical_bytes(bundle))
        return FORECAST_PATH
    except Exception as error:
        receipt = {"schema": "cascaded-tanks-week-development-fit-v1", "run_id": scoring.RUN_ID,
                   "stage": "development-fit", "status": "terminal_failure",
                   "marker_sha256": marker_hash, "manifest_sha256": manifest_hash,
                   "archive_sha256": data.archive_sha256,
                   "fit_simulator_calls": total_calls[0],
                   "failure": f"{type(error).__name__}: {error}"[:240]}
        try:
            _write_exclusive(run_dir / "development-fit.failure.json", scoring.canonical_bytes(receipt))
        finally:
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit-development", "score-development", "refit", "official-score"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--forecasts", type=Path, default=FORECAST_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    try:
        if args.stage in ("refit", "official-score"):
            scoring.reject_deferred_stage(args.stage)
        if args.stage == "freeze":
            if args.archive is None:
                parser.error("--archive is required for freeze")
            print(freeze_manifest(args.archive))
        elif args.stage == "fit-development":
            if args.archive is None:
                parser.error("--archive is required for fit-development")
            print(fit_development(args.archive, args.manifest))
        else:
            if args.archive is None:
                parser.error("--archive is required for score-development")
            print(json.dumps(scoring.score_development(args.archive, args.manifest, args.forecasts, RUN_DIR), sort_keys=True))
    except scoring.WeekStageError as error:
        parser.exit(2, f"week study stopped: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded simulator controls for observation mismatch and abstention.

This module is deliberately separate from the historical SBI/BDSS paths.  It
implements the 2026-09-17 observation-mismatch and nested-mechanism controls
with a transparent, deterministic likelihood-weighted parameter grid.  The
grid is a numerical reference, not an ABC posterior or a model-probability
claim.  Every discrepancy and validation score is computed on observed-scale
predictions produced by the model's declared observation map.

The development protocol uses contiguous train/validation/sealed roles.  The
sealed values are accepted only by the final scoring functions after model
selection, and the development receipt contains hashes for train and
validation inputs only.  The pilot driver is intentionally gated behind an
explicit review acknowledgement; unit tests exercise the information-flow
and abstention rules without running the six-cell pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import sqrt
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "observation_mismatch_abstention_control_v1"
BASELINE_KIND = "deterministic_likelihood_weighted_grid_reference"
DEFAULT_TIME_POINTS = np.arange(0.0, 8.000001, 0.25, dtype=float)
DEFAULT_FORECAST_TIMES = np.asarray((8.5, 9.0, 9.5, 10.0), dtype=float)
DEFAULT_TRAIN_COUNT = 16
DEFAULT_VALIDATION_COUNT = 8
PREDECLARED_DATA_SEEDS = (11, 29, 47)
PREDICTIVE_DRAW_COUNT = 512
MAX_GRID_POINTS = 100_000


class ObservationControlError(RuntimeError):
    """Base class for bounded observation-control failures."""


class ObservationControlIncomplete(ObservationControlError):
    """Raised when a bounded grid/reference run cannot produce a receipt."""


def _jsonable(value: Any) -> Any:
    """Convert NumPy values to strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("receipts cannot contain non-finite values")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a receipt deterministically with strict JSON semantics."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)}).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _freeze_array(value: Any, *, name: str, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite array with ndim={ndim}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _validate_time_points(value: Any, *, require_origin: bool = True) -> np.ndarray:
    times = _freeze_array(value, name="time_points")
    if np.any(times < 0) or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("time_points must be non-negative and strictly increasing")
    if require_origin and not np.isclose(times[0], 0.0):
        raise ValueError("the declared control grid must start at the original time origin t=0")
    return times


def _validate_observed(value: Any, *, name: str = "observed") -> np.ndarray:
    return _freeze_array(value, name=name)


@dataclass(frozen=True)
class ControlDataset:
    """One generated trajectory with a frozen train/validation/sealed split."""

    experiment: str
    data_seed: int
    time_points: np.ndarray
    latent: np.ndarray
    observed: np.ndarray
    sigma: float
    true_parameters: Mapping[str, float]
    train_count: int = DEFAULT_TRAIN_COUNT
    validation_count: int = DEFAULT_VALIDATION_COUNT
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.experiment not in {"observation_mismatch", "nested_mechanism"}:
            raise ValueError("unsupported control experiment")
        if isinstance(self.data_seed, bool) or int(self.data_seed) != self.data_seed:
            raise ValueError("data_seed must be an integer")
        times = _validate_time_points(self.time_points)
        latent = _validate_observed(self.latent, name="latent")
        observed = _validate_observed(self.observed)
        if latent.shape != times.shape or observed.shape != times.shape:
            raise ValueError("time_points, latent, and observed must have matching shapes")
        sigma = float(self.sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        train_count = int(self.train_count)
        validation_count = int(self.validation_count)
        if train_count <= 0 or validation_count <= 0 or train_count + validation_count >= len(times):
            raise ValueError("the frozen split must leave at least one sealed observation")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported control protocol version")
        object.__setattr__(self, "data_seed", int(self.data_seed))
        object.__setattr__(self, "time_points", times)
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "train_count", train_count)
        object.__setattr__(self, "validation_count", validation_count)
        object.__setattr__(self, "true_parameters", dict(self.true_parameters))

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_count)

    @property
    def validation_slice(self) -> slice:
        end = self.train_count + self.validation_count
        return slice(self.train_count, end)

    @property
    def sealed_slice(self) -> slice:
        return slice(self.train_count + self.validation_count, None)

    @property
    def train_times(self) -> np.ndarray:
        return self.time_points[self.train_slice]

    @property
    def validation_times(self) -> np.ndarray:
        return self.time_points[self.validation_slice]

    @property
    def sealed_times(self) -> np.ndarray:
        return self.time_points[self.sealed_slice]

    @property
    def train_observed(self) -> np.ndarray:
        return self.observed[self.train_slice]

    @property
    def validation_observed(self) -> np.ndarray:
        return self.observed[self.validation_slice]

    @property
    def sealed_observed(self) -> np.ndarray:
        return self.observed[self.sealed_slice]

    def with_sealed_observations(self, values: Sequence[float]) -> "ControlDataset":
        """Return a copy with only sealed outcomes replaced for sentinel tests."""

        replacement = _validate_observed(values, name="sealed_observations")
        if replacement.shape != self.sealed_observed.shape:
            raise ValueError("sealed replacement has the wrong length")
        observed = np.array(self.observed, copy=True)
        observed[self.sealed_slice] = replacement
        return ControlDataset(
            experiment=self.experiment,
            data_seed=self.data_seed,
            time_points=self.time_points,
            latent=self.latent,
            observed=observed,
            sigma=self.sigma,
            true_parameters=self.true_parameters,
            train_count=self.train_count,
            validation_count=self.validation_count,
            protocol_version=self.protocol_version,
        )

    def full_data_receipt(self) -> dict[str, Any]:
        """Return immutable-by-hash provenance for generated simulator data."""

        true_k = float(self.true_parameters.get("k", 0.65))
        if self.experiment == "observation_mismatch":
            true_c = float(self.true_parameters.get("c", 0.35))
            generator = {
                "latent_equation": f"x(t)=exp(-{true_k:g}*t)",
                "observation_equation": f"y(t)=x(t)/({true_c:g}+x(t))+epsilon",
                "noise": {"distribution": "Normal", "mean": 0.0, "sigma": float(self.sigma), "variance": float(self.sigma**2)},
            }
        else:
            generator = {
                "latent_equation": f"x(t)=exp(-{true_k:g}*t)",
                "observation_equation": "y(t)=x(t)+epsilon",
                "noise": {"distribution": "Normal", "mean": 0.0, "sigma": float(self.sigma), "variance": float(self.sigma**2)},
            }
        return {
            "experiment": self.experiment,
            "data_seed": int(self.data_seed),
            "protocol_version": self.protocol_version,
            "generator": generator,
            "time_points_sha256": sha256_array(self.time_points),
            "latent_sha256": sha256_array(self.latent),
            "observed_sha256": sha256_array(self.observed),
            "time_points_count": int(len(self.time_points)),
            "train_count": int(self.train_count),
            "validation_count": int(self.validation_count),
            "sealed_count": int(len(self.sealed_observed)),
            "sigma": float(self.sigma),
            "true_parameters": dict(self.true_parameters),
        }

    def development_input_receipt(self) -> dict[str, Any]:
        """Receipt safe for model fitting and selection.

        The sealed observation hash is intentionally absent.  Timestamps for
        the sealed suffix are protocol inputs, while its outcomes remain
        outside development selection.
        """

        payload = {
            "experiment": self.experiment,
            "protocol_version": self.protocol_version,
            "data_seed": int(self.data_seed),
            "time_grid_sha256": sha256_array(self.time_points),
            "train_time_sha256": sha256_array(self.train_times),
            "train_observed_sha256": sha256_array(self.train_observed),
            "validation_time_sha256": sha256_array(self.validation_times),
            "validation_observed_sha256": sha256_array(self.validation_observed),
            "sealed_outcomes_available_to_development": False,
            "sealed_times_are_declared_protocol_inputs": True,
            "roles": {
                "train": {
                    "index_start": 0,
                    "index_stop_exclusive": int(self.train_count),
                    "time_start": float(self.train_times[0]),
                    "time_end": float(self.train_times[-1]),
                    "time_role": "high_signal_calibration",
                },
                "validation": {
                    "index_start": int(self.train_count),
                    "index_stop_exclusive": int(self.train_count + self.validation_count),
                    "time_start": float(self.validation_times[0]),
                    "time_end": float(self.validation_times[-1]),
                    "time_role": "held_out_validation",
                },
                "sealed": {
                    "index_start": int(self.train_count + self.validation_count),
                    "index_stop_exclusive": int(len(self.time_points)),
                    "time_start": float(self.sealed_times[0]),
                    "time_end": float(self.sealed_times[-1]),
                    "time_role": "low_abundance_forecast_suffix",
                    "outcomes_in_receipt": False,
                },
            },
            "train_count": int(self.train_count),
            "validation_count": int(self.validation_count),
            "sealed_count": int(len(self.sealed_observed)),
        }
        payload["development_input_sha256"] = sha256_json(payload)
        return payload


def generate_observation_mismatch_dataset(
    data_seed: int,
    *,
    true_k: float = 0.65,
    true_c: float = 0.35,
    sigma: float = 0.025,
    time_points: Sequence[float] = tuple(DEFAULT_TIME_POINTS),
) -> ControlDataset:
    """Generate Experiment A: known saturating observation-map mismatch."""

    times = _validate_time_points(time_points)
    k = float(true_k)
    c = float(true_c)
    noise_scale = float(sigma)
    if not np.isfinite(k) or k <= 0 or not np.isfinite(c) or c <= 0:
        raise ValueError("true_k and true_c must be finite and positive")
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("sigma must be finite and positive")
    latent = np.exp(-k * times)
    mean = saturating_observation(latent, c)
    rng = np.random.default_rng(int(data_seed))
    observed = mean + rng.normal(0.0, noise_scale, size=len(times))
    return ControlDataset(
        experiment="observation_mismatch",
        data_seed=int(data_seed),
        time_points=times,
        latent=latent,
        observed=observed,
        sigma=noise_scale,
        true_parameters={"k": k, "c": c},
    )


def generate_nested_mechanism_dataset(
    data_seed: int,
    *,
    true_k: float = 0.65,
    sigma: float = 0.05,
    time_points: Sequence[float] = tuple(DEFAULT_TIME_POINTS),
) -> ControlDataset:
    """Generate Experiment B: direct one-rate decay negative control."""

    times = _validate_time_points(time_points)
    k = float(true_k)
    noise_scale = float(sigma)
    if not np.isfinite(k) or k <= 0 or not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("true_k and sigma must be finite and positive")
    latent = np.exp(-k * times)
    rng = np.random.default_rng(int(data_seed))
    observed = latent + rng.normal(0.0, noise_scale, size=len(times))
    return ControlDataset(
        experiment="nested_mechanism",
        data_seed=int(data_seed),
        time_points=times,
        latent=latent,
        observed=observed,
        sigma=noise_scale,
        true_parameters={"k": k},
    )


def saturating_observation(latent: np.ndarray | float, c: float) -> np.ndarray:
    """Declared observation map ``y=x/(c+x)``."""

    values = np.asarray(latent, dtype=float)
    scale = float(c)
    if not np.isfinite(scale) or scale <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("latent values and c must be finite, with c positive")
    return values / (scale + values)


def direct_observed_exponential(time_points: np.ndarray, amplitude: float, k: float) -> np.ndarray:
    """Misspecified observed-scale surrogate ``A exp(-k t)``."""

    times = np.asarray(time_points, dtype=float)
    return float(amplitude) * np.exp(-float(k) * times)


def one_rate_prediction(time_points: np.ndarray, k: float) -> np.ndarray:
    """Direct one-rate observed-scale decay with fixed initial value one."""

    return np.exp(-float(k) * np.asarray(time_points, dtype=float))


def two_rate_prediction(time_points: np.ndarray, weight: float, k1: float, k2: float) -> np.ndarray:
    """Nested two-rate observed-scale mixture with fixed initial value one."""

    times = np.asarray(time_points, dtype=float)
    return float(weight) * np.exp(-float(k1) * times) + (1.0 - float(weight)) * np.exp(-float(k2) * times)


@dataclass(frozen=True)
class _ModelDefinition:
    name: str
    parameter_names: tuple[str, ...]
    observation_model: Mapping[str, Any]
    grid_factory: Callable[[], np.ndarray]
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def grid(self) -> np.ndarray:
        values = np.asarray(self.grid_factory(), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.parameter_names) or len(values) == 0:
            raise ObservationControlIncomplete(f"model {self.name} produced an invalid parameter grid")
        if not np.all(np.isfinite(values)):
            raise ObservationControlIncomplete(f"model {self.name} produced a non-finite parameter grid")
        return values


def _rate_grid(step: float = 0.01) -> np.ndarray:
    return np.round(np.arange(0.05, 1.500001, step, dtype=float), 8)


def _saturating_grid() -> np.ndarray:
    rates = _rate_grid()
    c_values = np.round(np.arange(0.05, 1.000001, 0.01, dtype=float), 8)
    return np.asarray([(k, c) for k in rates for c in c_values], dtype=float)


def _direct_grid() -> np.ndarray:
    amplitudes = np.round(np.arange(0.10, 1.500001, 0.01, dtype=float), 8)
    rates = _rate_grid()
    return np.asarray([(amplitude, k) for amplitude in amplitudes for k in rates], dtype=float)


def _one_rate_grid() -> np.ndarray:
    return _rate_grid()[:, None]


def _two_rate_grid() -> np.ndarray:
    rates = np.round(np.arange(0.05, 1.500001, 0.025, dtype=float), 8)
    pairs = [(k1, k2) for k1 in rates for k2 in rates if k1 <= k2]
    weights = np.linspace(0.0, 1.0, 21, dtype=float)
    return np.asarray([(weight, k1, k2) for k1, k2 in pairs for weight in weights], dtype=float)


def _saturating_predictor(parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
    rates = parameters[:, 0, None]
    c_values = parameters[:, 1, None]
    latent = np.exp(-rates * times[None, :])
    return latent / (c_values + latent)


def _direct_predictor(parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
    return parameters[:, 0, None] * np.exp(-parameters[:, 1, None] * times[None, :])


def _one_rate_predictor(parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.exp(-parameters[:, 0, None] * times[None, :])


def _two_rate_predictor(parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
    weights = parameters[:, 0, None]
    return weights * np.exp(-parameters[:, 1, None] * times[None, :]) + (1.0 - weights) * np.exp(
        -parameters[:, 2, None] * times[None, :]
    )


def _model_definition(name: str) -> _ModelDefinition:
    if name == "correct_saturating_observation":
        return _ModelDefinition(
            name=name,
            parameter_names=("k", "c"),
            observation_model={
                "name": "saturating_observation",
                "formula": "y=x/(c+x)",
                "latent_state": "x=exp(-k*t)",
                "initial_condition": {"x0": 1.0, "estimated": False},
                "noise": {"distribution": "Normal", "sigma_source": "dataset_sigma"},
            },
            grid_factory=_saturating_grid,
            predictor=_saturating_predictor,
        )
    if name == "direct_observed_exponential_surrogate":
        return _ModelDefinition(
            name=name,
            parameter_names=("A", "k"),
            observation_model={
                "name": "direct_observed_exponential",
                "formula": "y=A*exp(-k*t)",
                "latent_state": "none; fit is directly on observed y",
                "initial_condition": {"A": "estimated_on_train_only"},
                "noise": {"distribution": "Normal", "sigma_source": "dataset_sigma"},
            },
            grid_factory=_direct_grid,
            predictor=_direct_predictor,
        )
    if name == "one_rate_exponential":
        return _ModelDefinition(
            name=name,
            parameter_names=("k",),
            observation_model={
                "name": "identity_observation",
                "formula": "y=exp(-k*t)+epsilon",
                "latent_state": "same as observed under this control",
                "initial_condition": {"y0": 1.0, "estimated": False},
                "noise": {"distribution": "Normal", "sigma_source": "dataset_sigma"},
            },
            grid_factory=_one_rate_grid,
            predictor=_one_rate_predictor,
        )
    if name == "two_rate_mixture":
        return _ModelDefinition(
            name=name,
            parameter_names=("w", "k1", "k2"),
            observation_model={
                "name": "identity_observation",
                "formula": "y=w*exp(-k1*t)+(1-w)*exp(-k2*t)+epsilon",
                "latent_state": "same as observed under this control",
                "initial_condition": {"y0": 1.0, "estimated": False},
                "constraints": ["0<=w<=1", "0.05<=k1<=k2<=1.5"],
                "nested_simple_case": "k1=k2 (or w in {0,1})",
                "noise": {"distribution": "Normal", "sigma_source": "dataset_sigma"},
            },
            grid_factory=_two_rate_grid,
            predictor=_two_rate_predictor,
        )
    raise ValueError(f"unknown observation-control model {name!r}")


def _normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ObservationControlIncomplete("grid log weights are invalid")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise ObservationControlIncomplete("all grid log weights are non-finite")
    weights = np.exp(values - maximum)
    normalizer = float(np.sum(weights))
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ObservationControlIncomplete("grid weights could not be normalized")
    weights /= normalizer
    if not np.all(np.isfinite(weights)) or not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise ObservationControlIncomplete("normalized grid weights are invalid")
    return weights


def _weighted_quantile_matrix(values: np.ndarray, weights: np.ndarray, probability: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    probabilities = float(probability)
    if matrix.ndim != 2 or len(matrix) == 0 or not 0.0 <= probabilities <= 1.0:
        raise ValueError("weighted quantile inputs are invalid")
    order = np.argsort(matrix, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(matrix, order, axis=0)
    sorted_weights = np.take_along_axis(np.broadcast_to(np.asarray(weights)[:, None], matrix.shape), order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    cumulative[-1, :] = 1.0
    index = np.argmax(cumulative >= probabilities, axis=0)
    columns = np.arange(matrix.shape[1])
    return sorted_values[index, columns]


def _rmse(predictions: np.ndarray, observed: np.ndarray) -> float:
    values = np.asarray(predictions, dtype=float)
    target = np.asarray(observed, dtype=float)
    if values.shape != target.shape or len(values) == 0 or not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
        return float("nan")
    return float(np.sqrt(np.mean((values - target) ** 2)))


@dataclass(frozen=True)
class _GridFit:
    model: _ModelDefinition
    parameters: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray
    train_prediction_median: np.ndarray
    evaluation_times: np.ndarray
    weighted_prediction_medians: np.ndarray
    posterior_predictive_draws: np.ndarray
    sigma: float
    fit_seed: int
    predictive_seed: int
    diagnostics: Mapping[str, Any]

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights * self.weights))

    def summary(self, *, phase: str, start: int, end: int | None = None) -> dict[str, Any]:
        stop = len(self.evaluation_times) if end is None else int(end)
        if start < 0 or stop <= start or stop > len(self.evaluation_times):
            raise ValueError("invalid forecast phase slice")
        draws = self.posterior_predictive_draws[:, start:stop]
        weighted = self.weighted_prediction_medians[start:stop]
        return {
            "phase": phase,
            "time_points": self.evaluation_times[start:stop],
            "weighted_observation_prediction_median": weighted,
            "posterior_predictive_median": np.median(draws, axis=0),
            "posterior_predictive_quantiles": np.quantile(draws, [0.1, 0.5, 0.9], axis=0).T,
            "posterior_predictive_draws": draws,
        }

    def receipt(self, *, include_grid: bool = True) -> dict[str, Any]:
        population = {
            "generation": 0,
            "status": "complete",
            "termination_reason": "grid_exhausted_complete",
            "epsilon": None,
            "epsilon_schedule": [],
            "attempts": int(len(self.parameters)),
            "grid_evaluations": int(len(self.parameters)),
            "simulator_calls": 0,
            "solver_failures": 0,
            "failure_categories": {},
            "weights": self.weights,
            "effective_sample_size": self.effective_sample_size,
        }
        payload: dict[str, Any] = {
            "status": "complete",
            "termination_reason": "grid_exhausted_complete",
            "posterior_summary": False,
            "baseline_kind": BASELINE_KIND,
            "model_name": self.model.name,
            "parameter_names": list(self.model.parameter_names),
            "parameter_grid_count": int(len(self.parameters)),
            "parameter_bounds": {
                name: [float(np.min(self.parameters[:, index])), float(np.max(self.parameters[:, index]))]
                for index, name in enumerate(self.model.parameter_names)
            },
            "parameter_grid_ordering": "fixed predeclared Cartesian grid; two-rate pairs constrained k1<=k2",
            "weights": self.weights,
            "effective_sample_size": self.effective_sample_size,
            "map_parameters": self.parameters[int(np.argmax(self.weights))],
            "weighted_parameter_mean": np.sum(self.parameters * self.weights[:, None], axis=0),
            "fit_seed": int(self.fit_seed),
            "predictive_seed": int(self.predictive_seed),
            "solver_config": {
                "method": "analytic_exp_decay",
                "time_origin": 0.0,
                "initial_condition_frozen": True,
                "solver_calls": 0,
            },
            "observation_model": self.model.observation_model,
            "inference_diagnostics": self.diagnostics,
            "epsilon_schedule": [],
            "population_receipts": [population],
            "evaluation_times": self.evaluation_times,
            "weighted_prediction_medians": self.weighted_prediction_medians,
            "posterior_predictive_draws": self.posterior_predictive_draws,
        }
        if include_grid:
            payload["parameter_grid"] = self.parameters
            payload["parameter_grid_sha256"] = sha256_array(self.parameters)
            payload["log_weights"] = self.log_weights
        return payload


def _fit_grid_model(
    model: _ModelDefinition,
    train_times: np.ndarray,
    train_observed: np.ndarray,
    sigma: float,
    evaluation_times: np.ndarray,
    *,
    fit_seed: int,
    predictive_seed: int,
    predictive_draw_count: int = PREDICTIVE_DRAW_COUNT,
    max_grid_points: int = MAX_GRID_POINTS,
) -> _GridFit:
    times = _validate_time_points(train_times)
    observed = _validate_observed(train_observed, name="train_observed")
    if times.shape != observed.shape:
        raise ObservationControlIncomplete("train times and observations have different shapes")
    scale = float(sigma)
    if not np.isfinite(scale) or scale <= 0:
        raise ObservationControlIncomplete("model sigma is invalid")
    eval_times = _validate_time_points(evaluation_times)
    if isinstance(predictive_draw_count, bool) or int(predictive_draw_count) <= 0:
        raise ObservationControlIncomplete("predictive_draw_count must be positive")
    if isinstance(max_grid_points, bool) or int(max_grid_points) <= 0:
        raise ObservationControlIncomplete("max_grid_points must be positive")
    parameters = model.grid()
    if len(parameters) > int(max_grid_points):
        raise ObservationControlIncomplete(
            f"model {model.name} grid has {len(parameters)} points; cap is {int(max_grid_points)}"
        )
    train_predictions = model.predictor(parameters, times)
    evaluation_predictions = model.predictor(parameters, eval_times)
    if train_predictions.shape != (len(parameters), len(times)) or evaluation_predictions.shape != (
        len(parameters), len(eval_times)
    ):
        raise ObservationControlIncomplete(f"model {model.name} returned an invalid prediction shape")
    if not np.all(np.isfinite(train_predictions)) or not np.all(np.isfinite(evaluation_predictions)):
        raise ObservationControlIncomplete(f"model {model.name} produced non-finite observed predictions")
    residual = train_predictions - observed[None, :]
    sse = np.sum(residual * residual, axis=1)
    log_weights = -0.5 * sse / (scale * scale)
    weights = _normalise_log_weights(log_weights)
    weighted_medians = _weighted_quantile_matrix(evaluation_predictions, weights, 0.5)
    rng = np.random.default_rng(int(predictive_seed))
    draws_index = rng.choice(len(parameters), size=int(predictive_draw_count), replace=True, p=weights)
    latent_draws = evaluation_predictions[draws_index]
    predictive_draws = latent_draws + rng.normal(0.0, scale, size=latent_draws.shape)
    diagnostics = {
        "baseline_kind": BASELINE_KIND,
        "epsilon": None,
        "attempts": int(len(parameters)),
        "grid_evaluations": int(len(parameters)),
        "simulator_calls": 0,
        "accepted": int(len(parameters)),
        "rejected": 0,
        "solver_failures": 0,
        "failure_categories": {},
        "counters_complete": True,
        "weight_semantics": "uniform_grid_prior_times_observed_scale_Gaussian_likelihood",
        "selection_prediction_semantics": "weighted_grid_median_without_new_observation_noise",
        "posterior_language_warning": "grid reference only; no calibrated posterior or model probability claim",
    }
    return _GridFit(
        model=model,
        parameters=parameters,
        weights=weights,
        log_weights=log_weights,
        train_prediction_median=_weighted_quantile_matrix(train_predictions, weights, 0.5),
        evaluation_times=eval_times,
        weighted_prediction_medians=weighted_medians,
        posterior_predictive_draws=predictive_draws,
        sigma=scale,
        fit_seed=int(fit_seed),
        predictive_seed=int(predictive_seed),
        diagnostics=diagnostics,
    )


def _fit_receipt_or_failure(
    model_name: str,
    dataset: ControlDataset,
    evaluation_times: np.ndarray,
    *,
    fit_seed: int,
    predictive_seed: int,
    predictive_draw_count: int,
    max_grid_points: int,
) -> tuple[_GridFit | None, dict[str, Any]]:
    model = _model_definition(model_name)
    try:
        fit = _fit_grid_model(
            model,
            dataset.train_times,
            dataset.train_observed,
            dataset.sigma,
            evaluation_times,
            fit_seed=fit_seed,
            predictive_seed=predictive_seed,
            predictive_draw_count=predictive_draw_count,
            max_grid_points=max_grid_points,
        )
        receipt = fit.receipt()
        receipt["seeds"] = {
            "data_seed": int(dataset.data_seed),
            "fit_seed": int(fit_seed),
            "proposal_seed": None,
            "simulator_noise_seed": None,
            "predictive_seed": int(predictive_seed),
        }
        return fit, receipt
    except (ObservationControlError, ValueError, TypeError, FloatingPointError) as exc:
        return None, {
            "status": "incomplete",
            "baseline_kind": BASELINE_KIND,
            "model_name": model_name,
            "error_category": type(exc).__name__,
            "error": str(exc),
            "weights": [],
            "effective_sample_size": None,
            "posterior_summary": False,
            "posterior_predictive_status": "unavailable_incomplete_fit",
            "posterior_predictive_draws": [],
            "predictive_summary": {
                "status": "unavailable",
                "posterior_summary": False,
                "reason": "incomplete_model_fit",
            },
            "solver_config": {
                "method": "analytic_exp_decay",
                "time_origin": 0.0,
                "initial_condition_frozen": True,
                "solver_calls": 0,
            },
            "observation_model": model.observation_model,
            "inference_diagnostics": {
                "epsilon": None,
                "epsilon_schedule": [],
                "attempts": 0,
                "grid_evaluations": 0,
                "simulator_calls": 0,
                "solver_failures": 0,
                "failure_categories": {},
                "counters_complete": False,
            },
            "termination_reason": "grid_fit_incomplete",
            "posterior_summary": False,
            "epsilon_schedule": [],
            "population_receipts": [],
            "fit_seed": int(fit_seed),
            "predictive_seed": int(predictive_seed),
            "seeds": {
                "data_seed": int(dataset.data_seed),
                "fit_seed": int(fit_seed),
                "proposal_seed": None,
                "simulator_noise_seed": None,
                "predictive_seed": int(predictive_seed),
            },
        }


def _validation_scores(fits: Mapping[str, _GridFit], dataset: ControlDataset) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, fit in fits.items():
        validation_start = dataset.train_count
        validation_end = dataset.train_count + dataset.validation_count
        validation_prediction = fit.weighted_prediction_medians[validation_start:validation_end]
        result[name] = {
            "validation_rmse": _rmse(validation_prediction, dataset.validation_observed),
            "validation_predictions": validation_prediction,
        }
    return result


def _sealed_scores(
    fits: Mapping[str, _GridFit],
    dataset: ControlDataset,
    sealed_observed: np.ndarray,
) -> dict[str, Any]:
    sealed = _validate_observed(sealed_observed, name="sealed_observed")
    if sealed.shape != dataset.sealed_observed.shape:
        raise ValueError("sealed observations have the wrong shape")
    result: dict[str, Any] = {}
    sealed_start = dataset.train_count + dataset.validation_count
    sealed_end = sealed_start + len(dataset.sealed_observed)
    for name, fit in fits.items():
        sealed_prediction = fit.weighted_prediction_medians[sealed_start:sealed_end]
        result[name] = {
            "sealed_rmse": _rmse(sealed_prediction, sealed),
            "sealed_predictions": sealed_prediction,
        }
    return result


def observation_mismatch_selection(
    correct_validation_rmse: float,
    surrogate_validation_rmse: float,
    practical_margin: float,
) -> dict[str, Any]:
    """Apply the frozen Experiment A validation rule to deterministic scores."""

    correct = float(correct_validation_rmse)
    surrogate = float(surrogate_validation_rmse)
    margin = float(practical_margin)
    if not np.isfinite(correct) or not np.isfinite(surrogate) or not np.isfinite(margin) or margin < 0:
        raise ValueError("selection scores and margin must be finite, with a non-negative margin")
    improvement = surrogate - correct
    separated = bool(improvement >= margin)
    return {
        "validation_improvement_surrogate_minus_correct": improvement,
        "practical_margin": margin,
        "separated": separated,
        "decision": "selected_correct_observation_model" if separated else "unresolved_retain_both",
        "selected_model": "correct_saturating_observation" if separated else None,
    }


def nested_mechanism_decision(
    simple_validation_rmse: float,
    flexible_validation_rmse: float,
    simple_sealed_rmse: float,
    flexible_sealed_rmse: float,
    practical_margin: float,
) -> dict[str, Any]:
    """Apply the frozen Experiment B promotion/abstention rule.

    The flexible model must clear the validation margin and reproduce that
    advantage on the sealed suffix.  A failed criterion keeps the simple model
    operationally selected while reporting an unresolved mechanism comparison.
    """

    values = [
        float(simple_validation_rmse),
        float(flexible_validation_rmse),
        float(simple_sealed_rmse),
        float(flexible_sealed_rmse),
        float(practical_margin),
    ]
    if not all(np.isfinite(value) for value in values) or values[-1] < 0:
        raise ValueError("nested decision scores and margin must be finite, with a non-negative margin")
    validation_improvement = values[0] - values[1]
    sealed_improvement = values[2] - values[3]
    validation_pass = bool(validation_improvement >= values[-1])
    sealed_reproduction = bool(sealed_improvement >= values[-1])
    promoted = bool(validation_pass and sealed_reproduction)
    return {
        "validation_improvement_simple_minus_flexible": validation_improvement,
        "sealed_improvement_simple_minus_flexible": sealed_improvement,
        "practical_margin": values[-1],
        "validation_margin_pass": validation_pass,
        "sealed_advantage_reproduced": sealed_reproduction,
        "promoted": promoted,
        "decision": "promote_two_rate" if promoted else "unresolved_retain_simpler",
        "selected_model": "two_rate_mixture" if promoted else "one_rate_exponential",
    }


def _forecast_receipt(fit: _GridFit, dataset: ControlDataset) -> dict[str, Any]:
    validation_start = dataset.train_count
    validation_end = dataset.train_count + dataset.validation_count
    sealed_end = validation_end + len(dataset.sealed_observed)
    future_start = sealed_end
    return {
        "model_name": fit.model.name,
        "prediction_semantics": {
            "weighted_observation_prediction_median": "declared observation map evaluated at weighted grid parameters; no new measurement noise",
            "posterior_predictive": "observed-scale prediction plus independent Normal(0,sigma^2) noise; no latent-state summary",
        },
        "validation": fit.summary(phase="validation", start=validation_start, end=validation_end),
        "sealed_forecast": fit.summary(phase="sealed_forecast", start=validation_end, end=sealed_end),
        "declared_low_signal_future": fit.summary(phase="declared_low_signal_future", start=future_start),
    }


def _receipt_with_hash(payload: Mapping[str, Any], *, hash_key: str = "receipt_sha256") -> dict[str, Any]:
    result = dict(payload)
    result[hash_key] = sha256_json(result)
    return result


def _seed_bundle(dataset: ControlDataset, fit_seed_base: int, predictive_seed_base: int) -> dict[str, Any]:
    """Record all seed roles, including unused stochastic-reference roles."""

    return {
        "data_seed": int(dataset.data_seed),
        "fit_seed_base": int(fit_seed_base),
        "proposal_seed": None,
        "simulator_noise_seed": None,
        "predictive_seed_base": int(predictive_seed_base),
    }


def _evaluation_times(dataset: ControlDataset, forecast_times: Sequence[float]) -> np.ndarray:
    future = _validate_time_points(forecast_times, require_origin=False)
    if np.any(future <= 0.0):
        raise ValueError("forecast times must be positive")
    if np.any(future <= dataset.time_points[-1]):
        raise ValueError("forecast times must be after the sealed time grid")
    return np.concatenate((dataset.time_points, future))


def run_observation_mismatch_control(
    dataset: ControlDataset,
    *,
    practical_margin_factor: float = 1.0,
    forecast_times: Sequence[float] = tuple(DEFAULT_FORECAST_TIMES),
    fit_seed_base: int = 17_001,
    predictive_seed_base: int = 27_001,
    predictive_draw_count: int = PREDICTIVE_DRAW_COUNT,
    max_grid_points: int = MAX_GRID_POINTS,
    sealed_observed_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run bounded Experiment A with sealed scoring only after selection."""

    if dataset.experiment != "observation_mismatch":
        raise ValueError("run_observation_mismatch_control requires an Experiment A dataset")
    factor = float(practical_margin_factor)
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("practical_margin_factor must be finite and non-negative")
    evaluation_times = _evaluation_times(dataset, forecast_times)
    models = ("correct_saturating_observation", "direct_observed_exponential_surrogate")
    fits: dict[str, _GridFit] = {}
    fit_receipts: dict[str, dict[str, Any]] = {}
    for index, model_name in enumerate(models):
        fit, receipt = _fit_receipt_or_failure(
            model_name,
            dataset,
            evaluation_times,
            fit_seed=int(fit_seed_base) + index,
            predictive_seed=int(predictive_seed_base) + index,
            predictive_draw_count=predictive_draw_count,
            max_grid_points=max_grid_points,
        )
        fit_receipts[model_name] = receipt
        if fit is not None:
            fits[model_name] = fit
    if len(fits) != len(models):
        payload = {
            "status": "incomplete",
            "complete": False,
            "termination_reason": "model_fit_incomplete",
            "experiment": dataset.experiment,
            "protocol_version": PROTOCOL_VERSION,
            "baseline_kind": BASELINE_KIND,
            "data_provenance": dataset.full_data_receipt(),
            "development_input": dataset.development_input_receipt(),
            "model_fits": fit_receipts,
            "selection": {"posterior_summary": False, "decision": "incomplete"},
            "final_evaluation": {"status": "not_run", "posterior_summary": False},
            "seeds": _seed_bundle(dataset, fit_seed_base, predictive_seed_base),
        }
        return _receipt_with_hash(payload)

    validation_scores = _validation_scores(fits, dataset)
    margin = factor * dataset.sigma / sqrt(float(dataset.validation_count))
    selection = observation_mismatch_selection(
        validation_scores[models[0]]["validation_rmse"],
        validation_scores[models[1]]["validation_rmse"],
        margin,
    )
    development_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": dataset.experiment,
        "baseline_kind": BASELINE_KIND,
        "data_provenance": dataset.development_input_receipt(),
        "selection_rule": {
            "metric": "validation_RMSE_on_observed_values",
            "improvement": "surrogate_rmse_minus_correct_rmse",
            "practical_margin_formula": "factor*sigma/sqrt(n_validation)",
            "practical_margin_factor": factor,
            "sealed_outcomes_used_for_selection": False,
            "declared_low_signal_future_times": np.asarray(forecast_times, dtype=float),
        },
        "selection": selection,
        "validation_scores": validation_scores,
        "model_fits": fit_receipts,
        "forecasts": {name: _forecast_receipt(fit, dataset) for name, fit in fits.items()},
    }
    development_receipt = _receipt_with_hash(development_payload, hash_key="development_receipt_sha256")
    selected_model = selection["selected_model"]
    if selected_model is None:
        final_evaluation = {
            "status": "not_run_selection_unresolved",
            "posterior_summary": False,
            "sealed_outcomes_used_for_selection": False,
            "selected_model": None,
        }
    else:
        final_observed = dataset.sealed_observed if sealed_observed_override is None else _validate_observed(
            sealed_observed_override, name="sealed_observed_override"
        )
        if final_observed.shape != dataset.sealed_observed.shape:
            raise ValueError("sealed_observed_override has the wrong shape")
        sealed_scores = _sealed_scores(fits, dataset, final_observed)
        final_prediction = sealed_scores[selected_model]["sealed_predictions"]
        final_evaluation = {
            "status": "evaluated_once_after_selection",
            "posterior_summary": False,
            "selected_model": selected_model,
            "sealed_outcomes_used_for_selection": False,
            "sealed_observed_sha256": sha256_array(final_observed),
            "sealed_prediction": final_prediction,
            "sealed_rmse": _rmse(final_prediction, final_observed),
            "evaluation_count": 1,
            "alternative_retained_without_sealed_score": True,
        }
    final_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": dataset.experiment,
        "sealed_role": "final",
        "development_receipt_sha256": development_receipt["development_receipt_sha256"],
        "selected_model": selected_model,
        "final_evaluation": final_evaluation,
    }
    final_receipt = _receipt_with_hash(final_payload, hash_key="final_receipt_sha256")
    payload = {
        "status": "complete",
        "complete": True,
        "termination_reason": "selection_and_final_evaluation_complete",
        "experiment": dataset.experiment,
        "protocol_version": PROTOCOL_VERSION,
        "baseline_kind": BASELINE_KIND,
        "data_provenance": dataset.full_data_receipt(),
        "development_receipt": development_receipt,
        "final_receipt": final_receipt,
        "selection": selection,
        "final_evaluation": final_evaluation,
        "model_fits": fit_receipts,
        "seeds": _seed_bundle(dataset, fit_seed_base, predictive_seed_base),
    }
    return _receipt_with_hash(payload)


def discriminating_measurement_rule(
    simple_posterior_predictive_median: Sequence[float],
    flexible_posterior_predictive_median: Sequence[float],
    candidate_times: Sequence[float],
    sigma: float,
) -> dict[str, Any]:
    """Apply the frozen observed-scale ``2*sigma`` measurement rule."""

    simple_median = _validate_observed(simple_posterior_predictive_median, name="simple_median")
    flexible_median = _validate_observed(flexible_posterior_predictive_median, name="flexible_median")
    times = _validate_time_points(candidate_times, require_origin=False)
    scale = float(sigma)
    if simple_median.shape != flexible_median.shape or simple_median.shape != times.shape:
        raise ValueError("predictive medians and candidate times must have matching shapes")
    if np.any(times <= 0.0) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("candidate times must be positive and sigma must be finite and positive")
    disagreement = np.abs(simple_median - flexible_median)
    threshold = 2.0 * scale
    informative = disagreement > threshold
    if np.any(informative):
        # The candidate order is part of the preregistered protocol.  Pick
        # the first qualifying time; never search the realized outcomes for
        # the largest disagreement.
        candidate = int(np.flatnonzero(informative)[0])
        recommendation = {
            "status": "informative_condition_available",
            "time": float(times[candidate]),
            "median_disagreement": float(disagreement[candidate]),
            "selection_order": "first_qualifying_predeclared_time",
        }
    else:
        recommendation = {
            "status": "no_informative_measurement_under_this_protocol",
            "time": None,
            "median_disagreement": float(np.max(disagreement)) if len(disagreement) else 0.0,
            "selection_order": "first_qualifying_predeclared_time",
        }
    return {
        "rule": "recommend only a predeclared low-signal future time whose posterior-predictive median disagreement exceeds 2*sigma",
        "candidate_times": times,
        "threshold": threshold,
        "simple_posterior_predictive_median": simple_median,
        "flexible_posterior_predictive_median": flexible_median,
        "absolute_median_disagreement": disagreement,
        "recommendation": recommendation,
        "post_hoc_time_search": False,
    }


def _discriminating_measurement_rule(
    fits: Mapping[str, _GridFit],
    dataset: ControlDataset,
    forecast_times: np.ndarray,
) -> dict[str, Any]:
    simple = fits["one_rate_exponential"]
    flexible = fits["two_rate_mixture"]
    start = len(dataset.time_points)
    return discriminating_measurement_rule(
        np.median(simple.posterior_predictive_draws[:, start:], axis=0),
        np.median(flexible.posterior_predictive_draws[:, start:], axis=0),
        forecast_times,
        dataset.sigma,
    )


def run_nested_mechanism_abstention_control(
    dataset: ControlDataset,
    *,
    practical_margin_factor: float = 1.0,
    forecast_times: Sequence[float] = tuple(DEFAULT_FORECAST_TIMES),
    fit_seed_base: int = 37_001,
    predictive_seed_base: int = 47_001,
    predictive_draw_count: int = PREDICTIVE_DRAW_COUNT,
    max_grid_points: int = MAX_GRID_POINTS,
    sealed_observed_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run Experiment B and gate promotion of the nested two-rate model."""

    if dataset.experiment != "nested_mechanism":
        raise ValueError("run_nested_mechanism_abstention_control requires an Experiment B dataset")
    factor = float(practical_margin_factor)
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("practical_margin_factor must be finite and non-negative")
    evaluation_times = _evaluation_times(dataset, forecast_times)
    models = ("one_rate_exponential", "two_rate_mixture")
    fits: dict[str, _GridFit] = {}
    fit_receipts: dict[str, dict[str, Any]] = {}
    for index, model_name in enumerate(models):
        fit, receipt = _fit_receipt_or_failure(
            model_name,
            dataset,
            evaluation_times,
            fit_seed=int(fit_seed_base) + index,
            predictive_seed=int(predictive_seed_base) + index,
            predictive_draw_count=predictive_draw_count,
            max_grid_points=max_grid_points,
        )
        fit_receipts[model_name] = receipt
        if fit is not None:
            fits[model_name] = fit
    if len(fits) != len(models):
        payload = {
            "status": "incomplete",
            "complete": False,
            "termination_reason": "model_fit_incomplete",
            "experiment": dataset.experiment,
            "protocol_version": PROTOCOL_VERSION,
            "baseline_kind": BASELINE_KIND,
            "data_provenance": dataset.full_data_receipt(),
            "development_input": dataset.development_input_receipt(),
            "model_fits": fit_receipts,
            "selection": {"posterior_summary": False, "decision": "incomplete"},
            "final_evaluation": {"status": "not_run", "posterior_summary": False},
            "discriminating_measurement": {"status": "not_run_incomplete"},
            "seeds": _seed_bundle(dataset, fit_seed_base, predictive_seed_base),
        }
        return _receipt_with_hash(payload)

    validation_scores = _validation_scores(fits, dataset)
    margin = factor * dataset.sigma / sqrt(float(dataset.validation_count))
    validation_improvement = validation_scores[models[0]]["validation_rmse"] - validation_scores[models[1]]["validation_rmse"]
    development_selection = {
        "validation_improvement_simple_minus_flexible": validation_improvement,
        "practical_margin": margin,
        "validation_margin_pass": bool(validation_improvement >= margin),
        "sealed_advantage_reproduced": None,
        "promoted": False,
        "decision": "await_sealed_reproduction",
        "selected_model": None,
    }
    development_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": dataset.experiment,
        "baseline_kind": BASELINE_KIND,
        "data_provenance": dataset.development_input_receipt(),
        "selection_rule": {
            "metric": "RMSE_on_observed_values",
            "practical_margin_formula": "factor*sigma/sqrt(n_validation)",
            "practical_margin_factor": factor,
            "promotion_requires": [
                "two_rate_validation_improvement_at_least_margin",
                "two_rate_advantage_reproduces_on_sealed_suffix_at_least_margin",
            ],
            "declared_low_signal_future_times": np.asarray(forecast_times, dtype=float),
            "rule_frozen_before_data_generation": True,
        },
        "selection": development_selection,
        "validation_scores": validation_scores,
        "model_fits": fit_receipts,
        "forecasts": {name: _forecast_receipt(fit, dataset) for name, fit in fits.items()},
    }
    development_receipt = _receipt_with_hash(development_payload, hash_key="development_receipt_sha256")
    final_observed = dataset.sealed_observed if sealed_observed_override is None else _validate_observed(
        sealed_observed_override, name="sealed_observed_override"
    )
    if final_observed.shape != dataset.sealed_observed.shape:
        raise ValueError("sealed_observed_override has the wrong shape")
    final_scores = {
        name: {
            "sealed_rmse": item["sealed_rmse"],
            "sealed_prediction": item["sealed_predictions"],
        }
        for name, item in _sealed_scores(fits, dataset, final_observed).items()
    }
    final_decision = nested_mechanism_decision(
        validation_scores[models[0]]["validation_rmse"],
        validation_scores[models[1]]["validation_rmse"],
        final_scores[models[0]]["sealed_rmse"],
        final_scores[models[1]]["sealed_rmse"],
        margin,
    )
    final_evaluation = {
        "status": "evaluated_once_after_validation",
        "posterior_summary": False,
        "sealed_observed_sha256": sha256_array(final_observed),
        "sealed_outcomes_not_available_to_development_selection": True,
        "evaluation_count": 1,
        "scores": final_scores,
        "decision": final_decision,
    }
    measurement = _discriminating_measurement_rule(
        fits, dataset, _validate_time_points(forecast_times, require_origin=False)
    )
    final_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": dataset.experiment,
        "sealed_role": "final",
        "development_receipt_sha256": development_receipt["development_receipt_sha256"],
        "final_evaluation": final_evaluation,
        "discriminating_measurement": measurement,
    }
    final_receipt = _receipt_with_hash(final_payload, hash_key="final_receipt_sha256")
    payload = {
        "status": "complete",
        "complete": True,
        "termination_reason": "selection_abstention_and_final_evaluation_complete",
        "experiment": dataset.experiment,
        "protocol_version": PROTOCOL_VERSION,
        "baseline_kind": BASELINE_KIND,
        "data_provenance": dataset.full_data_receipt(),
        "development_receipt": development_receipt,
        "final_receipt": final_receipt,
        "selection": final_decision,
        "final_evaluation": final_evaluation,
        "discriminating_measurement": measurement,
        "model_fits": fit_receipts,
        "seeds": _seed_bundle(dataset, fit_seed_base, predictive_seed_base),
    }
    return _receipt_with_hash(payload)


def predeclared_control_cells() -> tuple[dict[str, Any], ...]:
    """Return the six cells that a reviewed pilot may execute later."""

    cells: list[dict[str, Any]] = []
    for experiment in ("observation_mismatch", "nested_mechanism"):
        for data_seed in PREDECLARED_DATA_SEEDS:
            offset = 0 if experiment == "observation_mismatch" else 20_000
            cells.append(
                {
                    "cell_id": f"{experiment}_data{data_seed}",
                    "experiment": experiment,
                    "data_seed": int(data_seed),
                    "fit_seed_base": int(offset + 17_001 + data_seed),
                    "predictive_seed_base": int(offset + 27_001 + data_seed),
                    "status": "predeclared_not_run",
                }
            )
    return tuple(cells)


__all__ = [
    "BASELINE_KIND",
    "ControlDataset",
    "DEFAULT_FORECAST_TIMES",
    "DEFAULT_TIME_POINTS",
    "ObservationControlError",
    "ObservationControlIncomplete",
    "PREDECLARED_DATA_SEEDS",
    "PROTOCOL_VERSION",
    "canonical_json",
    "discriminating_measurement_rule",
    "direct_observed_exponential",
    "generate_nested_mechanism_dataset",
    "generate_observation_mismatch_dataset",
    "nested_mechanism_decision",
    "one_rate_prediction",
    "observation_mismatch_selection",
    "predeclared_control_cells",
    "run_nested_mechanism_abstention_control",
    "run_observation_mismatch_control",
    "saturating_observation",
    "sha256_array",
    "sha256_json",
    "two_rate_prediction",
]

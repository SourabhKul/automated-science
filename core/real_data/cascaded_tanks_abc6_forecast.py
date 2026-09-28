"""Source-free, target-free posterior forecasts for the synthetic ABC6 control.

The caller supplies one frozen ABC6 training-case view and its completed
training result. This module reads only the case's training inputs, fitted
particle parameters and baseline point. It never reads training outputs, opens
the deferred target gate, generates truth targets, or scores a forecast.

For the short window, the simulator carries each reconstructed state through
the 126 known zero-input transitions to state index 204 before forecasting.
Long-window cases already end at that common state index. Particle trajectories
are coherent simulator outputs; weighted pointwise summaries are separate
ensemble statistics and are generally not simulator trajectories.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.real_data.cascaded_tanks_abc6_cases import (
    PARAMETER_ORDER,
    PROSPECTIVE_INPUT,
    SHORT_LENGTH,
    TRAINING_INPUT_L,
    TRAINING_INPUT_S,
    TRAINING_LENGTH,
    ABC6TrainingCaseData,
    case_by_index,
)
from core.real_data.cascaded_tanks_abc6_training import ABC6TrainingResult
from core.real_data.cascaded_tanks_models import (
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

COMMON_STATE_INDEX = TRAINING_LENGTH
RECOVERY_LENGTH = TRAINING_LENGTH - SHORT_LENGTH
WEIGHT_SUM_ABS_TOL = 1.0e-12
QUANTILE_CONVENTION = "weighted left-inverse empirical CDF: first value with CDF >= q"


@dataclass(frozen=True, slots=True)
class ABC6ForecastFailure:
    """One particle or baseline reconstruction/forecast failure."""

    phase: Literal["training", "recovery", "forecast"]
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class ABC6ParticleForecast:
    """One terminal particle and its coherent prospective simulator path."""

    particle_index: int
    parameter_values: tuple[float, ...]
    weight: float
    common_time_state: TankState | None
    trajectory: tuple[float, ...] | None
    failure: ABC6ForecastFailure | None


@dataclass(frozen=True, slots=True)
class ABC6PosteriorBaselineForecast:
    """Forecast evidence without targets, scores, or implicit particle filtering."""

    status: Literal["complete", "abstained_n", "incomplete_abc_fit"]
    case_index: int
    case_id: str
    fit_window: str
    prospective_inputs: tuple[float, ...] | None
    common_state_index: int | None
    parameter_order: tuple[str, ...]
    particles: tuple[ABC6ParticleForecast, ...]
    weights: tuple[float, ...]
    particle_trajectories: tuple[tuple[float, ...] | None, ...]
    aggregate_status: Literal["complete", "suppressed_particle_failure", "unavailable"]
    pointwise_weighted_mean: tuple[float, ...] | None
    pointwise_weighted_median: tuple[float, ...] | None
    pointwise_q05: tuple[float, ...] | None
    pointwise_q95: tuple[float, ...] | None
    effective_sample_size: float | None
    quantile_convention: str
    pointwise_summaries_are_coherent_trajectories: Literal[False]
    baseline_parameter_values: tuple[float, ...] | None
    baseline_common_time_state: TankState | None
    baseline_trajectory: tuple[float, ...] | None
    baseline_failure: ABC6ForecastFailure | None


def _finite_vector(values: object, *, label: str, length: int) -> tuple[float, ...]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric vector") from error
    if array.ndim != 1 or array.dtype.kind not in "fiu" or array.size != length:
        raise ValueError(f"{label} must be a numeric vector of length {length}")
    result = tuple(float(value) for value in array)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _posterior_arrays(
    result: ABC6TrainingResult,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], float] | None:
    """Validate and copy every terminal posterior coordinate and weight."""

    if result.abc_status != "complete" or not isinstance(result.abc_result, Mapping):
        return None
    abc = result.abc_result
    if abc.get("complete") is not True:
        return None
    posterior = abc.get("posterior")
    if not isinstance(posterior, Mapping):
        return None
    if tuple(posterior.get("parameter_order", ())) != tuple(PARAMETER_ORDER):
        return None

    free_values = posterior.get("free_parameter_values")
    if not isinstance(free_values, Mapping) or set(free_values) != set(PARAMETER_ORDER):
        return None
    try:
        columns = [np.asarray(free_values[name]) for name in PARAMETER_ORDER]
        weights_array = np.asarray(posterior.get("weights"))
    except (TypeError, ValueError):
        return None
    if any(column.ndim != 1 or column.dtype.kind not in "fiu" for column in columns):
        return None
    particle_count = columns[0].size
    if particle_count < 1 or any(column.size != particle_count for column in columns):
        return None
    if (
        weights_array.ndim != 1
        or weights_array.dtype.kind not in "fiu"
        or weights_array.size != particle_count
    ):
        return None

    parameter_matrix = np.column_stack(columns).astype(float, copy=True)
    weights_copy = weights_array.astype(float, copy=True)
    if not np.all(np.isfinite(parameter_matrix)) or not np.all(np.isfinite(weights_copy)):
        return None
    if np.any(weights_copy < 0.0):
        return None
    weight_sum = float(np.sum(weights_copy))
    if not math.isfinite(weight_sum) or not math.isclose(
        weight_sum, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_ABS_TOL
    ):
        return None
    if not np.any(weights_copy > 0.0):
        return None

    parameters = tuple(
        tuple(float(value) for value in row) for row in parameter_matrix
    )
    weights = tuple(float(value) for value in weights_copy)
    ess = float(1.0 / np.sum(np.square(weights_copy)))
    if not math.isfinite(ess) or ess <= 0.0:
        return None
    return parameters, weights, ess


def _baseline_values(result: ABC6TrainingResult) -> tuple[float, ...] | None:
    """Decode a valid best normalized point from the baseline's saved record."""

    baseline = getattr(result, "baseline_result", None)
    if baseline is None:
        return None
    best = getattr(baseline, "best_evaluation", None)
    if best is None or not getattr(best, "succeeded", False):
        return None
    coordinates = _finite_vector(
        getattr(best, "coordinates", None), label="baseline coordinates", length=6
    )
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        return None
    bounds = np.asarray(case_by_index(result.case_index).prior_bounds, dtype=float)
    values = tuple(
        float(bounds[index, 0] + coordinates[index] * (bounds[index, 1] - bounds[index, 0]))
        for index in range(len(PARAMETER_ORDER))
    )
    return values if all(math.isfinite(value) for value in values) else None


def _parameters(values: tuple[float, ...]) -> tuple[TankParameters, TankState, float]:
    by_name = dict(zip(PARAMETER_ORDER, values, strict=True))
    return (
        TankParameters(by_name["a"], by_name["c"], by_name["p"]),
        TankState(by_name["x1_0"], by_name["x2_0"]),
        by_name["ceiling"],
    )


def _safe_simulation(
    inputs: Sequence[float],
    values: tuple[float, ...],
    *,
    data: ABC6TrainingCaseData,
    phase: Literal["training", "recovery", "forecast"],
    initial_state: TankState | None = None,
    simulator,
    max_magnitude: float,
) -> tuple[TankSimulationSuccess | None, ABC6ForecastFailure | None]:
    parameters, declared_initial, ceiling = _parameters(values)
    try:
        outcome = simulator(
            inputs,
            parameters,
            declared_initial if initial_state is None else initial_state,
            model=data.case.fit_model,
            ceiling=ceiling,
            limits=TankSimulationLimits(
                max_steps=len(inputs), max_magnitude=max_magnitude
            ),
        )
    except Exception as error:  # Preserve one particle failure without filtering it.
        return None, ABC6ForecastFailure(
            phase, "unexpected_exception", f"{type(error).__name__}: {error}"
        )
    if isinstance(outcome, TankSimulationFailure):
        return None, ABC6ForecastFailure(
            phase, outcome.category.value, outcome.message
        )
    if not isinstance(outcome, TankSimulationSuccess):
        return None, ABC6ForecastFailure(
            phase, "invalid_simulator_result", "simulator returned an unknown result"
        )
    try:
        observations = _finite_vector(
            outcome.observations, label=f"{phase} trajectory", length=len(inputs)
        )
        terminal_state = outcome.terminal_state
        state_values = _finite_vector(
            (terminal_state.x1, terminal_state.x2),
            label=f"{phase} terminal state",
            length=2,
        )
    except (AttributeError, TypeError, ValueError) as error:
        return None, ABC6ForecastFailure(
            phase, "invalid_trajectory", str(error)
        )
    if state_values[0] < 0.0 or state_values[1] < 0.0:
        return None, ABC6ForecastFailure(
            phase, "invalid_terminal_state", "terminal states must be nonnegative"
        )
    # Validate all returned observations but keep them only for the forecast phase.
    if phase == "forecast":
        # Constructing a replacement is unnecessary; caller reads the simulator result.
        pass
    return outcome, None


def _trajectory_from_values(
    values: tuple[float, ...],
    *,
    data: ABC6TrainingCaseData,
    simulator,
    max_magnitude: float,
) -> tuple[TankState | None, tuple[float, ...] | None, ABC6ForecastFailure | None]:
    fit_inputs = data.inputs
    training, failure = _safe_simulation(
        fit_inputs,
        values,
        data=data,
        phase="training",
        simulator=simulator,
        max_magnitude=max_magnitude,
    )
    if failure is not None or training is None:
        return None, None, failure
    state = training.terminal_state

    if data.case.input_window == "S":
        recovery, failure = _safe_simulation(
            (0.0,) * RECOVERY_LENGTH,
            values,
            data=data,
            phase="recovery",
            initial_state=state,
            simulator=simulator,
            max_magnitude=max_magnitude,
        )
        if failure is not None or recovery is None:
            return None, None, failure
        state = recovery.terminal_state

    forecast, failure = _safe_simulation(
        PROSPECTIVE_INPUT,
        values,
        data=data,
        phase="forecast",
        initial_state=state,
        simulator=simulator,
        max_magnitude=max_magnitude,
    )
    if failure is not None or forecast is None:
        return state, None, failure
    trajectory = _finite_vector(
        forecast.observations,
        label="prospective forecast",
        length=len(PROSPECTIVE_INPUT),
    )
    return state, trajectory, None


def weighted_left_inverse_quantile(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> float:
    """Return the first sorted value whose weighted empirical CDF reaches q."""

    value_array = np.asarray(values)
    weight_array = np.asarray(weights)
    if (
        value_array.ndim != 1
        or value_array.dtype.kind not in "fiu"
        or value_array.size == 0
        or not np.all(np.isfinite(value_array))
    ):
        raise ValueError("values must be a non-empty finite numeric vector")
    if (
        weight_array.ndim != 1
        or weight_array.dtype.kind not in "fiu"
        or weight_array.size != value_array.size
        or not np.all(np.isfinite(weight_array))
        or np.any(weight_array < 0.0)
    ):
        raise ValueError("weights must be finite, nonnegative, and match values")
    q = float(probability)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    total = float(np.sum(weight_array))
    if total <= 0.0 or not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_ABS_TOL
    ):
        raise ValueError("weights must sum to one")
    order = np.argsort(value_array, kind="stable")
    cumulative = np.cumsum(weight_array[order])
    index = int(np.searchsorted(cumulative, q, side="left"))
    if index >= order.size:
        index = order.size - 1
    return float(value_array[order[index]])


def _pointwise_summaries(
    trajectories: tuple[tuple[float, ...], ...], weights: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    matrix = np.asarray(trajectories, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    mean = tuple(float(value) for value in np.sum(matrix * weight_array[:, None], axis=0))
    median = tuple(
        weighted_left_inverse_quantile(matrix[:, index], weights, 0.50)
        for index in range(matrix.shape[1])
    )
    q05 = tuple(
        weighted_left_inverse_quantile(matrix[:, index], weights, 0.05)
        for index in range(matrix.shape[1])
    )
    q95 = tuple(
        weighted_left_inverse_quantile(matrix[:, index], weights, 0.95)
        for index in range(matrix.shape[1])
    )
    return mean, median, q05, q95


def forecast_abc6_posterior_and_baseline(
    data: ABC6TrainingCaseData,
    result: ABC6TrainingResult,
    *,
    simulator=simulate_cascaded_tanks,
    max_magnitude: float = 1.0e12,
) -> ABC6PosteriorBaselineForecast:
    """Reconstruct terminal states and forecast one completed non-N ABC6 fit.

    The training output field is deliberately never accessed. Only the frozen
    training inputs and fitted parameter points determine reconstructed
    states. An N case always abstains because the protocol declares no
    prospective target or forecast score for that case.
    """

    if not isinstance(data, ABC6TrainingCaseData):
        raise TypeError("data must be one ABC6TrainingCaseData case view")
    if not callable(simulator):
        raise TypeError("simulator must be callable")
    if isinstance(max_magnitude, bool):
        raise TypeError("max_magnitude must be a positive finite number")
    try:
        checked_max_magnitude = float(max_magnitude)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("max_magnitude must be a positive finite number") from error
    if not math.isfinite(checked_max_magnitude) or checked_max_magnitude <= 0.0:
        raise ValueError("max_magnitude must be a positive finite number")

    case = data.case
    if case != case_by_index(case.case_index):
        raise ValueError("case identity must match the frozen ABC6 roster")
    expected_inputs = TRAINING_INPUT_S if case.input_window == "S" else TRAINING_INPUT_L
    if data.inputs != expected_inputs:
        raise ValueError("case inputs must exactly match its frozen training window")
    if (
        result.case_index != case.case_index
        or result.case_id != case.case_id
        or result.model != case.fit_model.value
        or result.fit_length != case.input_length
    ):
        raise ValueError("training result identity must match the supplied case")

    if case.truth_id == "N":
        return ABC6PosteriorBaselineForecast(
            status="abstained_n",
            case_index=case.case_index,
            case_id=case.case_id,
            fit_window=case.input_window,
            prospective_inputs=None,
            common_state_index=None,
            parameter_order=tuple(PARAMETER_ORDER),
            particles=(),
            weights=(),
            particle_trajectories=(),
            aggregate_status="unavailable",
            pointwise_weighted_mean=None,
            pointwise_weighted_median=None,
            pointwise_q05=None,
            pointwise_q95=None,
            effective_sample_size=None,
            quantile_convention=QUANTILE_CONVENTION,
            pointwise_summaries_are_coherent_trajectories=False,
            baseline_parameter_values=None,
            baseline_common_time_state=None,
            baseline_trajectory=None,
            baseline_failure=None,
        )

    posterior = _posterior_arrays(result)
    if posterior is None:
        return ABC6PosteriorBaselineForecast(
            status="incomplete_abc_fit",
            case_index=case.case_index,
            case_id=case.case_id,
            fit_window=case.input_window,
            prospective_inputs=None,
            common_state_index=None,
            parameter_order=tuple(PARAMETER_ORDER),
            particles=(),
            weights=(),
            particle_trajectories=(),
            aggregate_status="unavailable",
            pointwise_weighted_mean=None,
            pointwise_weighted_median=None,
            pointwise_q05=None,
            pointwise_q95=None,
            effective_sample_size=None,
            quantile_convention=QUANTILE_CONVENTION,
            pointwise_summaries_are_coherent_trajectories=False,
            baseline_parameter_values=None,
            baseline_common_time_state=None,
            baseline_trajectory=None,
            baseline_failure=None,
        )

    parameter_rows, weights, ess = posterior
    particles: list[ABC6ParticleForecast] = []
    for particle_index, (values, weight) in enumerate(
        zip(parameter_rows, weights, strict=True)
    ):
        state, trajectory, failure = _trajectory_from_values(
            values,
            data=data,
            simulator=simulator,
            max_magnitude=checked_max_magnitude,
        )
        particles.append(
            ABC6ParticleForecast(
                particle_index=particle_index,
                parameter_values=values,
                weight=weight,
                common_time_state=state,
                trajectory=trajectory,
                failure=failure,
            )
        )

    particle_tuple = tuple(particles)
    all_particle_forecasts = all(particle.trajectory is not None for particle in particle_tuple)
    if all_particle_forecasts:
        complete_trajectories = tuple(
            particle.trajectory for particle in particle_tuple
        )
        # The `all` check above establishes that these optionals are tuples.
        trajectories = tuple(
            trajectory for trajectory in complete_trajectories if trajectory is not None
        )
        mean, median, q05, q95 = _pointwise_summaries(trajectories, weights)
        aggregate_status: Literal["complete", "suppressed_particle_failure"] = "complete"
    else:
        mean = median = q05 = q95 = None
        aggregate_status = "suppressed_particle_failure"

    baseline_values = _baseline_values(result)
    baseline_state: TankState | None = None
    baseline_trajectory: tuple[float, ...] | None = None
    baseline_failure: ABC6ForecastFailure | None = None
    if baseline_values is None:
        baseline_failure = ABC6ForecastFailure(
            "training", "no_valid_best_point", "baseline has no finite successful best point"
        )
    else:
        baseline_state, baseline_trajectory, baseline_failure = _trajectory_from_values(
            baseline_values,
            data=data,
            simulator=simulator,
            max_magnitude=checked_max_magnitude,
        )

    return ABC6PosteriorBaselineForecast(
        status="complete",
        case_index=case.case_index,
        case_id=case.case_id,
        fit_window=case.input_window,
        prospective_inputs=tuple(float(value) for value in PROSPECTIVE_INPUT),
        common_state_index=COMMON_STATE_INDEX,
        parameter_order=tuple(PARAMETER_ORDER),
        particles=particle_tuple,
        weights=weights,
        particle_trajectories=tuple(particle.trajectory for particle in particle_tuple),
        aggregate_status=aggregate_status,
        pointwise_weighted_mean=mean,
        pointwise_weighted_median=median,
        pointwise_q05=q05,
        pointwise_q95=q95,
        effective_sample_size=ess,
        quantile_convention=QUANTILE_CONVENTION,
        pointwise_summaries_are_coherent_trajectories=False,
        baseline_parameter_values=baseline_values,
        baseline_common_time_state=baseline_state,
        baseline_trajectory=baseline_trajectory,
        baseline_failure=baseline_failure,
    )


__all__ = [
    "ABC6ForecastFailure",
    "ABC6ParticleForecast",
    "ABC6PosteriorBaselineForecast",
    "COMMON_STATE_INDEX",
    "PROSPECTIVE_INPUT",
    "QUANTILE_CONVENTION",
    "RECOVERY_LENGTH",
    "forecast_abc6_posterior_and_baseline",
    "weighted_left_inverse_quantile",
]

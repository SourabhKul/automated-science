"""Synthetic-only discrete-time surrogate models for cascaded tanks.

This module has no source-data loading, fitting, priors, or inference code.  It
implements the S0, O2, and C2 equations from the 2026-09-26 design proposal at
the fixed four-second sample interval.  The coefficient on ``sqrt(x1)`` in the
lower-state inflow is fixed to one as the proposal's latent-scale convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

SAMPLE_INTERVAL_SECONDS = 4.0


class TankModel(str, Enum):
    """Supported deterministic model families."""

    S0 = "S0"
    O2 = "O2"
    C2 = "C2"


class TankFailureCategory(str, Enum):
    """Terminal reasons a simulator call could not produce a full trace."""

    INVALID_INPUT = "invalid_input"
    INVALID_PARAMETER = "invalid_parameter"
    INVALID_INITIAL_STATE = "invalid_initial_state"
    INVALID_CONFIGURATION = "invalid_configuration"
    TOO_MANY_STEPS = "too_many_steps"
    NON_FINITE = "non_finite"
    MAGNITUDE_LIMIT = "magnitude_limit"


@dataclass(frozen=True)
class TankParameters:
    """Positive per-sample coefficients; lower-state inflow gain is fixed at 1."""

    a: float
    c: float
    p: float


@dataclass(frozen=True)
class TankState:
    """Nonnegative effective states immediately before or after a transition."""

    x1: float
    x2: float


@dataclass(frozen=True)
class TankSimulationLimits:
    """Explicit resource and numerical bounds for one simulator call."""

    max_steps: int = 100_000
    max_magnitude: float = 1.0e12


_DEFAULT_LIMITS = TankSimulationLimits()


@dataclass(frozen=True)
class TankStep:
    """One observed sample and its transition to the next latent state."""

    index: int
    input_u: float
    state: TankState
    observation_y: float
    q12: float
    q2: float
    x1_next: float
    x2_raw: float
    next_state: TankState


@dataclass(frozen=True)
class TankSimulationSuccess:
    """A complete trajectory, including the state carried to the next segment."""

    model: TankModel
    parameters: TankParameters
    initial_state: TankState
    ceiling: float | None
    sample_interval_seconds: float
    trace: tuple[TankStep, ...]
    terminal_state: TankState

    @property
    def observations(self) -> tuple[float, ...]:
        """Observed output sequence in sample order."""

        return tuple(step.observation_y for step in self.trace)


@dataclass(frozen=True)
class TankSimulationFailure:
    """A categorized failure with the successfully completed prefix, if any."""

    category: TankFailureCategory
    message: str
    step_index: int | None
    trace: tuple[TankStep, ...]
    terminal_state: TankState | None


TankSimulationOutcome: TypeAlias = TankSimulationSuccess | TankSimulationFailure


class _SimulationAbort(Exception):
    def __init__(
        self,
        category: TankFailureCategory,
        message: str,
        *,
        step_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.step_index = step_index


def _finite_number(
    value: object, *, label: str, category: TankFailureCategory
) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise _SimulationAbort(category, f"{label} must be a finite real number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise _SimulationAbort(
            category, f"{label} must be a finite real number"
        ) from error
    if not math.isfinite(number):
        raise _SimulationAbort(category, f"{label} must be finite")
    return number


def _checked_magnitude(
    value: float,
    *,
    label: str,
    max_magnitude: float,
    step_index: int,
) -> float:
    if not math.isfinite(value):
        raise _SimulationAbort(
            TankFailureCategory.NON_FINITE,
            f"{label} became non-finite",
            step_index=step_index,
        )
    if abs(value) > max_magnitude:
        raise _SimulationAbort(
            TankFailureCategory.MAGNITUDE_LIMIT,
            f"{label} exceeded the magnitude limit {max_magnitude:g}",
            step_index=step_index,
        )
    return value


def _validate_configuration(
    model: TankModel | str,
    parameters: TankParameters,
    initial_state: TankState,
    ceiling: float | None,
    limits: TankSimulationLimits,
) -> tuple[TankModel, TankParameters, TankState, float | None, TankSimulationLimits]:
    if not isinstance(limits, TankSimulationLimits):
        raise _SimulationAbort(
            TankFailureCategory.INVALID_CONFIGURATION,
            "limits must be TankSimulationLimits",
        )
    if (
        isinstance(limits.max_steps, bool)
        or not isinstance(limits.max_steps, int)
        or limits.max_steps < 1
    ):
        raise _SimulationAbort(
            TankFailureCategory.INVALID_CONFIGURATION,
            "max_steps must be a positive integer",
        )
    max_magnitude = _finite_number(
        limits.max_magnitude,
        label="max_magnitude",
        category=TankFailureCategory.INVALID_CONFIGURATION,
    )
    if max_magnitude <= 0.0:
        raise _SimulationAbort(
            TankFailureCategory.INVALID_CONFIGURATION,
            "max_magnitude must be positive",
        )
    checked_limits = TankSimulationLimits(limits.max_steps, max_magnitude)

    try:
        checked_model = model if isinstance(model, TankModel) else TankModel(model)
    except (TypeError, ValueError) as error:
        raise _SimulationAbort(
            TankFailureCategory.INVALID_CONFIGURATION,
            "model must be one of S0, O2, or C2",
        ) from error

    if not isinstance(parameters, TankParameters):
        raise _SimulationAbort(
            TankFailureCategory.INVALID_PARAMETER, "parameters must be TankParameters"
        )
    checked_parameters = TankParameters(
        _finite_number(
            parameters.a, label="a", category=TankFailureCategory.INVALID_PARAMETER
        ),
        _finite_number(
            parameters.c, label="c", category=TankFailureCategory.INVALID_PARAMETER
        ),
        _finite_number(
            parameters.p, label="p", category=TankFailureCategory.INVALID_PARAMETER
        ),
    )
    for label, value in (
        ("a", checked_parameters.a),
        ("c", checked_parameters.c),
        ("p", checked_parameters.p),
    ):
        if value <= 0.0:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_PARAMETER,
                f"{label} must be positive",
            )
        if value > max_magnitude:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_PARAMETER,
                f"{label} exceeds the magnitude limit {max_magnitude:g}",
            )

    if not isinstance(initial_state, TankState):
        raise _SimulationAbort(
            TankFailureCategory.INVALID_INITIAL_STATE,
            "initial_state must be TankState",
        )
    checked_initial_state = TankState(
        _finite_number(
            initial_state.x1,
            label="initial_state.x1",
            category=TankFailureCategory.INVALID_INITIAL_STATE,
        ),
        _finite_number(
            initial_state.x2,
            label="initial_state.x2",
            category=TankFailureCategory.INVALID_INITIAL_STATE,
        ),
    )
    for label, value in (
        ("x1", checked_initial_state.x1),
        ("x2", checked_initial_state.x2),
    ):
        if value < 0.0:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_INITIAL_STATE,
                f"initial_state.{label} must be nonnegative",
            )
        if value > max_magnitude:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_INITIAL_STATE,
                f"initial_state.{label} exceeds the magnitude limit {max_magnitude:g}",
            )

    checked_ceiling: float | None
    if checked_model is TankModel.S0:
        if ceiling is not None:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_CONFIGURATION,
                "ceiling must be omitted for S0",
            )
        checked_ceiling = None
    else:
        if ceiling is None:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_CONFIGURATION,
                f"ceiling is required for {checked_model.value}",
            )
        checked_ceiling = _finite_number(
            ceiling,
            label="ceiling",
            category=TankFailureCategory.INVALID_CONFIGURATION,
        )
        if checked_ceiling < 0.0:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_CONFIGURATION,
                "ceiling must be nonnegative",
            )
        if checked_ceiling > max_magnitude:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_CONFIGURATION,
                f"ceiling exceeds the magnitude limit {max_magnitude:g}",
            )
        if checked_model is TankModel.O2 and checked_initial_state.x2 > checked_ceiling:
            raise _SimulationAbort(
                TankFailureCategory.INVALID_INITIAL_STATE,
                "O2 initial x2 must not exceed its state ceiling",
            )

    return (
        checked_model,
        checked_parameters,
        checked_initial_state,
        checked_ceiling,
        checked_limits,
    )


def simulate_cascaded_tanks(
    inputs: Sequence[float],
    parameters: TankParameters,
    initial_state: TankState,
    *,
    model: TankModel | str = TankModel.S0,
    ceiling: float | None = None,
    limits: TankSimulationLimits = _DEFAULT_LIMITS,
) -> TankSimulationOutcome:
    """Run a deterministic S0, O2, or C2 trajectory on synthetic inputs.

    For input ``u[k]``, ``y[k]`` is read from the state at index ``k`` before
    applying the transition to index ``k + 1``.  The transition uses the fixed
    lower-state inflow gain of one:

    ``q2 = c*sqrt(x2)``
    ``x1_next = max(0, x1 + p*u - a*sqrt(x1))``
    ``x2_raw = max(0, x2 + sqrt(x1) - q2)``

    S0 leaves the lower state uncapped.  O2 caps the next lower state at
    ``ceiling`` and requires its supplied initial lower state to respect that
    ceiling.  C2 leaves latent state uncapped and caps only its observation.
    Numerical failures return a categorized failure plus the valid trace prefix;
    the simulator never silently repairs values outside its declared floors or
    family-specific ceiling.
    """

    try:
        checked_model, checked_parameters, state, checked_ceiling, checked_limits = (
            _validate_configuration(model, parameters, initial_state, ceiling, limits)
        )
    except _SimulationAbort as error:
        return TankSimulationFailure(
            error.category, error.message, error.step_index, (), None
        )
    initial_state_checked = state

    if isinstance(inputs, (str, bytes, bytearray)):
        return TankSimulationFailure(
            TankFailureCategory.INVALID_INPUT,
            "inputs must be a numeric sequence, not text or bytes",
            None,
            (),
            state,
        )

    try:
        input_count = len(inputs)
    except (TypeError, AttributeError):
        return TankSimulationFailure(
            TankFailureCategory.INVALID_INPUT,
            "inputs must be a finite sequence",
            None,
            (),
            state,
        )
    if input_count > checked_limits.max_steps:
        return TankSimulationFailure(
            TankFailureCategory.TOO_MANY_STEPS,
            f"input length {input_count} exceeds max_steps {checked_limits.max_steps}",
            checked_limits.max_steps,
            (),
            state,
        )

    trace: list[TankStep] = []
    max_magnitude = checked_limits.max_magnitude
    for index in range(input_count):
        try:
            try:
                input_at_index = inputs[index]
            except (IndexError, KeyError, TypeError) as error:
                raise _SimulationAbort(
                    TankFailureCategory.INVALID_INPUT,
                    f"inputs[{index}] could not be read",
                    step_index=index,
                ) from error
            input_value = _finite_number(
                input_at_index,
                label=f"inputs[{index}]",
                category=TankFailureCategory.INVALID_INPUT,
            )
            if input_value < 0.0:
                raise _SimulationAbort(
                    TankFailureCategory.INVALID_INPUT,
                    f"inputs[{index}] must be nonnegative",
                    step_index=index,
                )
            _checked_magnitude(
                input_value,
                label=f"inputs[{index}]",
                max_magnitude=max_magnitude,
                step_index=index,
            )

            x1 = state.x1
            x2 = state.x2
            q12 = _checked_magnitude(
                checked_parameters.a * math.sqrt(x1),
                label="q12",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            q2 = _checked_magnitude(
                checked_parameters.c * math.sqrt(x2),
                label="q2",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            powered_input = _checked_magnitude(
                checked_parameters.p * input_value,
                label="p*u",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            x1_plus_input = _checked_magnitude(
                x1 + powered_input,
                label="x1+p*u",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            x1_candidate = _checked_magnitude(
                x1_plus_input - q12,
                label="x1+p*u-q12",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            x1_next = max(0.0, x1_candidate)

            x1_inflow = _checked_magnitude(
                math.sqrt(x1),
                label="sqrt(x1)",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            x2_plus_inflow = _checked_magnitude(
                x2 + x1_inflow,
                label="x2+sqrt(x1)",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            x2_candidate = _checked_magnitude(
                x2_plus_inflow - q2,
                label="x2+sqrt(x1)-q2",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            x2_raw = max(0.0, x2_candidate)

            if checked_model is TankModel.O2:
                assert checked_ceiling is not None
                x2_next = min(checked_ceiling, x2_raw)
                observation = x2
            elif checked_model is TankModel.C2:
                assert checked_ceiling is not None
                x2_next = x2_raw
                observation = min(checked_ceiling, x2)
            else:
                x2_next = x2_raw
                observation = x2

            _checked_magnitude(
                observation,
                label="observation",
                max_magnitude=max_magnitude,
                step_index=index,
            )
            next_state = TankState(x1_next, x2_next)
            trace.append(
                TankStep(
                    index=index,
                    input_u=input_value,
                    state=state,
                    observation_y=observation,
                    q12=q12,
                    q2=q2,
                    x1_next=x1_next,
                    x2_raw=x2_raw,
                    next_state=next_state,
                )
            )
            state = next_state
        except _SimulationAbort as error:
            return TankSimulationFailure(
                error.category,
                error.message,
                index if error.step_index is None else error.step_index,
                tuple(trace),
                state,
            )

    return TankSimulationSuccess(
        model=checked_model,
        parameters=checked_parameters,
        initial_state=initial_state_checked,
        ceiling=checked_ceiling,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        trace=tuple(trace),
        terminal_state=state,
    )

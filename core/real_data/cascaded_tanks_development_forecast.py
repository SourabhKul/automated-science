"""Target-deferred train-to-development forecasts for Cascaded Tanks.

The production entrypoint accepts a pinned archive path, loads the bounded
official development view itself, rechecks its source and stage receipts, and
forecasts every supplied particle. It has no development-target or official
test reader and performs no fitting or model scoring. A private fixture seam
supports synthetic mechanics tests while retaining an explicit fixture marker.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.real_data import cascaded_tanks_controlled as _source
from core.real_data.cascaded_tanks_controlled import (
    CONTRACT_SHA256,
    SAMPLE_INTERVAL_SECONDS,
    SOURCE_ARCHIVE_SHA256,
    SYNTHETIC_FIXTURE_CONTRACT_SHA256,
    SYNTHETIC_FIXTURE_TRACK_ID,
    TRACK_ID,
    CascadedTanksDevelopmentData,
    DevelopmentStageReceipts,
    SyntheticCascadedTanksDevelopmentData,
)
from core.real_data.cascaded_tanks_models import (
    TankFailureCategory,
    TankModel,
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

TRAIN_STOP = 768
FORECAST_STOP = 1_024
FORECAST_SCHEMA = "cascaded-tanks-development-forecast-v1"
_WEIGHT_SUM_ABS_TOL = 1.0e-12
_ALIGNMENT = (
    "y[k] reads state[k] before u[k] transitions to state[k+1]; "
    "training u[767] produces boundary state[768]; development u[768:1024] "
    "produces forecasts y[768:1024] from that carried state"
)


class CascadedTanksForecastError(ValueError):
    """A supplied ensemble, source view, or forecast receipt is invalid."""


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise TypeError(f"{label} must be a finite real number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class CascadedTanksForecastParticle:
    """One immutable family, parameter, initial-state, and weight declaration."""

    model: TankModel
    parameters: TankParameters
    initial_state: TankState
    weight: float
    ceiling: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, TankModel):
            raise TypeError("model must be an explicit TankModel member")
        if type(self.parameters) is not TankParameters:
            raise TypeError("parameters must be TankParameters")
        if type(self.initial_state) is not TankState:
            raise TypeError("initial_state must be TankState")

        parameters = TankParameters(
            _finite_real(self.parameters.a, "parameters.a"),
            _finite_real(self.parameters.c, "parameters.c"),
            _finite_real(self.parameters.p, "parameters.p"),
        )
        for name, value in (
            ("parameters.a", parameters.a),
            ("parameters.c", parameters.c),
            ("parameters.p", parameters.p),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        initial_state = TankState(
            _finite_real(self.initial_state.x1, "initial_state.x1"),
            _finite_real(self.initial_state.x2, "initial_state.x2"),
        )
        if initial_state.x1 < 0.0 or initial_state.x2 < 0.0:
            raise ValueError("initial states must be nonnegative")

        if self.model is TankModel.S0:
            if self.ceiling is not None:
                raise ValueError("S0 particles must omit ceiling")
            ceiling = None
        else:
            if self.ceiling is None:
                raise ValueError(f"{self.model.value} particles require a ceiling")
            ceiling = _finite_real(self.ceiling, "ceiling")
            if ceiling < 0.0:
                raise ValueError("ceiling must be nonnegative")
            if self.model is TankModel.O2 and initial_state.x2 > ceiling:
                raise ValueError("O2 initial_state.x2 must not exceed ceiling")

        weight = _finite_real(self.weight, "weight")
        if weight < 0.0:
            raise ValueError("weight must be nonnegative")

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "ceiling", ceiling)


@dataclass(frozen=True, slots=True)
class CascadedTanksForecastEnsemble:
    """Immutable caller-supplied particles with already normalized weights.

    This container makes no claim about how the particles were produced. In
    particular, it does not certify that they came from ABC or any real-data
    fit. Weights are checked and never altered or renormalized.
    """

    particles: tuple[CascadedTanksForecastParticle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.particles, tuple):
            raise TypeError("particles must be an immutable tuple")
        if not self.particles:
            raise ValueError("ensemble must contain at least one particle")
        if any(
            type(item) is not CascadedTanksForecastParticle for item in self.particles
        ):
            raise TypeError(
                "all particles must be CascadedTanksForecastParticle values"
            )
        weight_sum = math.fsum(item.weight for item in self.particles)
        if not math.isfinite(weight_sum) or not math.isclose(
            weight_sum, 1.0, rel_tol=0.0, abs_tol=_WEIGHT_SUM_ABS_TOL
        ):
            raise ValueError("declared particle weights must sum to 1 within 1e-12")


@dataclass(frozen=True, slots=True)
class ParticleForecastReceipt:
    """Compact alignment and hash evidence for one supplied particle."""

    particle_index: int
    model: str
    weight: float
    parameter_sha256: str
    training_status: str
    development_status: str
    boundary_state_index: int
    boundary_state_sha256: str | None
    forecast_indices: tuple[int, int]
    forecast_sha256: str | None


@dataclass(frozen=True, slots=True)
class ParticleForecastFailure:
    """Terminal simulator or trajectory-validation failure for a particle."""

    particle_index: int
    phase: Literal["training", "development"]
    category: str
    step_index: int | None


@dataclass(frozen=True, slots=True)
class DevelopmentForecastReceipt:
    """Hash-only source, model, code, and time-alignment record."""

    schema: str
    status: Literal["complete", "terminal_failure"]
    source_kind: Literal["official", "synthetic-fixture-only"]
    track_id: str
    contract_sha256: str
    archive_sha256: str | None
    source_visible_sha256: str
    training_stage_sha256: str
    forecast_inputs_stage_sha256: str
    source_receipt_sha256: str
    ensemble_parameter_sha256: str
    code_sha256: str
    sample_interval_seconds: float
    alignment: str
    training_indices: tuple[int, int]
    boundary_state_index: int
    forecast_indices: tuple[int, int]
    model_alignment: tuple[tuple[int, str], ...]
    simulation_limits: tuple[int, float]
    particle_receipts: tuple[ParticleForecastReceipt, ...]
    failures: tuple[ParticleForecastFailure, ...]
    weighted_median_forecast_sha256: str | None
    target_sha256: None


@dataclass(frozen=True, slots=True)
class DevelopmentForecastOutcome:
    """Forecast product plus its receipt; no target values or score are present."""

    forecast: tuple[float, ...] | None
    receipt: DevelopmentForecastReceipt

    @property
    def status(self) -> Literal["complete", "terminal_failure"]:
        return self.receipt.status


def run_cascaded_tanks_development_forecast(
    archive_path: str | Path,
    ensemble: CascadedTanksForecastEnsemble,
    *,
    limits: TankSimulationLimits,
) -> DevelopmentForecastOutcome:
    """Load the official archive and produce a complete 256-step free-run.

    The caller supplies all particle families, parameters, starting states,
    normalized weights, and simulation limits. This function loads the source
    by path itself, verifies the pinned archive and receipts, and never opens
    development targets or the official test columns.
    """

    if not isinstance(ensemble, CascadedTanksForecastEnsemble):
        raise TypeError("ensemble must be CascadedTanksForecastEnsemble")
    _validate_limits(limits)

    data = _source.load_development_data(archive_path)
    if type(data) is not CascadedTanksDevelopmentData:
        raise CascadedTanksForecastError(
            "production forecast requires a view returned by the official loader"
        )
    _validate_view(data, source_kind="official")
    return _forecast_view(data, ensemble, limits, source_kind="official")


def _forecast_synthetic_fixture(
    data: SyntheticCascadedTanksDevelopmentData,
    ensemble: CascadedTanksForecastEnsemble,
    *,
    limits: TankSimulationLimits,
) -> DevelopmentForecastOutcome:
    """Private synthetic-only seam; its receipt can never identify as official."""

    if type(data) is not SyntheticCascadedTanksDevelopmentData:
        raise CascadedTanksForecastError(
            "synthetic seam accepts only the fixture data type"
        )
    if not isinstance(ensemble, CascadedTanksForecastEnsemble):
        raise TypeError("ensemble must be CascadedTanksForecastEnsemble")
    _validate_limits(limits)
    _validate_view(data, source_kind="synthetic-fixture-only")
    return _forecast_view(data, ensemble, limits, source_kind="synthetic-fixture-only")


def _validate_limits(limits: TankSimulationLimits) -> None:
    if type(limits) is not TankSimulationLimits:
        raise TypeError("limits must be an explicit TankSimulationLimits value")
    if (
        isinstance(limits.max_steps, bool)
        or not isinstance(limits.max_steps, int)
        or limits.max_steps < 1
    ):
        raise ValueError("limits.max_steps must be a positive integer")
    max_magnitude = _finite_real(limits.max_magnitude, "limits.max_magnitude")
    if max_magnitude <= 0.0:
        raise ValueError("limits.max_magnitude must be positive")


def _validate_view(
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    *,
    source_kind: Literal["official", "synthetic-fixture-only"],
) -> None:
    if source_kind == "official":
        if type(data) is not CascadedTanksDevelopmentData:
            raise CascadedTanksForecastError("official source view type is required")
        expected_track = TRACK_ID
        expected_contract = CONTRACT_SHA256
        expected_archive: str | None = SOURCE_ARCHIVE_SHA256
    else:
        if type(data) is not SyntheticCascadedTanksDevelopmentData:
            raise CascadedTanksForecastError("synthetic fixture view type is required")
        if data.fixture_marker != "synthetic-fixture-only":
            raise CascadedTanksForecastError("synthetic fixture marker is missing")
        expected_track = SYNTHETIC_FIXTURE_TRACK_ID
        expected_contract = SYNTHETIC_FIXTURE_CONTRACT_SHA256
        expected_archive = None

    if data.track_id != expected_track or data.contract_sha256 != expected_contract:
        raise CascadedTanksForecastError(
            "source view track or contract identity is invalid"
        )
    if data.sample_interval_seconds != SAMPLE_INTERVAL_SECONDS:
        raise CascadedTanksForecastError("source view sample interval is invalid")
    if data.training_indices != tuple(range(TRAIN_STOP)):
        raise CascadedTanksForecastError("training indices must be exactly [0, 768)")
    if data.development_input_indices != tuple(range(TRAIN_STOP, FORECAST_STOP)):
        raise CascadedTanksForecastError(
            "forecast input indices must be exactly [768, 1024)"
        )
    if len(data.training_u_est) != TRAIN_STOP or len(data.training_y_est) != TRAIN_STOP:
        raise CascadedTanksForecastError(
            "source view must contain exactly 768 training pairs"
        )
    if len(data.development_u_est) != FORECAST_STOP - TRAIN_STOP:
        raise CascadedTanksForecastError(
            "source view must contain exactly 256 development inputs"
        )
    if not _is_sha256(data.source_visible_sha256):
        raise CascadedTanksForecastError("source-visible digest is malformed")

    training_inputs = _finite_sequence(
        data.training_u_est, TRAIN_STOP, "training inputs"
    )
    training_outputs = _finite_sequence(
        data.training_y_est, TRAIN_STOP, "training outputs"
    )
    development_inputs = _finite_sequence(
        data.development_u_est, FORECAST_STOP - TRAIN_STOP, "development inputs"
    )
    if any(value < 0.0 for value in (*training_inputs, *development_inputs)):
        raise CascadedTanksForecastError("tank inputs must be nonnegative")

    expected_receipts = _recompute_stage_receipts(
        training_inputs,
        training_outputs,
        data.source_visible_sha256,
        contract_sha256=expected_contract,
        archive_sha256=expected_archive,
    )
    if data.stage_receipts != expected_receipts:
        raise CascadedTanksForecastError(
            "source stage receipts do not match the supplied view"
        )


def _finite_sequence(
    values: Sequence[object], expected_length: int, label: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise CascadedTanksForecastError(f"{label} must be a numeric sequence")
    if len(values) != expected_length:
        raise CascadedTanksForecastError(
            f"{label} length does not match the fixed split"
        )
    try:
        checked = tuple(
            _finite_real(value, f"{label}[{index}]")
            for index, value in enumerate(values)
        )
    except (TypeError, ValueError) as error:
        raise CascadedTanksForecastError(
            f"{label} must contain finite real numbers"
        ) from error
    return checked


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_training_values(inputs: Sequence[float], outputs: Sequence[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"cascaded-tanks-training-values-v1\0")
    for name, values in ((b"uEst", inputs), (b"yEst", outputs)):
        digest.update(name)
        for value in values:
            digest.update(float(value).hex().encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _hash_fields(stage: str, fields: dict[str, object]) -> str:
    payload = {"stage": stage, **fields}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recompute_stage_receipts(
    training_inputs: Sequence[float],
    training_outputs: Sequence[float],
    source_visible_sha256: str,
    *,
    contract_sha256: str,
    archive_sha256: str | None,
) -> DevelopmentStageReceipts:
    bound_fields: dict[str, object] = {
        "contract_sha256": contract_sha256,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
    }
    if archive_sha256 is not None:
        bound_fields["archive_sha256"] = archive_sha256
    return DevelopmentStageReceipts(
        training_sha256=_hash_fields(
            "training-stage",
            {
                **bound_fields,
                "source_visible_sha256": _hash_training_values(
                    training_inputs, training_outputs
                ),
            },
        ),
        forecast_inputs_sha256=_hash_fields(
            "development-forecast-input-stage",
            {
                **bound_fields,
                "source_visible_sha256": source_visible_sha256,
            },
        ),
    )


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _particle_parameters(particle: CascadedTanksForecastParticle) -> dict[str, object]:
    return {
        "model": particle.model.value,
        "parameters": {
            "a": particle.parameters.a,
            "c": particle.parameters.c,
            "p": particle.parameters.p,
        },
        "initial_state": {
            "x1": particle.initial_state.x1,
            "x2": particle.initial_state.x2,
        },
        "ceiling": particle.ceiling,
        "weight": particle.weight,
    }


def _code_sha256() -> str:
    components = (
        ("cascaded_tanks_controlled.py", Path(_source.__file__)),
        (
            "cascaded_tanks_models.py",
            Path(_source.__file__).with_name("cascaded_tanks_models.py"),
        ),
        ("cascaded_tanks_development_forecast.py", Path(__file__)),
    )
    digest = hashlib.sha256()
    digest.update(b"cascaded-tanks-forecast-code-bundle-v1\0")
    for name, path in components:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _source_receipt_sha256(
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    source_kind: Literal["official", "synthetic-fixture-only"],
) -> str:
    return _canonical_json_hash(
        {
            "source_kind": source_kind,
            "track_id": data.track_id,
            "contract_sha256": data.contract_sha256,
            "archive_sha256": getattr(data, "archive_sha256", None),
            "source_visible_sha256": data.source_visible_sha256,
            "training_stage_sha256": data.stage_receipts.training_sha256,
            "forecast_inputs_stage_sha256": data.stage_receipts.forecast_inputs_sha256,
        }
    )


def _sequence_sha256(values: Sequence[float], *, start_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"cascaded-tanks-forecast-values-v1\0")
    digest.update(start_index.to_bytes(8, "big"))
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _state_sha256(state: TankState) -> str:
    return _canonical_json_hash({"x1": state.x1, "x2": state.x2, "index": TRAIN_STOP})


def _complete_success(
    outcome: object,
    *,
    model: TankModel,
    parameters: TankParameters,
    initial_state: TankState,
    ceiling: float | None,
    inputs: tuple[float, ...],
) -> tuple[float, ...] | None:
    if not isinstance(outcome, TankSimulationSuccess):
        return None
    if (
        outcome.model is not model
        or outcome.parameters != parameters
        or outcome.initial_state != initial_state
        or outcome.ceiling != ceiling
        or outcome.sample_interval_seconds != SAMPLE_INTERVAL_SECONDS
        or len(outcome.trace) != len(inputs)
        or len(outcome.observations) != len(inputs)
    ):
        return None
    state = initial_state
    for index, (step, input_value, observation) in enumerate(
        zip(outcome.trace, inputs, outcome.observations, strict=True)
    ):
        if (
            step.index != index
            or step.input_u != input_value
            or step.state != state
            or not math.isfinite(observation)
            or not _valid_state(step.next_state)
        ):
            return None
        expected_observation = (
            min(ceiling, step.state.x2)
            if model is TankModel.C2 and ceiling is not None
            else step.state.x2
        )
        if observation != expected_observation:
            return None
        state = step.next_state
    if not _valid_state(outcome.terminal_state) or outcome.terminal_state != state:
        return None
    return tuple(float(value) for value in outcome.observations)


def _valid_state(state: object) -> bool:
    return (
        isinstance(state, TankState)
        and math.isfinite(state.x1)
        and math.isfinite(state.x2)
        and state.x1 >= 0.0
        and state.x2 >= 0.0
    )


def _failure_fields(outcome: object) -> tuple[str, int | None]:
    if isinstance(outcome, TankSimulationFailure):
        category = (
            outcome.category.value
            if isinstance(outcome.category, TankFailureCategory)
            else str(outcome.category)
        )
        return category, outcome.step_index
    return "invalid_trajectory", None


def _weighted_median(
    trajectories: Sequence[tuple[float, ...]], weights: Sequence[float]
) -> tuple[float, ...]:
    if not trajectories or len(trajectories) != len(weights):
        raise ValueError(
            "weighted median requires matching non-empty trajectories and weights"
        )
    length = len(trajectories[0])
    if any(len(values) != length for values in trajectories):
        raise ValueError("weighted median trajectories must have equal lengths")
    total = math.fsum(weights)
    result: list[float] = []
    for index in range(length):
        ordered = sorted(
            (trajectories[particle][index], weights[particle], particle)
            for particle in range(len(trajectories))
        )
        cumulative = 0.0
        median = ordered[-1][0]
        for position, (value, weight, _particle_index) in enumerate(ordered):
            cumulative = math.fsum(item[1] for item in ordered[: position + 1])
            if cumulative >= total / 2.0:
                median = value
                break
        result.append(float(median))
    return tuple(result)


def _forecast_view(
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    ensemble: CascadedTanksForecastEnsemble,
    limits: TankSimulationLimits,
    *,
    source_kind: Literal["official", "synthetic-fixture-only"],
) -> DevelopmentForecastOutcome:
    _validate_view(data, source_kind=source_kind)
    training_inputs = _finite_sequence(
        data.training_u_est, TRAIN_STOP, "training inputs"
    )
    development_inputs = _finite_sequence(
        data.development_u_est, FORECAST_STOP - TRAIN_STOP, "development inputs"
    )
    parameter_hash = _canonical_json_hash(
        {
            "schema": "cascaded-tanks-supplied-ensemble-v1",
            "particles": [
                {"index": index, **_particle_parameters(particle)}
                for index, particle in enumerate(ensemble.particles)
            ],
        }
    )

    particle_receipts: list[ParticleForecastReceipt] = []
    failures: list[ParticleForecastFailure] = []
    particle_forecasts: list[tuple[float, ...]] = []
    for particle_index, particle in enumerate(ensemble.particles):
        particle_parameter_hash = _canonical_json_hash(_particle_parameters(particle))
        training_outcome = simulate_cascaded_tanks(
            training_inputs,
            particle.parameters,
            particle.initial_state,
            model=particle.model,
            ceiling=particle.ceiling,
            limits=limits,
        )
        training_trajectory = _complete_success(
            training_outcome,
            model=particle.model,
            parameters=particle.parameters,
            initial_state=particle.initial_state,
            ceiling=particle.ceiling,
            inputs=training_inputs,
        )
        if training_trajectory is None:
            category, step_index = _failure_fields(training_outcome)
            failures.append(
                ParticleForecastFailure(
                    particle_index, "training", category, step_index
                )
            )
            particle_receipts.append(
                ParticleForecastReceipt(
                    particle_index=particle_index,
                    model=particle.model.value,
                    weight=particle.weight,
                    parameter_sha256=particle_parameter_hash,
                    training_status="failed",
                    development_status="not_run",
                    boundary_state_index=TRAIN_STOP,
                    boundary_state_sha256=None,
                    forecast_indices=(TRAIN_STOP, FORECAST_STOP),
                    forecast_sha256=None,
                )
            )
            continue

        # The simulator's final transition consumed training u[767], so this
        # terminal state is state[768] and is the first dev output's state.
        training_terminal_state = training_outcome.terminal_state
        development_outcome = simulate_cascaded_tanks(
            development_inputs,
            particle.parameters,
            training_terminal_state,
            model=particle.model,
            ceiling=particle.ceiling,
            limits=limits,
        )
        development_trajectory = _complete_success(
            development_outcome,
            model=particle.model,
            parameters=particle.parameters,
            initial_state=training_terminal_state,
            ceiling=particle.ceiling,
            inputs=development_inputs,
        )
        if development_trajectory is None:
            category, step_index = _failure_fields(development_outcome)
            failures.append(
                ParticleForecastFailure(
                    particle_index, "development", category, step_index
                )
            )
            particle_receipts.append(
                ParticleForecastReceipt(
                    particle_index=particle_index,
                    model=particle.model.value,
                    weight=particle.weight,
                    parameter_sha256=particle_parameter_hash,
                    training_status="complete",
                    development_status="failed",
                    boundary_state_index=TRAIN_STOP,
                    boundary_state_sha256=_state_sha256(training_terminal_state),
                    forecast_indices=(TRAIN_STOP, FORECAST_STOP),
                    forecast_sha256=None,
                )
            )
            continue

        particle_forecasts.append(development_trajectory)
        particle_receipts.append(
            ParticleForecastReceipt(
                particle_index=particle_index,
                model=particle.model.value,
                weight=particle.weight,
                parameter_sha256=particle_parameter_hash,
                training_status="complete",
                development_status="complete",
                boundary_state_index=TRAIN_STOP,
                boundary_state_sha256=_state_sha256(training_terminal_state),
                forecast_indices=(TRAIN_STOP, FORECAST_STOP),
                forecast_sha256=_sequence_sha256(
                    development_trajectory, start_index=TRAIN_STOP
                ),
            )
        )

    all_complete = not failures and len(particle_forecasts) == len(ensemble.particles)
    forecast = (
        _weighted_median(
            particle_forecasts,
            tuple(particle.weight for particle in ensemble.particles),
        )
        if all_complete
        else None
    )
    forecast_hash = (
        _sequence_sha256(forecast, start_index=TRAIN_STOP)
        if forecast is not None
        else None
    )
    source_receipt_hash = _source_receipt_sha256(data, source_kind)
    receipt = DevelopmentForecastReceipt(
        schema=FORECAST_SCHEMA,
        status="complete" if all_complete else "terminal_failure",
        source_kind=source_kind,
        track_id=data.track_id,
        contract_sha256=data.contract_sha256,
        archive_sha256=getattr(data, "archive_sha256", None),
        source_visible_sha256=data.source_visible_sha256,
        training_stage_sha256=data.stage_receipts.training_sha256,
        forecast_inputs_stage_sha256=data.stage_receipts.forecast_inputs_sha256,
        source_receipt_sha256=source_receipt_hash,
        ensemble_parameter_sha256=parameter_hash,
        code_sha256=_code_sha256(),
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        alignment=_ALIGNMENT,
        training_indices=(0, TRAIN_STOP),
        boundary_state_index=TRAIN_STOP,
        forecast_indices=(TRAIN_STOP, FORECAST_STOP),
        model_alignment=tuple(
            (index, particle.model.value)
            for index, particle in enumerate(ensemble.particles)
        ),
        simulation_limits=(limits.max_steps, float(limits.max_magnitude)),
        particle_receipts=tuple(particle_receipts),
        failures=tuple(failures),
        weighted_median_forecast_sha256=forecast_hash,
        target_sha256=None,
    )
    return DevelopmentForecastOutcome(forecast=forecast, receipt=receipt)


__all__ = [
    "CascadedTanksForecastEnsemble",
    "CascadedTanksForecastError",
    "CascadedTanksForecastParticle",
    "DevelopmentForecastOutcome",
    "DevelopmentForecastReceipt",
    "ParticleForecastFailure",
    "ParticleForecastReceipt",
    "run_cascaded_tanks_development_forecast",
]

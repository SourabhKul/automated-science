"""Intended synthetic-only ABC-SMC control for cascaded-tanks models.

The caller supplies input and output arrays, whose origin this module cannot
verify. Synthetic use is an experimental-scope convention, not a security
boundary. This module does not load source archives or select data splits. Free
parameters are proposed in a normalized unit cube and decoded affinely into
their declared effective surrogate-coordinate ranges.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np

from core.abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference
from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

_BASE_PARAMETERS = ("a", "c", "p", "x1_0", "x2_0")
_CEILING_MODELS = (TankModel.O2, TankModel.C2)


class _TankABCSimulationError(RuntimeError):
    """Internal exception marking a counted simulator failure."""


@dataclass(frozen=True)
class CascadedTanksSyntheticABCConfig:
    """Frozen controls for one caller-array synthetic ABC-SMC run.

    ``free_parameter_bounds`` declares effective surrogate-coordinate bounds,
    not calibrated physical constants. The sampler is uniform in the
    corresponding normalized unit-cube coordinates, so its proposal prior
    density is exactly one throughout that cube.
    """

    model: TankModel
    fixed_parameters: Mapping[str, float]
    free_parameter_bounds: Mapping[str, tuple[float, float]]
    target_samples: int
    epsilon_schedule: Sequence[float]
    max_attempts_per_population: int | Sequence[int]
    seed: int
    output_units: str = "synthetic output units"
    limits: TankSimulationLimits = field(default_factory=TankSimulationLimits)


def _finite_array(values: Sequence[float] | np.ndarray, *, label: str) -> np.ndarray:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a numeric one-dimensional array")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric one-dimensional array") from exc
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a non-empty numeric one-dimensional array")
    result = raw.astype(float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a positive integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _checked_config(
    config: CascadedTanksSyntheticABCConfig,
) -> tuple[
    tuple[str, ...],
    dict[str, float],
    dict[str, tuple[float, float]],
    list[float],
    list[int],
]:
    if not isinstance(config.model, TankModel):
        raise TypeError("model must be an explicit TankModel member")
    if not isinstance(config.limits, TankSimulationLimits):
        raise TypeError("limits must be TankSimulationLimits")
    if (
        isinstance(config.seed, (bool, np.bool_))
        or not isinstance(config.seed, (int, np.integer))
        or config.seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(config.output_units, str) or not config.output_units.strip():
        raise ValueError("output_units must be a non-empty description")

    required = set(_BASE_PARAMETERS)
    if config.model in _CEILING_MODELS:
        required.add("ceiling")
    fixed: dict[str, float] = {}
    free: dict[str, tuple[float, float]] = {}
    try:
        for name, raw in config.fixed_parameters.items():
            if isinstance(raw, (bool, np.bool_, str, bytes, bytearray)):
                raise TypeError(f"fixed parameter {name!r} must be numeric")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"fixed parameter {name!r} must be finite")
            fixed[name] = value
        for name, raw_bounds in config.free_parameter_bounds.items():
            raw_pair = np.asarray(raw_bounds)
            if raw_pair.dtype.kind not in "fiu":
                raise TypeError(f"free parameter {name!r} bounds must be numeric")
            pair = raw_pair.astype(float)
            if (
                pair.shape != (2,)
                or not np.all(np.isfinite(pair))
                or pair[1] <= pair[0]
            ):
                raise ValueError(
                    f"free parameter {name!r} needs finite low < high bounds"
                )
            free[name] = (float(pair[0]), float(pair[1]))
    except AttributeError as exc:
        raise ValueError("fixed and free parameters must be mappings") from exc

    if set(fixed) & set(free):
        raise ValueError("a parameter cannot be both fixed and free")
    if set(fixed) | set(free) != required:
        raise ValueError(
            f"parameter declarations must cover exactly {sorted(required)}"
        )
    if not free:
        raise ValueError("the reference runner requires at least one free parameter")

    for name, value in fixed.items():
        if name in {"a", "c", "p"} and value <= 0.0:
            raise ValueError(f"fixed parameter {name!r} must be positive")
        if name in {"x1_0", "x2_0", "ceiling"} and value < 0.0:
            raise ValueError(f"fixed parameter {name!r} must be non-negative")
    for name, (low, high) in free.items():
        if name in {"a", "c", "p"} and low <= 0.0:
            raise ValueError(
                f"free parameter {name!r} must have a positive lower bound"
            )
        if name in {"x1_0", "x2_0", "ceiling"} and low < 0.0:
            raise ValueError(
                f"free parameter {name!r} must have a non-negative lower bound"
            )

    if config.model is TankModel.O2:
        highest_x2 = fixed.get("x2_0", free.get("x2_0", (0.0, 0.0))[1])
        lowest_ceiling = fixed.get("ceiling", free.get("ceiling", (0.0, 0.0))[0])
        if highest_x2 > lowest_ceiling:
            raise ValueError("all O2 prior points must satisfy x2_0 <= ceiling")

    if isinstance(config.epsilon_schedule, (str, bytes, bytearray)):
        raise TypeError("epsilon_schedule must be a non-empty sequence")
    try:
        epsilons = [float(value) for value in config.epsilon_schedule]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "epsilon_schedule must contain finite non-negative values"
        ) from exc
    if (
        not epsilons
        or not np.all(np.isfinite(epsilons))
        or np.any(np.asarray(epsilons) < 0)
    ):
        raise ValueError("epsilon_schedule must contain finite non-negative values")
    if any(epsilons[index] > epsilons[index - 1] for index in range(1, len(epsilons))):
        raise ValueError("epsilon_schedule must be non-increasing")

    raw_budget = config.max_attempts_per_population
    if np.isscalar(raw_budget):
        budgets = [_positive_integer(raw_budget, label="attempt budget")] * len(
            epsilons
        )
    else:
        try:
            values = list(raw_budget)
        except TypeError as exc:
            raise ValueError("attempt budgets must be an integer or sequence") from exc
        if len(values) != len(epsilons):
            raise ValueError("one attempt budget is required for each epsilon")
        budgets = [_positive_integer(value, label="attempt budget") for value in values]
    target = _positive_integer(config.target_samples, label="target_samples")
    if target != config.target_samples:
        raise ValueError("target_samples must be an integer")
    return tuple(free), fixed, free, epsilons, budgets


def _decode(
    unit_point: np.ndarray,
    order: tuple[str, ...],
    fixed: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    values = dict(fixed)
    for index, name in enumerate(order):
        low, high = bounds[name]
        values[name] = low + float(unit_point[index]) * (high - low)
    return values


def _canonical_hash(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8")
    return sha256(canonical.tobytes(order="C")).hexdigest()


def run_cascaded_tanks_synthetic_abc_smc(
    inputs: Sequence[float] | np.ndarray,
    observed_outputs: Sequence[float] | np.ndarray,
    config: CascadedTanksSyntheticABCConfig,
) -> dict[str, Any]:
    """Fit one declared family to caller-supplied arrays.

    The discrepancy is finite all-trajectory RMSE in ``output_units``. A
    returned ``posterior`` exists only when every reference population fills;
    partial reference arrays remain available as diagnostic evidence but are
    never promoted as terminal posterior or forecast products. Synthetic-only
    use is intended but not enforced: the origin of caller arrays is
    unverified, and this function is not a data-access security boundary.
    """

    input_values = _finite_array(inputs, label="inputs")
    target = _finite_array(observed_outputs, label="observed_outputs")
    if len(input_values) != len(target):
        raise ValueError("inputs and observed_outputs must have equal lengths")
    if np.any(input_values < 0.0):
        raise ValueError("inputs must be non-negative for the declared tank simulator")
    order, fixed, free_bounds, epsilons, budgets = _checked_config(config)
    _positive_integer(config.target_samples, label="target_samples")
    unit_bounds = np.tile(np.array([[0.0, 1.0]]), (len(order), 1))
    prior_sampler, prior_logpdf = make_uniform_prior(unit_bounds)
    simulation_failures: Counter[str] = Counter()
    discrepancy_failures: Counter[str] = Counter()

    def in_support(point: np.ndarray) -> bool:
        return (
            point.shape == (len(order),)
            and bool(np.all(np.isfinite(point)))
            and bool(np.all(point >= 0.0) and np.all(point <= 1.0))
        )

    def simulator(point: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
        categorized_failure = False
        try:
            parameters = _decode(point, order, fixed, free_bounds)
            outcome = simulate_cascaded_tanks(
                input_values,
                TankParameters(parameters["a"], parameters["c"], parameters["p"]),
                TankState(parameters["x1_0"], parameters["x2_0"]),
                model=config.model,
                ceiling=parameters.get("ceiling"),
                limits=config.limits,
            )
            if isinstance(outcome, TankSimulationFailure):
                simulation_failures[outcome.category.value] += 1
                categorized_failure = True
                raise _TankABCSimulationError(
                    f"tank simulation failed: {outcome.category.value}"
                )
            if not isinstance(outcome, TankSimulationSuccess):
                simulation_failures["invalid_simulator_result"] += 1
                categorized_failure = True
                raise _TankABCSimulationError(
                    "tank simulator returned an unknown result type"
                )
            generated = np.asarray(outcome.observations, dtype=float)
            if generated.shape != target.shape or not np.all(np.isfinite(generated)):
                simulation_failures["invalid_trajectory"] += 1
                categorized_failure = True
                raise _TankABCSimulationError(
                    "tank simulator did not return one finite full trajectory"
                )
            return generated
        except Exception:
            # Categorized simulator outcomes above retain their specific count.
            # Catch other unexpected wrapper/simulator errors for auditability.
            if not categorized_failure:
                simulation_failures["unexpected_exception"] += 1
            raise

    def discrepancy(generated: Any) -> float:
        prediction = np.asarray(generated, dtype=float)
        if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
            discrepancy_failures["invalid_trajectory"] += 1
            raise ValueError("discrepancy requires a finite full-length trajectory")
        distance = float(np.sqrt(np.mean(np.square(prediction - target))))
        if not math.isfinite(distance):
            discrepancy_failures["non_finite_rmse"] += 1
            raise ValueError("all-trajectory RMSE became non-finite")
        return distance

    reference = run_gaussian_abc_smc_reference(
        prior_sampler,
        simulator,
        discrepancy,
        target_samples=config.target_samples,
        epsilon_schedule=epsilons,
        max_attempts_per_population=budgets,
        bounds=unit_bounds,
        in_support=in_support,
        prior_logpdf=prior_logpdf,
        seed=int(config.seed),
    )
    complete = bool(reference["complete"])
    posterior: dict[str, Any] | None = None
    if complete:
        accepted = np.asarray(reference["accepted_params"], dtype=float)
        posterior = {
            "unit_parameters": accepted.copy(),
            "free_parameter_values": {
                name: np.asarray(
                    [low + point[index] * (high - low) for point in accepted],
                    dtype=float,
                )
                for index, name in enumerate(order)
                for low, high in (free_bounds[name],)
            },
            "parameter_order": order,
            "weights": np.asarray(reference["weights"], dtype=float).copy(),
            "effective_sample_size": float(reference["effective_sample_size"]),
            "distances": np.asarray(reference["distances"], dtype=float).copy(),
        }

    return {
        "status": reference["status"],
        "complete": complete,
        "termination_reason": reference["termination_reason"],
        "posterior": posterior,
        "reference_evidence": reference,
        "failure_categories": dict(sorted(simulation_failures.items())),
        "discrepancy_failure_categories": dict(sorted(discrepancy_failures.items())),
        "provenance": {
            "wrapper": "cascaded_tanks_synthetic_abc_smc_v1",
            "data_scope": "caller_supplied_unverified_arrays",
            "intended_use": "synthetic_only_method_control",
            "synthetic_origin_verified": False,
            "evidence_scope": "intended_synthetic_glue_control_only",
            "input_sha256": _canonical_hash(input_values),
            "observed_output_sha256": _canonical_hash(target),
            "trajectory_length": len(target),
            "model_family": config.model.value,
            "fixed_parameters": dict(fixed),
            "free_parameter_order": order,
            "free_parameter_surrogate_bounds": dict(free_bounds),
            "prior": "normalized uniform on [0, 1]^d; log density 0 inside support",
            "decoder": "surrogate_value = low + unit_coordinate * (high - low)",
            "discrepancy": "all_trajectory_rmse",
            "discrepancy_units": config.output_units,
            "epsilon_schedule": epsilons,
            "max_attempts_per_population": budgets,
            "target_samples": int(config.target_samples),
            "seed": int(config.seed),
            "simulation_limits": {
                "max_steps": config.limits.max_steps,
                "max_magnitude": config.limits.max_magnitude,
            },
            "reference_path": reference["reference_path"],
            "has_fixed_parameters": bool(fixed),
        },
    }


__all__ = [
    "CascadedTanksSyntheticABCConfig",
    "run_cascaded_tanks_synthetic_abc_smc",
]

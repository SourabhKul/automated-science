"""Reusable finite-difference sensitivity calculation for cascaded-tanks models.

The default roster reproduces the frozen synthetic N/S/L protocol. Callers may
pass a separate small :class:`SensitivityCase` roster for focused controls.
This module only computes and returns arrays and summaries: it does not access
files, source data, networks, inference code, or external model endpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

FloatArray = NDArray[np.float64]
SENSITIVITY_STEPS: tuple[float, ...] = (1.0e-3, 1.0e-4, 1.0e-5)
MODEL_ORDER: tuple[TankModel, ...] = (TankModel.S0, TankModel.O2, TankModel.C2)
BASE_PARAMETER_ORDER: tuple[str, ...] = ("a", "c", "p", "x1_0", "x2_0")
BASE_PARAMETER_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.375, 0.625),
    (0.30, 0.50),
    (0.375, 0.625),
    (0.0, 1.0),
    (0.0, 1.0),
)
OUTPUT_SCALE = np.float64(0.05)
MAX_MAGNITUDE = 1.0e6


@dataclass(frozen=True, slots=True)
class SensitivityCase:
    """One deterministic input trace and its intended capped-family ceiling."""

    case_id: str
    inputs: tuple[float, ...]
    ceiling: float
    ceiling_bounds: tuple[float, float]

    @property
    def input_array(self) -> FloatArray:
        """Return the exact case forcing as a fresh float64 array."""

        return np.asarray(self.inputs, dtype=np.float64)


PROTOCOL_CASES: tuple[SensitivityCase, ...] = (
    SensitivityCase(
        "N",
        (8.0,) * 24 + (0.0,) * 180,
        1000.0,
        (900.0, 1100.0),
    ),
    SensitivityCase(
        "S",
        (8.0,) * 24 + (0.0,) * 54,
        3.0,
        (2.5, 3.5),
    ),
    SensitivityCase(
        "L",
        (8.0,) * 24 + (0.0,) * 180,
        3.0,
        (2.5, 3.5),
    ),
)


@dataclass(frozen=True, slots=True)
class SimulationRecord:
    """Full output or completed prefix plus terminal/failure metadata."""

    observations: FloatArray | None
    completed_prefix: FloatArray
    terminal_state: tuple[float, float] | None
    failure_category: str | None
    failure_message: str | None
    output_sha256: str | None

    @property
    def succeeded(self) -> bool:
        return self.observations is not None and self.failure_category is None


@dataclass(frozen=True, slots=True)
class CoordinateAttempts:
    """The two simulations used for one central-difference coordinate."""

    parameter_name: str
    plus: SimulationRecord
    minus: SimulationRecord


@dataclass(frozen=True, slots=True)
class FiniteDifferenceStep:
    """Raw Jacobian and descriptive scaled-matrix diagnostics at one step."""

    h: float
    jacobian: FloatArray
    column_validity: tuple[bool, ...]
    coordinate_attempts: tuple[CoordinateAttempts, ...]
    scaled_matrix: FloatArray | None
    singular_values: FloatArray | None
    relative_rank_1e_3: int | None
    relative_rank_1e_6: int | None
    column_norms: tuple[float | None, ...]
    absolute_column_cosines: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True, slots=True)
class StepComparison:
    """Adjacent-step relative max-difference for one Jacobian column."""

    parameter_name: str
    larger_h: float
    smaller_h: float
    metric: float | None


@dataclass(frozen=True, slots=True)
class FamilyCaseSensitivity:
    """One family/case result, retaining every successful trajectory array."""

    model: TankModel
    case_id: str
    parameter_names: tuple[str, ...]
    parameter_bounds: tuple[tuple[float, float], ...]
    truth_coordinates: tuple[float, ...]
    truth: SimulationRecord
    steps: tuple[FiniteDifferenceStep, ...]
    comparisons: tuple[StepComparison, ...]

    @property
    def available(self) -> bool:
        return self.truth.succeeded


@dataclass(frozen=True, slots=True)
class PairwiseDivergence:
    """Exact truth-output comparison between two families for one case."""

    left_model: TankModel
    right_model: TankModel
    signed_difference: FloatArray | None
    byte_identical: bool | None
    first_differing_index: int | None
    differing_count: int | None
    maximum_absolute_difference: float | None


@dataclass(frozen=True, slots=True)
class CaseDivergenceSummary:
    """Pairwise truth-output divergence and exact capped-output counts."""

    case_id: str
    ceiling: float
    pairwise: tuple[PairwiseDivergence, ...]
    capped_output_counts: tuple[tuple[TankModel, int | None], ...]

    def comparison(
        self, left_model: TankModel, right_model: TankModel
    ) -> PairwiseDivergence:
        for comparison in self.pairwise:
            if (comparison.left_model, comparison.right_model) == (
                left_model,
                right_model,
            ):
                return comparison
        raise KeyError((left_model, right_model))

    @property
    def o2_c2(self) -> PairwiseDivergence:
        return self.comparison(TankModel.O2, TankModel.C2)


@dataclass(frozen=True, slots=True)
class CascadedTanksSensitivityResult:
    """Full arrays and compact, JSON-friendly diagnostics for one roster."""

    case_ids: tuple[str, ...]
    case_roster: tuple[SensitivityCase, ...]
    family_cases: tuple[FamilyCaseSensitivity, ...]
    divergence: tuple[CaseDivergenceSummary, ...]

    def family_case(self, model: TankModel, case_id: str) -> FamilyCaseSensitivity:
        for result in self.family_cases:
            if (result.model, result.case_id) == (model, case_id):
                return result
        raise KeyError((model, case_id))

    def case_divergence(self, case_id: str) -> CaseDivergenceSummary:
        for result in self.divergence:
            if result.case_id == case_id:
                return result
        raise KeyError(case_id)

    def compact_summary(self) -> dict[str, object]:
        """Return scalar/list diagnostics while full arrays remain on this result."""

        family_summaries: list[dict[str, object]] = []
        for result in self.family_cases:
            step_summaries: list[dict[str, object]] = []
            for step in result.steps:
                step_summaries.append(
                    {
                        "h": step.h,
                        "column_validity": dict(
                            zip(
                                result.parameter_names,
                                step.column_validity,
                                strict=True,
                            )
                        ),
                        "failure_categories": {
                            attempts.parameter_name: {
                                "plus": attempts.plus.failure_category,
                                "minus": attempts.minus.failure_category,
                            }
                            for attempts in step.coordinate_attempts
                        },
                        "scaled_matrix_available": step.scaled_matrix is not None,
                        "singular_values": _list_or_none(step.singular_values),
                        "relative_rank_1e_3": step.relative_rank_1e_3,
                        "relative_rank_1e_6": step.relative_rank_1e_6,
                        "column_norms": list(step.column_norms),
                        "absolute_column_cosines": [
                            list(row) for row in step.absolute_column_cosines
                        ],
                    }
                )
            family_summaries.append(
                {
                    "model": result.model.value,
                    "case_id": result.case_id,
                    "parameter_names": list(result.parameter_names),
                    "parameter_bounds": [
                        list(pair) for pair in result.parameter_bounds
                    ],
                    "available": result.available,
                    "truth_failure_category": result.truth.failure_category,
                    "truth_output_sha256": result.truth.output_sha256,
                    "terminal_state": result.truth.terminal_state,
                    "steps": step_summaries,
                    "step_comparisons": [
                        {
                            "parameter": comparison.parameter_name,
                            "larger_h": comparison.larger_h,
                            "smaller_h": comparison.smaller_h,
                            "metric": comparison.metric,
                        }
                        for comparison in result.comparisons
                    ],
                }
            )

        divergence_summaries: list[dict[str, object]] = []
        for case in self.divergence:
            divergence_summaries.append(
                {
                    "case_id": case.case_id,
                    "capped_output_counts": {
                        model.value: count for model, count in case.capped_output_counts
                    },
                    "pairwise": [
                        {
                            "left_model": item.left_model.value,
                            "right_model": item.right_model.value,
                            "byte_identical": item.byte_identical,
                            "first_differing_index": item.first_differing_index,
                            "differing_count": item.differing_count,
                            "maximum_absolute_difference": item.maximum_absolute_difference,
                        }
                        for item in case.pairwise
                    ],
                }
            )
        return {
            "case_ids": list(self.case_ids),
            "cases": [
                {
                    "case_id": case.case_id,
                    "input_length": len(case.inputs),
                    "ceiling": case.ceiling,
                    "ceiling_bounds": list(case.ceiling_bounds),
                }
                for case in self.case_roster
            ],
            "model_order": [model.value for model in MODEL_ORDER],
            "step_sizes": list(SENSITIVITY_STEPS),
            "base_parameter_order": list(BASE_PARAMETER_ORDER),
            "base_parameter_bounds": [list(pair) for pair in BASE_PARAMETER_BOUNDS],
            "output_scale": float(OUTPUT_SCALE),
            "max_magnitude": MAX_MAGNITUDE,
            "observation_convention": "pre_transition",
            "family_cases": family_summaries,
            "divergence": divergence_summaries,
        }


TankSimulator = Callable[..., TankSimulationSuccess | TankSimulationFailure]


def run_cascaded_tanks_synthetic_sensitivity(
    cases: Sequence[SensitivityCase] = PROTOCOL_CASES,
    *,
    simulator: TankSimulator = simulate_cascaded_tanks,
) -> CascadedTanksSensitivityResult:
    """Compute truth traces and normalized central-difference diagnostics.

    The default roster is the frozen N/S/L protocol. Supplying a different
    roster is intended for small unit controls; the exact parameter order,
    normalized truth coordinate, step sizes, scaling, simulator limits, and
    family order remain fixed.
    """

    validated_cases = _validate_cases(cases)
    if not callable(simulator):
        raise TypeError("simulator must be callable")

    family_cases: list[FamilyCaseSensitivity] = []
    truths: dict[tuple[TankModel, str], SimulationRecord] = {}
    for case, inputs in validated_cases:
        for model in MODEL_ORDER:
            parameter_names, bounds = _parameter_spec(model, case)
            coordinates = tuple(0.5 for _ in parameter_names)
            decoded = _decode_coordinates(coordinates, parameter_names, bounds)
            truth = _run_one(
                simulator, inputs, model, decoded, expected_length=len(inputs)
            )
            truths[(model, case.case_id)] = truth
            if not truth.succeeded:
                family_cases.append(
                    FamilyCaseSensitivity(
                        model=model,
                        case_id=case.case_id,
                        parameter_names=parameter_names,
                        parameter_bounds=bounds,
                        truth_coordinates=coordinates,
                        truth=truth,
                        steps=(),
                        comparisons=tuple(
                            StepComparison(name, larger, smaller, None)
                            for name in parameter_names
                            for larger, smaller in pairwise(SENSITIVITY_STEPS)
                        ),
                    )
                )
                continue

            assert truth.observations is not None
            step_results: list[FiniteDifferenceStep] = []
            for h in SENSITIVITY_STEPS:
                step_results.append(
                    _compute_step(
                        simulator,
                        inputs,
                        model,
                        parameter_names,
                        bounds,
                        truth_coordinates=coordinates,
                        h=h,
                        output_length=len(truth.observations),
                    )
                )
            comparisons = _compare_steps(parameter_names, step_results)
            family_cases.append(
                FamilyCaseSensitivity(
                    model=model,
                    case_id=case.case_id,
                    parameter_names=parameter_names,
                    parameter_bounds=bounds,
                    truth_coordinates=coordinates,
                    truth=truth,
                    steps=tuple(step_results),
                    comparisons=comparisons,
                )
            )

    divergence = tuple(
        _summarize_case_divergence(
            case,
            truths[(TankModel.S0, case.case_id)],
            truths[(TankModel.O2, case.case_id)],
            truths[(TankModel.C2, case.case_id)],
        )
        for case, _ in validated_cases
    )
    return CascadedTanksSensitivityResult(
        case_ids=tuple(case.case_id for case, _ in validated_cases),
        case_roster=tuple(case for case, _ in validated_cases),
        family_cases=tuple(family_cases),
        divergence=divergence,
    )


def _validate_cases(
    cases: Sequence[SensitivityCase],
) -> tuple[tuple[SensitivityCase, FloatArray], ...]:
    if isinstance(cases, (str, bytes, bytearray)):
        raise TypeError("cases must be a non-empty sequence of SensitivityCase values")
    try:
        roster = tuple(cases)
    except TypeError as exc:
        raise TypeError("cases must be a non-empty sequence") from exc
    if not roster:
        raise ValueError("cases must be non-empty")
    seen: set[str] = set()
    validated: list[tuple[SensitivityCase, FloatArray]] = []
    for case in roster:
        if not isinstance(case, SensitivityCase):
            raise TypeError("every case must be a SensitivityCase")
        if not isinstance(case.case_id, str) or not case.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        try:
            inputs = np.asarray(case.inputs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"case {case.case_id}: inputs must be numeric") from exc
        if inputs.ndim != 1 or inputs.size == 0 or inputs.dtype.kind not in "fiu":
            raise ValueError(
                f"case {case.case_id}: inputs must be a non-empty numeric 1-D sequence"
            )
        inputs = np.asarray(inputs, dtype=np.float64).copy()
        if not np.all(np.isfinite(inputs)) or np.any(inputs < 0.0):
            raise ValueError(
                f"case {case.case_id}: inputs must be finite and nonnegative"
            )
        if len(case.ceiling_bounds) != 2:
            raise ValueError(
                f"case {case.case_id}: ceiling_bounds must have two values"
            )
        low, high = (np.float64(value) for value in case.ceiling_bounds)
        ceiling = np.float64(case.ceiling)
        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or not np.isfinite(ceiling)
            or high <= low
            or low < 0.0
            or low + np.float64(0.5) * (high - low) != ceiling
        ):
            raise ValueError(
                f"case {case.case_id}: ceiling must equal the nonnegative bounds midpoint"
            )
        validated.append((case, inputs))
    return tuple(validated)


def _parameter_spec(
    model: TankModel, case: SensitivityCase
) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]]:
    if model is TankModel.S0:
        return BASE_PARAMETER_ORDER, BASE_PARAMETER_BOUNDS
    return (
        BASE_PARAMETER_ORDER + ("H",),
        BASE_PARAMETER_BOUNDS + (case.ceiling_bounds,),
    )


def _decode_coordinates(
    coordinates: Sequence[float],
    parameter_names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
) -> dict[str, np.float64]:
    if len(coordinates) != len(parameter_names) or len(bounds) != len(parameter_names):
        raise ValueError("coordinate, name, and bounds lengths must agree")
    result: dict[str, np.float64] = {}
    for name, coordinate, (low, high) in zip(
        parameter_names, coordinates, bounds, strict=True
    ):
        z = np.float64(coordinate)
        if not np.isfinite(z) or z < 0.0 or z > 1.0:
            raise ValueError("normalized coordinates must be finite and in [0, 1]")
        result[name] = np.float64(low) + z * (np.float64(high) - np.float64(low))
    return result


def _run_one(
    simulator: TankSimulator,
    inputs: FloatArray,
    model: TankModel,
    decoded: dict[str, np.float64],
    *,
    expected_length: int,
) -> SimulationRecord:
    parameters = TankParameters(
        float(decoded["a"]), float(decoded["c"]), float(decoded["p"])
    )
    initial_state = TankState(float(decoded["x1_0"]), float(decoded["x2_0"]))
    ceiling = None if model is TankModel.S0 else float(decoded["H"])
    limits = TankSimulationLimits(max_steps=len(inputs), max_magnitude=MAX_MAGNITUDE)
    try:
        outcome = simulator(
            inputs,
            parameters,
            initial_state,
            model=model,
            ceiling=ceiling,
            limits=limits,
        )
    except Exception as exc:  # noqa: BLE001 - injected simulator failures are data.
        return _failed_record("unexpected_exception", f"{type(exc).__name__}: {exc}")

    if isinstance(outcome, TankSimulationFailure):
        prefix = np.asarray(
            [step.observation_y for step in outcome.trace], dtype=np.float64
        )
        terminal = outcome.terminal_state
        return SimulationRecord(
            observations=None,
            completed_prefix=prefix,
            terminal_state=(float(terminal.x1), float(terminal.x2))
            if terminal is not None
            else None,
            failure_category=outcome.category.value,
            failure_message=outcome.message,
            output_sha256=None,
        )
    if not isinstance(outcome, TankSimulationSuccess):
        return _failed_record(
            "invalid_simulator_result", "simulator returned an unknown result type"
        )

    observations = np.asarray(outcome.observations, dtype=np.float64)
    if observations.shape != (expected_length,) or not np.all(
        np.isfinite(observations)
    ):
        return _failed_record(
            "invalid_trajectory",
            "simulator did not return one finite full-length trajectory",
        )
    terminal = outcome.terminal_state
    digest = sha256(
        np.asarray(observations, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    return SimulationRecord(
        observations=observations.copy(),
        completed_prefix=observations.copy(),
        terminal_state=(float(terminal.x1), float(terminal.x2)),
        failure_category=None,
        failure_message=None,
        output_sha256=digest,
    )


def _failed_record(category: str, message: str) -> SimulationRecord:
    return SimulationRecord(
        observations=None,
        completed_prefix=np.empty((0,), dtype=np.float64),
        terminal_state=None,
        failure_category=category,
        failure_message=message,
        output_sha256=None,
    )


def _compute_step(
    simulator: TankSimulator,
    inputs: FloatArray,
    model: TankModel,
    parameter_names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
    *,
    truth_coordinates: tuple[float, ...],
    h: float,
    output_length: int,
) -> FiniteDifferenceStep:
    jacobian = np.full((output_length, len(parameter_names)), np.nan, dtype=np.float64)
    validity: list[bool] = []
    attempts_by_coordinate: list[CoordinateAttempts] = []
    for column, name in enumerate(parameter_names):
        plus_coordinates = list(truth_coordinates)
        minus_coordinates = list(truth_coordinates)
        plus_coordinates[column] = np.float64(0.5) + np.float64(h)
        minus_coordinates[column] = np.float64(0.5) - np.float64(h)
        plus_parameters = _decode_coordinates(plus_coordinates, parameter_names, bounds)
        minus_parameters = _decode_coordinates(
            minus_coordinates, parameter_names, bounds
        )
        plus = _run_one(
            simulator, inputs, model, plus_parameters, expected_length=output_length
        )
        minus = _run_one(
            simulator, inputs, model, minus_parameters, expected_length=output_length
        )
        valid = plus.succeeded and minus.succeeded
        validity.append(valid)
        attempts_by_coordinate.append(CoordinateAttempts(name, plus, minus))
        if valid:
            assert plus.observations is not None and minus.observations is not None
            jacobian[:, column] = (plus.observations - minus.observations) / (
                np.float64(2.0) * np.float64(h)
            )

    complete = all(validity)
    divisor = OUTPUT_SCALE * np.sqrt(np.float64(output_length))
    scaled: FloatArray | None = jacobian / divisor if complete else None
    singular_values: FloatArray | None = None
    rank_1e_3: int | None = None
    rank_1e_6: int | None = None
    if scaled is not None:
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        if singular_values.size and singular_values[0] > 0.0:
            ratios = singular_values / singular_values[0]
            rank_1e_3 = int(np.count_nonzero(ratios > 1.0e-3))
            rank_1e_6 = int(np.count_nonzero(ratios > 1.0e-6))
        else:
            rank_1e_3 = 0
            rank_1e_6 = 0

    norms: list[float | None] = []
    for column, valid in enumerate(validity):
        norms.append(
            float(np.linalg.norm(jacobian[:, column] / divisor)) if valid else None
        )
    cosines: list[tuple[float | None, ...]] = []
    for i in range(len(parameter_names)):
        row: list[float | None] = []
        for j in range(len(parameter_names)):
            if not validity[i] or not validity[j]:
                row.append(None)
                continue
            norm_i = float(np.linalg.norm(jacobian[:, i]))
            norm_j = float(np.linalg.norm(jacobian[:, j]))
            if norm_i == 0.0 or norm_j == 0.0:
                row.append(None)
            else:
                row.append(
                    float(
                        abs(np.dot(jacobian[:, i], jacobian[:, j])) / (norm_i * norm_j)
                    )
                )
        cosines.append(tuple(row))

    return FiniteDifferenceStep(
        h=float(h),
        jacobian=jacobian,
        column_validity=tuple(validity),
        coordinate_attempts=tuple(attempts_by_coordinate),
        scaled_matrix=scaled,
        singular_values=singular_values,
        relative_rank_1e_3=rank_1e_3,
        relative_rank_1e_6=rank_1e_6,
        column_norms=tuple(norms),
        absolute_column_cosines=tuple(cosines),
    )


def _compare_steps(
    parameter_names: tuple[str, ...], steps: Sequence[FiniteDifferenceStep]
) -> tuple[StepComparison, ...]:
    comparisons: list[StepComparison] = []
    for column, name in enumerate(parameter_names):
        for larger, smaller in pairwise(steps):
            metric: float | None
            if larger.column_validity[column] and smaller.column_validity[column]:
                left = larger.jacobian[:, column]
                right = smaller.jacobian[:, column]
                denominator = max(
                    1.0,
                    float(np.max(np.abs(left))),
                    float(np.max(np.abs(right))),
                )
                metric = float(np.max(np.abs(left - right)) / denominator)
            else:
                metric = None
            comparisons.append(StepComparison(name, larger.h, smaller.h, metric))
    return tuple(comparisons)


def _summarize_case_divergence(
    case: SensitivityCase,
    s0: SimulationRecord,
    o2: SimulationRecord,
    c2: SimulationRecord,
) -> CaseDivergenceSummary:
    records = {TankModel.S0: s0, TankModel.O2: o2, TankModel.C2: c2}
    pairwise: list[PairwiseDivergence] = []
    for left, right in (
        (TankModel.S0, TankModel.O2),
        (TankModel.S0, TankModel.C2),
        (TankModel.O2, TankModel.C2),
    ):
        left_record = records[left]
        right_record = records[right]
        if (
            not left_record.succeeded
            or not right_record.succeeded
            or left_record.observations is None
            or right_record.observations is None
        ):
            pairwise.append(
                PairwiseDivergence(left, right, None, None, None, None, None)
            )
            continue
        if left_record.observations.shape != right_record.observations.shape:
            pairwise.append(
                PairwiseDivergence(left, right, None, None, None, None, None)
            )
            continue
        difference = left_record.observations - right_record.observations
        differing = np.flatnonzero(
            left_record.observations != right_record.observations
        )
        pairwise.append(
            PairwiseDivergence(
                left_model=left,
                right_model=right,
                signed_difference=difference,
                byte_identical=(
                    left_record.observations.dtype == right_record.observations.dtype
                    and left_record.observations.tobytes(order="C")
                    == right_record.observations.tobytes(order="C")
                ),
                first_differing_index=int(differing[0]) if differing.size else None,
                differing_count=int(differing.size),
                maximum_absolute_difference=float(np.max(np.abs(difference))),
            )
        )
    cap_counts = tuple(
        (
            model,
            int(np.count_nonzero(record.observations == case.ceiling))
            if record.succeeded and record.observations is not None
            else None,
        )
        for model, record in ((TankModel.O2, o2), (TankModel.C2, c2))
    )
    return CaseDivergenceSummary(
        case_id=case.case_id,
        ceiling=float(case.ceiling),
        pairwise=tuple(pairwise),
        capped_output_counts=cap_counts,
    )


def _list_or_none(values: FloatArray | None) -> list[float] | None:
    return None if values is None else [float(value) for value in values]

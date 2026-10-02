"""One-case source-free training integration for the reviewed ABC6 control.

This adapter accepts one prefix-limited case from
``cascaded_tanks_abc6_cases``.  It calibrates the frozen left-inverse RMSE
thresholds, runs the checked Gaussian ABC-SMC wrapper when calibration
resolves, and runs the deterministic cost-matched baseline.  It has no API
for prospective targets and makes no claim that caller arrays have verified
synthetic origin.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from core.real_data import cascaded_tanks_synthetic_abc as synthetic_abc
from core.real_data.cascaded_tanks_abc6_cases import (
    PARAMETER_ORDER,
    TRAINING_INPUT_L,
    TRAINING_INPUT_S,
    ABC6AuthorizedCaseData,
    ABC6Case,
    ABC6TrainingCaseData,
    case_by_index,
)
from core.real_data.cascaded_tanks_models import (
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)
from core.real_data.cascaded_tanks_pattern_search import (
    MatchedCostBudget,
    ObjectiveFailure,
    ObjectiveSuccess,
    PatternSearchResult,
    derive_cascaded_tanks_matched_cost_budget,
    run_normalized_cube_pattern_search,
)

CALIBRATION_DRAW_COUNT: Final = 256
PARTICLE_COUNT: Final = 48
POPULATION_COUNT: Final = 2
MAX_PROPOSALS_PER_POPULATION: Final = 4096
CALIBRATION_QUANTILES: Final = (0.25, 0.10)
CALIBRATION_QUANTILE_RULE: Final = "sorted finite values; index ceil(p*n)-1"


def _manifest_payload() -> dict[str, object]:
    return {
        "protocol_id": "cascaded_tanks_abc6_synthetic_v1_20260928",
        "case_scope": "one prefix-limited ABC6TrainingCaseData",
        "parameter_order": tuple(PARAMETER_ORDER),
        "calibration_draw_count": CALIBRATION_DRAW_COUNT,
        "calibration_quantiles": CALIBRATION_QUANTILES,
        "calibration_quantile_rule": CALIBRATION_QUANTILE_RULE,
        "minimum_finite_calibration_draws": CALIBRATION_DRAW_COUNT,
        "unresolved_rule": (
            "unresolved unless all calibration draws are finite and both "
            "thresholds are finite, positive, and non-increasing"
        ),
        "calibration_rng": "numpy.default_rng(case.calibration_seed); vectorized uniform draws",
        "target_particles_per_population": PARTICLE_COUNT,
        "population_count": POPULATION_COUNT,
        "max_proposals_per_population": MAX_PROPOSALS_PER_POPULATION,
        "epsilon_schedule": "fixed calibration q25 then q10",
        "baseline_budget": (
            "calibration calls plus sum(population simulated + failed_simulations)"
        ),
        "data_origin": "caller-supplied unverified arrays; intended synthetic only",
        "prospective_target_access": "not available in this training adapter",
    }


TRAINING_CONTROL_MANIFEST: Final[Mapping[str, object]] = MappingProxyType(
    _manifest_payload()
)
TRAINING_CONTROL_MANIFEST_SHA256: Final = hashlib.sha256(
    json.dumps(
        dict(TRAINING_CONTROL_MANIFEST),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class _ExecutionControls:
    """Private bounded controls; the public entry point uses only production values."""

    calibration_draws: int
    target_samples: int
    max_attempts_per_population: int

    def __post_init__(self) -> None:
        for name, value in (
            ("calibration_draws", self.calibration_draws),
            ("target_samples", self.target_samples),
            ("max_attempts_per_population", self.max_attempts_per_population),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


_PRODUCTION_CONTROLS: Final = _ExecutionControls(
    calibration_draws=CALIBRATION_DRAW_COUNT,
    target_samples=PARTICLE_COUNT,
    max_attempts_per_population=MAX_PROPOSALS_PER_POPULATION,
)


@dataclass(frozen=True, slots=True)
class ABC6CalibrationEvidence:
    """All prior points, discrepancies, failures, and calibrated tolerances."""

    seed: int
    prior_points: tuple[tuple[float, ...], ...]
    discrepancies: tuple[float | None, ...]
    failure_categories_by_draw: tuple[str | None, ...]
    failure_category_counts: tuple[tuple[str, int], ...]
    finite_count: int
    failed_count: int
    q25_index: int | None
    q10_index: int | None
    epsilon_0: float | None
    epsilon_1: float | None
    status: Literal["calibrated", "unresolved"]
    unresolved_reason: str | None
    simulator_calls: int
    completed_steps: int


@dataclass(frozen=True, slots=True)
class ABC6TrainingResult:
    """Complete evidence for one training case, including incomplete outcomes."""

    case_index: int
    case_id: str
    model: str
    fit_length: int
    status: Literal["complete", "incomplete", "unresolved"]
    abc_status: Literal["complete", "incomplete", "unresolved"]
    baseline_status: Literal["complete", "incomplete"]
    calibration: ABC6CalibrationEvidence
    abc_result: dict[str, Any] | None
    matched_cost_budget: MatchedCostBudget
    baseline_result: PatternSearchResult
    baseline_failure_category_counts: tuple[tuple[str, int], ...]
    baseline_completed_steps: int
    simulator_call_counts: tuple[tuple[str, int], ...]
    training_input_sha256: str
    training_output_sha256: str
    production_controls: tuple[tuple[str, object], ...]
    production_controls_sha256: str
    execution_controls: tuple[tuple[str, int], ...]
    production_controls_used: bool
    synthetic_origin_verified: Literal[False] = False


def _validated_case_data(
    data: ABC6TrainingCaseData,
) -> tuple[ABC6Case, np.ndarray, np.ndarray]:
    if not isinstance(data, ABC6TrainingCaseData):
        raise TypeError("data must be one ABC6TrainingCaseData case view")
    case = data.case
    if not isinstance(case, ABC6Case) or case != case_by_index(case.case_index):
        raise ValueError("case identity must match the frozen ABC6 roster")
    expected_inputs = TRAINING_INPUT_S if case.input_window == "S" else TRAINING_INPUT_L
    if data.inputs != expected_inputs:
        raise ValueError("case inputs must exactly match its frozen training window")
    inputs = np.asarray(data.inputs, dtype=float)
    observed = np.asarray(data.observed_outputs, dtype=float)
    expected_length = case.input_length
    if (
        inputs.ndim != 1
        or observed.ndim != 1
        or inputs.size != expected_length
        or observed.size != expected_length
        or not np.all(np.isfinite(inputs))
        or not np.all(np.isfinite(observed))
    ):
        raise ValueError("case arrays must be finite one-dimensional fit-length data")
    if np.any(inputs < 0.0):
        raise ValueError("tank inputs must be non-negative")
    return case, inputs.copy(), observed.copy()


def _actual_parameters(
    values: tuple[float, ...], case: ABC6Case
) -> tuple[TankParameters, TankState, float]:
    by_name = dict(zip(PARAMETER_ORDER, values, strict=True))
    return (
        TankParameters(by_name["a"], by_name["c"], by_name["p"]),
        TankState(by_name["x1_0"], by_name["x2_0"]),
        by_name["ceiling"],
    )


def _simulate_discrepancy(
    point: tuple[float, ...],
    *,
    case: ABC6Case,
    inputs: np.ndarray,
    observed: np.ndarray,
    limits: TankSimulationLimits,
) -> tuple[float | None, str | None, int]:
    parameters, initial_state, ceiling = _actual_parameters(point, case)
    try:
        outcome = simulate_cascaded_tanks(
            inputs,
            parameters,
            initial_state,
            model=case.fit_model,
            ceiling=ceiling,
            limits=limits,
        )
    except Exception:  # noqa: BLE001 - retain unexpected simulator-call failures as evidence.
        return None, "unexpected_exception", 0
    if isinstance(outcome, TankSimulationFailure):
        return None, outcome.category.value, len(outcome.trace)
    if not isinstance(outcome, TankSimulationSuccess):
        return None, "invalid_simulator_result", 0
    generated = np.asarray(outcome.observations, dtype=float)
    if generated.shape != observed.shape or not np.all(np.isfinite(generated)):
        return None, "invalid_trajectory", len(outcome.trace)
    with np.errstate(over="ignore", invalid="ignore"):
        discrepancy = float(np.sqrt(np.mean(np.square(generated - observed))))
    if not math.isfinite(discrepancy):
        return None, "non_finite_rmse", len(outcome.trace)
    return discrepancy, None, len(outcome.trace)


def _left_inverse_indices(finite_count: int) -> tuple[int, int]:
    if finite_count <= 0:
        raise ValueError("left-inverse calibration requires at least one finite value")
    return (
        math.ceil(CALIBRATION_QUANTILES[0] * finite_count) - 1,
        math.ceil(CALIBRATION_QUANTILES[1] * finite_count) - 1,
    )


def _calibrate(
    *,
    case: ABC6Case,
    inputs: np.ndarray,
    observed: np.ndarray,
    controls: _ExecutionControls,
    limits: TankSimulationLimits,
) -> ABC6CalibrationEvidence:
    rng = np.random.default_rng(case.calibration_seed)
    bounds = np.asarray(case.prior_bounds, dtype=float)
    points = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(controls.calibration_draws, 6)
    )
    discrepancies: list[float | None] = []
    failure_categories_by_draw: list[str | None] = []
    categories: Counter[str] = Counter()
    completed_steps = 0
    for row in points:
        point = tuple(float(value) for value in row)
        discrepancy, category, steps = _simulate_discrepancy(
            point,
            case=case,
            inputs=inputs,
            observed=observed,
            limits=limits,
        )
        discrepancies.append(discrepancy)
        failure_categories_by_draw.append(category)
        completed_steps += steps
        if category is not None:
            categories[category] += 1

    finite_values = sorted(value for value in discrepancies if value is not None)
    finite_count = len(finite_values)
    q25_index: int | None = None
    q10_index: int | None = None
    epsilon_0: float | None = None
    epsilon_1: float | None = None
    unresolved_reason: str | None = None
    if finite_count:
        q25_index, q10_index = _left_inverse_indices(finite_count)
        epsilon_0 = finite_values[q25_index]
        epsilon_1 = finite_values[q10_index]
    if finite_count < controls.calibration_draws:
        unresolved_reason = "fewer_than_required_finite_calibration_draws"
    elif epsilon_0 is None or epsilon_1 is None:
        unresolved_reason = "calibration_threshold_missing"
    elif not math.isfinite(epsilon_0) or not math.isfinite(epsilon_1):
        unresolved_reason = "non_finite_calibration_threshold"
    elif epsilon_0 <= 0.0 or epsilon_1 <= 0.0:
        unresolved_reason = "non_positive_calibration_threshold"
    elif epsilon_1 > epsilon_0:
        # A sorted empirical left-inverse rule makes this impossible unless
        # the quantile implementation or probability ordering is changed.
        unresolved_reason = "calibration_thresholds_not_non_increasing"
    status: Literal["calibrated", "unresolved"] = (
        "unresolved" if unresolved_reason is not None else "calibrated"
    )
    return ABC6CalibrationEvidence(
        seed=case.calibration_seed,
        prior_points=tuple(tuple(float(value) for value in row) for row in points),
        discrepancies=tuple(discrepancies),
        failure_categories_by_draw=tuple(failure_categories_by_draw),
        failure_category_counts=tuple(sorted(categories.items())),
        finite_count=finite_count,
        failed_count=controls.calibration_draws - finite_count,
        q25_index=q25_index,
        q10_index=q10_index,
        epsilon_0=epsilon_0,
        epsilon_1=epsilon_1,
        status=status,
        unresolved_reason=unresolved_reason,
        simulator_calls=controls.calibration_draws,
        completed_steps=completed_steps,
    )


def _baseline_objective(
    point: tuple[float, ...],
    *,
    case: ABC6Case,
    inputs: np.ndarray,
    observed: np.ndarray,
    limits: TankSimulationLimits,
) -> ObjectiveSuccess | ObjectiveFailure:
    bounds = np.asarray(case.prior_bounds, dtype=float)
    values = tuple(
        float(bounds[index, 0] + point[index] * (bounds[index, 1] - bounds[index, 0]))
        for index in range(6)
    )
    discrepancy, category, steps = _simulate_discrepancy(
        values,
        case=case,
        inputs=inputs,
        observed=observed,
        limits=limits,
    )
    if category is not None or discrepancy is None:
        return ObjectiveFailure(category or "invalid_objective", completed_steps=steps)
    return ObjectiveSuccess(discrepancy, completed_steps=steps)


def _run_one_case(
    data: ABC6TrainingCaseData,
    *,
    controls: _ExecutionControls,
) -> ABC6TrainingResult:
    case, inputs, observed = _validated_case_data(data)
    fit_length = len(inputs)
    limits = TankSimulationLimits(max_steps=fit_length)
    calibration = _calibrate(
        case=case,
        inputs=inputs,
        observed=observed,
        controls=controls,
        limits=limits,
    )

    abc_result: dict[str, Any] | None = None
    if calibration.status == "calibrated":
        assert calibration.epsilon_0 is not None and calibration.epsilon_1 is not None
        config = synthetic_abc.CascadedTanksSyntheticABCConfig(
            model=case.fit_model,
            fixed_parameters={},
            free_parameter_bounds=dict(
                zip(PARAMETER_ORDER, case.prior_bounds, strict=True)
            ),
            target_samples=controls.target_samples,
            epsilon_schedule=(calibration.epsilon_0, calibration.epsilon_1),
            max_attempts_per_population=(
                controls.max_attempts_per_population,
                controls.max_attempts_per_population,
            ),
            seed=case.abc_seed,
            output_units="synthetic output units",
            limits=limits,
        )
        abc_result = synthetic_abc.run_cascaded_tanks_synthetic_abc_smc(
            inputs, observed, config
        )

    populations = (
        ()
        if abc_result is None
        else tuple(abc_result["reference_evidence"]["populations"])
    )
    matched_budget = derive_cascaded_tanks_matched_cost_budget(
        calibration_calls=calibration.simulator_calls,
        population_records=populations,
        fit_length=fit_length,
    )
    baseline = run_normalized_cube_pattern_search(
        lambda point: _baseline_objective(
            point,
            case=case,
            inputs=inputs,
            observed=observed,
            limits=limits,
        ),
        dimension=6,
        max_calls=matched_budget.max_calls,
    )
    baseline_failure_categories = Counter(
        evaluation.failure_category
        for evaluation in baseline.evaluations
        if evaluation.failure_category is not None
    )
    baseline_completed_steps = sum(
        evaluation.completed_steps or 0 for evaluation in baseline.evaluations
    )
    baseline_status: Literal["complete", "incomplete"] = (
        "complete" if baseline.stop_reason == "all_starts_terminated" else "incomplete"
    )

    if calibration.status == "unresolved":
        abc_status: Literal["complete", "incomplete", "unresolved"] = "unresolved"
        status: Literal["complete", "incomplete", "unresolved"] = "unresolved"
    else:
        assert abc_result is not None
        abc_status = "complete" if abc_result["complete"] else "incomplete"
        status = (
            "complete"
            if abc_status == "complete" and baseline_status == "complete"
            else "incomplete"
        )

    abc_calls = sum(
        int(population["diagnostics"]["simulated"])
        + int(population["diagnostics"]["failed_simulations"])
        for population in populations
    )
    simulator_call_counts = (
        ("calibration", calibration.simulator_calls),
        ("abc", abc_calls),
        ("baseline", baseline.callback_calls),
    )
    return ABC6TrainingResult(
        case_index=case.case_index,
        case_id=case.case_id,
        model=case.fit_model.value,
        fit_length=fit_length,
        status=status,
        abc_status=abc_status,
        baseline_status=baseline_status,
        calibration=calibration,
        abc_result=abc_result,
        matched_cost_budget=matched_budget,
        baseline_result=baseline,
        baseline_failure_category_counts=tuple(
            sorted(
                (str(category), count)
                for category, count in baseline_failure_categories.items()
            )
        ),
        baseline_completed_steps=baseline_completed_steps,
        simulator_call_counts=simulator_call_counts,
        training_input_sha256=hashlib.sha256(
            np.asarray(inputs, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        training_output_sha256=hashlib.sha256(
            np.asarray(observed, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        production_controls=tuple(sorted(TRAINING_CONTROL_MANIFEST.items())),
        production_controls_sha256=TRAINING_CONTROL_MANIFEST_SHA256,
        execution_controls=(
            ("calibration_draws", controls.calibration_draws),
            ("target_samples", controls.target_samples),
            ("max_attempts_per_population", controls.max_attempts_per_population),
        ),
        production_controls_used=controls == _PRODUCTION_CONTROLS,
    )


def run_abc6_training_case(data: ABC6AuthorizedCaseData) -> ABC6TrainingResult:
    """Run one permit-authorized campaign case with frozen production controls.

    Plain case arrays remain usable through private offline numerical helpers,
    while this production entrypoint accepts only the one-use sealed value
    issued by ``cases.get_training_case_data``.
    """

    if type(data) is not ABC6AuthorizedCaseData:
        raise TypeError("production training requires sealed authorized case data")
    case_data = data._consume_for_fit()
    return _run_one_case(case_data, controls=_PRODUCTION_CONTROLS)


def _run_one_case_with_bounded_controls_for_test(
    data: ABC6TrainingCaseData,
    *,
    calibration_draws: int,
    target_samples: int,
    max_attempts_per_population: int,
) -> ABC6TrainingResult:
    """Exercise real consumers cheaply; never changes public production defaults."""

    controls = _ExecutionControls(
        calibration_draws=calibration_draws,
        target_samples=target_samples,
        max_attempts_per_population=max_attempts_per_population,
    )
    return _run_one_case(data, controls=controls)


__all__ = [
    "CALIBRATION_DRAW_COUNT",
    "CALIBRATION_QUANTILES",
    "MAX_PROPOSALS_PER_POPULATION",
    "PARTICLE_COUNT",
    "POPULATION_COUNT",
    "TRAINING_CONTROL_MANIFEST",
    "TRAINING_CONTROL_MANIFEST_SHA256",
    "ABC6CalibrationEvidence",
    "ABC6TrainingResult",
    "run_abc6_training_case",
]

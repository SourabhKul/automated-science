"""Deterministic normalized-cube coordinate pattern search for tank baselines.

This module is source-free and uses only the Python standard library.  It does
not import the tank simulator, read data, access the network, or run a research
campaign.  Callers provide a deterministic objective callback in normalized
``[0, 1]^d`` coordinates and a hard cap on callback invocations.

The frozen search is deliberately small and explicit: center then
lexicographic corners, starts sorted by ``(objective, coordinates)``, and
coordinate polls in ``+`` then ``-`` order.  Cached coordinates do not invoke
the callback again.  Failed/nonfinite objective results rank as ``+inf`` and
remain in the complete evaluation record.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Literal, TypeAlias

INITIAL_MESH = 0.25
MESH_STOP_THRESHOLD = 2.0**-10
SUPPORTED_TANK_FIT_LENGTHS = (78, 204)
_FAILURE_CATEGORIES_FOR_SUPPORT_REJECT = frozenset(
    {"outside_support", "out_of_support", "support_reject"}
)

NormalizedPoint: TypeAlias = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ObjectiveSuccess:
    """A finite objective and optional completed simulator-step count."""

    objective: float
    completed_steps: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.objective, bool) or not isinstance(self.objective, Real):
            raise TypeError("objective must be a real scalar")
        _validate_completed_steps(self.completed_steps)


@dataclass(frozen=True, slots=True)
class ObjectiveFailure:
    """A failed callback result retained with objective value ``+inf``."""

    category: str
    message: str | None = None
    completed_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("failure category must be non-empty text")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("failure message must be text or None")
        _validate_completed_steps(self.completed_steps)


ObjectiveCallbackResult: TypeAlias = (
    float | int | ObjectiveSuccess | ObjectiveFailure | None
)
ObjectiveCallback: TypeAlias = Callable[[NormalizedPoint], ObjectiveCallbackResult]
CandidateStatus: TypeAlias = Literal["evaluated", "cached", "budget_exhausted"]
CandidateRole: TypeAlias = Literal["center", "corner", "poll"]


@dataclass(frozen=True, slots=True)
class PatternSearchEvaluation:
    """One unique normalized point and its first callback result."""

    evaluation_index: int
    coordinates: NormalizedPoint
    objective: float
    failure_category: str | None
    failure_message: str | None
    completed_steps: int | None

    @property
    def succeeded(self) -> bool:
        return self.failure_category is None


@dataclass(frozen=True, slots=True)
class PatternSearchCandidate:
    """One ordered design/poll request, including cache and budget status."""

    attempt_index: int
    coordinates: NormalizedPoint
    role: CandidateRole
    status: CandidateStatus
    evaluation_index: int | None
    corner_bits: tuple[int, ...] | None = None
    coordinate_index: int | None = None
    direction: Literal["+", "-"] | None = None


@dataclass(frozen=True, slots=True)
class PatternSearchPoll:
    """One complete or budget-truncated coordinate poll."""

    restart_index: int
    poll_index: int
    mesh_before: float
    current_before: NormalizedPoint
    candidates: tuple[PatternSearchCandidate, ...]
    complete: bool
    blocked_candidate_position: int | None
    chosen_evaluation_index: int | None
    accepted: bool | None
    current_after: NormalizedPoint
    mesh_after: float


@dataclass(frozen=True, slots=True)
class PatternSearchRestart:
    """One frozen initial point's local search and termination state."""

    restart_index: int
    start_evaluation_index: int
    start_coordinates: NormalizedPoint
    end_evaluation_index: int
    end_coordinates: NormalizedPoint
    mesh_initial: float
    mesh_final: float
    poll_count: int
    termination_reason: Literal["mesh_below_threshold", "call_budget_exhausted"]


@dataclass(frozen=True, slots=True)
class PopulationCallAccounting:
    """Simulator-call counts retained from one ABC population record."""

    population_index: int
    complete: bool
    proposed: int
    simulated: int
    failed_simulations: int
    out_of_support: int
    simulator_calls: int


@dataclass(frozen=True, slots=True)
class MatchedCostBudget:
    """Realized ABC simulator cost converted to a baseline call/step cap."""

    calibration_calls: int
    populations: tuple[PopulationCallAccounting, ...]
    population_calls: int
    max_calls: int
    fit_length: int
    reserved_step_evaluations: int

    @property
    def excluded_out_of_support_proposals(self) -> int:
        """Total support rejects, which have no simulator-call charge."""

        return sum(population.out_of_support for population in self.populations)


def derive_cascaded_tanks_matched_cost_budget(
    *,
    calibration_calls: int,
    population_records: Iterable[Mapping[str, object]],
    fit_length: int,
) -> MatchedCostBudget:
    """Derive the ABC-matched baseline cap from every population record.

    ``population_records`` is the ordered ``reference_evidence["populations"]``
    sequence, including an incomplete final population.  For each population,
    the charged call count is exactly ``diagnostics.simulated +
    diagnostics.failed_simulations``.  ``proposed`` and ``out_of_support`` are
    retained for auditing but never enter that sum because support rejects do
    not invoke the simulator.  This adapter reads only caller-provided mappings
    and is independent of the ABC runner and all data sources.

    The synthetic protocol has fit lengths 78 (S) and 204 (L).  Reserved step
    evaluations use the full baseline call allowance, including failed traces:
    ``max_calls * fit_length``.
    """

    calibration_calls = _validate_nonnegative_integer(
        calibration_calls, "calibration_calls"
    )
    fit_length = _validate_nonnegative_integer(fit_length, "fit_length")
    if fit_length not in SUPPORTED_TANK_FIT_LENGTHS:
        raise ValueError(f"fit_length must be one of {SUPPORTED_TANK_FIT_LENGTHS}")

    accounting: list[PopulationCallAccounting] = []
    for population_index, record in enumerate(population_records):
        if not isinstance(record, Mapping):
            raise TypeError("each population record must be a mapping")
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise TypeError(
                f"population {population_index} must include diagnostics mapping"
            )
        complete = diagnostics.get("complete")
        if not isinstance(complete, bool):
            raise TypeError(
                f"population {population_index} diagnostics.complete must be bool"
            )
        proposed = _diagnostic_count(diagnostics, "proposed", population_index)
        simulated = _diagnostic_count(diagnostics, "simulated", population_index)
        failed_simulations = _diagnostic_count(
            diagnostics, "failed_simulations", population_index
        )
        out_of_support = _diagnostic_count(
            diagnostics, "out_of_support", population_index
        )
        simulator_calls = simulated + failed_simulations
        if out_of_support > proposed or simulator_calls + out_of_support > proposed:
            raise ValueError(
                f"population {population_index} simulator/support counts exceed proposed"
            )
        accounting.append(
            PopulationCallAccounting(
                population_index=population_index,
                complete=complete,
                proposed=proposed,
                simulated=simulated,
                failed_simulations=failed_simulations,
                out_of_support=out_of_support,
                simulator_calls=simulator_calls,
            )
        )

    populations = tuple(accounting)
    population_calls = sum(population.simulator_calls for population in populations)
    max_calls = calibration_calls + population_calls
    return MatchedCostBudget(
        calibration_calls=calibration_calls,
        populations=populations,
        population_calls=population_calls,
        max_calls=max_calls,
        fit_length=fit_length,
        reserved_step_evaluations=max_calls * fit_length,
    )


@dataclass(frozen=True, slots=True)
class PatternSearchResult:
    """Complete deterministic call, evaluation, poll, and restart record."""

    dimension: int
    call_budget: int
    callback_calls: int
    candidate_attempts: int
    cache_hits: int
    evaluations: tuple[PatternSearchEvaluation, ...]
    initial_design: tuple[PatternSearchCandidate, ...]
    initial_design_complete: bool
    initial_design_blocked_position: int | None
    start_evaluation_indices: tuple[int, ...]
    polls: tuple[PatternSearchPoll, ...]
    restarts: tuple[PatternSearchRestart, ...]
    best_evaluation_index: int | None
    stop_reason: Literal[
        "all_starts_terminated",
        "budget_exhausted_initial_design",
        "budget_exhausted_poll",
    ]

    @property
    def best_evaluation(self) -> PatternSearchEvaluation | None:
        """Return the best finite, successful evaluation, if one exists."""

        if self.best_evaluation_index is None:
            return None
        return self.evaluations[self.best_evaluation_index]

    @property
    def failure_count(self) -> int:
        return sum(not evaluation.succeeded for evaluation in self.evaluations)

    @property
    def support_reject_count(self) -> int:
        """Count callback-classified support failures separately.

        Search coordinates are always clipped to the normalized cube, so this
        is normally zero.  An adapter can still identify a simulator-side
        support rejection with one of the recognized failure categories.
        """

        return sum(
            evaluation.failure_category in _FAILURE_CATEGORIES_FOR_SUPPORT_REJECT
            for evaluation in self.evaluations
        )

    @property
    def unused_call_budget(self) -> int:
        return self.call_budget - self.callback_calls

    @property
    def budget_was_underused(self) -> bool:
        return (
            self.stop_reason == "all_starts_terminated" and self.unused_call_budget > 0
        )

    @property
    def completed_steps_total(self) -> int | None:
        """Sum completed steps when every callback result reported them."""

        if any(evaluation.completed_steps is None for evaluation in self.evaluations):
            return None
        return sum(evaluation.completed_steps or 0 for evaluation in self.evaluations)


def run_normalized_cube_pattern_search(
    objective: ObjectiveCallback,
    *,
    dimension: int,
    max_calls: int,
) -> PatternSearchResult:
    """Run the frozen center/corners multi-start coordinate pattern search.

    The callback receives an immutable tuple in ``[0, 1]^dimension`` and may
    return a finite number, :class:`ObjectiveSuccess`, :class:`ObjectiveFailure`,
    or ``None`` (recorded as a callback failure).  A nonfinite returned scalar
    is recorded as ``nonfinite_objective``.  Exceptions raised by the callback
    propagate so programming errors are not mistaken for simulation failures.
    The call budget counts actual callback invocations; cache hits remain
    visible as candidate attempts but do not consume it.

    If the allowance is too small to finish the initial design, no start order
    or local search is inferred.  If a poll runs out of allowance, its blocked
    candidate position and all preceding cache/evaluation events are retained.
    """

    if not callable(objective):
        raise TypeError("objective must be callable")
    dimension = _validate_integer(dimension, "dimension", minimum=1)
    max_calls = _validate_integer(max_calls, "max_calls", minimum=0)

    evaluations: list[PatternSearchEvaluation] = []
    evaluation_by_point: dict[NormalizedPoint, PatternSearchEvaluation] = {}
    design_events: list[PatternSearchCandidate] = []
    polls: list[PatternSearchPoll] = []
    restarts: list[PatternSearchRestart] = []
    candidate_attempts = 0
    cache_hits = 0
    blocked_initial_position: int | None = None

    def request(
        coordinates: NormalizedPoint,
        role: CandidateRole,
        *,
        corner_bits: tuple[int, ...] | None = None,
        coordinate_index: int | None = None,
        direction: Literal["+", "-"] | None = None,
    ) -> PatternSearchCandidate:
        nonlocal candidate_attempts, cache_hits
        attempt_index = candidate_attempts
        candidate_attempts += 1
        cached = evaluation_by_point.get(coordinates)
        if cached is not None:
            cache_hits += 1
            status: CandidateStatus = "cached"
            evaluation_index: int | None = cached.evaluation_index
        elif len(evaluations) >= max_calls:
            status = "budget_exhausted"
            evaluation_index = None
        else:
            record = _evaluate(objective, coordinates, len(evaluations))
            evaluations.append(record)
            evaluation_by_point[coordinates] = record
            status = "evaluated"
            evaluation_index = record.evaluation_index
        return PatternSearchCandidate(
            attempt_index=attempt_index,
            coordinates=coordinates,
            role=role,
            status=status,
            evaluation_index=evaluation_index,
            corner_bits=corner_bits,
            coordinate_index=coordinate_index,
            direction=direction,
        )

    center = (0.5,) * dimension
    design_events.append(request(center, "center"))
    if design_events[-1].status == "budget_exhausted":
        blocked_initial_position = 0
    else:
        for bits in product((0, 1), repeat=dimension):
            point = tuple(float(bit) for bit in bits)
            event = request(point, "corner", corner_bits=bits)
            design_events.append(event)
            if event.status == "budget_exhausted":
                blocked_initial_position = len(design_events) - 1
                break

    design_complete = blocked_initial_position is None
    if not design_complete:
        stop_reason: Literal[
            "all_starts_terminated",
            "budget_exhausted_initial_design",
            "budget_exhausted_poll",
        ] = "budget_exhausted_initial_design"
        start_indices: tuple[int, ...] = ()
    else:
        start_indices = tuple(
            event.evaluation_index
            for event in sorted(
                design_events,
                key=lambda event: _rank(evaluations[event.evaluation_index]),
            )
            if event.evaluation_index is not None
        )
        stop_reason = "all_starts_terminated"

        budget_stopped = False
        for restart_index, start_eval_index in enumerate(start_indices):
            current_eval = evaluations[start_eval_index]
            mesh = INITIAL_MESH
            restart_poll_start = len(polls)
            termination_reason: Literal[
                "mesh_below_threshold", "call_budget_exhausted"
            ] = "mesh_below_threshold"

            while mesh >= MESH_STOP_THRESHOLD:
                mesh_before = mesh
                current_before = current_eval.coordinates
                candidate_events: list[PatternSearchCandidate] = []
                blocked_position: int | None = None
                poll_index = len(polls) - restart_poll_start

                for coordinate_index in range(dimension):
                    for direction in ("+", "-"):
                        sign = 1.0 if direction == "+" else -1.0
                        candidate = list(current_before)
                        candidate[coordinate_index] = min(
                            1.0,
                            max(
                                0.0,
                                candidate[coordinate_index] + sign * mesh,
                            ),
                        )
                        event = request(
                            tuple(candidate),
                            "poll",
                            coordinate_index=coordinate_index,
                            direction=direction,
                        )
                        candidate_events.append(event)
                        if event.status == "budget_exhausted":
                            blocked_position = len(candidate_events) - 1
                            break
                    if blocked_position is not None:
                        break

                if blocked_position is not None:
                    polls.append(
                        PatternSearchPoll(
                            restart_index=restart_index,
                            poll_index=poll_index,
                            mesh_before=mesh,
                            current_before=current_before,
                            candidates=tuple(candidate_events),
                            complete=False,
                            blocked_candidate_position=blocked_position,
                            chosen_evaluation_index=None,
                            accepted=None,
                            current_after=current_before,
                            mesh_after=mesh,
                        )
                    )
                    termination_reason = "call_budget_exhausted"
                    stop_reason = "budget_exhausted_poll"
                    budget_stopped = True
                    break

                poll_evaluations = [
                    evaluations[event.evaluation_index]
                    for event in candidate_events
                    if event.evaluation_index is not None
                ]
                chosen = min(poll_evaluations, key=_rank)
                accepted = _rank(chosen) < _rank(current_eval)
                if accepted:
                    current_eval = chosen
                else:
                    mesh /= 2.0
                polls.append(
                    PatternSearchPoll(
                        restart_index=restart_index,
                        poll_index=poll_index,
                        mesh_before=mesh_before,
                        current_before=current_before,
                        candidates=tuple(candidate_events),
                        complete=True,
                        blocked_candidate_position=None,
                        chosen_evaluation_index=chosen.evaluation_index,
                        accepted=accepted,
                        current_after=current_eval.coordinates,
                        mesh_after=mesh,
                    )
                )

            restarts.append(
                PatternSearchRestart(
                    restart_index=restart_index,
                    start_evaluation_index=start_eval_index,
                    start_coordinates=evaluations[start_eval_index].coordinates,
                    end_evaluation_index=current_eval.evaluation_index,
                    end_coordinates=current_eval.coordinates,
                    mesh_initial=INITIAL_MESH,
                    mesh_final=mesh,
                    poll_count=len(polls) - restart_poll_start,
                    termination_reason=termination_reason,
                )
            )
            if budget_stopped:
                break

    valid_evaluations = [
        evaluation for evaluation in evaluations if evaluation.succeeded
    ]
    best = (
        min(valid_evaluations, key=_rank).evaluation_index
        if valid_evaluations
        else None
    )
    return PatternSearchResult(
        dimension=dimension,
        call_budget=max_calls,
        callback_calls=len(evaluations),
        candidate_attempts=candidate_attempts,
        cache_hits=cache_hits,
        evaluations=tuple(evaluations),
        initial_design=tuple(design_events),
        initial_design_complete=design_complete,
        initial_design_blocked_position=blocked_initial_position,
        start_evaluation_indices=start_indices,
        polls=tuple(polls),
        restarts=tuple(restarts),
        best_evaluation_index=best,
        stop_reason=stop_reason,
    )


def _evaluate(
    objective: ObjectiveCallback,
    coordinates: NormalizedPoint,
    evaluation_index: int,
) -> PatternSearchEvaluation:
    result = objective(coordinates)

    if isinstance(result, ObjectiveFailure):
        return PatternSearchEvaluation(
            evaluation_index=evaluation_index,
            coordinates=coordinates,
            objective=math.inf,
            failure_category=result.category,
            failure_message=result.message,
            completed_steps=result.completed_steps,
        )

    if result is None:
        return PatternSearchEvaluation(
            evaluation_index=evaluation_index,
            coordinates=coordinates,
            objective=math.inf,
            failure_category="callback_failure",
            failure_message=None,
            completed_steps=None,
        )

    if isinstance(result, ObjectiveSuccess):
        raw_objective = result.objective
        completed_steps = result.completed_steps
    elif isinstance(result, Real) and not isinstance(result, bool):
        raw_objective = result
        completed_steps = None
    else:
        raise TypeError(
            "objective callback must return a real scalar, ObjectiveSuccess, "
            "ObjectiveFailure, or None"
        )

    value = float(raw_objective)
    if not math.isfinite(value):
        return PatternSearchEvaluation(
            evaluation_index=evaluation_index,
            coordinates=coordinates,
            objective=math.inf,
            failure_category="nonfinite_objective",
            failure_message="callback returned a nonfinite objective",
            completed_steps=completed_steps,
        )
    return PatternSearchEvaluation(
        evaluation_index=evaluation_index,
        coordinates=coordinates,
        objective=value,
        failure_category=None,
        failure_message=None,
        completed_steps=completed_steps,
    )


def _rank(evaluation: PatternSearchEvaluation) -> tuple[float, NormalizedPoint]:
    return evaluation.objective, evaluation.coordinates


def _validate_integer(value: int, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _diagnostic_count(
    diagnostics: Mapping[str, object], label: str, population_index: int
) -> int:
    try:
        value = diagnostics[label]
    except KeyError as error:
        raise ValueError(
            f"population {population_index} diagnostics.{label} is required"
        ) from error
    return _validate_nonnegative_integer(
        value, f"population {population_index} diagnostics.{label}"
    )


def _validate_completed_steps(value: int | None) -> None:
    if value is not None:
        _validate_integer(value, "completed_steps", minimum=0)


__all__ = [
    "INITIAL_MESH",
    "MESH_STOP_THRESHOLD",
    "SUPPORTED_TANK_FIT_LENGTHS",
    "MatchedCostBudget",
    "ObjectiveFailure",
    "ObjectiveSuccess",
    "PatternSearchCandidate",
    "PatternSearchEvaluation",
    "PatternSearchPoll",
    "PatternSearchRestart",
    "PatternSearchResult",
    "PopulationCallAccounting",
    "derive_cascaded_tanks_matched_cost_budget",
    "run_normalized_cube_pattern_search",
]

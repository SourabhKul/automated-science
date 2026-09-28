"""Synthetic tiny-objective controls for the source-free tank baseline."""

from __future__ import annotations

import math

from core.real_data.cascaded_tanks_pattern_search import (
    ObjectiveFailure,
    ObjectiveSuccess,
    derive_cascaded_tanks_matched_cost_budget,
    run_normalized_cube_pattern_search,
)


def test_six_dimensional_initial_design_stops_cleanly_below_65_call_allowance() -> None:
    calls: list[tuple[float, ...]] = []

    def objective(point: tuple[float, ...]) -> float:
        calls.append(point)
        return sum(point)

    result = run_normalized_cube_pattern_search(objective, dimension=6, max_calls=64)

    assert len(calls) == result.callback_calls == 64
    assert result.candidate_attempts == 65
    assert result.cache_hits == 0
    assert not result.initial_design_complete
    assert result.initial_design_blocked_position == 64
    assert result.stop_reason == "budget_exhausted_initial_design"
    assert not result.start_evaluation_indices
    assert not result.restarts
    assert result.initial_design[0].role == "center"
    assert result.initial_design[0].coordinates == (0.5,) * 6
    assert result.initial_design[1].corner_bits == (0, 0, 0, 0, 0, 0)
    assert result.initial_design[-2].corner_bits == (1, 1, 1, 1, 1, 0)
    assert result.initial_design[-1].corner_bits == (1, 1, 1, 1, 1, 1)
    assert result.initial_design[-1].status == "budget_exhausted"
    assert result.initial_design[-1].evaluation_index is None
    assert result.best_evaluation is not None
    assert result.best_evaluation.coordinates == (0.0,) * 6


def test_center_precedes_lexicographic_corners_and_starts_use_rmse_then_zlex() -> None:
    seen: list[tuple[float, ...]] = []

    def objective(point: tuple[float, ...]) -> float:
        seen.append(point)
        return 1.0  # Every start ties, so normalized lexicographic order decides.

    result = run_normalized_cube_pattern_search(objective, dimension=2, max_calls=5)

    assert seen == [
        (0.5, 0.5),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
    assert result.initial_design_complete
    assert tuple(
        result.evaluations[index].coordinates
        for index in result.start_evaluation_indices
    ) == ((0.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0))


def test_poll_chooses_lexicographically_first_candidate_when_objectives_tie() -> None:
    def objective(point: tuple[float, ...]) -> float:
        return 0.0 if point[0] in {0.25, 0.5, 0.75} else 1.0

    result = run_normalized_cube_pattern_search(objective, dimension=1, max_calls=5)

    first_poll = result.polls[0]
    assert first_poll.complete
    assert tuple(candidate.coordinates for candidate in first_poll.candidates) == (
        (0.75,),
        (0.25,),
    )
    assert result.evaluations[first_poll.chosen_evaluation_index].coordinates == (0.25,)
    assert first_poll.accepted
    assert first_poll.current_after == (0.25,)
    assert result.best_evaluation is not None
    assert result.best_evaluation.coordinates == (0.25,)


def test_boundary_polls_reuse_cache_without_charging_callback_again() -> None:
    def objective(point: tuple[float, ...]) -> float:
        return 0.0

    result = run_normalized_cube_pattern_search(objective, dimension=1, max_calls=10)

    assert result.callback_calls == 10
    assert len({evaluation.coordinates for evaluation in result.evaluations}) == 10
    assert result.cache_hits > 0
    assert result.candidate_attempts > result.callback_calls
    assert any(
        candidate.status == "cached"
        for poll in result.polls
        for candidate in poll.candidates
    )
    assert result.stop_reason == "budget_exhausted_poll"


def test_budget_expiry_records_exact_mid_poll_position_without_completing_poll() -> (
    None
):
    def objective(point: tuple[float, ...]) -> float:
        return sum((coordinate - 0.5) ** 2 for coordinate in point)

    result = run_normalized_cube_pattern_search(objective, dimension=2, max_calls=6)

    assert result.callback_calls == 6
    assert result.stop_reason == "budget_exhausted_poll"
    poll = result.polls[-1]
    assert not poll.complete
    assert poll.blocked_candidate_position == 1
    assert len(poll.candidates) == 2
    assert poll.candidates[0].coordinates == (0.75, 0.5)
    assert poll.candidates[0].status == "evaluated"
    assert poll.candidates[1].coordinates == (0.25, 0.5)
    assert poll.candidates[1].status == "budget_exhausted"
    assert poll.candidates[1].evaluation_index is None
    assert poll.current_after == poll.current_before == (0.5, 0.5)
    assert poll.accepted is None
    assert result.restarts[-1].termination_reason == "call_budget_exhausted"


def test_all_failed_objectives_rank_at_infinity_and_produce_no_best_point() -> None:
    def objective(point: tuple[float, ...]) -> ObjectiveFailure:
        return ObjectiveFailure("simulator_failure", "injected", completed_steps=2)

    result = run_normalized_cube_pattern_search(objective, dimension=1, max_calls=3)

    assert result.callback_calls == 3
    assert result.failure_count == 3
    assert all(math.isinf(item.objective) for item in result.evaluations)
    assert all(
        item.failure_category == "simulator_failure" for item in result.evaluations
    )
    assert result.best_evaluation is None
    assert result.completed_steps_total == 6


def test_success_results_record_completed_steps_and_complete_search_underuses_budget() -> (
    None
):
    def objective(point: tuple[float, ...]) -> ObjectiveSuccess:
        return ObjectiveSuccess(point[0], completed_steps=3)

    result = run_normalized_cube_pattern_search(objective, dimension=1, max_calls=1000)

    assert result.stop_reason == "all_starts_terminated"
    assert len(result.restarts) == 3
    assert all(
        restart.termination_reason == "mesh_below_threshold"
        for restart in result.restarts
    )
    assert result.budget_was_underused
    assert result.unused_call_budget > 0
    assert result.completed_steps_total == result.callback_calls * 3
    assert result.restarts[0].mesh_final < 2.0**-10


def test_nonfinite_callback_result_is_a_failed_objective_not_a_search_crash() -> None:
    result = run_normalized_cube_pattern_search(
        lambda _: float("nan"), dimension=1, max_calls=3
    )

    assert result.best_evaluation is None
    assert result.failure_count == 3
    assert all(
        item.failure_category == "nonfinite_objective" for item in result.evaluations
    )


def test_matched_budget_sums_population_zero_and_one_simulator_calls() -> None:
    populations = (
        {
            "diagnostics": {
                "complete": True,
                "proposed": 20,
                "simulated": 7,
                "failed_simulations": 2,
                "out_of_support": 4,
            }
        },
        {
            "diagnostics": {
                "complete": True,
                "proposed": 13,
                "simulated": 5,
                "failed_simulations": 3,
                "out_of_support": 3,
            }
        },
    )

    budget = derive_cascaded_tanks_matched_cost_budget(
        calibration_calls=256, population_records=populations, fit_length=78
    )

    assert tuple(item.population_index for item in budget.populations) == (0, 1)
    assert tuple(item.simulator_calls for item in budget.populations) == (9, 8)
    assert budget.population_calls == 17
    assert budget.calibration_calls == 256
    assert budget.max_calls == 273
    assert budget.reserved_step_evaluations == 273 * 78


def test_matched_budget_includes_diagnostics_from_an_incomplete_population() -> None:
    populations = (
        {
            "diagnostics": {
                "complete": True,
                "proposed": 2,
                "simulated": 1,
                "failed_simulations": 1,
                "out_of_support": 0,
            }
        },
        {
            "diagnostics": {
                "complete": False,
                "proposed": 12,
                "simulated": 2,
                "failed_simulations": 3,
                "out_of_support": 4,
            }
        },
    )

    budget = derive_cascaded_tanks_matched_cost_budget(
        calibration_calls=10, population_records=populations, fit_length=204
    )

    assert tuple(item.complete for item in budget.populations) == (True, False)
    assert tuple(item.simulator_calls for item in budget.populations) == (2, 5)
    assert budget.population_calls == 7
    assert budget.max_calls == 17
    assert budget.reserved_step_evaluations == 17 * 204


def test_matched_budget_excludes_support_rejects_but_counts_failed_simulations() -> (
    None
):
    populations = (
        {
            "diagnostics": {
                "complete": False,
                "proposed": 15,
                "simulated": 2,
                "failed_simulations": 4,
                "out_of_support": 9,
            }
        },
    )

    budget = derive_cascaded_tanks_matched_cost_budget(
        calibration_calls=10, population_records=populations, fit_length=78
    )

    assert budget.populations[0].simulated == 2
    assert budget.populations[0].failed_simulations == 4
    assert budget.populations[0].out_of_support == 9
    assert budget.excluded_out_of_support_proposals == 9
    assert budget.populations[0].simulator_calls == 6
    assert budget.max_calls == 16  # proposed=15 and support rejects are not calls


def test_matched_budget_reserved_steps_use_the_78_or_204_fit_length() -> None:
    populations = (
        {
            "diagnostics": {
                "complete": True,
                "proposed": 15,
                "simulated": 3,
                "failed_simulations": 2,
                "out_of_support": 5,
            }
        },
    )
    short = derive_cascaded_tanks_matched_cost_budget(
        calibration_calls=256, population_records=populations, fit_length=78
    )
    long = derive_cascaded_tanks_matched_cost_budget(
        calibration_calls=256, population_records=populations, fit_length=204
    )

    assert short.max_calls == long.max_calls == 261
    assert short.reserved_step_evaluations == 261 * 78 == 20358
    assert long.reserved_step_evaluations == 261 * 204 == 53244

"""Synthetic known-parameter checks for the discrete cascaded-tanks maps."""

from __future__ import annotations

import math

import pytest

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


def test_s0_short_transitions_match_independent_hand_calculation_and_alignment() -> (
    None
):
    # With a=c=0.5, p=1, x=(4,4), and u=(1,0):
    # k=0 reads y=4, then x1'=4 and x2'=4+sqrt(4)-0.5*sqrt(4)=5.
    # k=1 reads that x2'=5 before applying u[1]; x1'=3 and
    # x2'=5+2-0.5*sqrt(5).
    outcome = simulate_cascaded_tanks(
        [1.0, 0.0],
        TankParameters(a=0.5, c=0.5, p=1.0),
        TankState(x1=4.0, x2=4.0),
    )

    assert isinstance(outcome, TankSimulationSuccess)
    assert outcome.sample_interval_seconds == 4.0
    assert outcome.observations == pytest.approx((4.0, 5.0))
    assert outcome.trace[0].q12 == pytest.approx(1.0)
    assert outcome.trace[0].q2 == pytest.approx(1.0)
    assert outcome.trace[0].x1_next == pytest.approx(4.0)
    assert outcome.trace[0].x2_raw == pytest.approx(5.0)
    assert outcome.trace[0].next_state == TankState(4.0, 5.0)
    assert outcome.trace[1].state == outcome.trace[0].next_state
    assert outcome.trace[1].x1_next == pytest.approx(3.0)
    assert outcome.trace[1].x2_raw == pytest.approx(5.0 + 2.0 - 0.5 * math.sqrt(5.0))
    assert outcome.terminal_state == outcome.trace[-1].next_state
    assert outcome.trace[0].x2_raw == pytest.approx(
        outcome.trace[0].state.x2
        + math.sqrt(outcome.trace[0].state.x1)
        - outcome.trace[0].q2
    )


def test_declared_dry_floor_sets_both_negative_state_candidates_to_zero() -> None:
    # At x=(1,1), a=2 and c=3 give x1 candidate 1-2=-1 and
    # x2 candidate 1+sqrt(1)-3=-1. The surrogate's declared dry floors
    # therefore set both next states to zero.
    outcome = simulate_cascaded_tanks(
        [0.0],
        TankParameters(a=2.0, c=3.0, p=1.0),
        TankState(x1=1.0, x2=1.0),
    )

    assert isinstance(outcome, TankSimulationSuccess)
    assert outcome.observations == (1.0,)
    assert 1.0 - 2.0 < 0.0
    assert 1.0 + math.sqrt(1.0) - 3.0 < 0.0
    assert outcome.trace[0].x1_next == 0.0
    assert outcome.trace[0].x2_raw == 0.0
    assert outcome.terminal_state == TankState(x1=0.0, x2=0.0)


@pytest.mark.parametrize("model", [TankModel.S0, TankModel.O2, TankModel.C2])
def test_each_family_reads_output_before_using_current_input(model: TankModel) -> None:
    kwargs = {"ceiling": 4.0} if model is not TankModel.S0 else {}
    outcome = simulate_cascaded_tanks(
        [100.0],
        TankParameters(a=0.5, c=0.5, p=1.0),
        TankState(x1=4.0, x2=3.0),
        model=model,
        **kwargs,
    )

    assert isinstance(outcome, TankSimulationSuccess)
    assert outcome.observations == (3.0,)
    assert outcome.trace[0].input_u == 100.0
    assert outcome.trace[0].next_state.x1 == pytest.approx(103.0)
    assert outcome.terminal_state == outcome.trace[0].next_state


def test_o2_state_overflow_and_c2_sensor_ceiling_have_different_memory() -> None:
    parameters = TankParameters(a=0.5, c=0.5, p=1.0)
    initial = TankState(x1=4.0, x2=4.0)
    inputs = [0.0] * 10

    o2 = simulate_cascaded_tanks(
        inputs, parameters, initial, model=TankModel.O2, ceiling=4.0
    )
    c2 = simulate_cascaded_tanks(
        inputs, parameters, initial, model=TankModel.C2, ceiling=4.0
    )

    assert isinstance(o2, TankSimulationSuccess)
    assert isinstance(c2, TankSimulationSuccess)
    assert all(step.state.x2 <= 4.0 for step in o2.trace)
    assert all(step.next_state.x2 <= 4.0 for step in o2.trace)
    assert any(step.x2_raw > 4.0 for step in o2.trace)
    assert any(step.next_state.x2 > 4.0 for step in c2.trace)
    assert all(step.observation_y <= 4.0 for step in c2.trace)
    assert c2.terminal_state.x2 > o2.terminal_state.x2


def test_o2_and_c2_observations_match_when_lower_state_stays_below_ceiling() -> None:
    parameters = TankParameters(a=0.5, c=0.5, p=1.0)
    initial = TankState(x1=0.0, x2=1.0)
    inputs = [0.0] * 12

    o2 = simulate_cascaded_tanks(
        inputs, parameters, initial, model=TankModel.O2, ceiling=2.0
    )
    c2 = simulate_cascaded_tanks(
        inputs, parameters, initial, model=TankModel.C2, ceiling=2.0
    )

    assert isinstance(o2, TankSimulationSuccess)
    assert isinstance(c2, TankSimulationSuccess)
    assert all(step.x2_raw <= 2.0 for step in o2.trace)
    assert o2.observations == c2.observations
    assert o2.terminal_state == c2.terminal_state


def test_o2_and_c2_recovery_outputs_separate_after_a_ceiling_crossing() -> None:
    # u[0] raises x1 from zero to 100 without adding lower-state inflow yet.
    # On the following low-input step, x1 drains to zero while its fixed-unit
    # sqrt(x1) inflow raises x2_raw above H=2. O2 discards the excess; C2 keeps
    # it hidden. O2 therefore recovers below H one sample before C2 does.
    parameters = TankParameters(a=10.0, c=0.5, p=1.0)
    initial = TankState(x1=0.0, x2=1.0)
    inputs = [100.0, 0.0, 0.0, 0.0]

    o2 = simulate_cascaded_tanks(
        inputs, parameters, initial, model=TankModel.O2, ceiling=2.0
    )
    c2 = simulate_cascaded_tanks(
        inputs, parameters, initial, model=TankModel.C2, ceiling=2.0
    )

    assert isinstance(o2, TankSimulationSuccess)
    assert isinstance(c2, TankSimulationSuccess)
    assert o2.trace[1].x2_raw == pytest.approx(10.5 - 0.5 * math.sqrt(0.5))
    assert o2.trace[1].next_state.x2 == 2.0
    assert c2.trace[1].next_state.x2 == pytest.approx(o2.trace[1].x2_raw)
    assert o2.observations[:3] == pytest.approx(c2.observations[:3])
    assert o2.observations[3] == pytest.approx(2.0 - 0.5 * math.sqrt(2.0))
    assert c2.observations[3] == 2.0
    assert o2.observations[3] < c2.observations[3]


def test_long_replay_is_deterministic_and_carries_terminal_state() -> None:
    inputs = tuple((index % 9) / 8.0 for index in range(4_000))
    parameters = TankParameters(a=0.04, c=0.1, p=0.03)
    initial = TankState(x1=1.5, x2=0.25)

    first = simulate_cascaded_tanks(inputs, parameters, initial)
    second = simulate_cascaded_tanks(inputs, parameters, initial)

    assert isinstance(first, TankSimulationSuccess)
    assert first == second
    assert len(first.trace) == len(inputs)
    assert first.trace[0].state == initial
    assert first.trace[-1].next_state == first.terminal_state
    assert all(step.state.x1 >= 0.0 and step.state.x2 >= 0.0 for step in first.trace)


@pytest.mark.parametrize(
    ("inputs", "parameters", "initial", "kwargs", "limits", "category", "step_index"),
    [
        (
            [-1.0],
            TankParameters(1.0, 1.0, 1.0),
            TankState(1.0, 1.0),
            {},
            TankSimulationLimits(),
            TankFailureCategory.INVALID_INPUT,
            0,
        ),
        (
            [math.nan],
            TankParameters(1.0, 1.0, 1.0),
            TankState(1.0, 1.0),
            {},
            TankSimulationLimits(),
            TankFailureCategory.INVALID_INPUT,
            0,
        ),
        (
            [0.0],
            TankParameters(0.0, 1.0, 1.0),
            TankState(1.0, 1.0),
            {},
            TankSimulationLimits(),
            TankFailureCategory.INVALID_PARAMETER,
            None,
        ),
        (
            [0.0],
            TankParameters(1.0, 1.0, 1.0),
            TankState(-1.0, 1.0),
            {},
            TankSimulationLimits(),
            TankFailureCategory.INVALID_INITIAL_STATE,
            None,
        ),
        (
            [0.0, 0.0],
            TankParameters(1.0, 1.0, 1.0),
            TankState(1.0, 1.0),
            {},
            TankSimulationLimits(max_steps=1),
            TankFailureCategory.TOO_MANY_STEPS,
            1,
        ),
        (
            [1.0e308],
            TankParameters(1.0, 1.0, 1.0e308),
            TankState(0.0, 0.0),
            {},
            TankSimulationLimits(max_magnitude=1.0e308),
            TankFailureCategory.NON_FINITE,
            0,
        ),
        (
            [10.0],
            TankParameters(1.0, 1.0, 10.0),
            TankState(0.0, 0.0),
            {},
            TankSimulationLimits(max_magnitude=50.0),
            TankFailureCategory.MAGNITUDE_LIMIT,
            0,
        ),
    ],
)
def test_invalid_and_numerical_failures_are_categorized(
    inputs: list[float],
    parameters: TankParameters,
    initial: TankState,
    kwargs: dict[str, object],
    limits: TankSimulationLimits,
    category: TankFailureCategory,
    step_index: int | None,
) -> None:
    outcome = simulate_cascaded_tanks(
        inputs, parameters, initial, limits=limits, **kwargs
    )

    assert isinstance(outcome, TankSimulationFailure)
    assert outcome.category is category
    assert outcome.step_index == step_index


def test_magnitude_failure_returns_completed_prefix_and_boundary_state() -> None:
    outcome = simulate_cascaded_tanks(
        [0.0, 10.0],
        TankParameters(a=1.0, c=1.0, p=10.0),
        TankState(x1=1.0, x2=1.0),
        limits=TankSimulationLimits(max_magnitude=50.0),
    )

    assert isinstance(outcome, TankSimulationFailure)
    assert outcome.category is TankFailureCategory.MAGNITUDE_LIMIT
    assert outcome.step_index == 1
    assert len(outcome.trace) == 1
    assert outcome.terminal_state == outcome.trace[0].next_state


def test_o2_cannot_hide_a_pre_cap_magnitude_crossing() -> None:
    outcome = simulate_cascaded_tanks(
        [0.0],
        TankParameters(a=0.001, c=0.001, p=0.1),
        TankState(x1=20.0, x2=20.0),
        model=TankModel.O2,
        ceiling=20.0,
        limits=TankSimulationLimits(max_magnitude=20.0),
    )

    assert isinstance(outcome, TankSimulationFailure)
    assert outcome.category is TankFailureCategory.MAGNITUDE_LIMIT
    assert outcome.step_index == 0
    assert outcome.trace == ()


def test_model_configuration_requires_the_declared_ceiling_only_for_o2_and_c2() -> None:
    parameters = TankParameters(a=0.5, c=0.5, p=1.0)
    initial = TankState(x1=1.0, x2=1.0)

    missing = simulate_cascaded_tanks([], parameters, initial, model=TankModel.C2)
    extraneous = simulate_cascaded_tanks(
        [], parameters, initial, model=TankModel.S0, ceiling=3.0
    )
    invalid_o2_start = simulate_cascaded_tanks(
        [], parameters, initial, model=TankModel.O2, ceiling=0.5
    )

    assert isinstance(missing, TankSimulationFailure)
    assert missing.category is TankFailureCategory.INVALID_CONFIGURATION
    assert isinstance(extraneous, TankSimulationFailure)
    assert extraneous.category is TankFailureCategory.INVALID_CONFIGURATION
    assert isinstance(invalid_o2_start, TankSimulationFailure)
    assert invalid_o2_start.category is TankFailureCategory.INVALID_INITIAL_STATE


def test_numeric_strings_are_not_accepted_as_synthetic_input() -> None:
    outcome = simulate_cascaded_tanks(
        ["1.0"],  # type: ignore[list-item]
        TankParameters(a=0.5, c=0.5, p=1.0),
        TankState(x1=1.0, x2=1.0),
    )

    assert isinstance(outcome, TankSimulationFailure)
    assert outcome.category is TankFailureCategory.INVALID_INPUT
    assert outcome.step_index == 0

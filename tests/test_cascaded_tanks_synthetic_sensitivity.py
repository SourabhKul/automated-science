"""Focused controls for the reusable synthetic sensitivity computation."""

from __future__ import annotations

import numpy as np

from core.real_data.cascaded_tanks_models import (
    TankFailureCategory,
    TankModel,
    TankSimulationFailure,
    simulate_cascaded_tanks,
)
from core.real_data.cascaded_tanks_synthetic_sensitivity import (
    PROTOCOL_CASES,
    SENSITIVITY_STEPS,
    SensitivityCase,
    run_cascaded_tanks_synthetic_sensitivity,
)


def _small_crossing_case() -> SensitivityCase:
    # Deliberately short, unrelated fixture. It is not the frozen N/S/L roster.
    return SensitivityCase(
        case_id="tiny-crossing-fixture",
        inputs=(8.0,) + (0.0,) * 14,
        ceiling=2.0,
        ceiling_bounds=(1.5, 2.5),
    )


def test_protocol_roster_is_exact_without_running_its_simulations() -> None:
    assert tuple(case.case_id for case in PROTOCOL_CASES) == ("N", "S", "L")
    assert tuple(len(case.inputs) for case in PROTOCOL_CASES) == (204, 78, 204)
    assert all(case.inputs[:24] == (8.0,) * 24 for case in PROTOCOL_CASES)
    assert PROTOCOL_CASES[0].inputs[24:] == (0.0,) * 180
    assert PROTOCOL_CASES[1].inputs[24:] == (0.0,) * 54
    assert PROTOCOL_CASES[2].inputs[24:] == (0.0,) * 180
    assert tuple(case.ceiling for case in PROTOCOL_CASES) == (1000.0, 3.0, 3.0)
    assert tuple(case.ceiling_bounds for case in PROTOCOL_CASES) == (
        (900.0, 1100.0),
        (2.5, 3.5),
        (2.5, 3.5),
    )
    assert all(case.input_array.dtype == np.float64 for case in PROTOCOL_CASES)
    assert tuple(case.input_array.size for case in PROTOCOL_CASES) == (204, 78, 204)


def test_small_fixture_returns_full_jacobians_metrics_and_exact_divergence() -> None:
    simulator_calls = []

    def recording_simulator(inputs, parameters, initial_state, **kwargs):
        simulator_calls.append((kwargs["model"], kwargs["ceiling"], kwargs["limits"]))
        return simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)

    case = _small_crossing_case()
    result = run_cascaded_tanks_synthetic_sensitivity(
        (case,), simulator=recording_simulator
    )

    assert result.case_ids == (case.case_id,)
    assert len(result.family_cases) == 3
    assert [item.model for item in result.family_cases] == [
        TankModel.S0,
        TankModel.O2,
        TankModel.C2,
    ]
    s0 = result.family_case(TankModel.S0, case.case_id)
    o2 = result.family_case(TankModel.O2, case.case_id)
    c2 = result.family_case(TankModel.C2, case.case_id)
    assert s0.available and o2.available and c2.available
    assert s0.parameter_names == ("a", "c", "p", "x1_0", "x2_0")
    assert o2.parameter_names == s0.parameter_names + ("H",)
    assert s0.truth_coordinates == (0.5,) * 5
    assert o2.truth_coordinates == (0.5,) * 6
    assert s0.truth.observations is not None
    assert s0.truth.observations.dtype == np.float64
    assert len(s0.steps) == 3
    assert tuple(step.h for step in s0.steps) == SENSITIVITY_STEPS
    for step in s0.steps:
        assert step.jacobian.shape == (len(case.inputs), 5)
        assert step.jacobian.dtype == np.float64
        assert step.scaled_matrix is not None
        expected_scaled = step.jacobian / (0.05 * np.sqrt(len(case.inputs)))
        assert np.array_equal(step.scaled_matrix, expected_scaled)
        expected_singular_values = np.linalg.svd(expected_scaled, compute_uv=False)
        assert np.allclose(step.singular_values, expected_singular_values)
        assert step.relative_rank_1e_3 == int(
            np.count_nonzero(
                expected_singular_values / expected_singular_values[0] > 1e-3
            )
        )
        assert step.relative_rank_1e_6 == int(
            np.count_nonzero(
                expected_singular_values / expected_singular_values[0] > 1e-6
            )
        )

    expected_metric = np.max(
        np.abs(s0.steps[0].jacobian[:, 0] - s0.steps[1].jacobian[:, 0])
    ) / max(
        1.0,
        float(np.max(np.abs(s0.steps[0].jacobian[:, 0]))),
        float(np.max(np.abs(s0.steps[1].jacobian[:, 0]))),
    )
    assert s0.comparisons[0].parameter_name == "a"
    assert s0.comparisons[0].metric == expected_metric

    divergence = result.case_divergence(case.case_id)
    o2_c2 = divergence.o2_c2
    expected_o2_c2_diff = o2.truth.observations - c2.truth.observations
    expected_indices = np.flatnonzero(o2.truth.observations != c2.truth.observations)
    assert len(expected_indices) > 0
    assert o2_c2.first_differing_index == int(expected_indices[0])
    assert o2_c2.differing_count == len(expected_indices)
    assert not o2_c2.byte_identical
    assert o2_c2.maximum_absolute_difference is not None
    assert o2_c2.maximum_absolute_difference == float(
        np.max(np.abs(expected_o2_c2_diff))
    )
    assert o2_c2.signed_difference is not None
    assert o2_c2.signed_difference.dtype == np.float64
    assert dict(divergence.capped_output_counts) == {
        TankModel.O2: int(np.count_nonzero(o2.truth.observations == case.ceiling)),
        TankModel.C2: int(np.count_nonzero(c2.truth.observations == case.ceiling)),
    }

    # One truth and 2 * coordinates * 3 step sizes per model; all invocations
    # use exact trace-length limits, 1e6 magnitude, and omit S0's ceiling.
    assert len(simulator_calls) == 3 + 2 * (5 + 6 + 6) * 3
    assert all(call[2].max_magnitude == 1.0e6 for call in simulator_calls)
    assert all(call[2].max_steps == len(case.inputs) for call in simulator_calls)
    assert all(
        ceiling is None
        for model, ceiling, _ in simulator_calls
        if model is TankModel.S0
    )
    assert (
        next(ceiling for model, ceiling, _ in simulator_calls if model is TankModel.O2)
        == case.ceiling
    )
    assert (
        next(ceiling for model, ceiling, _ in simulator_calls if model is TankModel.C2)
        == case.ceiling
    )
    assert all(
        case.ceiling_bounds[0] <= ceiling <= case.ceiling_bounds[1]
        for model, ceiling, _ in simulator_calls
        if model is not TankModel.S0
    )

    compact = result.compact_summary()
    assert compact["case_ids"] == [case.case_id]
    assert compact["model_order"] == ["S0", "O2", "C2"]
    assert compact["divergence"][0]["pairwise"][2]["first_differing_index"] == int(
        expected_indices[0]
    )


def test_failed_plus_perturbation_invalidates_only_its_column_and_svd() -> None:
    case = SensitivityCase(
        case_id="tiny-failure-fixture",
        inputs=(0.0, 0.25, 0.0),
        ceiling=2.0,
        ceiling_bounds=(1.5, 2.5),
    )

    def fail_positive_x1(inputs, parameters, initial_state, **kwargs):
        if kwargs["model"] is TankModel.S0 and initial_state.x1 > 0.5:
            return TankSimulationFailure(
                TankFailureCategory.INVALID_INITIAL_STATE,
                "injected plus-side failure",
                None,
                (),
                initial_state,
            )
        return simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)

    result = run_cascaded_tanks_synthetic_sensitivity(
        (case,), simulator=fail_positive_x1
    )
    s0 = result.family_case(TankModel.S0, case.case_id)
    assert s0.available
    assert s0.steps
    for step in s0.steps:
        assert step.column_validity == (True, True, True, False, True)
        assert np.all(np.isfinite(step.jacobian[:, [0, 1, 2, 4]]))
        assert np.all(np.isnan(step.jacobian[:, 3]))
        assert step.scaled_matrix is None
        assert step.singular_values is None
        assert step.relative_rank_1e_3 is None
        assert step.relative_rank_1e_6 is None
        assert step.column_norms[3] is None
        assert all(value is None for value in step.absolute_column_cosines[3])
        assert (
            step.coordinate_attempts[3].plus.failure_category == "invalid_initial_state"
        )
        assert step.coordinate_attempts[3].minus.succeeded
    x1_comparisons = [item for item in s0.comparisons if item.parameter_name == "x1_0"]
    assert len(x1_comparisons) == 2
    assert all(item.metric is None for item in x1_comparisons)
    assert len(s0.steps[0].coordinate_attempts[0].plus.observations) == len(case.inputs)


def test_zero_norm_column_uses_null_cosine() -> None:
    case = SensitivityCase(
        case_id="tiny-zero-column-fixture",
        inputs=(0.0, 0.0, 0.0),
        ceiling=100.0,
        ceiling_bounds=(90.0, 110.0),
    )
    result = run_cascaded_tanks_synthetic_sensitivity((case,))
    s0 = result.family_case(TankModel.S0, case.case_id)
    o2 = result.family_case(TankModel.O2, case.case_id)
    c2 = result.family_case(TankModel.C2, case.case_id)
    step = s0.steps[0]
    p_column = s0.parameter_names.index("p")
    assert np.array_equal(step.jacobian[:, p_column], np.zeros(3, dtype=np.float64))
    assert step.column_norms[p_column] == 0.0
    assert step.absolute_column_cosines[p_column][p_column] is None
    assert step.absolute_column_cosines[0][p_column] is None

    divergence = result.case_divergence(case.case_id)
    assert all(item.byte_identical for item in divergence.pairwise)
    assert dict(divergence.capped_output_counts) == {
        TankModel.O2: 0,
        TankModel.C2: 0,
    }
    for model_result in (o2, c2):
        h_column = model_result.parameter_names.index("H")
        for model_step in model_result.steps:
            assert np.array_equal(
                model_step.jacobian[:, h_column], np.zeros(3, dtype=np.float64)
            )

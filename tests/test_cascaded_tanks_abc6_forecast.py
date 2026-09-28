"""Synthetic tests for source-free, target-free ABC6 forecasting."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from core.real_data.cascaded_tanks_abc6_cases import (
    PARAMETER_ORDER,
    build_synthetic_training_bundle,
)
from core.real_data.cascaded_tanks_abc6_forecast import (
    PROSPECTIVE_INPUT,
    forecast_abc6_posterior_and_baseline,
    weighted_left_inverse_quantile,
)
from core.real_data.cascaded_tanks_models import (
    TankFailureCategory,
    TankSimulationFailure,
    simulate_cascaded_tanks,
)


def _result(case_data, rows, weights, *, baseline_values=None, abc_status="complete"):
    case = case_data.case
    columns = {
        name: np.asarray([row[index] for row in rows], dtype=float)
        for index, name in enumerate(PARAMETER_ORDER)
    }
    baseline = None
    if baseline_values is not None:
        bounds = np.asarray(case.prior_bounds, dtype=float)
        coordinates = tuple(
            (baseline_values[index] - bounds[index, 0])
            / (bounds[index, 1] - bounds[index, 0])
            for index in range(6)
        )
        best = SimpleNamespace(succeeded=True, coordinates=coordinates)
        baseline = SimpleNamespace(best_evaluation=best)
    posterior = {
        "parameter_order": tuple(PARAMETER_ORDER),
        "free_parameter_values": columns,
        "weights": np.asarray(weights, dtype=float),
    }
    return SimpleNamespace(
        case_index=case.case_index,
        case_id=case.case_id,
        model=case.fit_model.value,
        fit_length=case.input_length,
        abc_status=abc_status,
        abc_result={"complete": abc_status == "complete", "posterior": posterior},
        baseline_result=baseline,
    )


def _values(a=0.50, c=0.40, p=0.50, x1=0.50, x2=0.50, ceiling=3.0):
    return (a, c, p, x1, x2, ceiling)


def test_short_and_long_reconstruction_reach_the_same_common_state_and_forecast():
    bundle = build_synthetic_training_bundle()
    short_data = bundle.data_for_case(0)
    long_data = bundle.data_for_case(2)
    fitted_values = _values()
    short_result = _result(short_data, [fitted_values], [1.0], baseline_values=fitted_values)
    long_result = _result(long_data, [fitted_values], [1.0], baseline_values=fitted_values)

    short = forecast_abc6_posterior_and_baseline(short_data, short_result)
    long = forecast_abc6_posterior_and_baseline(long_data, long_result)

    assert short.status == long.status == "complete"
    assert short.common_state_index == long.common_state_index == 204
    assert short.particles[0].common_time_state == long.particles[0].common_time_state
    assert short.particles[0].trajectory == long.particles[0].trajectory
    assert short.baseline_common_time_state == long.baseline_common_time_state
    assert short.baseline_trajectory == long.baseline_trajectory
    assert len(short.particles[0].trajectory) == len(PROSPECTIVE_INPUT) == 60


def test_mutating_training_outputs_cannot_change_forecast_or_particle_weights():
    bundle = build_synthetic_training_bundle()
    altered_bundle = build_synthetic_training_bundle(
        training_output_mutator=lambda truth_id, outputs: (
            outputs[:78]
            + tuple(value + 500.0 + index for index, value in enumerate(outputs[78:]))
            if truth_id == "A"
            else outputs
        )
    )
    data = bundle.data_for_case(0)
    hidden_suffix_changed = altered_bundle.data_for_case(0)
    assert hidden_suffix_changed.observed_outputs == data.observed_outputs
    assert (
        altered_bundle.data_for_case(2).observed_outputs
        != bundle.data_for_case(2).observed_outputs
    )
    changed = replace(
        data,
        observed_outputs=tuple(value + 1_000_000.0 for value in data.observed_outputs),
    )
    rows = [_values(a=0.48), _values(a=0.53)]
    result = _result(data, rows, [0.35, 0.65], baseline_values=_values(a=0.49))

    ordinary = forecast_abc6_posterior_and_baseline(data, result)
    suffix_mutated = forecast_abc6_posterior_and_baseline(
        hidden_suffix_changed, result
    )
    mutated = forecast_abc6_posterior_and_baseline(changed, result)

    assert ordinary == suffix_mutated
    assert ordinary == mutated
    assert ordinary.weights == (0.35, 0.65)


def test_weighted_quantile_uses_the_left_inverse_empirical_cdf():
    values = (1.0, 2.0, 3.0)
    weights = (0.25, 0.25, 0.50)

    # The cumulative mass reaches exactly 0.50 at value 2; the left inverse
    # returns that value instead of an interpolated midpoint.
    assert weighted_left_inverse_quantile(values, weights, 0.50) == 2.0
    assert weighted_left_inverse_quantile(values, weights, 0.05) == 1.0
    assert weighted_left_inverse_quantile(values, weights, 0.95) == 3.0


def test_one_failed_particle_suppresses_all_abc_summaries_without_reweighting():
    bundle = build_synthetic_training_bundle()
    data = bundle.data_for_case(0)
    rows = [_values(a=0.41), _values(a=0.53)]
    weights = [0.2, 0.8]
    result = _result(data, rows, weights, baseline_values=_values(a=0.49))

    def one_particle_fails(inputs, parameters, initial_state, **kwargs):
        if parameters.a == 0.41:
            return TankSimulationFailure(
                category=TankFailureCategory.MAGNITUDE_LIMIT,
                message="synthetic particle failure",
                step_index=0,
                trace=(),
                terminal_state=None,
            )
        return simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)

    forecast = forecast_abc6_posterior_and_baseline(
        data, result, simulator=one_particle_fails
    )

    assert forecast.aggregate_status == "suppressed_particle_failure"
    assert forecast.pointwise_weighted_mean is None
    assert forecast.pointwise_weighted_median is None
    assert forecast.pointwise_q05 is None
    assert forecast.pointwise_q95 is None
    assert forecast.weights == (0.2, 0.8)
    assert forecast.particles[0].trajectory is None
    assert forecast.particles[0].failure is not None
    assert forecast.particles[1].trajectory is not None
    assert forecast.baseline_trajectory is not None


def test_baseline_failure_does_not_suppress_complete_abc_aggregate():
    bundle = build_synthetic_training_bundle()
    data = bundle.data_for_case(0)
    rows = [_values(a=0.41), _values(a=0.53)]
    result = _result(
        data, rows, [0.2, 0.8], baseline_values=_values(a=0.49)
    )

    def baseline_fails(inputs, parameters, initial_state, **kwargs):
        if abs(parameters.a - 0.49) < 1e-12:
            return TankSimulationFailure(
                category=TankFailureCategory.NON_FINITE,
                message="synthetic baseline failure",
                step_index=0,
                trace=(),
                terminal_state=None,
            )
        return simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)

    forecast = forecast_abc6_posterior_and_baseline(
        data, result, simulator=baseline_fails
    )

    assert forecast.aggregate_status == "complete"
    assert forecast.pointwise_weighted_mean is not None
    assert forecast.baseline_trajectory is None
    assert forecast.baseline_failure is not None
    assert forecast.baseline_failure.category == "non_finite"


def test_n_case_abstains_without_calling_simulator_or_returning_forecasts():
    bundle = build_synthetic_training_bundle()
    data = bundle.data_for_case(8)
    result = _result(data, [_values(ceiling=1000.0)], [1.0])
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("N abstention must not simulate a prospective forecast")

    forecast = forecast_abc6_posterior_and_baseline(
        data, result, simulator=forbidden_simulator
    )

    assert forecast.status == "abstained_n"
    assert forecast.aggregate_status == "unavailable"
    assert forecast.prospective_inputs is None
    assert forecast.particles == ()
    assert forecast.baseline_trajectory is None
    assert calls == []


def test_abc_pointwise_summaries_are_labelled_separately_from_trajectories():
    bundle = build_synthetic_training_bundle()
    data = bundle.data_for_case(0)
    rows = [_values(a=0.45), _values(a=0.55)]
    result = _result(data, rows, [0.5, 0.5], baseline_values=_values(a=0.50))

    forecast = forecast_abc6_posterior_and_baseline(data, result)

    assert forecast.aggregate_status == "complete"
    assert len(forecast.particle_trajectories) == 2
    assert all(path is not None and len(path) == 60 for path in forecast.particle_trajectories)
    assert forecast.pointwise_summaries_are_coherent_trajectories is False
    assert forecast.effective_sample_size == 2.0
    assert forecast.baseline_trajectory is not None
    assert forecast.baseline_trajectory not in forecast.particle_trajectories

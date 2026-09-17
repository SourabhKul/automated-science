"""Focused tests for the bounded numerical ODE reference control.

This file intentionally runs with the system Python and NumPy only.  The
target-side noncentral-chi-square calculation is repeated here in a separate
implementation so the acceptance test does not merely compare a helper with
itself.  The stochastic ABC check is deliberately small; the six-cell pilot
has a separate bounded driver and is not run by the test suite.
"""

from __future__ import annotations

from math import erf, exp, lgamma, log, sqrt
import time

import numpy as np

from core.abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference
from core.ode_reference_controls import (
    DecayObservationSimulator,
    ODESolverConfig,
    ODESolverFailure,
    finite_epsilon_ode_target,
    gaussian_likelihood_ode_target,
    generate_decay_observation,
    integrate_decay_ode,
    noncentral_chi_square_cdf,
    run_ode_decay_control_cell,
    solver_accuracy_check,
    weighted_decay_prediction_summary,
)


def _independent_central_df4(x: float, index: int) -> float:
    # Independent implementation of the Gamma(shape=df/2, scale=2) CDF.
    # The scale conversion to x/2 is intentionally written separately from
    # the production helper.
    value = float(x) / 2.0
    shape = int(index) + 2
    if shape <= value:
        term = 1.0
        total = 1.0
        for m in range(1, shape):
            term *= value / m
            total += term
        return 1.0 - np.exp(-value) * total
    term = np.exp(-value + shape * log(value) - lgamma(shape + 1.0))
    total = term
    m = shape
    for _ in range(10_000):
        m += 1
        term *= value / m
        total += term
        if term <= total * 1e-16 or term == 0.0:
            break
    return total


def _independent_ncx2_df4(x: float, noncentrality: float) -> float:
    a = float(noncentrality) / 2.0
    if a == 0.0:
        return _independent_central_df4(x, 0)
    maximum = int(np.ceil(a + 12.0 * np.sqrt(a + 1.0) + 50.0))
    total = 0.0
    poisson = np.exp(-a)
    for index in range(maximum + 1):
        total += poisson * _independent_central_df4(x, index)
        poisson *= a / (index + 1.0)
    return float(total)


def _independent_finite_target_density(
    rates: np.ndarray,
    observed: np.ndarray,
    time_points: np.ndarray,
    sigma: float,
    epsilon: float,
) -> np.ndarray:
    means = np.exp(-rates[:, None] * time_points[None, :])
    noncentralities = np.sum(((means - observed[None, :]) / sigma) ** 2, axis=1)
    return np.asarray(
        [_independent_ncx2_df4(epsilon * epsilon, float(value)) for value in noncentralities],
        dtype=float,
    )


def _independent_normal_cdf(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray(
        [0.5 * (1.0 + erf(float(value) / sqrt(2.0))) for value in array.flat],
        dtype=float,
    ).reshape(array.shape)


def _independent_density_weighted_observation_cdf(
    future_time: float,
    value: float,
    observed: np.ndarray,
    time_points: np.ndarray,
    sigma: float,
    epsilon: float,
    bounds: tuple[float, float] = (0.05, 1.5),
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(192)
    lower, upper = bounds
    rates = 0.5 * (upper - lower) * (nodes + 1.0) + lower
    density = _independent_finite_target_density(rates, observed, time_points, sigma, epsilon)
    scale = 0.5 * (upper - lower)
    numerator = scale * np.sum(
        weights
        * density
        * _independent_normal_cdf((value - np.exp(-rates * future_time)) / sigma)
    )
    denominator = scale * np.sum(weights * density)
    return float(numerator / denominator)


def _independent_density_weighted_observation_quantile(
    future_time: float,
    probability: float,
    observed: np.ndarray,
    time_points: np.ndarray,
    sigma: float,
    epsilon: float,
    bounds: tuple[float, float] = (0.05, 1.5),
) -> float:
    lower, upper = bounds
    low = exp(-upper * future_time) - 10.0 * sigma
    high = exp(-lower * future_time) + 10.0 * sigma
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        cdf = _independent_density_weighted_observation_cdf(
            future_time,
            midpoint,
            observed,
            time_points,
            sigma,
            epsilon,
            bounds,
        )
        if cdf < probability:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def test_solver_accuracy_is_below_measurement_scale_and_preserves_origin():
    check = solver_accuracy_check()
    assert check["passed"] is True
    assert check["max_abs_error"] < 1e-6
    one_future = integrate_decay_ode(0.8, [4.0])
    full = integrate_decay_ode(0.8, [0.0, 4.0, 5.0])
    assert one_future.shape == (1,)
    assert np.allclose(one_future[0], full[1], atol=1e-10)
    assert np.allclose(one_future[0], np.exp(-0.8 * 4.0), atol=1e-6)


def test_noncentral_chi_square_target_is_independent_poisson_quadrature():
    for x in (4.0, 9.0):
        for noncentrality in (0.0, 0.5, 8.0, 40.0):
            expected = _independent_ncx2_df4(x, noncentrality)
            actual = noncentral_chi_square_cdf(x, noncentrality=noncentrality)
            assert np.allclose(actual, expected, rtol=2e-12, atol=2e-14)


def test_noncentral_chi_square_matches_scipy_when_available():
    """Regression against an independent reference implementation.

    SciPy is intentionally optional for the dependency-light repository test
    command.  The same test is exercised by the review command in an isolated
    environment with SciPy installed.
    """

    try:
        from scipy.stats import ncx2  # type: ignore
    except ModuleNotFoundError:
        return
    for threshold in (4.0, 9.0):
        for noncentrality in (0.0, 0.5, 8.0, 40.0):
            expected = float(ncx2.cdf(threshold, 4, noncentrality))
            actual = float(noncentral_chi_square_cdf(threshold, noncentrality=noncentrality))
            assert np.allclose(actual, expected, rtol=2e-11, atol=2e-13)


def test_finite_epsilon_target_matches_independent_quadrature():
    dataset = generate_decay_observation(0.35, 1101)
    target = finite_epsilon_ode_target(
        dataset.observed,
        dataset.time_points,
        sigma=dataset.sigma,
        epsilon=2.0,
        quadrature_order=96,
    )
    nodes, weights = np.polynomial.legendre.leggauss(192)
    low, high = 0.05, 1.5
    grid = 0.5 * (high - low) * (nodes + 1.0) + low
    means = np.exp(-grid[:, None] * dataset.time_points[None, :])
    lam = np.sum(((means - dataset.observed[None, :]) / dataset.sigma) ** 2, axis=1)
    independent_density = np.asarray([_independent_ncx2_df4(4.0, value) for value in lam])
    independent_normalizer = 0.5 * (high - low) * np.sum(weights * independent_density)
    assert np.allclose(target.normalizer, independent_normalizer, rtol=1e-10, atol=1e-12)
    for point in (0.1, 0.35, 0.8, 1.4):
        means_at_point = np.exp(-point * dataset.time_points)
        lam_at_point = np.sum(((means_at_point - dataset.observed) / dataset.sigma) ** 2)
        assert np.allclose(
            target.density(point),
            _independent_ncx2_df4(4.0, lam_at_point),
            rtol=2e-12,
            atol=2e-14,
        )


def test_targets_enforce_support_and_finite_epsilon_differs_from_exact_likelihood():
    dataset = generate_decay_observation(0.35, 1101)
    finite_target = finite_epsilon_ode_target(
        dataset.observed,
        dataset.time_points,
        sigma=dataset.sigma,
        epsilon=2.0,
        quadrature_order=64,
    )
    exact_target = gaussian_likelihood_ode_target(
        dataset.observed,
        dataset.time_points,
        sigma=dataset.sigma,
        quadrature_order=64,
    )
    for target in (finite_target, exact_target):
        values = np.asarray(target.density([-0.1, 0.05, 1.5, 1.6]))
        assert values[0] == 0.0
        assert values[3] == 0.0
        assert values[1] > 0.0
        assert values[2] > 0.0
    assert finite_target._target_kind == "finite_epsilon_abc"
    assert exact_target._target_kind == "exact_gaussian_likelihood"
    point = 0.35
    lam = np.sum(((np.exp(-point * dataset.time_points) - dataset.observed) / dataset.sigma) ** 2)
    assert np.allclose(exact_target.density(point), np.exp(-0.5 * lam), rtol=1e-12, atol=1e-15)
    assert not np.isclose(finite_target.mean, exact_target.mean, atol=1e-5)
    assert not np.allclose(
        finite_target.quantiles([0.1, 0.5, 0.9]),
        exact_target.quantiles([0.1, 0.5, 0.9]),
        atol=1e-5,
    )


def test_target_observation_predictive_uses_density_weighted_quadrature():
    dataset = generate_decay_observation(0.35, 1101)
    target = finite_epsilon_ode_target(
        dataset.observed,
        dataset.time_points,
        sigma=dataset.sigma,
        epsilon=2.0,
        quadrature_order=96,
    )
    future_time = 4.0
    for value in (0.18, 0.24, 0.30, 0.36):
        expected = _independent_density_weighted_observation_cdf(
            future_time,
            value,
            dataset.observed,
            dataset.time_points,
            dataset.sigma,
            2.0,
        )
        actual = target._observation_predictive_cdf(future_time, value)
        assert np.allclose(actual, expected, rtol=2e-6, atol=2e-7)
    for probability in (0.1, 0.5, 0.9):
        expected = _independent_density_weighted_observation_quantile(
            future_time,
            probability,
            dataset.observed,
            dataset.time_points,
            dataset.sigma,
            2.0,
        )
        actual = target._observation_predictive_quantile(future_time, probability)
        assert np.allclose(actual, expected, rtol=2e-6, atol=2e-7)

    summary = target.prediction_summary([future_time])
    assert "no measurement noise" in summary["semantics"]["latent"]
    assert "independent Normal" in summary["semantics"]["observation_predictive"]
    assert summary["observation_predictive_quantiles"].shape == (1, 3)


def test_simulator_noise_stream_is_fresh_and_separate_from_proposals():
    dataset = generate_decay_observation(0.35, 1102)
    first = DecayObservationSimulator(dataset.time_points, simulator_noise_seed=3101)
    second = DecayObservationSimulator(dataset.time_points, simulator_noise_seed=3101)
    proposal_rng_a = np.random.default_rng(7)
    proposal_rng_baseline = np.random.default_rng(7)
    proposal_rng_b = np.random.default_rng(71)
    a1 = first(np.array([0.35]), proposal_rng_a)
    b1 = second(np.array([0.35]), proposal_rng_b)
    a2 = first(np.array([0.35]), proposal_rng_a)
    b2 = second(np.array([0.35]), proposal_rng_b)
    assert np.array_equal(a1, b1)
    assert np.array_equal(a2, b2)
    assert not np.array_equal(a1, a2)
    # The simulator owns its noise stream and must not consume the proposal
    # generator passed by the ABC reference.
    assert np.array_equal(proposal_rng_a.random(4), proposal_rng_baseline.random(4))
    assert not np.array_equal(proposal_rng_a.random(4), proposal_rng_b.random(4))
    assert first.diagnostics()["noise_draws"] == 2 * len(dataset.time_points)


def test_weighted_prediction_semantics_use_unequal_weights():
    result = weighted_decay_prediction_summary(
        [0.1, 1.0],
        [0.9, 0.1],
        future_times=[4.0],
        sigma=0.05,
        probabilities=[0.1, 0.5, 0.9],
    )
    expected_mean = 0.9 * exp(-0.4) + 0.1 * exp(-4.0)
    assert np.allclose(result["latent_mean"], [expected_mean])
    assert result["latent_quantiles"][0, 1] == exp(-0.4)
    assert result["observation_predictive_quantiles"][0, 0] < result["observation_predictive_quantiles"][0, 2]
    assert "measurement noise included" in result["semantics"]["observation_predictive"]


def test_small_ode_reference_cell_reports_weights_and_recomputed_transition():
    dataset = generate_decay_observation(0.8, 1103)
    result = run_ode_decay_control_cell(
        dataset,
        proposal_seed=2103,
        simulator_noise_seed=3103,
        target_samples=24,
        epsilon_schedule=[3.0, 2.0],
        max_attempts_per_population=[3_000, 3_000],
        quadrature_order=48,
    )
    assert result["status"] == "complete"
    populations = result["reference"]["populations"]
    assert len(populations) == 2
    for population in populations:
        weights = np.asarray(population["weights"], dtype=float)
        assert len(weights) == 24
        assert np.all(np.isfinite(weights))
        assert np.isclose(np.sum(weights), 1.0)
    check = result["transition_recheck"]["checks"][0]
    assert check["status"] == "recomputed"
    assert check["max_abs_log_proposal_error"] < 1e-10
    assert check["max_abs_weight_error"] < 1e-10
    assert result["particle_prediction"]["semantics"]["latent"].startswith("weighted particles")
    for generation in (0, 1):
        diagnostics = result["reference"]["populations"][generation]["diagnostics"]
        assert diagnostics["elapsed_seconds"] >= 0.0
        assert diagnostics["solver_failures"] == 0
        assert diagnostics["simulator_calls_timed"] == diagnostics["simulated"]


def test_incomplete_cell_cannot_emit_posterior_like_particle_summaries():
    dataset = generate_decay_observation(0.35, 1106)
    result = run_ode_decay_control_cell(
        dataset,
        proposal_seed=2106,
        simulator_noise_seed=3106,
        target_samples=3,
        epsilon_schedule=[8.0],
        max_attempts_per_population=1,
        quadrature_order=32,
    )
    assert result["status"] == "incomplete"
    assert result["termination_reason"] == "attempt_budget_exhausted"
    assert result["parameter_diagnostics"]["status"] == "unavailable"
    assert result["particle_prediction"]["status"] == "unavailable"
    assert result["parameter_diagnostics"]["posterior_summary"] is False
    assert result["particle_prediction"]["posterior_summary"] is False
    # The independent targets remain available as reference calculations.
    assert result["target"]["target_kind"] == "finite_epsilon_abc"
    assert result["exact_gaussian_likelihood_target"]["target_kind"] == "exact_gaussian_likelihood"


def test_wall_deadline_returns_incomplete_without_particle_summary():
    dataset = generate_decay_observation(0.8, 1107)
    result = run_ode_decay_control_cell(
        dataset,
        proposal_seed=2107,
        simulator_noise_seed=3107,
        target_samples=3,
        epsilon_schedule=[3.0, 2.0],
        max_attempts_per_population=100,
        quadrature_order=32,
        wall_deadline=time.monotonic() - 1.0,
    )
    assert result["status"] == "incomplete"
    assert result["termination_reason"] == "wall_budget_exhausted"
    assert result["parameter_diagnostics"]["posterior_summary"] is False
    assert result["particle_prediction"]["posterior_summary"] is False
    assert result["reference"]["populations"][0]["diagnostics"]["termination_reason"] == "wall_budget_exhausted"
    assert result["population_timing"]["0"]["elapsed_seconds"] >= 0.0


def test_deadline_rechecked_after_slow_simulator_call():
    prior_sampler, prior_logpdf = make_uniform_prior(((0.05, 1.5),))

    def slow_simulator(_theta, _rng):
        time.sleep(0.05)
        return np.zeros(1, dtype=float)

    deadline = time.monotonic() + 0.02
    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        slow_simulator,
        lambda _simulated: 0.0,
        target_samples=1,
        epsilon_schedule=[1.0],
        max_attempts_per_population=50,
        bounds=((0.05, 1.5),),
        prior_logpdf=prior_logpdf,
        seed=2108,
        stop_requested=lambda: time.monotonic() >= deadline,
    )
    assert result["status"] == "incomplete"
    assert result["termination_reason"] == "wall_budget_exhausted"
    assert result["diagnostics"]["proposed"] == 1
    assert result["diagnostics"]["simulated"] == 1
    assert result["diagnostics"]["accepted"] == 0


def test_forced_solver_failure_is_counted_and_budget_bounded():
    dataset = generate_decay_observation(0.35, 1104)
    config = ODESolverConfig(max_step=0.05, max_steps=1, error_tolerance=1e-6)
    simulator = DecayObservationSimulator(dataset.time_points, simulator_noise_seed=3104, solver_config=config)
    prior_sampler, prior_logpdf = make_uniform_prior(((0.05, 1.5),))
    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        simulator,
        lambda _simulated: 0.0,
        target_samples=3,
        epsilon_schedule=[3.0],
        max_attempts_per_population=7,
        bounds=((0.05, 1.5),),
        prior_logpdf=prior_logpdf,
        seed=2104,
    )
    assert result["status"] == "incomplete"
    assert result["diagnostics"]["proposed"] == 7
    assert result["diagnostics"]["failed_simulations"] == 7
    assert simulator.diagnostics()["failure_categories"] == {"max_steps_exceeded": 7}


def test_unattainable_discrepancy_preserves_explicit_partial_status():
    prior_sampler, prior_logpdf = make_uniform_prior(((0.05, 1.5),))
    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        lambda _theta, _rng: 1.0,
        lambda _simulated: 1.0,
        target_samples=3,
        epsilon_schedule=[0.0],
        max_attempts_per_population=7,
        bounds=((0.05, 1.5),),
        prior_logpdf=prior_logpdf,
        seed=2105,
    )
    assert result["status"] == "incomplete"
    assert result["termination_reason"] == "attempt_budget_exhausted"
    assert result["diagnostics"]["proposed"] == 7
    assert len(result["accepted_params"]) == 0


if __name__ == "__main__":
    test_solver_accuracy_is_below_measurement_scale_and_preserves_origin()
    test_noncentral_chi_square_target_is_independent_poisson_quadrature()
    test_finite_epsilon_target_matches_independent_quadrature()
    test_targets_enforce_support_and_finite_epsilon_differs_from_exact_likelihood()
    test_target_observation_predictive_uses_density_weighted_quadrature()
    test_simulator_noise_stream_is_fresh_and_separate_from_proposals()
    test_weighted_prediction_semantics_use_unequal_weights()
    test_small_ode_reference_cell_reports_weights_and_recomputed_transition()
    test_forced_solver_failure_is_counted_and_budget_bounded()
    test_unattainable_discrepancy_preserves_explicit_partial_status()
    test_wall_deadline_returns_incomplete_without_particle_summary()
    print("SUCCESS: ODE reference controls passed")

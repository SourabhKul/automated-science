"""Focused controls for the opt-in Gaussian ABC-SMC reference path."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np

from core.abc_reference_controls import run_scalar_finite_epsilon_control, weighted_empirical_cdf, weighted_quantile
from core.abc_smc_reference import (
    gaussian_logpdf,
    gaussian_mixture_logpdf,
    make_uniform_prior,
    run_gaussian_abc_smc_reference,
)


def _independent_scalar_target(y: float, epsilon: float, sigma: float = 0.1):
    """Independent test-side quadrature, separate from the control helper."""

    nodes, weights = np.polynomial.legendre.leggauss(320)

    def density(theta):
        theta = np.asarray(theta, dtype=float)
        upper = 0.5 * (1.0 + np.vectorize(lambda x: erf(float(x) / sqrt(2.0)))(
            (y + epsilon - theta) / sigma
        ))
        lower = 0.5 * (1.0 + np.vectorize(lambda x: erf(float(x) / sqrt(2.0)))(
            (y - epsilon - theta) / sigma
        ))
        return upper - lower

    def integrate(function, upper=1.0):
        mapped = 0.5 * upper * (nodes + 1.0)
        return float(0.5 * upper * np.sum(weights * function(mapped)))

    normalizer = integrate(density)
    mean = integrate(lambda theta: theta * density(theta)) / normalizer

    def cdf(x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return integrate(density, x) / normalizer

    def quantile(probability):
        low, high = 0.0, 1.0
        for _ in range(80):
            midpoint = (low + high) / 2.0
            if cdf(midpoint) < probability:
                low = midpoint
            else:
                high = midpoint
        return (low + high) / 2.0

    return normalizer, mean, cdf, quantile


def test_reference_weights_are_normalized_and_reported_for_each_population():
    bounds = [(0.0, 1.0)]
    prior_sampler, prior_logpdf = make_uniform_prior(bounds)

    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        lambda theta, _rng: float(theta[0]),
        lambda simulated: abs(simulated - 0.5),
        target_samples=12,
        epsilon_schedule=[0.8, 0.5],
        max_attempts_per_population=[2_000, 2_000],
        bounds=bounds,
        prior_logpdf=prior_logpdf,
        seed=31,
    )

    assert result["status"] == "complete"
    assert result["complete"] is True
    assert len(result["populations"]) == 2
    for population in result["populations"]:
        weights = np.asarray(population["weights"])
        assert len(weights) == 12
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.isclose(population["effective_sample_size"], 1.0 / np.sum(weights**2))

    # The transition density is the same Gaussian mixture used to generate
    # the second population.  A one-component check gives a closed form value.
    second = result["populations"][1]
    covariance = np.asarray(second["proposal_covariance"])
    previous = result["populations"][0]
    point = np.array([0.4])
    mixture = gaussian_mixture_logpdf(
        point,
        np.asarray(previous["accepted_params"]),
        np.asarray(previous["weights"]),
        covariance,
    )
    assert np.isfinite(mixture)
    assert np.allclose(np.asarray(second["log_proposal_mixture_density"]), [
        gaussian_mixture_logpdf(p, np.asarray(previous["accepted_params"]), np.asarray(previous["weights"]), covariance)
        for p in np.asarray(second["accepted_params"])
    ])


def test_gaussian_density_uses_supplied_covariance_without_hidden_regularization():
    point = np.array([0.75])
    means = np.array([[0.5]])
    covariance = np.array([[0.25]])
    expected = -0.5 * (np.log(2.0 * np.pi * 0.25) + (0.25**2) / 0.25)
    assert np.allclose(gaussian_logpdf(point, means, covariance), [expected])


def test_reference_rejects_out_of_support_without_boundary_atoms_or_clipping():
    bounds = [(0.0, 1.0)]
    prior_sampler, prior_logpdf = make_uniform_prior(bounds)
    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        lambda theta, _rng: float(theta[0]),
        lambda _simulated: 0.0,
        target_samples=40,
        epsilon_schedule=[1.0, 1.0],
        max_attempts_per_population=[5_000, 5_000],
        bounds=bounds,
        prior_logpdf=prior_logpdf,
        seed=4,
        nugget=0.08,
    )

    assert result["status"] == "complete"
    second = result["populations"][1]
    diagnostics = second["diagnostics"]
    assert diagnostics["out_of_support"] > 0
    accepted = np.asarray(second["accepted_params"])[:, 0]
    assert np.all((accepted >= 0.0) & (accepted <= 1.0))
    assert not np.any(np.isclose(accepted, 0.0, atol=1e-14))
    assert not np.any(np.isclose(accepted, 1.0, atol=1e-14))
    ancestors = [value for value in diagnostics["ancestor_indices"] if value is not None]
    assert len(ancestors) == diagnostics["proposed"]
    # Retries choose ancestors inside the attempt loop.  This seed has both
    # rejected and accepted proposals and therefore records multiple choices.
    assert len(set(ancestors)) > 1


def test_weighted_summary_helpers_use_declared_empirical_conventions():
    values = np.array([0.0, 1.0, 2.0])
    weights = np.array([0.2, 0.3, 0.5])
    assert np.allclose(weighted_quantile(values, weights, [0.1, 0.5, 0.9]), [0.0, 1.0, 2.0])
    assert np.allclose(weighted_empirical_cdf(values, weights, [0.0, 0.5, 1.0, 2.0]), [0.2, 0.2, 0.5, 1.0])


def test_budget_failure_returns_partial_population_instead_of_previous_success():
    bounds = [(0.0, 1.0)]
    prior_sampler, prior_logpdf = make_uniform_prior(bounds)
    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        lambda _theta, _rng: 1.0,
        lambda simulated: float(simulated),
        target_samples=6,
        epsilon_schedule=[1.0, 0.0],
        max_attempts_per_population=[100, 7],
        bounds=bounds,
        prior_logpdf=prior_logpdf,
        seed=9,
    )

    assert result["status"] == "incomplete"
    assert result["complete"] is False
    assert result["termination_reason"] == "attempt_budget_exhausted"
    assert len(result["populations"]) == 2
    assert result["populations"][0]["diagnostics"]["complete"] is True
    failed = result["populations"][1]
    assert failed["diagnostics"]["proposed"] == 7
    assert failed["diagnostics"]["accepted"] == 0
    assert np.asarray(result["accepted_params"]).shape == (0, 1)
    assert np.asarray(result["weights"]).size == 0


def test_custom_support_without_bounds_infers_dimension_from_first_prior_draw():
    def prior_sampler(rng):
        return np.array([rng.uniform(-1.0, 1.0)])

    def prior_logpdf(theta):
        return 0.0 if -1.0 <= float(theta[0]) <= 1.0 else float("-inf")

    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        lambda theta, _rng: float(theta[0]),
        lambda _simulated: 0.0,
        target_samples=5,
        epsilon_schedule=[0.0],
        max_attempts_per_population=100,
        in_support=lambda theta: -1.0 <= float(theta[0]) <= 1.0,
        prior_logpdf=prior_logpdf,
        seed=15,
    )

    assert result["status"] == "complete"
    assert np.asarray(result["accepted_params"]).shape == (5, 1)


def test_scalar_finite_epsilon_controls_match_independent_target_at_mc_scale():
    # This is a compact six-cell control over both support boundaries and the
    # interior.  The acceptance target is finite-epsilon ABC, not the exact
    # zero-epsilon posterior.  We assess errors in estimated Monte Carlo units
    # and keep the bound deliberately broad for a small deterministic control.
    for y in (0.1, 0.5, 0.9):
        for epsilon in (0.1, 0.025):
            normalizer, target_mean, target_cdf, target_quantile = _independent_scalar_target(y, epsilon)
            control = run_scalar_finite_epsilon_control(
                y,
                epsilon,
                sigma=0.1,
                target_samples=240,
                max_attempts=30_000,
                seed=100 + int(round(100 * y)) + int(round(1_000 * epsilon)),
                epsilon_schedule=[0.2, epsilon],
            )
            assert control["reference"]["status"] == "complete"
            diagnostics = control["diagnostics"]
            assert diagnostics["effective_sample_size"] > 100
            assert np.isclose(control["target"].normalizer, normalizer, atol=1e-10)
            assert np.isclose(diagnostics["target_mean"], target_mean, atol=1e-10)
            target_grid = np.asarray(diagnostics["cdf_grid"])
            assert np.allclose(diagnostics["target_cdf"], [target_cdf(float(x)) for x in target_grid], atol=1e-10)
            assert np.allclose(
                diagnostics["target_quantiles"],
                [target_quantile(float(probability)) for probability in (0.1, 0.5, 0.9)],
                atol=1e-10,
            )
            assert diagnostics["mean_error_over_mc_se"] < 8.0
            assert np.all(np.asarray(diagnostics["quantile_error_over_mc_se"]) < 8.0)
            assert diagnostics["cdf_max_error_over_mc_se"] < 12.0


if __name__ == "__main__":
    test_reference_weights_are_normalized_and_reported_for_each_population()
    test_reference_rejects_out_of_support_without_boundary_atoms_or_clipping()
    test_weighted_summary_helpers_use_declared_empirical_conventions()
    test_budget_failure_returns_partial_population_instead_of_previous_success()
    test_custom_support_without_bounds_infers_dimension_from_first_prior_draw()
    test_scalar_finite_epsilon_controls_match_independent_target_at_mc_scale()
    print("SUCCESS: opt-in Gaussian ABC-SMC reference controls passed")

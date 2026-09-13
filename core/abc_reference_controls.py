"""Independent scalar controls for the opt-in Gaussian ABC-SMC reference.

The finite-epsilon target is evaluated with fixed Gauss-Legendre quadrature
over the prior support.  It does not reuse the sampler's proposal or accepted
particles, so the control can distinguish implementation errors from ordinary
Monte Carlo variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt
from typing import Callable, Sequence

import numpy as np

from .abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference


def _normal_cdf(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=float)
    flat = np.fromiter(
        (0.5 * (1.0 + erf(float(item) / sqrt(2.0))) for item in array.flat),
        dtype=float,
        count=array.size,
    ).reshape(array.shape)
    return float(flat) if array.ndim == 0 else flat


@dataclass(frozen=True)
class FiniteEpsilonABCTarget:
    """Independent one-dimensional finite-epsilon ABC target."""

    y: float
    sigma: float
    epsilon: float
    normalizer: float
    mean: float
    density: Callable[[float | np.ndarray], float | np.ndarray]
    cdf: Callable[[float], float]

    def quantile(self, probability: float, *, iterations: int = 80) -> float:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if probability == 0:
            return 0.0
        if probability == 1:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(iterations):
            midpoint = 0.5 * (low + high)
            if self.cdf(midpoint) < probability:
                low = midpoint
            else:
                high = midpoint
        return 0.5 * (low + high)

    def quantiles(self, probabilities: Sequence[float]) -> np.ndarray:
        return np.asarray([self.quantile(float(p)) for p in probabilities], dtype=float)


def _integrate_unit_interval(
    function: Callable[[np.ndarray], np.ndarray],
    upper: float,
    *,
    order: int,
) -> float:
    """Integrate a smooth scalar function on ``[0, upper]`` independently."""

    if upper <= 0.0:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mapped = 0.5 * upper * (nodes + 1.0)
    values = np.asarray(function(mapped), dtype=float)
    return float(0.5 * upper * np.sum(weights * values))


def finite_epsilon_abc_target(
    y: float,
    *,
    sigma: float = 0.1,
    epsilon: float = 0.1,
    quadrature_order: int = 256,
) -> FiniteEpsilonABCTarget:
    """Compute the exact scalar finite-epsilon ABC target by quadrature.

    With ``theta ~ Uniform(0, 1)`` and ``Y | theta ~ Normal(theta, sigma)``,
    accepting ``|Y - y| <= epsilon`` gives an unnormalized density

    ``Phi((y + epsilon - theta)/sigma) - Phi((y - epsilon - theta)/sigma)``.
    """

    y = float(y)
    sigma = float(sigma)
    epsilon = float(epsilon)
    if not np.isfinite(y) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("y must be finite and sigma must be positive")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or int(quadrature_order) < 8:
        raise ValueError("quadrature_order must be an integer >= 8")
    quadrature_order = int(quadrature_order)
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(quadrature_order)

    def integrate(function: Callable[[np.ndarray], np.ndarray], upper: float) -> float:
        if upper <= 0.0:
            return 0.0
        mapped = 0.5 * upper * (quadrature_nodes + 1.0)
        values = np.asarray(function(mapped), dtype=float)
        return float(0.5 * upper * np.sum(quadrature_weights * values))

    def density(point: float | np.ndarray) -> float | np.ndarray:
        theta = np.asarray(point, dtype=float)
        result = _normal_cdf((y + epsilon - theta) / sigma) - _normal_cdf((y - epsilon - theta) / sigma)
        return float(result) if theta.ndim == 0 else np.asarray(result, dtype=float)

    normalizer = integrate(lambda grid: np.asarray(density(grid)), 1.0)
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("finite-epsilon target has non-positive normalizer")
    mean = integrate(
        lambda grid: grid * np.asarray(density(grid)),
        1.0,
    ) / normalizer

    def cdf(point: float) -> float:
        x = float(point)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return float(
            np.clip(
            integrate(lambda grid: np.asarray(density(grid)), x)
                / normalizer,
                0.0,
                1.0,
            )
        )

    return FiniteEpsilonABCTarget(
        y=y,
        sigma=sigma,
        epsilon=epsilon,
        normalizer=normalizer,
        mean=float(mean),
        density=density,
        cdf=cdf,
    )


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    probabilities: Sequence[float],
) -> np.ndarray:
    """Return weighted quantiles by inverting the weighted empirical CDF."""

    samples = np.asarray(values, dtype=float)
    normalized = np.asarray(weights, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if samples.ndim != 1 or normalized.shape != samples.shape or len(samples) == 0:
        raise ValueError("values and weights must be non-empty one-dimensional arrays")
    if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValueError("values and weights must be finite; weights must be non-negative")
    if not np.isclose(np.sum(normalized), 1.0, atol=1e-10):
        raise ValueError("weights must be normalized")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    order = np.argsort(samples, kind="mergesort")
    sorted_values = samples[order]
    cumulative = np.cumsum(normalized[order])
    cumulative[-1] = 1.0
    # The left-continuous inverse is deterministic and records the empirical
    # quantile convention used by the control.
    return sorted_values[np.searchsorted(cumulative, probs, side="left")]


def weighted_empirical_cdf(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    points: Sequence[float] | np.ndarray,
) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    normalized = np.asarray(weights, dtype=float)
    grid = np.asarray(points, dtype=float)
    if samples.ndim != 1 or normalized.shape != samples.shape:
        raise ValueError("values and weights must have matching one-dimensional shapes")
    if not np.isclose(np.sum(normalized), 1.0, atol=1e-10):
        raise ValueError("weights must be normalized")
    order = np.argsort(samples, kind="mergesort")
    sorted_values = samples[order]
    cumulative = np.cumsum(normalized[order])
    indices = np.searchsorted(sorted_values, grid, side="right")
    result = np.zeros(len(grid), dtype=float)
    positive = indices > 0
    result[positive] = cumulative[indices[positive] - 1]
    return result


def weighted_empirical_diagnostics(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    target: FiniteEpsilonABCTarget,
    *,
    probabilities: Sequence[float] = (0.1, 0.5, 0.9),
    cdf_grid: Sequence[float] | np.ndarray | None = None,
) -> dict[str, object]:
    """Compare weighted summaries to an independent target.

    The returned ``*_over_mc_se`` fields are diagnostics based on effective
    sample size.  They are not formal confidence guarantees for dependent
    weighted SMC particles.
    """

    samples = np.asarray(values, dtype=float)
    normalized = np.asarray(weights, dtype=float)
    if samples.ndim != 1 or normalized.shape != samples.shape or len(samples) == 0:
        raise ValueError("values and weights must be non-empty one-dimensional arrays")
    if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValueError("values and weights must be finite; weights must be non-negative")
    total = float(np.sum(normalized))
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError("weights must be normalized")
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.linspace(0.0, 1.0, 101) if cdf_grid is None else np.asarray(cdf_grid, dtype=float)
    empirical_cdf = weighted_empirical_cdf(samples, normalized, grid)
    target_cdf = np.asarray([target.cdf(float(point)) for point in grid], dtype=float)
    effective_sample_size = float(1.0 / np.sum(normalized * normalized))
    weighted_mean = float(np.sum(normalized * samples))
    weighted_variance = float(np.sum(normalized * (samples - weighted_mean) ** 2))
    mean_se = sqrt(max(weighted_variance, 0.0) / max(effective_sample_size, 1.0))
    quantiles = weighted_quantile(samples, normalized, probabilities)
    target_quantiles = target.quantiles(probabilities)
    quantile_errors = quantiles - target_quantiles
    quantile_se = []
    for probability, quantile in zip(probabilities, target_quantiles):
        cdf_se = sqrt(max(float(probability) * (1.0 - float(probability)), 1e-15) / max(effective_sample_size, 1.0))
        density = max(float(target.density(float(quantile))), 1e-15)
        quantile_se.append(cdf_se / density)
    quantile_se = np.asarray(quantile_se, dtype=float)
    cdf_se = np.sqrt(np.maximum(target_cdf * (1.0 - target_cdf), 1e-15) / max(effective_sample_size, 1.0))
    cdf_errors = empirical_cdf - target_cdf
    return {
        "effective_sample_size": effective_sample_size,
        "weighted_mean": weighted_mean,
        "target_mean": target.mean,
        "mean_error": weighted_mean - target.mean,
        "mean_mc_se": mean_se,
        "mean_error_over_mc_se": abs(weighted_mean - target.mean) / max(mean_se, 1e-15),
        "probabilities": probabilities,
        "weighted_quantiles": quantiles,
        "target_quantiles": target_quantiles,
        "quantile_errors": quantile_errors,
        "quantile_mc_se": quantile_se,
        "quantile_error_over_mc_se": np.abs(quantile_errors) / np.maximum(quantile_se, 1e-15),
        "cdf_grid": grid,
        "weighted_cdf": empirical_cdf,
        "target_cdf": target_cdf,
        "cdf_errors": cdf_errors,
        "cdf_mc_se": cdf_se,
        "cdf_max_abs_error": float(np.max(np.abs(cdf_errors))),
        "cdf_max_error_over_mc_se": float(np.max(np.abs(cdf_errors) / np.maximum(cdf_se, 1e-15))),
    }


def run_scalar_finite_epsilon_control(
    y: float,
    epsilon: float,
    *,
    sigma: float = 0.1,
    target_samples: int = 500,
    max_attempts: int = 50_000,
    seed: int = 0,
    epsilon_schedule: Sequence[float] | None = None,
) -> dict[str, object]:
    """Run one bounded scalar control and return independent diagnostics."""

    bounds = [(0.0, 1.0)]
    prior_sampler, prior_logpdf = make_uniform_prior(bounds)

    def simulator(theta: np.ndarray, rng: np.random.Generator) -> float:
        return float(rng.normal(float(theta[0]), sigma))

    schedule = [float(epsilon)] if epsilon_schedule is None else [float(value) for value in epsilon_schedule]
    if not schedule or not np.isclose(schedule[-1], float(epsilon)):
        raise ValueError("epsilon_schedule must end at epsilon")
    result = run_gaussian_abc_smc_reference(
        prior_sampler,
        simulator,
        lambda simulated: abs(float(simulated) - float(y)),
        target_samples=target_samples,
        epsilon_schedule=schedule,
        max_attempts_per_population=max_attempts,
        bounds=bounds,
        prior_logpdf=prior_logpdf,
        seed=seed,
    )
    target = finite_epsilon_abc_target(y, sigma=sigma, epsilon=epsilon)
    diagnostics: dict[str, object]
    if len(result["accepted_params"]):
        diagnostics = weighted_empirical_diagnostics(
            np.asarray(result["accepted_params"])[:, 0],
            np.asarray(result["weights"]),
            target,
        )
    else:
        diagnostics = {"effective_sample_size": 0.0}
    return {
        "y": float(y),
        "sigma": float(sigma),
        "epsilon": float(epsilon),
        "seed": int(seed),
        "target_samples": int(target_samples),
        "max_attempts": int(max_attempts),
        "reference": result,
        "target": target,
        "diagnostics": diagnostics,
    }


__all__ = [
    "FiniteEpsilonABCTarget",
    "finite_epsilon_abc_target",
    "run_scalar_finite_epsilon_control",
    "weighted_empirical_cdf",
    "weighted_empirical_diagnostics",
    "weighted_quantile",
]

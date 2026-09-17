"""Opt-in Gaussian ABC-SMC reference implementation.

This module is deliberately separate from :mod:`core.sbi_engine`.  The
historical engine remains available for replay and its default behaviour is
unchanged.  Call :func:`run_gaussian_abc_smc_reference` explicitly when a
population is meant to follow the Gaussian ABC-SMC reference contract:

* draw a new ancestor and a new Gaussian proposal for every attempt;
* reject proposals outside the declared prior support without clipping;
* use one covariance matrix for both proposal draws and mixture densities in
  a population; and
* calculate prior-over-mixture importance weights in log space.

The callable is simulator-agnostic.  A simulator receives ``(theta, rng)``
and a discrepancy callable maps its output to a finite scalar.  The returned
mapping is intentionally serializable after converting NumPy arrays to lists,
but arrays are retained in the in-memory result for scientific use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import log, pi
from typing import Any

import numpy as np


def _logsumexp(values: np.ndarray) -> float:
    """Small dependency-free log-sum-exp for one-dimensional arrays."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("-inf")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


ArrayLike = Sequence[float] | np.ndarray


def _as_bounds(bounds: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Validate and return bounds as a ``(n_parameters, 2)`` array."""

    result = np.asarray(bounds, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] == 0:
        raise ValueError("bounds must have shape (n_parameters, 2) with at least one parameter")
    if not np.all(np.isfinite(result)) or np.any(result[:, 1] <= result[:, 0]):
        raise ValueError("each bound must be finite with high > low")
    return result


def make_uniform_prior(bounds: Sequence[Sequence[float]] | np.ndarray) -> tuple[Callable, Callable]:
    """Return a uniform prior sampler and log-density for ``bounds``.

    The sampler and density are useful for small controls and make the
    constant prior density explicit.  The log-density is ``-inf`` outside
    support and the normalizing constant is retained (it cancels in a
    population, but retaining it makes the callable correct on its own).
    """

    checked = _as_bounds(bounds)
    lows = checked[:, 0].copy()
    highs = checked[:, 1].copy()
    log_density = -float(np.sum(np.log(highs - lows)))

    def sample(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(lows, highs)

    def logpdf(theta: ArrayLike) -> float:
        point = np.asarray(theta, dtype=float)
        if point.shape != (len(checked),) or not np.all(np.isfinite(point)):
            return float("-inf")
        if np.any(point < lows) or np.any(point > highs):
            return float("-inf")
        return log_density

    return sample, logpdf


def _validate_point(theta: Any, dimension: int) -> np.ndarray:
    point = np.asarray(theta, dtype=float)
    if point.shape != (dimension,):
        raise ValueError(f"prior/proposal point must have shape {(dimension,)}, got {point.shape}")
    return point


def _weighted_covariance(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Population covariance under normalized weights."""

    if points.ndim != 2 or len(points) == 0:
        raise ValueError("points must be a non-empty two-dimensional array")
    mean = np.sum(points * weights[:, None], axis=0)
    centered = points - mean
    return (centered * weights[:, None]).T @ centered


def make_gaussian_kernel_covariance(
    accepted_params: np.ndarray,
    accepted_weights: np.ndarray,
    *,
    covariance_scale: float = 2.0,
    lambda_noise: float = 0.01,
    nugget: float = 1e-9,
) -> np.ndarray:
    """Build and validate the fixed covariance for one transition.

    This is the only covariance construction used by the reference path.
    The returned matrix is passed unchanged to both
    ``Generator.multivariate_normal`` and
    :func:`gaussian_mixture_logpdf`.
    """

    points = np.asarray(accepted_params, dtype=float)
    weights = np.asarray(accepted_weights, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("accepted_params must be a non-empty (n_particles, n_parameters) array")
    if weights.shape != (len(points),) or not np.all(np.isfinite(weights)):
        raise ValueError("accepted_weights must be finite with one entry per particle")
    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("accepted_weights must be non-negative and normalized")
    if not np.isfinite(covariance_scale) or covariance_scale <= 0:
        raise ValueError("covariance_scale must be finite and positive")
    if not np.isfinite(lambda_noise) or lambda_noise < 0:
        raise ValueError("lambda_noise must be finite and non-negative")
    if not np.isfinite(nugget) or nugget < 0:
        raise ValueError("nugget must be finite and non-negative")

    empirical = _weighted_covariance(points, weights)
    dimension = points.shape[1]
    covariance = (
        covariance_scale * empirical
        + lambda_noise * np.diag(np.diag(empirical))
        + np.eye(dimension) * nugget
    )
    covariance = np.asarray(covariance, dtype=float)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("proposal covariance is non-finite")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "proposal covariance is not positive definite; increase nugget or provide a less-degenerate population"
        ) from exc
    return covariance


def gaussian_logpdf(point: ArrayLike, means: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian density using the exact supplied covariance.

    ``means`` may be one point or a matrix of mixture component means.  A
    Cholesky factor is computed from the supplied matrix; no covariance
    regularization or clipping is performed here.
    """

    x = np.asarray(point, dtype=float)
    centers = np.asarray(means, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if centers.ndim == 1:
        centers = centers[None, :]
    if x.ndim != 1 or centers.ndim != 2 or centers.shape[1] != len(x):
        raise ValueError("point and means have incompatible shapes")
    if cov.shape != (len(x), len(x)):
        raise ValueError("covariance has incompatible shape")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    delta = centers - x[None, :]
    solved = np.linalg.solve(chol, delta.T).T
    mahalanobis = np.sum(solved * solved, axis=1)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (len(x) * log(2.0 * pi) + logdet + mahalanobis)


def gaussian_mixture_logpdf(
    point: ArrayLike,
    means: np.ndarray,
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Evaluate ``log(sum_j weights[j] K(point | means[j], covariance))``."""

    mixture_weights = np.asarray(weights, dtype=float)
    if mixture_weights.ndim != 1 or len(means) != len(mixture_weights):
        raise ValueError("mixture means and weights must have matching lengths")
    if np.any(mixture_weights < 0) or not np.all(np.isfinite(mixture_weights)):
        raise ValueError("mixture weights must be finite and non-negative")
    if not np.isclose(np.sum(mixture_weights), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("mixture weights must be normalized")
    positive = mixture_weights > 0
    if not np.any(positive):
        return float("-inf")
    component_logs = gaussian_logpdf(point, np.asarray(means)[positive], covariance)
    return float(_logsumexp(np.log(mixture_weights[positive]) + component_logs))


def _normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    if len(log_weights) == 0:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("importance log-weights must be finite")
    normalizer = float(_logsumexp(log_weights))
    weights = np.exp(log_weights - normalizer)
    if not np.all(np.isfinite(weights)) or not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise ValueError("importance weights could not be normalized")
    return weights


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    denominator = float(np.sum(values * values))
    return float(1.0 / denominator) if denominator > 0 else float("nan")


def _attempt_budgets(
    schedule_length: int,
    max_attempts_per_population: int | Sequence[int] | None,
    max_attempts: int | Sequence[int] | None,
) -> list[int]:
    if max_attempts_per_population is not None and max_attempts is not None:
        if max_attempts_per_population != max_attempts:
            raise ValueError("provide only one of max_attempts_per_population and max_attempts")
    supplied = max_attempts_per_population if max_attempts_per_population is not None else max_attempts
    if supplied is None:
        raise ValueError("a finite max_attempts_per_population budget is required")
    if np.isscalar(supplied):
        raw = [supplied] * schedule_length
    else:
        raw = list(supplied)
        if len(raw) != schedule_length:
            raise ValueError("one max-attempt budget is required for each epsilon")
    budgets: list[int] = []
    for value in raw:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError("max-attempt budgets must be positive integers")
        budgets.append(int(value))
    return budgets


def _support_checker(
    bounds: np.ndarray | None,
    in_support: Callable[[np.ndarray], bool] | None,
    dimension: int,
) -> Callable[[np.ndarray], bool]:
    if bounds is None and in_support is None:
        raise ValueError("bounds or an in_support callable must be provided")

    def check(point: np.ndarray) -> bool:
        if point.shape != (dimension,) or not np.all(np.isfinite(point)):
            return False
        if bounds is not None and (np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1])):
            return False
        if in_support is not None:
            try:
                return bool(in_support(point))
            except Exception:
                return False
        return True

    return check


def _population_record(
    *,
    generation: int,
    epsilon: float,
    accepted_params: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
    log_weights: np.ndarray,
    log_prior: np.ndarray,
    log_proposal: np.ndarray,
    covariance: np.ndarray | None,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generation": generation,
        "epsilon": float(epsilon),
        "accepted_params": accepted_params,
        "distances": distances,
        "weights": weights,
        "log_weights": log_weights,
        "log_prior_density": log_prior,
        "log_proposal_mixture_density": log_proposal,
        "effective_sample_size": _effective_sample_size(weights),
        "proposal_covariance": covariance,
        "diagnostics": diagnostics,
    }


def run_gaussian_abc_smc_reference(
    prior_sampler: Callable[[np.random.Generator], ArrayLike],
    simulator: Callable[[np.ndarray, np.random.Generator], Any],
    discrepancy: Callable[[Any], float],
    *,
    target_samples: int,
    epsilon_schedule: Sequence[float],
    max_attempts_per_population: int | Sequence[int] | None = None,
    max_attempts: int | Sequence[int] | None = None,
    bounds: Sequence[Sequence[float]] | np.ndarray | None = None,
    in_support: Callable[[np.ndarray], bool] | None = None,
    prior_logpdf: Callable[[np.ndarray], float] | None = None,
    covariance_scale: float = 2.0,
    lambda_noise: float = 0.01,
    nugget: float = 1e-9,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    population_event: Callable[[str, int, dict[str, Any] | None], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the explicit Gaussian ABC-SMC reference path.

    ``epsilon_schedule[0]`` is the generation-zero discrepancy threshold and
    each later value is the threshold for one Gaussian transition.  No
    threshold is inferred from particle quantiles.  Every attempt counts
    against that population's finite budget, including out-of-support
    proposals and failed simulations.  If a target population is not filled,
    the function returns ``status == "incomplete"`` with partial evidence and
    does not substitute a previous successful population.

    The canonical :class:`core.sbi_engine.SBIEngine` runner does not call this
    function.  It is an explicit opt-in reference for isolated validation.
    """

    if isinstance(target_samples, bool) or int(target_samples) != target_samples or int(target_samples) <= 0:
        raise ValueError("target_samples must be a positive integer")
    target_samples = int(target_samples)

    schedule = list(epsilon_schedule)
    if not schedule:
        raise ValueError("epsilon_schedule must contain at least one threshold")
    try:
        schedule = [float(value) for value in schedule]
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon_schedule must contain finite non-negative values") from exc
    if not np.all(np.isfinite(schedule)) or np.any(np.asarray(schedule) < 0):
        raise ValueError("epsilon_schedule must contain finite non-negative values")
    if any(schedule[index] > schedule[index - 1] for index in range(1, len(schedule))):
        raise ValueError("epsilon_schedule must be non-increasing")
    budgets = _attempt_budgets(len(schedule), max_attempts_per_population, max_attempts)

    checked_bounds = _as_bounds(bounds) if bounds is not None else None
    if checked_bounds is None:
        if in_support is None:
            raise ValueError("bounds or an in_support callable must be provided")
        # Dimension is learned from the first prior draw below.  A custom
        # prior density is required because there is no default support volume.
        dimension: int | None = None
    else:
        dimension = len(checked_bounds)

    if prior_logpdf is None:
        if checked_bounds is None:
            raise ValueError("prior_logpdf is required when bounds are omitted")
        _, uniform_logpdf = make_uniform_prior(checked_bounds)
        prior_logpdf = uniform_logpdf

    if rng is not None and seed is not None:
        raise ValueError("provide either seed or rng, not both")
    random = rng if rng is not None else np.random.default_rng(seed)
    if population_event is not None and not callable(population_event):
        raise TypeError("population_event must be callable when provided")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable when provided")
    checker: Callable[[np.ndarray], bool] | None = None
    if dimension is not None:
        checker = _support_checker(checked_bounds, in_support, dimension)

    populations: list[dict[str, Any]] = []
    previous_params: np.ndarray | None = None
    previous_weights: np.ndarray | None = None
    final_population: dict[str, Any] | None = None
    top_level_reason = "completed"

    for generation, (epsilon, budget) in enumerate(zip(schedule, budgets)):
        if population_event is not None:
            population_event("start", generation, None)
        if generation == 0:
            covariance = None
        else:
            assert previous_params is not None and previous_weights is not None
            covariance = make_gaussian_kernel_covariance(
                previous_params,
                previous_weights,
                covariance_scale=covariance_scale,
                lambda_noise=lambda_noise,
                nugget=nugget,
            )
            # Factor once per population.  The exact covariance matrix above
            # is retained in the record and used by both draw and density.

        accepted: list[np.ndarray] = []
        accepted_distances: list[float] = []
        ancestor_indices: list[int | None] = []
        attempts = 0
        simulated = 0
        out_of_support = 0
        failed_prior_draws = 0
        failed_proposals = 0
        failed_simulations = 0
        failed_discrepancies = 0
        wall_budget_exhausted = False

        while len(accepted) < target_samples and attempts < budget:
            if stop_requested is not None and stop_requested():
                wall_budget_exhausted = True
                break
            attempts += 1
            ancestor_index: int | None = None
            try:
                if generation == 0:
                    raw_theta = np.asarray(prior_sampler(random), dtype=float)
                    if dimension is None:
                        if raw_theta.ndim != 1 or raw_theta.size == 0:
                            raise ValueError("prior sampler must return a non-empty one-dimensional point")
                        theta = raw_theta
                    else:
                        theta = _validate_point(raw_theta, dimension)
                else:
                    assert previous_params is not None and previous_weights is not None
                    # This choice is intentionally inside the retry loop:
                    # every rejected proposal gets a fresh ancestor.
                    ancestor_index = int(
                        random.choice(len(previous_params), p=previous_weights)
                    )
                    theta = _validate_point(
                        random.multivariate_normal(previous_params[ancestor_index], covariance),
                        len(previous_params[ancestor_index]),
                    )
            except Exception:
                if generation == 0:
                    failed_prior_draws += 1
                else:
                    failed_proposals += 1
                ancestor_indices.append(ancestor_index)
                continue
            ancestor_indices.append(ancestor_index)

            if dimension is None:
                dimension = len(theta)
                checker = _support_checker(checked_bounds, in_support, dimension)
            assert checker is not None
            if not checker(theta):
                out_of_support += 1
                continue

            try:
                simulated_value = simulator(theta, random)
                simulated += 1
            except Exception:
                failed_simulations += 1
                # A bounded simulator may return an exception after the
                # deadline. Preserve the timeout as the population outcome
                # rather than spending another attempt or presenting the
                # partial population as complete.
                if stop_requested is not None and stop_requested():
                    wall_budget_exhausted = True
                    break
                continue
            # The simulator itself can consume the remaining wall budget.
            # Recheck immediately after it returns, before discrepancy
            # evaluation or accepting the proposal into this population.
            if stop_requested is not None and stop_requested():
                wall_budget_exhausted = True
                break
            try:
                distance = float(discrepancy(simulated_value))
            except Exception:
                failed_discrepancies += 1
                continue
            if not np.isfinite(distance):
                failed_discrepancies += 1
                continue
            if distance <= epsilon:
                accepted.append(theta.copy())
                accepted_distances.append(distance)

        params = np.asarray(accepted, dtype=float)
        if params.size == 0:
            params = np.empty((0, dimension or 0), dtype=float)
        else:
            params = params.reshape(len(accepted), -1)
        distances = np.asarray(accepted_distances, dtype=float)
        accepted_count = len(accepted)
        complete = accepted_count == target_samples

        weight_failures = 0
        if generation == 0:
            if accepted_count:
                weights = np.full(accepted_count, 1.0 / accepted_count, dtype=float)
                log_weights = np.log(weights)
                log_prior = np.full(accepted_count, np.nan, dtype=float)
                log_proposal = np.full(accepted_count, np.nan, dtype=float)
            else:
                weights = np.empty(0, dtype=float)
                log_weights = np.empty(0, dtype=float)
                log_prior = np.empty(0, dtype=float)
                log_proposal = np.empty(0, dtype=float)
        else:
            assert previous_params is not None and previous_weights is not None and covariance is not None
            if accepted_count:
                raw_log_prior = []
                raw_log_proposal = []
                for point in params:
                    try:
                        prior_value = float(prior_logpdf(point))
                        proposal_value = gaussian_mixture_logpdf(
                            point,
                            previous_params,
                            previous_weights,
                            covariance,
                        )
                        if not np.isfinite(prior_value) or not np.isfinite(proposal_value):
                            raise ValueError("non-finite prior or proposal log-density")
                    except Exception:
                        weight_failures += 1
                        raw_log_prior.append(float("nan"))
                        raw_log_proposal.append(float("nan"))
                        continue
                    raw_log_prior.append(prior_value)
                    raw_log_proposal.append(proposal_value)
                log_prior = np.asarray(raw_log_prior, dtype=float)
                log_proposal = np.asarray(raw_log_proposal, dtype=float)
                if weight_failures:
                    # Preserve all accepted points as partial evidence, but
                    # expose unusable weights instead of presenting a finite
                    # posterior summary for a population whose density failed.
                    weights = np.full(accepted_count, np.nan, dtype=float)
                    log_weights = np.full(accepted_count, np.nan, dtype=float)
                    complete = False
                else:
                    log_weights = log_prior - log_proposal
                    try:
                        weights = _normalise_log_weights(log_weights)
                    except ValueError:
                        weight_failures = accepted_count
                        weights = np.full(accepted_count, np.nan, dtype=float)
                        log_weights = np.full(accepted_count, np.nan, dtype=float)
                        complete = False
            else:
                weights = np.empty(0, dtype=float)
                log_weights = np.empty(0, dtype=float)
                log_prior = np.empty(0, dtype=float)
                log_proposal = np.empty(0, dtype=float)

        if wall_budget_exhausted:
            termination_reason = "wall_budget_exhausted"
        elif weight_failures:
            termination_reason = "weight_evaluation_failed"
        elif complete:
            termination_reason = "target_reached"
        else:
            termination_reason = "attempt_budget_exhausted"
        diagnostics = {
            "proposed": attempts,
            "simulated": simulated,
            "accepted": accepted_count,
            "out_of_support": out_of_support,
            "failed_prior_draws": failed_prior_draws,
            "failed_proposals": failed_proposals,
            "failed_simulations": failed_simulations,
            "failed_discrepancies": failed_discrepancies,
            "weight_failures": weight_failures,
            "max_attempts": budget,
            "complete": complete,
            "termination_reason": termination_reason,
            # Keeping ancestor choices makes the fresh-ancestor property
            # auditable without retaining every simulated trajectory.
            "ancestor_indices": ancestor_indices,
        }
        record = _population_record(
            generation=generation,
            epsilon=epsilon,
            accepted_params=params,
            distances=distances,
            weights=weights,
            log_weights=log_weights,
            log_prior=log_prior,
            log_proposal=log_proposal,
            covariance=covariance,
            diagnostics=diagnostics,
        )
        populations.append(record)
        final_population = record
        if population_event is not None:
            population_event("end", generation, record)
        if not complete:
            top_level_reason = termination_reason
            break
        previous_params = params
        previous_weights = weights

    assert final_population is not None
    complete_run = len(populations) == len(schedule) and top_level_reason == "completed"
    return {
        "status": "complete" if complete_run else "incomplete",
        "complete": complete_run,
        "termination_reason": top_level_reason,
        "reference_path": "gaussian_abc_smc_reference_opt_in",
        "canonical_runner_integrated": False,
        "target_samples": target_samples,
        "epsilon_schedule": schedule,
        "max_attempts_per_population": budgets,
        "seed": seed,
        "accepted_params": final_population["accepted_params"],
        "distances": final_population["distances"],
        "accepted_distances": final_population["distances"],
        "weights": final_population["weights"],
        "accepted_weights": final_population["weights"],
        "effective_sample_size": final_population["effective_sample_size"],
        "diagnostics": final_population["diagnostics"],
        "populations": populations,
    }


# Short alias for callers that already use the generic ABC-SMC naming.
run_abc_smc_reference = run_gaussian_abc_smc_reference


__all__ = [
    "gaussian_logpdf",
    "gaussian_mixture_logpdf",
    "make_gaussian_kernel_covariance",
    "make_uniform_prior",
    "run_abc_smc_reference",
    "run_gaussian_abc_smc_reference",
]

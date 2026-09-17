"""Bounded numerical ODE controls for the opt-in Gaussian ABC-SMC reference.

The historical :mod:`core.sbi_engine` path is intentionally not imported by
this module.  These controls provide a small, fully specified decay problem
that can be used to check numerical integration, fresh simulator noise,
finite-epsilon ABC targets, and weighted prediction summaries.

The inference simulator integrates ``dx/dt = -k*x`` with a fixed-step
classical RK4 solver starting at ``t=0``.  The analytic exponential solution
is used only by the independent target and solver-accuracy checks; it is never
used to produce a simulated ABC dataset.  Measurement noise is drawn by a
separate generator owned by :class:`DecayObservationSimulator`, so the ABC
proposal stream and simulator-noise stream cannot accidentally share draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, lgamma, log, sqrt
from typing import Any, Callable, Sequence

import numpy as np

from .abc_smc_reference import (
    gaussian_mixture_logpdf,
    make_uniform_prior,
    run_gaussian_abc_smc_reference,
)


ArrayLike = Sequence[float] | np.ndarray


class ODEControlError(RuntimeError):
    """Base class for an explicit, bounded ODE-control failure."""


class ODESolverFailure(ODEControlError):
    """A numerical solve failed with an auditable category."""

    def __init__(self, category: str, message: str):
        self.category = str(category)
        super().__init__(f"{self.category}: {message}")


@dataclass(frozen=True)
class ODESolverConfig:
    """Predeclared fixed-step RK4 settings used by the control.

    ``error_tolerance`` is a contract for the independent solver-versus-
    analytic check.  Fixed-step RK4 does not adapt its step size from this
    value; changing it without rerunning that check is therefore invalid.
    """

    method: str = "rk4_fixed_step"
    max_step: float = 0.05
    max_steps: int = 100_000
    error_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.method != "rk4_fixed_step":
            raise ValueError("only the fixed-step RK4 control solver is supported")
        if not np.isfinite(self.max_step) or self.max_step <= 0:
            raise ValueError("max_step must be finite and positive")
        if isinstance(self.max_steps, bool) or int(self.max_steps) != self.max_steps or int(self.max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not np.isfinite(self.error_tolerance) or self.error_tolerance <= 0:
            raise ValueError("error_tolerance must be finite and positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "max_step": float(self.max_step),
            "max_steps": int(self.max_steps),
            "error_tolerance": float(self.error_tolerance),
        }


def _as_time_points(time_points: ArrayLike) -> np.ndarray:
    values = np.asarray(time_points, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("time_points must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("time_points must be finite and non-negative")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("time_points must be strictly increasing")
    return values


def _as_state(y0: ArrayLike | float) -> np.ndarray:
    state = np.asarray(y0, dtype=float)
    if state.ndim == 0:
        state = state.reshape(1)
    if state.ndim != 1 or len(state) == 0 or not np.all(np.isfinite(state)):
        raise ValueError("y0 must be a finite scalar or one-dimensional state")
    return state.copy()


def _rhs_state(rhs: Callable[[float, np.ndarray, Any], ArrayLike], t: float, y: np.ndarray, args: Any) -> np.ndarray:
    try:
        value = np.asarray(rhs(float(t), y.copy(), args), dtype=float)
    except ODESolverFailure:
        raise
    except Exception as exc:
        raise ODESolverFailure("rhs_failure", f"right-hand side raised {type(exc).__name__}: {exc}") from exc
    if value.shape != y.shape:
        raise ODESolverFailure("rhs_shape_failure", f"right-hand side returned {value.shape}, expected {y.shape}")
    if not np.all(np.isfinite(value)):
        raise ODESolverFailure("nonfinite_rhs", "right-hand side returned NaN or Inf")
    return value


def integrate_ode(
    rhs: Callable[[float, np.ndarray, Any], ArrayLike],
    time_points: ArrayLike,
    y0: ArrayLike | float,
    args: Any = None,
    *,
    solver_config: ODESolverConfig | None = None,
) -> np.ndarray:
    """Integrate a numerical ODE from ``t=0`` and return requested times.

    If the first requested time is positive, ``t=0`` is inserted internally
    and the returned array contains only the originally requested times.  This
    behavior makes a one-point future prediction unambiguous and prevents a
    future horizon from silently restarting with the initial condition.
    """

    requested = _as_time_points(time_points)
    config = ODESolverConfig() if solver_config is None else solver_config
    if not isinstance(config, ODESolverConfig):
        raise TypeError("solver_config must be an ODESolverConfig")
    initial = _as_state(y0)
    prepended_origin = bool(requested[0] > 0.0)
    integration_times = np.concatenate(([0.0], requested)) if prepended_origin else requested.copy()
    state = initial.copy()
    outputs: list[np.ndarray] = []
    total_steps = 0
    current_time = float(integration_times[0])
    if not prepended_origin:
        outputs.append(state.copy())

    for target_time in integration_times[1:]:
        delta = float(target_time - current_time)
        n_steps = max(1, int(np.ceil(delta / config.max_step)))
        if total_steps + n_steps > config.max_steps:
            raise ODESolverFailure(
                "max_steps_exceeded",
                f"interval requires {n_steps} steps after {total_steps} completed steps, cap is {config.max_steps}",
            )
        step = delta / n_steps
        for step_index in range(n_steps):
            t = current_time + step_index * step
            k1 = _rhs_state(rhs, t, state, args)
            k2 = _rhs_state(rhs, t + 0.5 * step, state + 0.5 * step * k1, args)
            k3 = _rhs_state(rhs, t + 0.5 * step, state + 0.5 * step * k2, args)
            k4 = _rhs_state(rhs, t + step, state + step * k3, args)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(state)):
                raise ODESolverFailure("nonfinite_state", "integrator produced NaN or Inf")
        total_steps += n_steps
        current_time = float(target_time)
        outputs.append(state.copy())

    if prepended_origin:
        return np.asarray(outputs, dtype=float)
    return np.asarray(outputs, dtype=float)


def decay_rhs(_time: float, state: np.ndarray, k: float) -> np.ndarray:
    """Numerical RHS for ``dx/dt = -k*x``."""

    value = float(k)
    if not np.isfinite(value) or value < 0:
        raise ODESolverFailure("invalid_parameter", "decay rate k must be finite and non-negative")
    return -value * state


def integrate_decay_ode(
    k: float,
    time_points: ArrayLike,
    *,
    y0: float = 1.0,
    solver_config: ODESolverConfig | None = None,
) -> np.ndarray:
    """Numerically integrate the scalar decay ODE from the original time origin."""

    return integrate_ode(decay_rhs, time_points, y0, float(k), solver_config=solver_config)[:, 0]


@dataclass(frozen=True)
class DecayDataset:
    """One independently generated observed decay trajectory."""

    true_k: float
    time_points: np.ndarray
    latent: np.ndarray
    observed: np.ndarray
    sigma: float
    data_seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "true_k": float(self.true_k),
            "time_points": self.time_points.tolist(),
            "latent": self.latent.tolist(),
            "observed": self.observed.tolist(),
            "sigma": float(self.sigma),
            "data_seed": int(self.data_seed),
        }


def generate_decay_observation(
    true_k: float,
    data_seed: int,
    *,
    time_points: ArrayLike = (0.5, 1.0, 2.0, 3.0),
    sigma: float = 0.05,
    y0: float = 1.0,
    solver_config: ODESolverConfig | None = None,
) -> DecayDataset:
    """Generate one observed dataset with an independent data RNG stream."""

    times = _as_time_points(time_points)
    scale = float(sigma)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("sigma must be finite and positive")
    if isinstance(data_seed, bool) or int(data_seed) != data_seed:
        raise ValueError("data_seed must be an integer")
    latent = integrate_decay_ode(true_k, times, y0=y0, solver_config=solver_config)
    data_rng = np.random.default_rng(int(data_seed))
    observed = latent + data_rng.normal(0.0, scale, size=len(times))
    return DecayDataset(
        true_k=float(true_k),
        time_points=times,
        latent=latent,
        observed=np.asarray(observed, dtype=float),
        sigma=scale,
        data_seed=int(data_seed),
    )


class DecayObservationSimulator:
    """ABC simulator using its own fresh-noise generator for every call."""

    def __init__(
        self,
        time_points: ArrayLike,
        *,
        sigma: float = 0.05,
        simulator_noise_seed: int = 0,
        y0: float = 1.0,
        solver_config: ODESolverConfig | None = None,
    ) -> None:
        self.time_points = _as_time_points(time_points)
        self.sigma = float(sigma)
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if isinstance(simulator_noise_seed, bool) or int(simulator_noise_seed) != simulator_noise_seed:
            raise ValueError("simulator_noise_seed must be an integer")
        self.simulator_noise_seed = int(simulator_noise_seed)
        self.y0 = float(y0)
        if not np.isfinite(self.y0):
            raise ValueError("y0 must be finite")
        self.solver_config = ODESolverConfig() if solver_config is None else solver_config
        self._noise_rng = np.random.default_rng(self.simulator_noise_seed)
        self.calls = 0
        self.noise_draws = 0
        self.solver_failures = 0
        self.failure_categories: dict[str, int] = {}

    def __call__(self, theta: np.ndarray, _proposal_rng: np.random.Generator) -> np.ndarray:
        point = np.asarray(theta, dtype=float)
        if point.shape != (1,) or not np.all(np.isfinite(point)):
            raise ODESolverFailure("invalid_parameter", "simulator theta must be one finite scalar")
        self.calls += 1
        try:
            latent = integrate_decay_ode(
                float(point[0]),
                self.time_points,
                y0=self.y0,
                solver_config=self.solver_config,
            )
        except ODESolverFailure as exc:
            self.solver_failures += 1
            self.failure_categories[exc.category] = self.failure_categories.get(exc.category, 0) + 1
            raise
        noise = self._noise_rng.normal(0.0, self.sigma, size=len(self.time_points))
        self.noise_draws += len(noise)
        return latent + noise

    def diagnostics(self) -> dict[str, object]:
        return {
            "calls": int(self.calls),
            "noise_draws": int(self.noise_draws),
            "simulator_noise_seed": int(self.simulator_noise_seed),
            "solver_failures": int(self.solver_failures),
            "failure_categories": dict(self.failure_categories),
        }


def _central_chi_square_cdf_df4(x: float, poisson_index: int) -> float:
    """Central chi-square CDF for df ``4 + 2*poisson_index``.

    The integer-shape gamma formula is implemented directly so the target is
    independent of SciPy and of the inference simulator.  The upper-tail
    complement is used when it is stable; otherwise the equivalent Poisson
    upper-tail series is summed from its first term.
    """

    # A chi-square(df) variable is Gamma(shape=df/2, scale=2).  The
    # regularized-gamma argument is therefore x/2; retaining x here would
    # double the acceptance probability and shift every finite-epsilon target.
    value = float(x) / 2.0
    if value <= 0.0:
        return 0.0
    n = int(poisson_index) + 2  # shape = df/2 + poisson_index for df=4
    if n <= value:
        term = 1.0
        total = 1.0
        for m in range(1, n):
            term *= value / m
            total += term
        result = 1.0 - exp(-value) * total
    else:
        # P(Gamma(n, 1) <= x) = exp(-x) * sum_{m=n}^inf x^m/m!.
        term = exp(-value + n * log(value) - lgamma(n + 1.0))
        total = term
        m = n
        for _ in range(10_000):
            m += 1
            term *= value / m
            total += term
            if term <= total * 1e-16:
                break
            if term == 0.0:
                break
    return float(np.clip(result if n <= value else total, 0.0, 1.0))


def noncentral_chi_square_cdf(
    x: float | np.ndarray,
    *,
    df: int = 4,
    noncentrality: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate the noncentral-chi-square CDF via a Poisson mixture.

    The ODE control only permits ``df=4``.  For the small finite tolerances in
    the pilot this direct mixture is numerically stable and avoids sharing a
    SciPy implementation with the independent review calculation.
    """

    if int(df) != df or int(df) != 4:
        raise ValueError("the decay control requires df=4")
    threshold = float(x)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("x must be finite and non-negative")
    lambdas = np.asarray(noncentrality, dtype=float)
    if not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0):
        raise ValueError("noncentrality must be finite and non-negative")
    if threshold == 0.0:
        output = np.zeros_like(lambdas, dtype=float)
        return float(output) if output.ndim == 0 else output
    a = lambdas / 2.0
    max_a = float(np.max(a)) if a.size else 0.0
    # The pilot has small lambda values.  This branch remains vectorized for
    # the ordinary case.  The fallback handles extreme values in log space so
    # the callable still has an explicit bounded behavior.
    if max_a <= 700.0:
        max_index = int(np.ceil(max_a + 12.0 * sqrt(max_a + 1.0) + 50.0))
        result = np.zeros_like(a, dtype=float)
        poisson = np.exp(-a)
        for index in range(max_index + 1):
            result += poisson * _central_chi_square_cdf_df4(threshold, index)
            poisson *= a / float(index + 1)
        output = np.clip(result, 0.0, 1.0)
    else:
        flat = [
            _noncentral_chi_square_cdf_scalar(threshold, float(value))
            for value in lambdas.flat
        ]
        output = np.asarray(flat, dtype=float).reshape(lambdas.shape)
    return float(output) if output.ndim == 0 else output


def _noncentral_chi_square_cdf_scalar(x: float, noncentrality: float) -> float:
    a = noncentrality / 2.0
    if a == 0.0:
        return _central_chi_square_cdf_df4(x, 0)
    maximum = int(np.ceil(a + 12.0 * sqrt(a + 1.0) + 50.0))
    logs = np.asarray([-a + index * log(a) - lgamma(index + 1.0) for index in range(maximum + 1)])
    scale = float(np.max(logs))
    terms = np.exp(logs - scale)
    central = np.asarray([_central_chi_square_cdf_df4(x, index) for index in range(maximum + 1)])
    value = exp(scale) * float(np.sum(terms * central))
    return float(np.clip(value, 0.0, 1.0))


def _normal_cdf(values: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = np.fromiter(
        (0.5 * (1.0 + erf(float(item) / sqrt(2.0))) for item in array.flat),
        dtype=float,
        count=array.size,
    ).reshape(array.shape)
    return float(flat) if array.ndim == 0 else flat


class FiniteEpsilonODETarget:
    """Independent finite-epsilon target for one observed decay trajectory."""

    def __init__(
        self,
        observed: ArrayLike,
        time_points: ArrayLike,
        *,
        sigma: float = 0.05,
        epsilon: float = 2.0,
        bounds: Sequence[Sequence[float]] = ((0.05, 1.5),),
        quadrature_order: int = 256,
        _target_kind: str = "finite_epsilon_abc",
    ) -> None:
        self.observed = np.asarray(observed, dtype=float)
        self.time_points = _as_time_points(time_points)
        if self.observed.shape != self.time_points.shape or not np.all(np.isfinite(self.observed)):
            raise ValueError("observed and time_points must have matching finite shapes")
        self.sigma = float(sigma)
        self._target_kind = str(_target_kind)
        if self._target_kind not in {"finite_epsilon_abc", "exact_gaussian_likelihood"}:
            raise ValueError("unknown ODE target kind")
        self.epsilon = float(epsilon)
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if self._target_kind == "finite_epsilon_abc" and (not np.isfinite(self.epsilon) or self.epsilon < 0):
            raise ValueError("epsilon must be finite and non-negative")
        checked = np.asarray(bounds, dtype=float)
        if checked.shape != (1, 2) or not np.all(np.isfinite(checked)) or checked[0, 1] <= checked[0, 0]:
            raise ValueError("bounds must contain one finite (low, high) pair")
        self.lower = float(checked[0, 0])
        self.upper = float(checked[0, 1])
        if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or int(quadrature_order) < 16:
            raise ValueError("quadrature_order must be an integer >= 16")
        self.quadrature_order = int(quadrature_order)
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        self._nodes = nodes
        self._weights = weights
        self.normalizer = self._integrate(self._density_values, self.upper)
        if not np.isfinite(self.normalizer) or self.normalizer <= 0:
            raise ValueError(f"{self._target_kind} ODE target has non-positive normalizer")
        self.mean = self._integrate(lambda grid: grid * self._density_values(grid), self.upper) / self.normalizer

    def _integrate(self, function: Callable[[np.ndarray], np.ndarray], upper: float) -> float:
        if upper <= self.lower:
            return 0.0
        mapped = 0.5 * (upper - self.lower) * (self._nodes + 1.0) + self.lower
        values = np.asarray(function(mapped), dtype=float)
        return float(0.5 * (upper - self.lower) * np.sum(self._weights * values))

    def _density_values(self, k_values: np.ndarray) -> np.ndarray:
        rates = np.asarray(k_values, dtype=float)
        means = np.exp(-rates[:, None] * self.time_points[None, :])
        standardized = (means - self.observed[None, :]) / self.sigma
        noncentrality = np.sum(standardized * standardized, axis=1)
        if self._target_kind == "exact_gaussian_likelihood":
            return np.exp(-0.5 * noncentrality)
        return np.asarray(
            noncentral_chi_square_cdf(
                self.epsilon * self.epsilon,
                df=4,
                noncentrality=noncentrality,
            ),
            dtype=float,
        )

    def density(self, k: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(k, dtype=float)
        flat = values.reshape(-1)
        output = np.zeros(flat.shape, dtype=float)
        supported = np.isfinite(flat) & (flat >= self.lower) & (flat <= self.upper)
        if np.any(supported):
            output[supported] = self._density_values(flat[supported])
        output = output.reshape(values.shape)
        return float(output) if values.ndim == 0 else output

    def cdf(self, k: float) -> float:
        point = float(k)
        if point <= self.lower:
            return 0.0
        if point >= self.upper:
            return 1.0
        value = self._integrate(self._density_values, point) / self.normalizer
        return float(np.clip(value, 0.0, 1.0))

    def quantile(self, probability: float, *, iterations: int = 70) -> float:
        p = float(probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if p == 0.0:
            return self.lower
        if p == 1.0:
            return self.upper
        low, high = self.lower, self.upper
        for _ in range(iterations):
            midpoint = 0.5 * (low + high)
            if self.cdf(midpoint) < p:
                low = midpoint
            else:
                high = midpoint
        return 0.5 * (low + high)

    def quantiles(self, probabilities: Sequence[float]) -> np.ndarray:
        return np.asarray([self.quantile(float(p)) for p in probabilities], dtype=float)

    def _observation_predictive_cdf(self, future_time: float, value: float) -> float:
        t = float(future_time)
        if not np.isfinite(t) or t <= 0:
            raise ValueError("future times must be finite and positive")
        y = float(value)
        integral = self._integrate(
            lambda grid: (
                np.asarray(_normal_cdf((y - np.exp(-grid * t)) / self.sigma))
                * self._density_values(grid)
            ),
            self.upper,
        )
        return float(np.clip(integral / self.normalizer, 0.0, 1.0))

    def _observation_predictive_quantile(self, future_time: float, probability: float) -> float:
        t = float(future_time)
        p = float(probability)
        low = exp(-self.upper * t) - 10.0 * self.sigma
        high = exp(-self.lower * t) + 10.0 * self.sigma
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            if self._observation_predictive_cdf(t, midpoint) < p:
                low = midpoint
            else:
                high = midpoint
        return 0.5 * (low + high)

    def prediction_summary(
        self,
        future_times: ArrayLike = (4.0, 5.0),
        *,
        probabilities: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> dict[str, object]:
        times = np.asarray(future_times, dtype=float)
        if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("future_times must be a non-empty finite positive vector")
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 1 or np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probabilities must lie in [0, 1]")
        latent_means = []
        latent_quantiles = []
        observation_quantiles = []
        for time in times:
            latent_means.append(
                self._integrate(lambda grid: np.exp(-grid * time) * self._density_values(grid), self.upper)
                / self.normalizer
            )
            # k -> exp(-k*t) is decreasing, so reverse the parameter quantile.
            latent_quantiles.append([exp(-float(time) * self.quantile(1.0 - p)) for p in probs])
            observation_quantiles.append([self._observation_predictive_quantile(float(time), p) for p in probs])
        return {
            "future_times": times,
            "probabilities": probs,
            "latent_mean": np.asarray(latent_means, dtype=float),
            "latent_quantiles": np.asarray(latent_quantiles, dtype=float),
            "observation_predictive_quantiles": np.asarray(observation_quantiles, dtype=float),
            "semantics": {
                "latent": "weighted uncertainty in exp(-k*t); no measurement noise",
                "observation_predictive": "latent exp(-k*t) convolved with independent Normal(0, sigma^2) noise",
            },
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed.tolist(),
            "time_points": self.time_points.tolist(),
            "sigma": float(self.sigma),
            "target_kind": self._target_kind,
            "epsilon": float(self.epsilon) if self._target_kind == "finite_epsilon_abc" else None,
            "bounds": [[self.lower, self.upper]],
            "quadrature_order": int(self.quadrature_order),
            "normalizer": float(self.normalizer),
            "mean": float(self.mean),
        }


class GaussianLikelihoodODETarget(FiniteEpsilonODETarget):
    """Independent quadrature reference for the ordinary Gaussian likelihood.

    This target is proportional to ``exp(-lambda(k)/2)`` and is deliberately
    kept separate from the finite-tolerance ABC target.  It is a zero-
    tolerance likelihood reference, not the target sampled by the pilot.
    """

    def __init__(
        self,
        observed: ArrayLike,
        time_points: ArrayLike,
        *,
        sigma: float = 0.05,
        bounds: Sequence[Sequence[float]] = ((0.05, 1.5),),
        quadrature_order: int = 256,
    ) -> None:
        super().__init__(
            observed,
            time_points,
            sigma=sigma,
            epsilon=0.0,
            bounds=bounds,
            quadrature_order=quadrature_order,
            _target_kind="exact_gaussian_likelihood",
        )


def finite_epsilon_ode_target(
    observed: ArrayLike,
    time_points: ArrayLike,
    *,
    sigma: float = 0.05,
    epsilon: float = 2.0,
    bounds: Sequence[Sequence[float]] = ((0.05, 1.5),),
    quadrature_order: int = 256,
) -> FiniteEpsilonODETarget:
    """Construct the independent noncentral-chi-square quadrature target."""

    return FiniteEpsilonODETarget(
        observed,
        time_points,
        sigma=sigma,
        epsilon=epsilon,
        bounds=bounds,
        quadrature_order=quadrature_order,
    )


def gaussian_likelihood_ode_target(
    observed: ArrayLike,
    time_points: ArrayLike,
    *,
    sigma: float = 0.05,
    bounds: Sequence[Sequence[float]] = ((0.05, 1.5),),
    quadrature_order: int = 256,
) -> GaussianLikelihoodODETarget:
    """Construct the separately labeled ordinary Gaussian-likelihood target."""

    return GaussianLikelihoodODETarget(
        observed,
        time_points,
        sigma=sigma,
        bounds=bounds,
        quadrature_order=quadrature_order,
    )


def _validated_weights(values: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(values, dtype=float).reshape(-1)
    normalized = np.asarray(weights, dtype=float).reshape(-1)
    if len(samples) == 0 or normalized.shape != samples.shape:
        raise ValueError("values and weights must have matching non-empty vectors")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValueError("values and weights must be finite with non-negative weights")
    if not np.isclose(np.sum(normalized), 1.0, atol=1e-10):
        raise ValueError("weights must be normalized")
    return samples, normalized


def weighted_quantile(values: ArrayLike, weights: ArrayLike, probabilities: Sequence[float]) -> np.ndarray:
    samples, normalized = _validated_weights(values, weights)
    probs = np.asarray(probabilities, dtype=float)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    order = np.argsort(samples, kind="mergesort")
    sorted_values = samples[order]
    cumulative = np.cumsum(normalized[order])
    cumulative[-1] = 1.0
    return sorted_values[np.searchsorted(cumulative, probs, side="left")]


def weighted_empirical_cdf(values: ArrayLike, weights: ArrayLike, points: ArrayLike) -> np.ndarray:
    samples, normalized = _validated_weights(values, weights)
    grid = np.asarray(points, dtype=float).reshape(-1)
    order = np.argsort(samples, kind="mergesort")
    sorted_values = samples[order]
    cumulative = np.cumsum(normalized[order])
    indices = np.searchsorted(sorted_values, grid, side="right")
    output = np.zeros(len(grid), dtype=float)
    positive = indices > 0
    output[positive] = cumulative[indices[positive] - 1]
    return output


def weighted_parameter_diagnostics(
    values: ArrayLike,
    weights: ArrayLike,
    target: FiniteEpsilonODETarget,
    *,
    probabilities: Sequence[float] = (0.1, 0.5, 0.9),
    cdf_grid: ArrayLike | None = None,
) -> dict[str, object]:
    """Compare weighted parameter particles with one independent target."""

    samples, normalized = _validated_weights(values, weights)
    probs = np.asarray(probabilities, dtype=float)
    grid = np.linspace(target.lower, target.upper, 101) if cdf_grid is None else np.asarray(cdf_grid, dtype=float)
    empirical = weighted_empirical_cdf(samples, normalized, grid)
    target_cdf = np.asarray([target.cdf(float(point)) for point in grid], dtype=float)
    ess = float(1.0 / np.sum(normalized * normalized))
    mean = float(np.sum(normalized * samples))
    variance = float(np.sum(normalized * (samples - mean) ** 2))
    mean_se = sqrt(max(variance, 0.0) / max(ess, 1.0))
    quantiles = weighted_quantile(samples, normalized, probs)
    target_quantiles = target.quantiles(probs)
    quantile_errors = quantiles - target_quantiles
    return {
        "effective_sample_size": ess,
        "weighted_mean": mean,
        "target_mean": float(target.mean),
        "mean_error": mean - float(target.mean),
        "mean_mc_se": mean_se,
        "mean_error_over_mc_se": abs(mean - float(target.mean)) / max(mean_se, 1e-15),
        "probabilities": probs,
        "weighted_quantiles": quantiles,
        "target_quantiles": target_quantiles,
        "quantile_errors": quantile_errors,
        "cdf_grid": grid,
        "weighted_cdf": empirical,
        "target_cdf": target_cdf,
        "cdf_errors": empirical - target_cdf,
        "cdf_max_abs_error": float(np.max(np.abs(empirical - target_cdf))),
        "diagnostic_note": "ESS-scaled errors are descriptive for dependent weighted SMC particles, not confidence guarantees.",
    }


def weighted_decay_prediction_summary(
    values: ArrayLike,
    weights: ArrayLike,
    *,
    future_times: ArrayLike = (4.0, 5.0),
    sigma: float = 0.05,
    probabilities: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, object]:
    """Summarize weighted latent and observation-predictive trajectories."""

    rates, normalized = _validated_weights(values, weights)
    times = np.asarray(future_times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("future_times must be a non-empty finite positive vector")
    scale = float(sigma)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("sigma must be finite and positive")
    probs = np.asarray(probabilities, dtype=float)
    latent_values = np.exp(-times[:, None] * rates[None, :])
    latent_means = np.sum(latent_values * normalized[None, :], axis=1)
    latent_quantiles = np.asarray([weighted_quantile(row, normalized, probs) for row in latent_values])
    observation_quantiles = []
    for row in latent_values:
        def predictive_cdf(value: float) -> float:
            return float(np.sum(normalized * np.asarray(_normal_cdf((float(value) - row) / scale))))

        low = float(np.min(row) - 10.0 * scale)
        high = float(np.max(row) + 10.0 * scale)
        row_quantiles = []
        for probability in probs:
            # Reset the bracket for each quantile.  Reusing a bracket after
            # the lower quantile would silently force every later quantile to
            # remain below that first root.
            low = float(np.min(row) - 10.0 * scale)
            high = float(np.max(row) + 10.0 * scale)
            for _ in range(80):
                midpoint = 0.5 * (low + high)
                if predictive_cdf(midpoint) < probability:
                    low = midpoint
                else:
                    high = midpoint
            row_quantiles.append(0.5 * (low + high))
        observation_quantiles.append(row_quantiles)
    return {
        "future_times": times,
        "probabilities": probs,
        "latent_mean": latent_means,
        "latent_quantiles": latent_quantiles,
        "observation_predictive_quantiles": np.asarray(observation_quantiles, dtype=float),
        "semantics": {
            "latent": "weighted particles of exp(-k*t); no measurement noise",
            "observation_predictive": "weighted mixture Normal(exp(-k*t), sigma^2); measurement noise included",
        },
    }


def _transition_recheck(
    reference: dict[str, object],
    *,
    bounds: Sequence[Sequence[float]] = ((0.05, 1.5),),
) -> dict[str, object]:
    populations = reference.get("populations", [])
    checks: list[dict[str, object]] = []
    for generation in range(1, len(populations)):
        previous = populations[generation - 1]
        current = populations[generation]
        params = np.asarray(current["accepted_params"], dtype=float)
        stored_weights = np.asarray(current["weights"], dtype=float)
        stored_log_prior = np.asarray(current["log_prior_density"], dtype=float)
        stored_log_proposal = np.asarray(current["log_proposal_mixture_density"], dtype=float)
        covariance = np.asarray(current["proposal_covariance"], dtype=float)
        previous_params = np.asarray(previous["accepted_params"], dtype=float)
        previous_weights = np.asarray(previous["weights"], dtype=float)
        if len(params) == 0 or not np.all(np.isfinite(stored_weights)):
            checks.append({"generation": generation, "status": "not_recomputed", "reason": "partial_or_nonfinite_weights"})
            continue
        _, prior_logpdf = make_uniform_prior(bounds)
        recomputed_prior = np.asarray([prior_logpdf(point) for point in params], dtype=float)
        recomputed_proposal = np.asarray(
            [gaussian_mixture_logpdf(point, previous_params, previous_weights, covariance) for point in params],
            dtype=float,
        )
        raw = recomputed_prior - recomputed_proposal
        recomputed_weights = np.exp(raw - np.max(raw))
        recomputed_weights /= np.sum(recomputed_weights)
        checks.append(
            {
                "generation": generation,
                "status": "recomputed",
                "particle_count": int(len(params)),
                "max_abs_log_prior_error": float(np.max(np.abs(recomputed_prior - stored_log_prior))),
                "max_abs_log_proposal_error": float(np.max(np.abs(recomputed_proposal - stored_log_proposal))),
                "max_abs_weight_error": float(np.max(np.abs(recomputed_weights - stored_weights))),
            }
        )
    return {"checks": checks}


def run_ode_decay_control_cell(
    dataset: DecayDataset,
    *,
    proposal_seed: int,
    simulator_noise_seed: int,
    target_samples: int = 400,
    epsilon_schedule: Sequence[float] = (3.0, 2.0),
    max_attempts_per_population: int | Sequence[int] = 50_000,
    bounds: Sequence[Sequence[float]] = ((0.05, 1.5),),
    solver_config: ODESolverConfig | None = None,
    quadrature_order: int = 256,
    wall_deadline: float | None = None,
) -> dict[str, object]:
    """Run one fully bounded ODE control cell and attach audit diagnostics."""

    config = ODESolverConfig() if solver_config is None else solver_config
    prior_sampler, prior_logpdf = make_uniform_prior(bounds)
    simulator = DecayObservationSimulator(
        dataset.time_points,
        sigma=dataset.sigma,
        simulator_noise_seed=simulator_noise_seed,
        solver_config=config,
    )
    observed = np.asarray(dataset.observed, dtype=float)
    population_timing: dict[int, dict[str, object]] = {}

    def population_event(event: str, generation: int, _record: dict[str, object] | None) -> None:
        if event == "start":
            population_timing[generation] = {
                "started_monotonic": __import__("time").monotonic(),
                "simulator_calls_before": simulator.calls,
                "solver_failures_before": simulator.solver_failures,
            }
            return
        if event != "end":
            raise ValueError(f"unknown population event {event!r}")
        started_record = population_timing[generation]
        failure_counts = dict(simulator.failure_categories)
        before_failures = dict(started_record.get("failure_categories_before", {}))
        started_record.update(
            {
                "elapsed_seconds": __import__("time").monotonic() - float(started_record["started_monotonic"]),
                "simulator_calls": simulator.calls - int(started_record["simulator_calls_before"]),
                "solver_failures": simulator.solver_failures - int(started_record["solver_failures_before"]),
                "solver_failure_categories": {
                    key: int(value) - int(before_failures.get(key, 0))
                    for key, value in failure_counts.items()
                    if int(value) - int(before_failures.get(key, 0))
                },
            }
        )

    # Retain category counters at each population boundary so the end event
    # reports per-population deltas rather than cumulative simulator totals.
    def timed_population_event(event: str, generation: int, record: dict[str, object] | None) -> None:
        if event == "start":
            population_event(event, generation, record)
            population_timing[generation]["failure_categories_before"] = dict(simulator.failure_categories)
        else:
            population_event(event, generation, record)

    def discrepancy(simulated: ArrayLike) -> float:
        values = np.asarray(simulated, dtype=float)
        if values.shape != observed.shape or not np.all(np.isfinite(values)):
            return float("nan")
        return float(np.linalg.norm((values - observed) / dataset.sigma))

    started = __import__("time").monotonic()
    reference = run_gaussian_abc_smc_reference(
        prior_sampler,
        simulator,
        discrepancy,
        target_samples=target_samples,
        epsilon_schedule=epsilon_schedule,
        max_attempts_per_population=max_attempts_per_population,
        bounds=bounds,
        prior_logpdf=prior_logpdf,
        seed=int(proposal_seed),
        population_event=timed_population_event,
        stop_requested=(None if wall_deadline is None else lambda: __import__("time").monotonic() >= wall_deadline),
    )
    elapsed = __import__("time").monotonic() - started
    final_params = np.asarray(reference["accepted_params"], dtype=float)
    final_weights = np.asarray(reference["weights"], dtype=float)
    for generation, timing in population_timing.items():
        if generation < len(reference["populations"]):
            diagnostics = reference["populations"][generation]["diagnostics"]
            diagnostics.update(
                {
                    "elapsed_seconds": float(timing.get("elapsed_seconds", 0.0)),
                    "solver_failures": int(timing.get("solver_failures", 0)),
                    "solver_failure_categories": dict(timing.get("solver_failure_categories", {})),
                    "simulator_calls_timed": int(timing.get("simulator_calls", 0)),
                }
            )
    target = finite_epsilon_ode_target(
        dataset.observed,
        dataset.time_points,
        sigma=dataset.sigma,
        epsilon=float(list(epsilon_schedule)[-1]),
        bounds=bounds,
        quadrature_order=quadrature_order,
    )
    exact_target = gaussian_likelihood_ode_target(
        dataset.observed,
        dataset.time_points,
        sigma=dataset.sigma,
        bounds=bounds,
        quadrature_order=quadrature_order,
    )
    if (
        reference["status"] == "complete"
        and final_params.ndim == 2
        and final_params.shape[1] == 1
        and len(final_params)
        and np.all(np.isfinite(final_weights))
    ):
        parameter_diagnostics: dict[str, object] = weighted_parameter_diagnostics(
            final_params[:, 0], final_weights, target
        )
        predictions = weighted_decay_prediction_summary(
            final_params[:, 0],
            final_weights,
            future_times=(4.0, 5.0),
            sigma=dataset.sigma,
        )
    else:
        parameter_diagnostics = {
            "status": "unavailable",
            "reason": "incomplete_or_nonfinite_final_weights",
            "posterior_summary": False,
        }
        predictions = {
            "status": "unavailable",
            "reason": "incomplete_or_nonfinite_final_weights",
            "posterior_summary": False,
        }
    return {
        "status": reference["status"],
        "termination_reason": reference["termination_reason"],
        "dataset": dataset.as_dict(),
        "seeds": {
            "data_seed": int(dataset.data_seed),
            "proposal_seed": int(proposal_seed),
            "simulator_noise_seed": int(simulator_noise_seed),
        },
        "solver_config": config.as_dict(),
        "reference": reference,
        "target": target.as_dict(),
        "exact_gaussian_likelihood_target": exact_target.as_dict(),
        "exact_gaussian_likelihood_summary": {
            "mean": float(exact_target.mean),
            "quantiles": exact_target.quantiles((0.1, 0.5, 0.9)),
        },
        "target_distinction": {
            "finite_epsilon_target": "proportional to P(D <= epsilon | k) with noncentral-chi-square D^2",
            "exact_gaussian_likelihood_target": "proportional to exp(-lambda(k)/2), reported as a separate zero-tolerance reference",
        },
        "parameter_diagnostics": parameter_diagnostics,
        "target_prediction": target.prediction_summary((4.0, 5.0)),
        "particle_prediction": predictions,
        "transition_recheck": _transition_recheck(reference, bounds=bounds),
        "simulator": simulator.diagnostics(),
        "population_timing": {
            str(generation): {
                key: value
                for key, value in timing.items()
                if key not in {"started_monotonic", "simulator_calls_before", "solver_failures_before", "failure_categories_before"}
            }
            for generation, timing in population_timing.items()
        },
        "elapsed_seconds": float(elapsed),
    }


def solver_accuracy_check(
    *,
    solver_config: ODESolverConfig | None = None,
    rates: Sequence[float] = (0.05, 0.35, 0.8, 1.5),
    time_points: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
) -> dict[str, object]:
    """Compare numerical RK4 trajectories with the analytic solution."""

    config = ODESolverConfig() if solver_config is None else solver_config
    times = _as_time_points(time_points)
    errors: list[float] = []
    for rate in rates:
        numerical = integrate_decay_ode(float(rate), times, solver_config=config)
        analytic = np.exp(-float(rate) * times)
        errors.append(float(np.max(np.abs(numerical - analytic))))
    maximum = max(errors) if errors else float("nan")
    return {
        "solver_config": config.as_dict(),
        "rates": [float(rate) for rate in rates],
        "time_points": times,
        "max_abs_errors": np.asarray(errors, dtype=float),
        "max_abs_error": float(maximum),
        "measurement_sigma": 0.05,
        "passed": bool(np.isfinite(maximum) and maximum <= config.error_tolerance),
    }


__all__ = [
    "DecayDataset",
    "DecayObservationSimulator",
    "FiniteEpsilonODETarget",
    "GaussianLikelihoodODETarget",
    "ODEControlError",
    "ODESolverConfig",
    "ODESolverFailure",
    "decay_rhs",
    "finite_epsilon_ode_target",
    "gaussian_likelihood_ode_target",
    "generate_decay_observation",
    "integrate_decay_ode",
    "integrate_ode",
    "noncentral_chi_square_cdf",
    "run_ode_decay_control_cell",
    "solver_accuracy_check",
    "weighted_decay_prediction_summary",
    "weighted_empirical_cdf",
    "weighted_parameter_diagnostics",
    "weighted_quantile",
]

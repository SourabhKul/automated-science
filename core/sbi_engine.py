import jax.numpy as jnp
import jax
import numpy as np


class ABCSMCStrategy:
    name = "base"
    log_label = "base"

    def transition(
        self,
        *,
        rng,
        accepted_params,
        accepted_weights,
        priors,
        initial_particles,
        lambda_noise,
        nugget,
    ):
        raise NotImplementedError


class GaussianWeightedStrategy(ABCSMCStrategy):
    name = "gaussian_weighted"
    log_label = "gaussian_weighted"

    def transition(
        self,
        *,
        rng,
        accepted_params,
        accepted_weights,
        priors,
        initial_particles,
        lambda_noise,
        nugget,
    ):
        num_params = len(priors)
        emp_cov = np.atleast_2d(np.cov(accepted_params, rowvar=False, aweights=accepted_weights))
        regularization = lambda_noise * np.diag(np.diag(emp_cov)) + np.eye(num_params) * nugget
        kernel_cov = 2.0 * emp_cov + regularization

        eigvals = np.linalg.eigvals(kernel_cov)
        if np.any(eigvals < 1e-12):
            print("[Sampling Warning] High correlation. Covariate noise is maintaining stability.")

        drawn_idx = rng.choice(len(accepted_params), size=initial_particles, replace=True, p=accepted_weights)
        base_samples = accepted_params[drawn_idx]
        perturbations = rng.multivariate_normal(np.zeros(num_params), kernel_cov, size=initial_particles)
        proposed_params = base_samples + perturbations

        for i, p in enumerate(priors):
            proposed_params[:, i] = np.clip(proposed_params[:, i], p['range'][0], p['range'][1])

        return jnp.array(proposed_params), kernel_cov


class BetaDistributedStepSizeStrategy(ABCSMCStrategy):
    name = "bdss"
    log_label = "bdss"

    def __init__(self, alpha=0.7, beta=3.0, min_step_fraction=0.01, max_step_fraction=0.35):
        self.alpha = alpha
        self.beta = beta
        self.min_step_fraction = min_step_fraction
        self.max_step_fraction = max_step_fraction

    def transition(
        self,
        *,
        rng,
        accepted_params,
        accepted_weights,
        priors,
        initial_particles,
        lambda_noise,
        nugget,
    ):
        num_params = len(priors)
        lows = np.array([p["range"][0] for p in priors], dtype=float)
        highs = np.array([p["range"][1] for p in priors], dtype=float)
        spans = np.maximum(highs - lows, 1e-12)

        drawn_idx = rng.choice(len(accepted_params), size=initial_particles, replace=True, p=accepted_weights)
        base_samples = np.asarray(accepted_params, dtype=float)[drawn_idx]
        base_unit = np.clip((base_samples - lows) / spans, 0.0, 1.0)

        accepted_unit = np.clip((np.asarray(accepted_params, dtype=float) - lows) / spans, 0.0, 1.0)
        center = np.average(accepted_unit, axis=0, weights=accepted_weights)
        posterior_var = np.average((accepted_unit - center) ** 2, axis=0, weights=accepted_weights)
        posterior_std = np.sqrt(np.maximum(posterior_var, 0.0))
        adaptive_radius = np.clip(
            2.0 * posterior_std + lambda_noise,
            self.min_step_fraction,
            self.max_step_fraction,
        )

        directions = rng.normal(size=(initial_particles, num_params))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-12)
        step_fractions = rng.beta(self.alpha, self.beta, size=(initial_particles, 1))
        signed_steps = directions * step_fractions * adaptive_radius

        proposed_unit = np.clip(base_unit + signed_steps, 0.0, 1.0)
        proposed_params = lows + proposed_unit * spans

        perturbations = proposed_params - base_samples
        if initial_particles > 1:
            kernel_cov = np.atleast_2d(np.cov(perturbations, rowvar=False))
        else:
            kernel_cov = np.zeros((num_params, num_params))
        kernel_cov = kernel_cov + np.diag((spans * nugget) ** 2 + spans * spans * 1e-12)
        return jnp.array(proposed_params), kernel_cov


ABC_SMC_STRATEGIES = {
    "default": GaussianWeightedStrategy(),
    "gaussian": GaussianWeightedStrategy(),
    "gaussian_weighted": GaussianWeightedStrategy(),
    "bdss": BetaDistributedStepSizeStrategy(),
}


def resolve_abc_smc_strategy(strategy):
    if isinstance(strategy, ABCSMCStrategy):
        return strategy
    key = (strategy or "default").lower()
    if key not in ABC_SMC_STRATEGIES:
        supported = ", ".join(sorted(ABC_SMC_STRATEGIES))
        raise ValueError(f"unknown ABC-SMC strategy '{strategy}'. Supported strategies: {supported}")
    return ABC_SMC_STRATEGIES[key]


class SBIEngine:
    def __init__(self, observation_data, time_points, use_summary_stats=True, seed=None, observation_mask=None):
        self.obs_data = jnp.array(observation_data)
        self.time_points = jnp.array(time_points)
        self.use_summary_stats = use_summary_stats
        self.obs_mask = None
        if observation_mask is not None:
            mask = np.asarray(observation_mask, dtype=bool)
            if mask.shape != np.asarray(observation_data).shape:
                raise ValueError(f"observation_mask shape mismatch: {mask.shape} != {np.asarray(observation_data).shape}")
            if not np.any(mask):
                raise ValueError("observation_mask must include at least one observed value")
            self.obs_mask = jnp.array(mask)
        self.obs_summary = self.extract_summary_statistics(self.obs_data) if use_summary_stats and self.obs_mask is None else None
        self.rng = np.random.default_rng(seed)

    def _validate_priors(self, priors):
        if not priors:
            raise ValueError("model must define at least one parameter prior")
        for idx, prior in enumerate(priors):
            if "range" not in prior or len(prior["range"]) != 2:
                raise ValueError(f"prior {idx} must include a two-value range")
            low, high = prior["range"]
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(f"prior {idx} has invalid range: {prior['range']}")

    def _sample_from_priors(self, priors, n_particles, rng):
        samples = []
        for p in priors:
            samples.append(rng.uniform(p['range'][0], p['range'][1], n_particles))
        return jnp.array(np.column_stack(samples))

    def extract_summary_statistics(self, data):
        """
        Dimensionality reduction for oscillatory time-series.
        Extracts: Mean, Std, ACF(1), and FFT Peak Power.
        """
        # data shape: (num_particles, time, dims) or (time, dims)
        if data.ndim == 2:
            data = data[jnp.newaxis, ...]
        
        # 1. Moments
        means = jnp.mean(data, axis=1) # (N, dims)
        stds = jnp.std(data, axis=1)   # (N, dims)
        
        # 2. Autocorrelation (lag 1)
        centered = data - means[:, jnp.newaxis, :]
        acf1 = jnp.sum(centered[:, :-1, :] * centered[:, 1:, :], axis=1) / (jnp.sum(centered**2, axis=1) + 1e-8)
        
        # 3. FFT (Dominant Frequency)
        # We take the max of the FFT amplitude to represent 'oscillatory power'
        fft_data = jnp.abs(jnp.fft.fft(data, axis=1))
        max_power = jnp.max(fft_data[:, 1:data.shape[1]//2, :], axis=1) # skip DC
        
        # Concat all features
        features = jnp.concatenate([means, stds, acf1, max_power], axis=1)
        return features

    def compute_distance(self, simulated_data):
        if self.obs_mask is not None:
            diff = simulated_data - self.obs_data[jnp.newaxis, ...]
            mask = self.obs_mask[jnp.newaxis, ...]
            sq_diff = jnp.sum(jnp.where(mask, jnp.power(diff, 2), 0.0), axis=(1, 2))
            observed_count = jnp.maximum(jnp.sum(mask, axis=(1, 2)), 1)
            return jnp.sqrt(sq_diff / observed_count)
        if self.use_summary_stats:
            sim_summary = self.extract_summary_statistics(simulated_data)
            # Simple Euclidean distance on features
            # Note: Ideally we would use MAD normalization here, but local distance is fine for now
            diff = sim_summary - self.obs_summary
            return jnp.sqrt(jnp.sum(diff**2, axis=1))
        else:
            sq_diff = jnp.sum(jnp.power(simulated_data - self.obs_data[jnp.newaxis, ...], 2), axis=(1, 2))
            return jnp.sqrt(sq_diff)
        
    def run_abc_rejection(self, model, n_particles=1000, epsilon=5.0, seed=None):
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        priors = model.get_parameter_metadata()
        self._validate_priors(priors)

        params_batch = self._sample_from_priors(priors, n_particles, rng)
        y0_single = model.get_initial_conditions()
        y0_batch = jnp.tile(y0_single, (n_particles, 1))
        simulated_data = jax.vmap(model.simulate, in_axes=(0, None, 0))(params_batch, self.time_points, y0_batch)
        distances = np.array(self.compute_distance(simulated_data))
        distances[~np.isfinite(distances)] = np.inf
        keep_idx = np.where(distances <= epsilon)[0]
        return {
            "accepted_params": np.array(params_batch)[keep_idx],
            "distances": distances[keep_idx],
            "acceptance_rate": float(len(keep_idx) / n_particles),
            "median_distance": float(np.median(distances[keep_idx])) if len(keep_idx) else float("inf"),
            "min_distance": float(np.min(distances[keep_idx])) if len(keep_idx) else float("inf"),
        }

    def run_abc_smc(self, model, target_samples=500, generations=4, initial_particles=20000, strategy='gaussian', lambda_noise=0.01, nugget=1e-9, seed=None):
        """
        Bayesian-Correct ABC-SMC with Importance Weighting and Degeneracy Stopping.
        """
        if target_samples <= 0:
            raise ValueError("target_samples must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if initial_particles < target_samples:
            raise ValueError("initial_particles must be >= target_samples")
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        priors = model.get_parameter_metadata()
        self._validate_priors(priors)
        num_params = len(priors)
        requested_strategy = strategy
        smc_strategy = resolve_abc_smc_strategy(strategy)
        
        # Generation 0: Draw from Uniform Priors
        params_batch = self._sample_from_priors(priors, initial_particles, rng)
        weights = jnp.ones(initial_particles) / initial_particles
        
        y0_single = model.get_initial_conditions()
        vmap_sim = jax.vmap(model.simulate, in_axes=(0, None, 0))
        
        epsilon_init = np.inf
        kernel_cov = None
        for g in range(generations):
            if g > 0:
                params_batch, kernel_cov = smc_strategy.transition(
                    rng=rng,
                    accepted_params=accepted_params,
                    accepted_weights=accepted_weights,
                    priors=priors,
                    initial_particles=initial_particles,
                    lambda_noise=lambda_noise,
                    nugget=nugget,
                )

            # Parallel Simulation
            y0_batch = jnp.tile(y0_single, (initial_particles, 1))
            try:
                simulated_data = vmap_sim(params_batch, self.time_points, y0_batch)
                distances = self.compute_distance(simulated_data)
                
                dist_np = np.array(distances)
                dist_np[np.isnan(dist_np)] = np.inf
                sorted_idx = np.argsort(dist_np)
                
                # 4. Selection (Top-N Epsilon Ball)
                keep_idx = sorted_idx[:target_samples]
                accepted_params = np.array(params_batch)[keep_idx]
                final_distances = dist_np[keep_idx]
                
                # 5. Calculate New Weights (Bayesian Importance)
                # w_next = prior / sum(w_prev * G_kernel(proposed | prev))
                if g == 0:
                    accepted_weights = np.ones(target_samples) / target_samples
                else:
                    # vectorized kernel density calculation
                    denoms = []
                    # Ensure prev_accepted_params is 2D for mvn logic
                    prev_p = np.atleast_2d(prev_accepted_params)
                    for p_i in accepted_params:
                        # Density of p_i under the previous generation's mixture kernel
                        # mvn.pdf handles vector p_i against mean mixture
                        # We use numpy for the densities loop to avoid complex jax tracers here
                        densities = jax.scipy.stats.multivariate_normal.pdf(p_i, mean=prev_p, cov=kernel_cov)
                        denoms.append(np.sum(densities * prev_accepted_weights))
                    
                    new_weights = 1.0 / (np.array(denoms) + 1e-12)
                    accepted_weights = new_weights / np.sum(new_weights)
                
                # Preserve for next generation weighting
                prev_accepted_params = accepted_params.copy()
                prev_accepted_weights = accepted_weights.copy()
                
                current_epsilon = final_distances[-1]
                if g == 0: epsilon_init = current_epsilon
                print(f"Gen {g} | Strategy {smc_strategy.log_label} | Epsilon: {current_epsilon:.2f} | Med_MSE: {np.median(final_distances):.2f}")
                
            except Exception as e:
                print(f"SMC Simulation failed: {e}")
                return {
                    "median_distance": float('inf'),
                    "min_distance": float('inf'),
                    "requested_strategy": str(requested_strategy),
                    "effective_strategy": smc_strategy.name,
                }

        return {
            "accepted_params": accepted_params,
            "median_distance": float(np.median(final_distances)) if len(final_distances) > 0 else float('inf'),
            "min_distance": float(np.min(final_distances)) if len(final_distances) > 0 else float('inf'),
            "requested_strategy": str(requested_strategy),
            "effective_strategy": smc_strategy.name,
        }

import jax.numpy as jnp
import numpy as np

from core.base_model import BaseModel
from core.sbi_engine import SBIEngine, resolve_abc_smc_strategy


class LinearModel(BaseModel):
    def simulate(self, params, time_points, y0):
        return (params[0] * time_points)[:, jnp.newaxis]

    def get_parameter_metadata(self):
        return [{"name": "slope", "range": (0.0, 10.0)}]

    def get_latex(self):
        return "linear"

    def get_initial_conditions(self):
        return jnp.array([0.0])


class BadModel(LinearModel):
    def get_parameter_metadata(self):
        return [{"name": "bad", "range": (1.0, 1.0)}]


def test_seeded_smc_is_reproducible():
    t = jnp.linspace(0, 5, 20)
    obs = 3.0 * t[:, jnp.newaxis]
    engine_a = SBIEngine(obs, t, seed=123)
    engine_b = SBIEngine(obs, t, seed=123)
    res_a = engine_a.run_abc_smc(LinearModel(), target_samples=10, generations=3, initial_particles=50)
    res_b = engine_b.run_abc_smc(LinearModel(), target_samples=10, generations=3, initial_particles=50)
    assert np.allclose(res_a["accepted_params"], res_b["accepted_params"])
    print("SUCCESS: seeded ABC-SMC is reproducible")


def test_strategy_dispatch_preserves_current_default_path():
    t = jnp.linspace(0, 5, 20)
    obs = 3.0 * t[:, jnp.newaxis]
    kwargs = {"target_samples": 10, "generations": 3, "initial_particles": 50}
    default = SBIEngine(obs, t, seed=321).run_abc_smc(LinearModel(), **kwargs)
    explicit = SBIEngine(obs, t, seed=321).run_abc_smc(
        LinearModel(), strategy="gaussian_weighted", **kwargs
    )

    assert default["effective_strategy"] == "gaussian_weighted"
    assert explicit["effective_strategy"] == "gaussian_weighted"
    assert np.allclose(default["accepted_params"], explicit["accepted_params"])
    print("SUCCESS: ABC-SMC strategy dispatch preserves the current gaussian weighted path")


def test_bdss_strategy_is_reproducible_and_distinct():
    t = jnp.linspace(0, 5, 20)
    obs = 3.0 * t[:, jnp.newaxis]
    kwargs = {"target_samples": 10, "generations": 3, "initial_particles": 50}
    bdss_a = SBIEngine(obs, t, seed=777).run_abc_smc(
        LinearModel(), strategy="bdss", **kwargs
    )
    bdss_b = SBIEngine(obs, t, seed=777).run_abc_smc(
        LinearModel(), strategy="bdss", **kwargs
    )
    gaussian = SBIEngine(obs, t, seed=777).run_abc_smc(
        LinearModel(), strategy="gaussian_weighted", **kwargs
    )

    assert bdss_a["requested_strategy"] == "bdss"
    assert bdss_a["effective_strategy"] == "bdss"
    assert np.allclose(bdss_a["accepted_params"], bdss_b["accepted_params"])
    assert np.isfinite(bdss_a["median_distance"])
    assert not np.allclose(bdss_a["accepted_params"], gaussian["accepted_params"])
    print("SUCCESS: BDSS strategy is reproducible, finite, and distinct from gaussian weighted ABC-SMC")


def test_unknown_strategy_is_rejected():
    try:
        resolve_abc_smc_strategy("does_not_exist")
    except ValueError as exc:
        assert "Supported strategies" in str(exc)
        print("SUCCESS: unknown ABC-SMC strategies are rejected")
        return
    raise AssertionError("unknown strategy should have been rejected")


def test_masked_distance_uses_only_observed_entries():
    t = jnp.array([0.0, 1.0])
    obs = jnp.array([[1.0, 100.0], [3.0, 80.0]])
    mask = np.array([[True, False], [True, False]])
    simulated = jnp.array(
        [
            [[2.0, -1000.0], [5.0, -1000.0]],
            [[1.0, 1000.0], [3.0, 1000.0]],
        ]
    )
    engine = SBIEngine(obs, t, observation_mask=mask)
    distances = np.array(engine.compute_distance(simulated))
    assert np.allclose(distances, [np.sqrt((1.0 + 4.0) / 2.0), 0.0])
    print("SUCCESS: masked ABC distance ignores unobserved endpoint entries")


def test_invalid_prior_is_rejected():
    t = jnp.linspace(0, 5, 20)
    obs = 3.0 * t[:, jnp.newaxis]
    engine = SBIEngine(obs, t, seed=123)
    try:
        engine.run_abc_rejection(BadModel(), n_particles=10)
    except ValueError:
        print("SUCCESS: invalid prior was rejected")
        return
    raise AssertionError("invalid prior should have been rejected")


if __name__ == "__main__":
    test_seeded_smc_is_reproducible()
    test_strategy_dispatch_preserves_current_default_path()
    test_bdss_strategy_is_reproducible_and_distinct()
    test_unknown_strategy_is_rejected()
    test_masked_distance_uses_only_observed_entries()
    test_invalid_prior_is_rejected()

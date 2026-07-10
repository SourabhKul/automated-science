import numpy as np
import jax.numpy as jnp
from core.sbi_engine import SBIEngine
from core.base_model import BaseModel

class MockModel(BaseModel):
    def get_parameter_metadata(self):
        return [{'name': 'p1', 'range': (0, 10)}]
    def get_initial_conditions(self):
        return jnp.array([1.0])
    def get_latex(self):
        return "mock"
    def simulate(self, params, time_points, y0):
        return (params[0] * time_points)[:, jnp.newaxis]

def test_stress():
    print("Testing Covariate Noise (Nugget Effect) Stability...")
    t = jnp.linspace(0, 10, 50)
    obs_data = 5.0 * t[:, jnp.newaxis]
    
    engine = SBIEngine(obs_data, t, use_summary_stats=True)
    model = MockModel()
    
    # Run for 15 generations to force degeneracy
    print("Running 15-generation ABC-SMC (Stress Test)...")
    res = engine.run_abc_smc(model, target_samples=50, generations=15, initial_particles=200)
    
    best_params = res['accepted_params']
    median_p1 = np.median(best_params)
    var_p1 = np.var(best_params)
    print(f"Final Gen Results: Median p1 = {median_p1:.4f}, Variance = {var_p1:.2e}")
    
    if len(best_params) == 50:
        print("SUCCESS: Engine completed all 15 generations without crashing or early-stopping.")
    else:
        print("FAILURE: Engine did not complete the full generation loop.")

if __name__ == "__main__":
    test_stress()

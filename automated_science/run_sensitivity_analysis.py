import numpy as np
import jax.numpy as jnp
import os
import json
import time
from core.sbi_engine import SBIEngine
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

# Define a simple Lotka-Volterra model for sensitivity testing
def dynamics(t, y, args):
    r, a, b, m = args
    H, L = y
    dH = r * H - a * H * L
    dL = b * H * L - m * L
    return jnp.array([dH, dL])

class TestEcologyModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        try:
            sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0=float(time_points[0]), t1=float(time_points[-1]), dt0=0.5, y0=y0, args=params, saveat=saveat, max_steps=1000)
            return sol.ys
        except Exception:
            return jnp.zeros((len(time_points), y0.shape[0]))
            
    def get_parameter_metadata(self):
        return [
            {'name': 'r', 'range': (0.1, 2.0)},
            {'name': 'a', 'range': (0.01, 0.5)},
            {'name': 'b', 'range': (0.01, 0.5)},
            {'name': 'm', 'range': (0.1, 2.0)}
        ]
        
    def get_initial_conditions(self):
        return jnp.array([40.0, 9.0])
        
    def get_latex(self):
        return "Lotka-Volterra Test Model"

def main():
    print("=== Starting ABC-SMC Sensitivity Analysis ===")
    
    # Load or generate test data
    data_path = "data/ecology_ground_truth.npy"
    time_path = "data/ecology_time_points.npy"
    if not os.path.exists(data_path) or not os.path.exists(time_path):
        import ecology_data_loader as dl
        dl.generate_data()
        
    gt_data = np.load(data_path)
    t_points = np.load(time_path)
    
    engine = SBIEngine(gt_data, t_points)
    model = TestEcologyModel()
    
    # Parameters to test
    lambda_values = [0.001, 0.01, 0.1]
    particle_counts = [1000, 5000, 10000]
    
    results = []
    
    # 1. Test Lambda Noise Sensitivity
    for l_val in lambda_values:
        print(f"\nEvaluating lambda_noise = {l_val}...")
        start_time = time.time()
        res = engine.run_abc_smc(
            model, 
            target_samples=200, 
            generations=5, 
            initial_particles=5000, 
            lambda_noise=l_val
        )
        elapsed = time.time() - start_time
        results.append({
            "test_type": "lambda_noise",
            "value": l_val,
            "median_mse": res["median_distance"],
            "min_mse": res["min_distance"],
            "time_seconds": elapsed
        })
        
    # 2. Test Particle Count Sensitivity
    for p_count in particle_counts:
        print(f"\nEvaluating initial_particles = {p_count}...")
        start_time = time.time()
        res = engine.run_abc_smc(
            model, 
            target_samples=200, 
            generations=5, 
            initial_particles=p_count, 
            lambda_noise=0.01
        )
        elapsed = time.time() - start_time
        results.append({
            "test_type": "particle_count",
            "value": p_count,
            "median_mse": res["median_distance"],
            "min_mse": res["min_distance"],
            "time_seconds": elapsed
        })
        
    print("\n=== Sensitivity Results ===")
    print(json.dumps(results, indent=2))
    
    with open("data/sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to data/sensitivity_results.json")

if __name__ == "__main__":
    main()

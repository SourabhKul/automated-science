"""
Benchmark: Warne 2020 ODE SIARD model (thesis published academic model)
Parameters: [α₀, α, n, β, γ, δ, η, κ]  (8 params, from Warne et al. 2020)
Infection rate: g(A,R,D) = α₀ + α / (1 + (A+R+D)^n)

Same ABC-SMC config as LLM experiments: 15 generations, 100k particles, 500 samples.
"""

import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from automated_science.core.sbi_engine import SBIEngine

import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

# ── Warne 2020 ODE dynamics ──────────────────────────────────────────────────
def warne_dynamics(t, y, args):
    alpha_0, alpha, n, beta, gamma, delta, eta, kappa = args
    S, I, A, R, D, Ru = y
    P = 60.36e6

    # Behaviorally-adaptive infection rate (key thesis contribution)
    g = alpha_0 + alpha / (1.0 + (A + R + D) ** n)

    infection = g * S * I / P

    dS  = -infection
    dI  = infection - gamma * I - beta * eta * I
    dA  = gamma * I - (beta + delta) * A
    dR  = beta * A
    dD  = delta * A
    dRu = beta * eta * I

    return jnp.array([dS, dI, dA, dR, dD, dRu])

warne_metadata = [
    {'name': 'alpha_0', 'range': (0.0,   1.0)},
    {'name': 'alpha',   'range': (0.0, 100.0)},
    {'name': 'n',       'range': (0.1,   2.0)},
    {'name': 'beta',    'range': (0.0,   1.0)},
    {'name': 'gamma',   'range': (0.0,   1.0)},
    {'name': 'delta',   'range': (0.0,   1.0)},
    {'name': 'eta',     'range': (0.0,   1.0)},
    {'name': 'kappa',   'range': (0.0,   2.0)},
]

class Warne2020Model(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        try:
            sol = diffeqsolve(
                ODETerm(warne_dynamics), Tsit5(),
                t0=float(time_points[0]), t1=float(time_points[-1]),
                dt0=0.1, y0=y0, args=params, saveat=saveat, max_steps=4000
            )
            # Return observed states: Active (col 2), Recovered (col 3), Deaths (col 4)
            return jnp.stack([sol.ys[:, 2], sol.ys[:, 3], sol.ys[:, 4]], axis=-1)
        except Exception:
            return jnp.zeros((len(time_points), 3))

    def get_parameter_metadata(self):
        return warne_metadata

    def get_initial_conditions(self):
        P = 60.36e6
        A0, R0, D0 = 155.0, 2.0, 3.0
        I0 = 1000.0        # approx κ * A0, κ~6.5 (mean of prior range 0-2 scaled)
        S0 = P - (A0 + R0 + D0 + I0)
        return jnp.array([S0, I0, A0, R0, D0, 0.0])

    def get_latex(self):
        return r"Warne 2020: g=\alpha_0 + \alpha/(1+(A+R+D)^n)"


if __name__ == "__main__":
    print("=" * 65)
    print("  Warne 2020 SIARD — Published Academic Model Benchmark")
    print("  ABC-SMC: 15 generations · 100k particles · 500 samples")
    print("=" * 65)

    gt_data  = np.load("automated_science/data/covid_ground_truth.npy")
    t_points = np.load("automated_science/data/covid_time_points.npy")
    engine   = SBIEngine(gt_data, t_points)

    model  = Warne2020Model()
    result = engine.run_abc_smc(model, target_samples=500, generations=15, initial_particles=100000)
    loss   = result["median_distance"]

    print(f"\n  Warne 2020 Median MSE   : {loss:.4f}")
    print(f"  Simple SEIRD seed       : ~191,699")
    print(f"  Qwen 35B best           :  16,563")
    print(f"  Gemma-4 26B best        :  15,440")
    print()
    if loss < 15440:
        print("  → Academic model BEATS both LLMs")
    elif loss < 16563:
        print("  → Academic model beats Qwen, loses to Gemma")
    else:
        print(f"  → LLMs outperform published academic model (delta={loss-15440:.1f})")
    print("=" * 65)

    # Save result
    import csv
    from datetime import datetime
    with open("automated_science/warne2020_benchmark_result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Model", "Median_MSE"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Warne2020_SIARD", loss])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SimpleSEIRD_seed", 191699.0])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Qwen35B_best", 16563.79])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Gemma4_26B_best", 15440.51])

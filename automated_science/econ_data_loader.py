import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    alpha, beta, gamma, rho = args
    v, u = y
    dv = v * (1 - u) - alpha * v
    du = u * (rho * v - gamma) - beta * u
    return jnp.array([dv, du])

metadata = [
    {'name': 'alpha', 'range': (0.01, 0.1)},
    {'name': 'beta', 'range': (0.01, 0.2)},
    {'name': 'gamma', 'range': (0.05, 0.5)},
    {'name': 'rho', 'range': (0.05, 0.5)}
]

def generate_data():
    t0 = 0.0
    t1 = 20.0
    ts = jnp.linspace(t0, t1, 200)
    y0 = jnp.array([0.8, 0.6])
    args = jnp.array([0.02, 0.05, 0.1, 0.1])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/econ_ground_truth.npy", gt_data)
    np.save("data/econ_time_points.npy", ts)
    print("Saved econ ground truth data.")

if __name__ == "__main__":
    generate_data()

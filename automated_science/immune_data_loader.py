import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    r, a, c, d = args
    P, T = y
    dP = r * P - a * P * T
    dT = c * P * T - d * T
    return jnp.array([dP, dT])

metadata = [
    {'name': 'r', 'range': (0.1, 1.0)},
    {'name': 'a', 'range': (0.01, 0.5)},
    {'name': 'c', 'range': (0.01, 0.5)},
    {'name': 'd', 'range': (0.01, 0.2)}
]

def generate_data():
    t0 = 0.0
    t1 = 15.0
    ts = jnp.linspace(t0, t1, 150)
    y0 = jnp.array([1.0, 0.1])
    args = jnp.array([0.5, 0.1, 0.2, 0.05])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/immune_ground_truth.npy", gt_data)
    np.save("data/immune_time_points.npy", ts)
    print("Saved immune ground truth data.")

if __name__ == "__main__":
    generate_data()

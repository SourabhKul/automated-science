import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    a1, a2, gamma, n = args
    U, V = y
    dU = a1 / (1 + V**n) - gamma * U
    dV = a2 / (1 + U**n) - gamma * V
    return jnp.array([dU, dV])

metadata = [
    {'name': 'a1', 'range': (0.5, 5.0)},
    {'name': 'a2', 'range': (0.5, 5.0)},
    {'name': 'gamma', 'range': (0.5, 5.0)},
    {'name': 'n', 'range': (1.0, 5.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([2.0, 0.1])
    args = jnp.array([2.0, 2.0, 2.0, 2.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/synbio_ground_truth.npy", gt_data)
    np.save("data/synbio_time_points.npy", ts)
    print("Saved synbio ground truth data.")

if __name__ == "__main__":
    generate_data()

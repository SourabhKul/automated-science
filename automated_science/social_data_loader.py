import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    p, q, M = args
    A = y[0]
    dA = (p + q * (A / M)) * (M - A)
    return jnp.array([dA])

metadata = [
    {'name': 'p', 'range': (0.01, 0.1)},
    {'name': 'q', 'range': (0.1, 0.9)},
    {'name': 'M', 'range': (0.5, 2.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 5.0
    ts = jnp.linspace(t0, t1, 50)
    y0 = jnp.array([0.01])
    args = jnp.array([0.03, 0.38, 1.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/social_ground_truth.npy", gt_data)
    np.save("data/social_time_points.npy", ts)
    print("Saved social ground truth data.")

if __name__ == "__main__":
    generate_data()

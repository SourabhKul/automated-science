import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    r, alpha, m = args
    B, N = y
    dB = r * B * N - m * B
    dN = -alpha * r * B * N
    return jnp.array([dB, dN])

metadata = [
    {'name': 'r', 'range': (0.1, 1.0)},
    {'name': 'alpha', 'range': (0.01, 0.5)},
    {'name': 'm', 'range': (0.01, 0.2)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([0.1, 10.0])
    args = jnp.array([0.5, 0.1, 0.05])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/agriculture_ground_truth.npy", gt_data)
    np.save("data/agriculture_time_points.npy", ts)
    print("Saved agriculture ground truth data.")

if __name__ == "__main__":
    generate_data()

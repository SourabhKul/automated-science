import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    r, a, b, m = args
    H, L = y
    dH = r * H - a * H * L
    dL = b * H * L - m * L
    return jnp.array([dH, dL])

metadata = [
    {'name': 'r', 'range': (0.1, 2.0)},
    {'name': 'a', 'range': (0.01, 0.5)},
    {'name': 'b', 'range': (0.01, 0.5)},
    {'name': 'm', 'range': (0.1, 2.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([40.0, 9.0])
    args = jnp.array([1.0, 0.1, 0.05, 0.5])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/ecology_ground_truth.npy", gt_data)
    np.save("data/ecology_time_points.npy", ts)
    print("Saved ecology ground truth data.")

if __name__ == "__main__":
    generate_data()

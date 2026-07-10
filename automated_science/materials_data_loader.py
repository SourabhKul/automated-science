import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    C, m, dK = args
    a = jnp.maximum(y[0], 1e-6)
    da = C * (dK * jnp.sqrt(a))**m
    return jnp.array([da])

metadata = [
    {'name': 'C', 'range': (1e-8, 1e-4)},
    {'name': 'm', 'range': (2.0, 5.0)},
    {'name': 'dK', 'range': (10.0, 100.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 30.0
    ts = jnp.linspace(t0, t1, 300)
    y0 = jnp.array([0.01])
    args = jnp.array([1e-6, 3.0, 50.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/materials_ground_truth.npy", gt_data)
    np.save("data/materials_time_points.npy", ts)
    print("Saved materials ground truth data.")

if __name__ == "__main__":
    generate_data()

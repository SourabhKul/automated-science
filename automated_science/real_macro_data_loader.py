import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    g = args[0]
    G = y[0]
    dG = g * G
    return jnp.array([dG])

metadata = [
    {'name': 'g', 'range': (0.001, 0.1)}
]

def generate_data():
    t0 = 0.0
    t1 = 10.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([2710.0])
    args = jnp.array([0.03])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/real_macro_ground_truth.npy", gt_data)
    np.save("data/real_macro_time_points.npy", ts)
    print("Saved real_macro ground truth data.")

if __name__ == "__main__":
    generate_data()

import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    k, V_mean = args
    V = y[0]
    dV = k * (V_mean - V)
    return jnp.array([dV])

metadata = [
    {'name': 'k', 'range': (0.01, 1.0)},
    {'name': 'V_mean', 'range': (800.0, 1200.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([1120.0])
    args = jnp.array([0.1, 1000.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/real_nile_ground_truth.npy", gt_data)
    np.save("data/real_nile_time_points.npy", ts)
    print("Saved real_nile ground truth data.")

if __name__ == "__main__":
    generate_data()

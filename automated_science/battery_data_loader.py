import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    k_sei, n, k_aml = args
    Q, S = y
    dS = k_sei * (t ** n)
    dQ = -k_aml * Q - 0.1 * dS
    return jnp.array([dQ, dS])

metadata = [
    {'name': 'k_sei', 'range': (0.001, 0.1)},
    {'name': 'n', 'range': (0.1, 0.9)},
    {'name': 'k_aml', 'range': (0.01, 0.1)}
]

def generate_data():
    t0 = 0.0
    t1 = 50.0
    ts = jnp.linspace(t0, t1, 500)
    y0 = jnp.array([100.0, 1.0])
    args = jnp.array([0.01, 0.5, 0.05])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/battery_ground_truth.npy", gt_data)
    np.save("data/battery_time_points.npy", ts)
    print("Saved battery ground truth data.")

if __name__ == "__main__":
    generate_data()

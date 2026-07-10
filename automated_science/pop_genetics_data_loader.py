import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    s, mu, nu = args
    p = y[0]
    dp = s * p * (1.0 - p) - mu * p + nu * (1.0 - p)
    return jnp.array([dp])

metadata = [
    {'name': 's', 'range': (0.01, 0.2)},
    {'name': 'mu', 'range': (0.001, 0.05)},
    {'name': 'nu', 'range': (0.001, 0.05)}
]

def generate_data():
    t0 = 0.0
    t1 = 20.0
    ts = jnp.linspace(t0, t1, 200)
    y0 = jnp.array([0.5])
    args = jnp.array([0.05, 0.01, 0.01])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/pop_genetics_ground_truth.npy", gt_data)
    np.save("data/pop_genetics_time_points.npy", ts)
    print("Saved pop_genetics ground truth data.")

if __name__ == "__main__":
    generate_data()

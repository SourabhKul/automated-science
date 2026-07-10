import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    ka, ke = args
    C = y[0]
    dC = ka * 4.02 * jnp.exp(-ka * t) - ke * C
    return jnp.array([dC])

metadata = [
    {'name': 'ka', 'range': (0.1, 5.0)},
    {'name': 'ke', 'range': (0.01, 1.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 1.1
    ts = jnp.linspace(t0, t1, 11)
    y0 = jnp.array([0.74])
    args = jnp.array([1.5, 0.1])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/real_theophylline_ground_truth.npy", gt_data)
    np.save("data/real_theophylline_time_points.npy", ts)
    print("Saved real_theophylline ground truth data.")

if __name__ == "__main__":
    generate_data()

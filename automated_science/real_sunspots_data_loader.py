import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    omega, A, d = args
    S = y[0]
    dS = A * jnp.sin(omega * t) - d * S
    return jnp.array([dS])

metadata = [
    {'name': 'omega', 'range': (0.01, 1.0)},
    {'name': 'A', 'range': (1.0, 50.0)},
    {'name': 'd', 'range': (0.01, 0.5)}
]

def generate_data():
    t0 = 0.0
    t1 = 10.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([5.0])
    args = jnp.array([0.1, 10.0, 0.05])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/real_sunspots_ground_truth.npy", gt_data)
    np.save("data/real_sunspots_time_points.npy", ts)
    print("Saved real_sunspots ground truth data.")

if __name__ == "__main__":
    generate_data()

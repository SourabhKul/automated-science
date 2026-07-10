import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    q, f, eps, alpha, beta = args
    X, Y, Z = y
    dX = (q*Y - X*Y + X*(1-X)) / eps
    dY = (-q*Y - X*Y + f*Z) / alpha
    dZ = (X - Z) / beta
    return jnp.array([dX, dY, dZ])

metadata = [
    {'name': 'q', 'range': (0.01, 0.5)},
    {'name': 'f', 'range': (0.1, 2.0)},
    {'name': 'eps', 'range': (0.01, 2.0)},
    {'name': 'alpha','range': (0.1, 2.0)},
    {'name': 'beta', 'range': (0.1, 2.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 20.0
    ts = jnp.linspace(t0, t1, 200)
    y0 = jnp.array([0.1, 0.1, 0.1])
    args = jnp.array([0.2, 0.1, 1.0, 0.5, 0.2])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/bz_chem_ground_truth.npy", gt_data)
    np.save("data/bz_chem_time_points.npy", ts)
    print("Saved bz_chem ground truth data.")

if __name__ == "__main__":
    generate_data()

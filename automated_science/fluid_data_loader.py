import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    sigma, rho, beta = args
    X, Y, Z = y
    dX = sigma * (Y - X)
    dY = X * (rho - Z) - Y
    dZ = X * Y - beta * Z
    return jnp.array([dX, dY, dZ])

metadata = [
    {'name': 'sigma', 'range': (5.0, 15.0)},
    {'name': 'rho', 'range': (15.0, 35.0)},
    {'name': 'beta', 'range': (1.0, 5.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 40.0
    ts = jnp.linspace(t0, t1, 400)
    y0 = jnp.array([1.0, 1.0, 1.0])
    args = jnp.array([10.0, 28.0, 2.66])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/fluid_ground_truth.npy", gt_data)
    np.save("data/fluid_time_points.npy", ts)
    print("Saved fluid ground truth data.")

if __name__ == "__main__":
    generate_data()

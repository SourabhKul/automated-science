import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    GM = args[0]
    x, y, vx, vy = y
    r3 = jnp.maximum((x**2 + y**2)**1.5, 1e-6)
    dx = vx
    dy = vy
    dvx = -GM * x / r3
    dvy = -GM * y / r3
    return jnp.array([dx, dy, dvx, dvy])

metadata = [
    {'name': 'GM', 'range': (0.1, 5.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 50.0
    ts = jnp.linspace(t0, t1, 500)
    y0 = jnp.array([1.0, 0.0, 0.0, 1.0])
    args = jnp.array([1.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/astro_ground_truth.npy", gt_data)
    np.save("data/astro_time_points.npy", ts)
    print("Saved astro ground truth data.")

if __name__ == "__main__":
    generate_data()

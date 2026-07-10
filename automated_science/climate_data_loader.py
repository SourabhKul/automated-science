import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    a, b, c, d, e = args
    I, mu, theta = y
    dI = -I + a * mu + b * theta
    dmu = c * I - d * mu
    dtheta = -e * I + 0.1 * mu
    return jnp.array([dI, dmu, dtheta])

metadata = [
    {'name': 'a', 'range': (0.01, 1.0)},
    {'name': 'b', 'range': (0.01, 1.0)},
    {'name': 'c', 'range': (0.01, 1.0)},
    {'name': 'd', 'range': (0.01, 1.0)},
    {'name': 'e', 'range': (0.01, 1.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 40.0
    ts = jnp.linspace(t0, t1, 400)
    y0 = jnp.array([1.0, 1.0, 1.0])
    args = jnp.array([0.1, 0.2, 0.1, 0.5, 0.2])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/climate_ground_truth.npy", gt_data)
    np.save("data/climate_time_points.npy", ts)
    print("Saved climate ground truth data.")

if __name__ == "__main__":
    generate_data()

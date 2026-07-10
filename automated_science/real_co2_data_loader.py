import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    r, a = args
    C = y[0]
    dC = r * (C - 280.0) + a
    return jnp.array([dC])

metadata = [
    {'name': 'r', 'range': (0.001, 0.1)},
    {'name': 'a', 'range': (0.1, 5.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 10.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([315.0])
    args = jnp.array([0.01, 1.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/real_co2_ground_truth.npy", gt_data)
    np.save("data/real_co2_time_points.npy", ts)
    print("Saved real_co2 ground truth data.")

if __name__ == "__main__":
    generate_data()

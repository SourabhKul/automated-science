import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    R_p, C, R_c = args
    P, Q = y
    # Simplified mock forced flow
    flow_in = 100 * jnp.sin(t * jnp.pi)
    dP = (flow_in - P/R_p) / C
    dQ = P / R_c - Q * 2.0
    return jnp.array([dP, dQ])

metadata = [
    {'name': 'R_p', 'range': (0.5, 3.0)},
    {'name': 'C', 'range': (0.01, 0.2)},
    {'name': 'R_c', 'range': (0.01, 0.5)}
]

def generate_data():
    t0 = 0.0
    t1 = 20.0
    ts = jnp.linspace(t0, t1, 200)
    y0 = jnp.array([80.0, 0.0])
    args = jnp.array([1.0, 0.05, 0.1])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/cardio_ground_truth.npy", gt_data)
    np.save("data/cardio_time_points.npy", ts)
    print("Saved cardio ground truth data.")

if __name__ == "__main__":
    generate_data()

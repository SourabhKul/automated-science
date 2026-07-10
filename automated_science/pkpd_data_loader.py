import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    k10, k12, k21, k13, k31 = args
    C1, C2, C3 = y
    dC1 = -(k10 + k12 + k13)*C1 + k21*C2 + k31*C3
    dC2 = k12*C1 - k21*C2
    dC3 = k13*C1 - k31*C3
    return jnp.array([dC1, dC2, dC3])

metadata = [
    {'name': 'k10', 'range': (0.01, 0.5)},
    {'name': 'k12', 'range': (0.01, 0.5)},
    {'name': 'k21', 'range': (0.01, 0.5)},
    {'name': 'k13', 'range': (0.01, 0.5)},
    {'name': 'k31', 'range': (0.01, 0.5)}
]

def generate_data():
    t0 = 0.0
    t1 = 12.0
    ts = jnp.linspace(t0, t1, 120)
    y0 = jnp.array([10.0, 0.0, 0.0])
    args = jnp.array([0.1, 0.05, 0.02, 0.05, 0.01])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/pkpd_ground_truth.npy", gt_data)
    np.save("data/pkpd_time_points.npy", ts)
    print("Saved pkpd ground truth data.")

if __name__ == "__main__":
    generate_data()

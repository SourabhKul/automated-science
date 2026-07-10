import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    k, eps_sig, T_env = args
    T = y[0]
    dT = -k * (T - T_env) - eps_sig * (T**4 - T_env**4)
    return jnp.array([dT])

metadata = [
    {'name': 'k', 'range': (0.01, 0.5)},
    {'name': 'eps_sig', 'range': (1e-10, 1e-8)},
    {'name': 'T_env', 'range': (250.0, 350.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([500.0])
    args = jnp.array([0.1, 1e-9, 300.0])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/thermo_ground_truth.npy", gt_data)
    np.save("data/thermo_time_points.npy", ts)
    print("Saved thermo ground truth data.")

if __name__ == "__main__":
    generate_data()

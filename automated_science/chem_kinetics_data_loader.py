import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    k1, k_1, k2 = args
    S, E, ES, P = y
    dS = -k1 * S * E + k_1 * ES
    dE = -k1 * S * E + (k_1 + k2) * ES
    dES = k1 * S * E - (k_1 + k2) * ES
    dP = k2 * ES
    return jnp.array([dS, dE, dES, dP])

metadata = [
    {'name': 'k1', 'range': (0.1, 2.0)},
    {'name': 'k_1', 'range': (0.01, 1.0)},
    {'name': 'k2', 'range': (0.01, 1.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([10.0, 1.0, 0.0, 0.0])
    args = jnp.array([1.0, 0.5, 0.2])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/chem_kinetics_ground_truth.npy", gt_data)
    np.save("data/chem_kinetics_time_points.npy", ts)
    print("Saved chem_kinetics ground truth data.")

if __name__ == "__main__":
    generate_data()

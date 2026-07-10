import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    beta1, beta2, gamma1, gamma2 = args
    S, I1, I2, R = y
    dS = -beta1 * S * I1 - beta2 * S * I2
    dI1 = beta1 * S * I1 - gamma1 * I1
    dI2 = beta2 * S * I2 - gamma2 * I2
    dR = gamma1 * I1 + gamma2 * I2
    return jnp.array([dS, dI1, dI2, dR])

metadata = [
    {'name': 'beta1', 'range': (0.1, 1.0)},
    {'name': 'beta2', 'range': (0.1, 1.0)},
    {'name': 'gamma1', 'range': (0.05, 0.3)},
    {'name': 'gamma2', 'range': (0.05, 0.3)}
]

def generate_data():
    t0 = 0.0
    t1 = 15.0
    ts = jnp.linspace(t0, t1, 150)
    y0 = jnp.array([0.9, 0.05, 0.05, 0.0])
    args = jnp.array([0.4, 0.6, 0.1, 0.1])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/epidemiology_ground_truth.npy", gt_data)
    np.save("data/epidemiology_time_points.npy", ts)
    print("Saved epidemiology ground truth data.")

if __name__ == "__main__":
    generate_data()

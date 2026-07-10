import os
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import matplotlib.pyplot as plt
import numpy as np

def toggle_switch(t, y, args):
    U, V = y
    alpha1, alpha2, n, m, d1, d2 = args
    du = alpha1 / (1 + jnp.power(V, n)) - d1 * U
    dv = alpha2 / (1 + jnp.power(U, m)) - d2 * V
    return jnp.array([du, dv])

def generate_data():
    true_params = jnp.array([10.0, 10.0, 2.0, 2.0, 1.0, 1.0])
    y0 = jnp.array([0.1, 2.0])
    t0, t1, dt0 = 0.0, 10.0, 0.1
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 20))
    sol = diffeqsolve(ODETerm(toggle_switch), Tsit5(), t0=t0, t1=t1, dt0=dt0, y0=y0, args=true_params, saveat=saveat)
    
    noise_sigma = 0.05
    noisy_data = sol.ys + noise_sigma * np.random.randn(*sol.ys.shape)
    
    os.makedirs("automated_science/data", exist_ok=True)
    np.save("automated_science/data/ground_truth.npy", noisy_data)
    np.save("automated_science/data/time_points.npy", sol.ts)
    print(f"Generated ground truth data with shape {noisy_data.shape}")

if __name__ == "__main__":
    generate_data()

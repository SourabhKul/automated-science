import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    alpha, b, d = args
    V, K = y
    dV = alpha * V * jnp.log(K / V)
    dK = b * V - d * K * (V ** (2/3))
    return jnp.array([dV, dK])

metadata = [
    {'name': 'alpha', 'range': (0.01, 0.5)},
    {'name': 'b', 'range': (0.01, 0.5)},
    {'name': 'd', 'range': (0.01, 0.5)}
]

def generate_data():
    t0 = 0.0
    t1 = 100.0
    ts = jnp.linspace(t0, t1, 100)
    y0 = jnp.array([10.0, 20.0])
    args = jnp.array([0.1, 0.05, 0.02])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/oncology_ground_truth.npy", gt_data)
    np.save("data/oncology_time_points.npy", ts)
    print("Saved oncology ground truth data.")

if __name__ == "__main__":
    generate_data()

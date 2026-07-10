import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import os

def dynamics(t, y, args):
    a, b, tau, I_ext = args
    V, W = y
    dV = V - (V**3)/3.0 - W + I_ext
    dW = (V + a - b * W) / tau
    return jnp.array([dV, dW])

metadata = [
    {'name': 'a', 'range': (0.1, 1.0)},
    {'name': 'b', 'range': (0.1, 1.0)},
    {'name': 'tau', 'range': (0.01, 0.2)},
    {'name': 'I_ext', 'range': (0.1, 1.0)}
]

def generate_data():
    t0 = 0.0
    t1 = 20.0
    ts = jnp.linspace(t0, t1, 200)
    y0 = jnp.array([-1.0, 0.0])
    args = jnp.array([0.7, 0.8, 0.08, 0.5])

    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0, t1, 0.1, y0, args=args, saveat=saveat)
    
    gt_data = sol.ys + np.random.normal(0, 0.05, sol.ys.shape)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/neuro_ground_truth.npy", gt_data)
    np.save("data/neuro_time_points.npy", ts)
    print("Saved neuro ground truth data.")

if __name__ == "__main__":
    generate_data()

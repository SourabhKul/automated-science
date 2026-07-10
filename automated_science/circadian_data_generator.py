"""
Goodwin Circadian Oscillator — Ground Truth Generator
Published model: Leloup & Goldbeter 1999 (simplified Goodwin 1965)

State vector: [M, P_C, P_N]
  M   = mRNA transcript level
  P_C = cytoplasmic protein (CLOCK/PER)  [observed]
  P_N = nuclear protein (inhibitor)      [observed]

Ground truth parameters (Leloup & Goldbeter canonical values):
  v_s=0.76, K_I=1.0, n=3, v_m=0.65, K_m=0.5,
  k_s=0.38, v_d=0.95, K_d=0.2, k1=1.9, k2=1.3

These produce a ~24h limit cycle.
"""

import numpy as np
import os

try:
    from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
    import jax.numpy as jnp
except ImportError:
    print("Install diffrax: pip install diffrax")
    raise

def goodwin_dynamics(t, y, args):
    v_s, K_I, n, v_m, K_m, k_s, v_d, K_d, k1, k2 = args
    M, P_C, P_N = y

    dM   = v_s / (1.0 + (P_N / K_I) ** n) - v_m * M / (K_m + M)
    dP_C = k_s * M - v_d * P_C / (K_d + P_C) - k1 * P_C + k2 * P_N
    dP_N = k1 * P_C - k2 * P_N

    return jnp.array([dM, dP_C, dP_N])


# Published canonical parameters
TRUE_PARAMS = (
    0.76,   # v_s  — max transcription rate
    1.0,    # K_I  — nuclear inhibition constant
    3.0,    # n    — Hill cooperativity
    0.65,   # v_m  — max mRNA degradation
    0.5,    # K_m  — Michaelis for mRNA
    0.38,   # k_s  — protein synthesis rate
    0.95,   # v_d  — max protein degradation
    0.2,    # K_d  — Michaelis for protein
    1.9,    # k1   — cytoplasm → nucleus transport
    1.3,    # k2   — nucleus → cytoplasm transport
)

# Observe hourly over 96 hours (4 oscillation cycles)
T_END     = 96.0
T_POINTS  = np.arange(0.0, T_END + 1.0, 1.0)   # 97 timepoints (0–96 h)
Y0        = jnp.array([0.5, 0.3, 0.2])          # arbitrary initial conditions
NOISE_STD = 0.03                                  # 3% Gaussian noise

def generate():
    saveat = SaveAt(ts=jnp.array(T_POINTS))
    sol = diffeqsolve(
        ODETerm(goodwin_dynamics), Tsit5(),
        t0=0.0, t1=T_END, dt0=0.01,
        y0=Y0, args=TRUE_PARAMS, saveat=saveat, max_steps=50000
    )

    # Discard first 24h (transient) and use hours 24-96
    data_full = np.array(sol.ys)   # shape (97, 3)
    data = data_full[24:, :]       # 73 timepoints, 3 states [M, P_C, P_N]

    # Add measurement noise
    np.random.seed(42)
    noise = np.random.normal(0.0, NOISE_STD, data.shape)
    data_noisy = np.clip(data + noise * data, 0.0, None)  # proportional noise, keep positive

    # We observe [P_C, P_N] (the two protein compartments, mRNA is latent)
    gt = data_noisy[:, 1:]   # shape (73, 2)
    t  = T_POINTS[24:]       # hours 24-96

    out_dir = "automated_science/data"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/circadian_ground_truth.npy", gt)
    np.save(f"{out_dir}/circadian_time_points.npy",  t)
    np.save(f"{out_dir}/circadian_true_params.npy",  np.array(TRUE_PARAMS))

    print(f"Ground truth shape : {gt.shape}  (timepoints × [P_C, P_N])")
    print(f"Time points        : {t[0]:.0f}h – {t[-1]:.0f}h  ({len(t)} points)")
    print(f"P_C range          : [{gt[:,0].min():.3f}, {gt[:,0].max():.3f}]")
    print(f"P_N range          : [{gt[:,1].min():.3f}, {gt[:,1].max():.3f}]")
    print(f"Saved to {out_dir}/circadian_*.npy")
    return gt, t


if __name__ == "__main__":
    print("Generating Goodwin circadian ground truth...")
    generate()

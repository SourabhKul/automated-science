import os
import sys
import numpy as np
import requests
import csv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from automated_science.core.sbi_engine import SBIEngine
    from automated_science.core.generated_code import extract_validated_code_tuple as extract_code
    from automated_science.core.run_config import RunConfig, write_run_config
except ImportError:
    from core.sbi_engine import SBIEngine
    from core.generated_code import extract_validated_code_tuple as extract_code
    from core.run_config import RunConfig, write_run_config

# ============================================================
# MODEL CONFIG — Google Gemma-4 26B (LM Studio port 1234)
# Application: Goodwin Circadian Oscillator (Chronobiology)
# Published ref: Leloup & Goldbeter 1999 / Goodwin 1965
# ============================================================
LLM_ENDPOINT   = "http://localhost:1234/v1/chat/completions"
MODEL_ID       = "gemma-4-26b-it"
RUN_SEED      = int(os.environ["E3_SEED"]) if os.environ.get("E3_SEED") else None
EXPERIMENT_TAG = "gemma4_26b_circadian"

MODELS_DIR      = f"models/{EXPERIMENT_TAG}"
LOG_FILE        = f"{EXPERIMENT_TAG}_run_history.csv"
BEST_LOGIC_PATH = f"{MODELS_DIR}/best_logic.py"
RUN_LOG_PATH    = f"{EXPERIMENT_TAG}_run.log"
RUN_CONFIG_PATH = f"{MODELS_DIR}/run_config.json"

os.makedirs(MODELS_DIR, exist_ok=True)

# ── tee stdout ────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_log_fh = open(RUN_LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)
# ─────────────────────────────────────────────────────────────


def call_llm(prompt):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a mathematical chronobiology AI. "
                    "Output ONLY a raw Python code block with the `dynamics(t, y, args)` function "
                    "and `metadata` list as requested. No prose, no markdown outside the code fence."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
    }
    try:
        r = requests.post(LLM_ENDPOINT, json=payload, timeout=900)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        err_msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        print(f"LLM Error: {err_msg}")
        return None


def run_loop():
    print("=" * 70)
    print(f"  GEMMA-4 31B  ·  Goodwin Circadian Recursive Scientist")
    print(f"  Domain      : Chronobiology (circadian gene expression)")
    print(f"  Model       : {MODEL_ID}")
    print(f"  Started at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    write_run_config(
        RunConfig(
            experiment_tag=EXPERIMENT_TAG,
            domain=EXPERIMENT_TAG,
            model_id=MODEL_ID,
            llm_endpoint=LLM_ENDPOINT,
            seed=RUN_SEED,
        ),
        RUN_CONFIG_PATH,
    )


    # ── CSV ledger ───────────────────────────────────────────
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Iteration", "Action", "Proposed_Loss", "Baseline_Loss"])

    def log_action(iteration, action, prop_loss, base_loss):
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                iteration, action, prop_loss, base_loss
            ])

    # ── Load data ────────────────────────────────────────────
    gt_data  = np.load("data/circadian_ground_truth.npy")
    t_points = np.load("data/circadian_time_points.npy")
    engine   = SBIEngine(gt_data, t_points, seed=RUN_SEED)

    print(f"  Data shape  : {gt_data.shape}  ([P_C, P_N] over {len(t_points)} hourly timepoints)")
    print(f"  Time range  : {t_points[0]:.0f}h – {t_points[-1]:.0f}h")

    # ── Evaluator ────────────────────────────────────────────
    def evaluate_llm_logic(logic_code_string):
        full_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{logic_code_string}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        try:
            sol = diffeqsolve(
                ODETerm(dynamics), Tsit5(),
                t0=float(time_points[0]), t1=float(time_points[-1]),
                dt0=0.5, y0=y0, args=params, saveat=saveat, max_steps=20000
            )
            # Return observed outputs: [P_C (col 1), P_N (col 2)]
            return jnp.stack([sol.ys[:, 1], sol.ys[:, 2]], axis=-1)
        except Exception:
            return jnp.zeros((len(time_points), 2))

    def get_parameter_metadata(self):
        return metadata

    def get_initial_conditions(self):
        # Near steady-state initial conditions for Goodwin system
        return jnp.array([0.5, 0.3, 0.2])   # [M, P_C, P_N]

    def get_latex(self):
        return "Gemma-4 Generated Goodwin Circadian"
"""
        import importlib.util
        module_name = f"dynamic_circadian_{np.random.randint(100000)}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod  = importlib.util.module_from_spec(spec)
        try:
            exec(full_module, mod.__dict__)
            model = mod.CandidateModel()
            # 15 generations, 100k particles — same as COVID experiment
            res = engine.run_abc_smc(model, target_samples=500, generations=15, initial_particles=100000)
            return res["median_distance"], full_module
        except Exception as e:
            print(f"Compilation/Simulation Error: {e}")
            return float('inf'), full_module

    # ── Seed: published Leloup & Goldbeter Goodwin model ─────
    current_logic = """def dynamics(t, y, args):
    v_s, K_I, n, v_m, K_m, k_s, v_d, K_d, k1, k2 = args
    M, P_C, P_N = y

    # mRNA transcription (Hill repression by nuclear protein P_N)
    dM   = v_s / (1.0 + (P_N / K_I) ** n) - v_m * M / (K_m + M)
    # Cytoplasmic protein (synthesized from mRNA, degraded, transported)
    dP_C = k_s * M - v_d * P_C / (K_d + P_C) - k1 * P_C + k2 * P_N
    # Nuclear protein (transported from cytoplasm)
    dP_N = k1 * P_C - k2 * P_N

    return jnp.array([dM, dP_C, dP_N])

metadata = [
    {'name': 'v_s', 'range': (0.1,  2.0)},
    {'name': 'K_I', 'range': (0.1,  5.0)},
    {'name': 'n',   'range': (1.0,  6.0)},
    {'name': 'v_m', 'range': (0.1,  2.0)},
    {'name': 'K_m', 'range': (0.01, 2.0)},
    {'name': 'k_s', 'range': (0.1,  2.0)},
    {'name': 'v_d', 'range': (0.1,  2.0)},
    {'name': 'K_d', 'range': (0.01, 1.0)},
    {'name': 'k1',  'range': (0.1,  5.0)},
    {'name': 'k2',  'range': (0.1,  5.0)},
]"""

    best_loss, _ = evaluate_llm_logic(current_logic)
    print(f"\n[Seed — Leloup & Goldbeter 1999] Best Loss: {best_loss:.6f}")
    log_action(0, "SEED_EVALUATED", best_loss, best_loss)

    # ── Main loop: 100 iterations ─────────────────────────────
    for iteration in range(1, 101):
        print(f"\n{'─'*60}")
        print(f"  [Gemma-4] Circadian  ·  Iteration {iteration}/100")
        print(f"  Current Best Loss: {best_loss:.6f}")
        print(f"{'─'*60}")

        prompt = f"""We are fitting a mathematical model of the mammalian circadian clock to experimental data.
The dataset tracks [P_C (cytoplasmic CLOCK/PER protein), P_N (nuclear protein)] concentration over 73 hourly timepoints (hours 24–96) showing ~24h oscillations.
State vector has exactly 3 elements: [M (mRNA), P_C, P_N].
Current best model ABC-SMC Median MSE: {best_loss:.6f}

Current model dynamics:
```python
{current_logic}
```

Task:
Propose an improved `dynamics(t, y, args)` function and `metadata` list.

Biology hints for improvement:
- The real circadian clock has phosphorylation cascades (CKI-mediated degradation of PER)
- Consider multi-site phosphorylation: separate P_C into unphosphorylated/phosphorylated forms
  BUT you MUST keep the output state vector at exactly 3 elements [M, P_C, P_N]
- Light coupling or temperature compensation (amplitude/period scaling terms)
- Cooperative transcriptional repression (higher Hill coefficient n, or dual Hill terms)
- Non-Michaelis-Menten degradation (ubiquitin-mediated degradation: P^2 terms)
- mRNA stability regulation (AU-rich element decay: Michaelis + linear)
- Phase-response characteristics (PER protein half-life asymmetry)

Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`. Keep state vector = 3 elements. No class boilerplate. No prose.
"""

        # ── 3-attempt retry ──────────────────────────────────
        new_logic, thoughts = None, ""
        for attempt in range(1, 4):
            reply = call_llm(prompt)
            new_logic, thoughts = extract_code(reply)
            if new_logic:
                if attempt > 1:
                    print(f"  ↳ Code extracted on attempt {attempt}.")
                break
            print(f"  ✗ No code on attempt {attempt}/3 — retrying...")

        if not new_logic:
            print("  ✗ All 3 attempts failed. Skipping iteration.")
            log_action(iteration, "ERROR_NO_CODE", float('inf'), best_loss)
            continue

        print("  ↳ Evaluating via ABC-SMC (15 gen, 500 samples, 100k particles)...")
        new_loss, full_module = evaluate_llm_logic(new_logic)

        # Save candidate
        with open(f"{MODELS_DIR}/proposed_{iteration}.py", "w") as f:
            f.write(full_module)
        if thoughts:
            with open(f"{MODELS_DIR}/proposed_{iteration}_thoughts.md", "w") as f:
                f.write(thoughts)

        print(f"  Proposed Loss : {new_loss:.6f}")
        print(f"  Baseline Loss : {best_loss:.6f}")

        if new_loss < best_loss:
            print("  ✓ UPDATE ACCEPTED — new best hypothesis!")
            log_action(iteration, "ACCEPTED", new_loss, best_loss)
            best_loss     = new_loss
            current_logic = new_logic
            with open(BEST_LOGIC_PATH, "w") as f:
                f.write(current_logic)
        else:
            print("  ✗ UPDATE REJECTED — keeping baseline.")
            log_action(iteration, "REJECTED", new_loss, best_loss)

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT COMPLETE — Goodwin Circadian")
    print(f"  Seed (Leloup & Goldbeter 1999) Loss : {best_loss:.6f}  ← compare")
    print(f"  Final Best Loss (Gemma-4 26B)       : {best_loss:.6f}")
    print(f"  Finished at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_loop()

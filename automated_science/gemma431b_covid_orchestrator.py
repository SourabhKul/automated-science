import os
import sys
import numpy as np
import requests
import json
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
# MODEL CONFIG — Google Gemma 4 26B (via LM Studio port 1234)
# Load "gemma-4-31b-it" in LM Studio before running.
# ============================================================
LLM_ENDPOINT = "http://localhost:1234/v1/chat/completions"
MODEL_ID      = "gemma-4-31b-it"
RUN_SEED      = int(os.environ["E3_SEED"]) if os.environ.get("E3_SEED") else None
EXPERIMENT_TAG = "gemma431b_covid"  # used to namespace all outputs

# Output paths — isolated from the Qwen run
MODELS_DIR   = f"automated_science/models/{EXPERIMENT_TAG}"
LOG_FILE     = f"automated_science/{EXPERIMENT_TAG}_run_history.csv"
BEST_LOGIC_PATH = f"{MODELS_DIR}/best_logic.py"
RUN_LOG_PATH    = f"automated_science/{EXPERIMENT_TAG}_run.log"
RUN_CONFIG_PATH = f"{MODELS_DIR}/run_config.json"

os.makedirs(MODELS_DIR, exist_ok=True)

# ---- redirect stdout to both file and console ----------------
import sys as _sys

class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_fh = open(RUN_LOG_PATH, "a", buffering=1)
_sys.stdout = _Tee(_sys.__stdout__, _log_fh)
_sys.stderr = _Tee(_sys.__stderr__, _log_fh)
# --------------------------------------------------------------


def call_llm(prompt):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a mathematical physics AI. "
                    "Output ONLY a raw Python code block containing the `dynamics` function "
                    "and `metadata` dictionary as requested. No prose, no markdown outside the code fence."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 8192
    }
    try:
        r = requests.post(LLM_ENDPOINT, json=payload, timeout=900)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        err_msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        print(f"LLM Error: {err_msg}")
        return None


def run_loop():
    print("=" * 70)
    print(f"  GEMMA-4 31B IT  ·  COVID SEIRD Recursive Scientist")
    print(f"  Model       : {MODEL_ID}")
    print(f"  Output tag  : {EXPERIMENT_TAG}")
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


    # ---- Init CSV ledger ----
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Iteration", "Action", "Proposed_Loss", "Baseline_Loss"])

    def log_action(iteration, action, prop_loss, base_loss):
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                iteration, action,
                prop_loss, base_loss
            ])

    # ---- Load data ----
    gt_data  = np.load("automated_science/data/covid_ground_truth.npy")
    t_points = np.load("automated_science/data/covid_time_points.npy")
    engine   = SBIEngine(gt_data, t_points, seed=RUN_SEED)

    # ---- Model evaluator ----
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
                dt0=0.1, y0=y0, args=params, saveat=saveat, max_steps=4000
            )
            return jnp.stack([sol.ys[:, 2], sol.ys[:, 3], sol.ys[:, 4]], axis=-1)
        except Exception:
            return jnp.zeros((len(time_points), 3))

    def get_parameter_metadata(self):
        return metadata

    def get_initial_conditions(self):
        N = 60.36e6
        return jnp.array([N - (1000 + 155 + 2 + 3), 1000.0, 155.0, 2.0, 3.0])

    def get_latex(self):
        return "Gemma-4 Generated SEIRD"
"""
        import importlib.util
        module_name = f"dynamic_gemma4_module_{np.random.randint(100000)}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod  = importlib.util.module_from_spec(spec)
        try:
            exec(full_module, mod.__dict__)
            model = mod.CandidateModel()
            # Same rigor as Qwen run: 15 generations, 500 target samples
            res = engine.run_abc_smc(model, target_samples=500, generations=15, initial_particles=100000, strategy='bdss')
            return res["median_distance"], full_module
        except Exception as e:
            print(f"Compilation/Simulation Error: {e}")
            return float('inf'), full_module

    # ---- Seed model (identical to Qwen baseline for fair comparison) ----
    current_logic = """def dynamics(t, y, args):
    S, E, I, R, D = y
    alpha_0, beta, gamma, delta = args[0], args[1], args[2], args[3]
    N = 60.36e6
    dS = -alpha_0 * (S * I) / N
    dE = alpha_0 * (S * I) / N - beta * E
    dI = beta * E - (gamma + delta) * I
    dR = gamma * I
    dD = delta * I
    return jnp.array([dS, dE, dI, dR, dD])

metadata = [
    {'name': 'alpha_0', 'range': (0.0, 1.0)},
    {'name': 'beta',    'range': (0.0, 1.0)},
    {'name': 'gamma',   'range': (0.0, 1.0)},
    {'name': 'delta',   'range': (0.0, 0.5)}
]"""

    best_loss, _ = evaluate_llm_logic(current_logic)
    print(f"\n[Seed] Best Loss: {best_loss:.4f}")
    log_action(0, "SEED_EVALUATED", best_loss, best_loss)

    # ---- Main loop: 100 iterations ----
    for iteration in range(1, 101):
        print(f"\n{'─'*60}")
        print(f"  [Gemma-4] COVID SEIRD  ·  Iteration {iteration}/100")
        print(f"  Current Best Loss: {best_loss:.4f}")
        print(f"{'─'*60}")

        prompt = f"""We are fitting a macroscopic SEIRD epidemic model to empirical COVID-19 data.
The dataset tracks [Active Infections, Recovered, Deaths] over 49 days in the UK (population N=60.36M).
Current best model SMC Median Prediction MSE: {best_loss:.4f}

Current physics kernel:
```python
{current_logic}
```

Task:
Propose an improved `dynamics(t, y, args)` function and a `metadata` list of parameter dicts (each with 'name' and 'range' keys).

Guidelines for improvement:
- Consider time-varying transmission: lockdowns, behavioural fatigue, seasonal forcing
- Non-linear saturation effects (Hill functions, logistic damping on α)
- Heterogeneous mixing (compartment-specific susceptibility)
- Disease progression refinements (pre-symptomatic, asymptomatic, hospitalisation)
- Keep the state vector as (S, E, I, R, D) — 5 elements — so the diffrax scaffolding stays valid

Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`. No class boilerplate. No prose.
"""

        new_logic, thoughts = None, ""
        for attempt in range(1, 4):  # up to 3 retries
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

        print("  ↳ Evaluating proposed model via ABC-SMC (15 gen, 500 samples)...")
        new_loss, full_module = evaluate_llm_logic(new_logic)

        # Save candidate
        prop_path = f"{MODELS_DIR}/proposed_{iteration}.py"
        with open(prop_path, "w") as f:
            f.write(full_module)
        if thoughts:
            with open(f"{MODELS_DIR}/proposed_{iteration}_thoughts.md", "w") as f:
                f.write(thoughts)

        print(f"  Proposed Loss : {new_loss:.4f}")
        print(f"  Baseline Loss : {best_loss:.4f}")

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
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Final Best Loss (Gemma-4 26B) : {best_loss:.4f}")
    print(f"  Qwen 35B Final Best Loss      : 16,563.79  (from run_history.csv)")
    delta = best_loss - 16563.7890625
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta vs Qwen 35B             : {sign}{delta:.2f} (negative = Gemma better)")
    print(f"  Finished at                   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_loop()

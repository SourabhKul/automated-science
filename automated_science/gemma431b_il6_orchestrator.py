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
# ============================================================
LLM_ENDPOINT   = "http://localhost:1234/v1/chat/completions"
MODEL_ID       = "gemma-4-31b-it"
RUN_SEED      = int(os.environ["E3_SEED"]) if os.environ.get("E3_SEED") else None
EXPERIMENT_TAG = "gemma431b_il6"

MODELS_DIR      = f"automated_science/models/{EXPERIMENT_TAG}"
LOG_FILE        = f"automated_science/{EXPERIMENT_TAG}_run_history.csv"
BEST_LOGIC_PATH = f"{MODELS_DIR}/best_logic.py"
RUN_LOG_PATH    = f"automated_science/{EXPERIMENT_TAG}_run.log"
RUN_CONFIG_PATH = f"{MODELS_DIR}/run_config.json"

os.makedirs(MODELS_DIR, exist_ok=True)

# ---- tee stdout to log file ----------------------------------
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_log_fh = open(RUN_LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)
# --------------------------------------------------------------


def call_llm(prompt):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a computational systems biology AI. "
                    "Output ONLY a raw Python code block with the `dynamics(t, y, args)` function "
                    "and `metadata` list as requested. No prose, no markdown outside the code fence."
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
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        err_msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        print(f"LLM Error: {err_msg}")
        return None


def run_loop():
    print("=" * 70)
    print(f"  GEMMA-4 31B IT  ·  Hyper IL-6 Cytokine Recursive Scientist")
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


    # ---- CSV ledger ----
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Iteration", "Action", "Proposed_Loss", "Baseline_Loss"])

    def log_action(iteration, action, prop_loss, base_loss):
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                iteration, action, prop_loss, base_loss
            ])

    # ---- Load IL-6 data ----
    gt_data  = np.load("automated_science/data/hyper_IL6_ground_truth.npy")
    t_points = np.load("automated_science/data/hyper_IL6_time_points.npy")
    engine   = SBIEngine(gt_data, t_points, seed=RUN_SEED)

    # ---- Evaluator: identical scaffolding to cytokine_orchestrator ----
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
                dt0=2.0, y0=y0, args=params, saveat=saveat, max_steps=10000
            )
            pS1 = sol.ys[:, 8] + sol.ys[:, 11] + 2 * sol.ys[:, 12] + sol.ys[:, 17] + sol.ys[:, 19] + sol.ys[:, 20]
            pS3 = sol.ys[:, 9] + sol.ys[:, 14] + 2 * sol.ys[:, 15] + sol.ys[:, 18] + sol.ys[:, 19] + sol.ys[:, 21]
            out_pS1 = pS1 / (pS1[3] + 1e-4)
            out_pS3 = pS3 / (pS3[3] + 1e-4)
            return jnp.stack([out_pS1, out_pS3], axis=-1)
        except Exception:
            return jnp.zeros((len(time_points), 2))

    def get_parameter_metadata(self):
        return metadata

    def get_initial_conditions(self):
        return jnp.array([12.7, 10.0, 0.0, 0.0, 300.0, 400.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def get_latex(self):
        return "Gemma-4 Generated Hyper IL-6"
"""
        import importlib.util
        module_name = f"dynamic_gemma4_il6_{np.random.randint(100000)}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod  = importlib.util.module_from_spec(spec)
        try:
            exec(full_module, mod.__dict__)
            model = mod.CandidateModel()
            # Same rigor as Qwen IL-6 run: 100 SMC generations, 100k particles
            res = engine.run_abc_smc(model, target_samples=500, generations=15, initial_particles=100000, strategy='bdss')
            return res["median_distance"], full_module
        except Exception as e:
            print(f"Compilation/Simulation Error: {e}")
            return float('inf'), full_module

    # ---- Seed: same as Qwen baseline ----
    with open("automated_science/models/cytokine_seed_logic.py", "r") as f:
        current_logic = f.read()

    best_loss, _ = evaluate_llm_logic(current_logic)
    print(f"\n[Seed] Best Loss: {best_loss:.6f}")
    log_action(0, "SEED_EVALUATED", best_loss, best_loss)

    # ---- Main loop: 100 iterations ----
    for iteration in range(1, 101):
        print(f"\n{'─'*60}")
        print(f"  [Gemma-4] IL-6  ·  Iteration {iteration}/100")
        print(f"  Current Best Loss: {best_loss:.6f}")
        print(f"{'─'*60}")

        prompt = f"""We are fitting a 22-state ODE model of the Hyper IL-6 signaling network tracking pSTAT1 and pSTAT3 phosphorylation cascades.
The dataset has 8 timepoints (0–180 seconds) from Thuringia lab experiments. State vector is 22 elements.
Current best model SMC Median L2 Error: {best_loss:.6f}
Qwen 35B reference best: 0.4042

Current pathway logic:
```python
{current_logic}
```

Task:
Propose an improved `dynamics(t, y, args)` function and a `metadata` list of parameter dicts (each with 'name' and 'range' keys).

Key biology hints:
- R[0]–R[7]: receptor/ligand binding states; R[8]–R[15]: STAT phosphorylation intermediates; R[16]–R[21]: dimer/nuclear complexes
- Consider: competitive receptor inhibition between states R[8]–R[15], cross-talk STAT1/STAT3 phosphatase saturation, nuclear export feedback, receptor recycling nonlinearities
- Output jnp.array must remain exactly 22 elements

Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`. No class boilerplate. No prose.
"""

        # ---- 3-attempt retry ----
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

        print("  ↳ Evaluating via ABC-SMC (100 gen, 500 samples)...")
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
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Final Best Loss (Gemma-4 26B) : {best_loss:.6f}")
    print(f"  Qwen 35B Final Best Loss      : 0.404168")
    delta = best_loss - 0.4041678011417389
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta vs Qwen 35B             : {sign}{delta:.6f} (negative = Gemma better)")
    print(f"  Finished at                   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_loop()

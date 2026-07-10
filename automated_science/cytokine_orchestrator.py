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

USE_122B_SCIENTIST = False

if USE_122B_SCIENTIST:
    LLM_ENDPOINT = "http://localhost:1235/v1/chat/completions"
    MODEL_ID = "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx"
else:
    LLM_ENDPOINT = "http://localhost:1234/v1/chat/completions"
    MODEL_ID = "qwen3.5-35b-a3b-mlx-5"

RUN_SEED = int(os.environ["E3_SEED"]) if os.environ.get("E3_SEED") else None
EXPERIMENT_TAG = "legacy_cytokine"
RUN_CONFIG_PATH = f"automated_science/models/{EXPERIMENT_TAG}_run_config.json"

def call_llm(prompt):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are Andrej Karpathy's AutoResearch AI. Maximize token efficiency. Output pure python code blocks containing ONLY the biological mathematical differential function requested."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 16384
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
    print("Initializing Phase 5 [Deep Extrapolation] (Hyper IL-6 Cytokine Network)...")
    
    log_file = "automated_science/cytokine_run_history.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Iteration", "Action", "Proposed_Loss", "Baseline_Loss"])
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


    def log_action(iteration, action, prop_loss, base_loss):
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iteration, action, prop_loss, base_loss])

    # Multiply dataset values scaling to allow variance visibility
    # The Cytokine data is mapped precisely across 8 specific timepoints (0,5,15,30,60,90,120,180)
    gt_data = np.load("automated_science/data/hyper_IL6_ground_truth.npy")
    t_points = np.load("automated_science/data/hyper_IL6_time_points.npy")
    engine = SBIEngine(gt_data, t_points, seed=RUN_SEED)
    
    def evaluate_llm_logic(logic_code_string):
        full_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{logic_code_string}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        # We save precisely at the 8 timepoints matching empirical arrays
        saveat = SaveAt(ts=time_points)
        try:
            sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0=float(time_points[0]), t1=float(time_points[-1]), dt0=2.0, y0=y0, args=params, saveat=saveat, max_steps=10000)
            
            # Map the exact R.T legacy outputs from ABC_SMC_TH1.py
            # pS1 is structurally matched across R[8], R[11], R[12], R[17], R[19], R[20]
            pS1 = sol.ys[:, 8] + sol.ys[:, 11] + 2 * sol.ys[:, 12] + sol.ys[:, 17] + sol.ys[:, 19] + sol.ys[:, 20]
            pS3 = sol.ys[:, 9] + sol.ys[:, 14] + 2 * sol.ys[:, 15] + sol.ys[:, 18] + sol.ys[:, 19] + sol.ys[:, 21]
            
            # Standardize output (we mimic IL27 cross-normalization directly by capping division)
            out_pS1 = pS1 / (pS1[3] + 1e-4)
            out_pS3 = pS3 / (pS3[3] + 1e-4)

            return jnp.stack([out_pS1, out_pS3], axis=-1)
        except Exception:
            return jnp.zeros((len(time_points), 2))
            
    def get_parameter_metadata(self):
        return metadata
        
    def get_initial_conditions(self):
        # Map r10, s10, s30 initial conditions. They are typically priors, but we pin averages for stability.
        # r10=12.7, s10=300, s30=400 
        return jnp.array([12.7, 10.0, 0.0, 0.0, 300.0, 400.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    def get_latex(self):
        return "LLM Generated Hyper IL-6"
"""
        import importlib.util
        module_name = f"dynamic_LLM_module_{np.random.randint(100000)}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        try:
            exec(full_module, mod.__dict__)
            model = mod.CandidateModel()
            # PHASE 5 Eval: 100 SMC Generations, 100,000 parallel baseline particles
            res = engine.run_abc_smc(model, target_samples=500, generations=100, initial_particles=100000)
            return res["median_distance"], full_module
        except Exception as e:
            print(f"Compilation/Simulation Error: {e}")
            return float('inf'), full_module

    with open("automated_science/models/cytokine_seed_logic.py", "r") as f:
        current_logic = f.read()

    best_loss, _ = evaluate_llm_logic(current_logic)
    print(f"Hyper IL-6 Seed Model Best Loss: {best_loss:.4f}")
    log_action(0, "SEED_EVALUATED", best_loss, best_loss)
    
    for iteration in range(1, 101):
        print(f"\n--- Hyper IL-6 Iteration {iteration}/100 ---")
        prompt = f"""We are fitting a complex 22-state ordinary differential equation model tracking Hyper IL-6 interactions across the intracellular pSTAT1 and pSTAT3 signaling cascades.
The dataset tracks [pSTAT1, pSTAT3] variance against Thuringia lab experiments across 8 distinct temporal limits up to 180 seconds.
Current best model's SMC Median Prediction L2 Error: {best_loss:.4f}.

Current Critical Pathway Logic:
```python
{current_logic}
```

Task:
Provide ONLY the `dynamics(t, y, args)` differential system matrix and the `metadata` bounds dictionary containing structural parameters.
We suspect cross-talk mechanisms or competitive receptor inhibition might be missing between `R[8]` through `R[15]` which limits convergence. 
Alter the dynamics algebraically to synthesize hypothesized non-linear interactions across states. Do not exceed 22 elements in the output `jnp.array`.
Do NOT include any generic diffrax boilerplate. We will automatically inject your logic.
"""
        
        reply = call_llm(prompt)
        new_logic, thoughts = extract_code(reply)
        if not new_logic:
            print("Failed to get code from LLM.")
            log_action(iteration, "ERROR_NO_CODE", float('inf'), best_loss)
            continue
            
        print("Scaffolding proposed IL-6 pathways into JAX and transitioning...")
        new_loss, full_module = evaluate_llm_logic(new_logic)
        
        prop_path = f"automated_science/models/il6_proposed_{iteration}.py"
        with open(prop_path, "w") as f:
            f.write(full_module)

        if thoughts:
            with open(f"automated_science/models/il6_proposed_{iteration}_thoughts.md", "w") as f:
                f.write(thoughts)
        
        print(f"Proposed IL-6 Loss: {new_loss:.4f} vs Baseline: {best_loss:.4f}")
        
        if new_loss < best_loss:
            print(">>> UPDATE ACCEPTED! We found a better pathway hypothesis. <<<")
            log_action(iteration, "ACCEPTED", new_loss, best_loss)
            best_loss = new_loss
            current_logic = new_logic
            with open("automated_science/models/cytokine_lean_best_logic.py", "w") as f:
                f.write(current_logic)
        else:
            print(">>> UPDATE REJECTED. Rollback to baseline. <<<")
            log_action(iteration, "REJECTED", new_loss, best_loss)

if __name__ == "__main__":
    run_loop()

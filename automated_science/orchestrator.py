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
EXPERIMENT_TAG = "legacy_covid"
RUN_CONFIG_PATH = f"automated_science/models/{EXPERIMENT_TAG}_run_config.json"

def call_llm(prompt):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are Andrej Karpathy's AutoResearch AI. Maximize token efficiency. Output pure python code blocks containing ONLY the mathematical function requested."},
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
    print("Initializing Phase 4 [Deep Execution] (COVID SEIRD)...")
    
    # Initialize Log Ledger
    log_file = "automated_science/run_history.csv"
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

    gt_data = np.load("automated_science/data/covid_ground_truth.npy")
    t_points = np.load("automated_science/data/covid_time_points.npy")
    engine = SBIEngine(gt_data, t_points, seed=RUN_SEED)
    
    def evaluate_llm_logic(logic_code_string):
        full_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{logic_code_string}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        try:
            sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0=float(time_points[0]), t1=float(time_points[-1]), dt0=0.1, y0=y0, args=params, saveat=saveat, max_steps=4000)
            return jnp.stack([sol.ys[:, 2], sol.ys[:, 3], sol.ys[:, 4]], axis=-1)
        except Exception:
            return jnp.zeros((len(time_points), 3))
            
    def get_parameter_metadata(self):
        return metadata
        
    def get_initial_conditions(self):
        N = 60.36e6
        return jnp.array([N-(1000+155+2+3), 1000.0, 155.0, 2.0, 3.0])
        
    def get_latex(self):
        return "LLM Generated SEIRD"
"""
        import importlib.util
        module_name = f"dynamic_LLM_module_{np.random.randint(100000)}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        try:
            exec(full_module, mod.__dict__)
            model = mod.CandidateModel()
            # PHASE 4: 15 Generations constraint for deep parameter convergence checks.
            res = engine.run_abc_smc(model, target_samples=500, generations=15, initial_particles=100000)
            return res["median_distance"], full_module
        except Exception as e:
            print(f"Compilation/Simulation Error: {e}")
            return float('inf'), full_module

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
    {'name': 'beta', 'range': (0.0, 1.0)},
    {'name': 'gamma', 'range': (0.0, 1.0)},
    {'name': 'delta', 'range': (0.0, 0.5)}
]"""

    best_loss, _ = evaluate_llm_logic(current_logic)
    print(f"Seed Model Best Loss: {best_loss:.4f}")
    log_action(0, "SEED_EVALUATED", best_loss, best_loss)
    
    # PHASE 4: Execute explicitly 100 generations simulating an overnight 8-hour stretch.
    for iteration in range(1, 101):
        print(f"\n--- COVID SEIRD Iteration {iteration}/100 ---")
        prompt = f"""We are fitting a macroscopic SEIRD model to empirical COVID-19 data.
The data tracks [Active, Recovered, Deaths] over 49 days.
Current best model's SMC Median Prediction MSE (across converged posteriors): {best_loss:.4f}.

Current Critical Physics Logic:
```python
{current_logic}
```

Task:
Provide ONLY the `dynamics(t, y, args)` function and the `metadata` dictionary declaring its structural parameters. 
Account for socio-behavioral lockdown interventions or natural transmission limits algebraically (like scaling `alpha_0` into a time-varying non-linear function utilizing population `N` or Hill constants).
Do NOT include any class boilerplate. We will automatically scaffold the `simulate()` interface around your response. Just write raw mathematics.
"""
        
        reply = call_llm(prompt)
        new_logic, thoughts = extract_code(reply)
        if not new_logic:
            print("Failed to get code from LLM.")
            log_action(iteration, "ERROR_NO_CODE", float('inf'), best_loss)
            continue
            
        print("Scaffolding proposed logic into JAX solver and testing (5 generations SMC)...")
        new_loss, full_module = evaluate_llm_logic(new_logic)
        
        prop_path = f"automated_science/models/covid_lean_proposed_{iteration}.py"
        with open(prop_path, "w") as f:
            f.write(full_module)

        if thoughts:
            with open(f"automated_science/models/covid_lean_proposed_{iteration}_thoughts.md", "w") as f:
                f.write(thoughts)
        
        print(f"Proposed Loss: {new_loss:.4f} vs Baseline: {best_loss:.4f}")
        
        if new_loss < best_loss:
            print(">>> UPDATE ACCEPTED! We found a better hypothesis. <<<")
            log_action(iteration, "ACCEPTED", new_loss, best_loss)
            best_loss = new_loss
            current_logic = new_logic
            with open("automated_science/models/covid_lean_best_logic.py", "w") as f:
                f.write(current_logic)
        else:
            print(">>> UPDATE REJECTED. Rollback to baseline. <<<")
            log_action(iteration, "REJECTED", new_loss, best_loss)

if __name__ == "__main__":
    run_loop()

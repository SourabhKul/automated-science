import os
import subprocess
import shutil

DOMAINS = [
    "ecology", "oncology", "econ", "battery", "pkpd",
    "synbio", "climate", "cardio", "bz_chem", "epidemiology",
    "neuro", "fluid", "astro", "chem_kinetics", "immune",
    "pop_genetics", "thermo", "materials", "agriculture", "social",
    "real_sunspots", "real_nile", "real_macro", "real_theophylline", "real_co2"
]

def main():
    print("Starting 10-Independent Run Circuit for 25 Domains...")
    for run_idx in range(1, 11):
        print(f"\n{'='*40}\n  INITIATING SERIAL RUN {run_idx} / 10  \n{'='*40}")
        
        for d in DOMAINS:
            print(f"\n> Launching {d} pipeline... (This will take a long time)")
            
            if run_idx > 1:
                import requests
                prev_run = run_idx - 1
                prev_best_logic = f"scheduler_logs/run_{prev_run}/models_qwen36_27b_{d}/best_logic.py"
                if os.path.exists(prev_best_logic):
                    print(f"  > Meta-Contemplation: Reviewing Run {prev_run} discoveries for {d}...")
                    with open(prev_best_logic, "r") as f:
                        logic_text = f.read()
                    payload = {
                        "model": "qwen3.6-27b-nvfp4",
                        "messages": [
                            {"role": "system", "content": "You are a Senior Research Director specializing in mathematical biology and dynamical systems. Your goal is to extract deep mechanistic invariants from successful models."},
                            {"role": "user", "content": f"Analyze this validated mathematical model for {d}:\n\n```python\n{logic_text}\n```\n1. Mechanistic Summary: Why did this specific mathematical structure succeed?\n2. Missing Invariants: What biological/physical complexity is still absent?\n3. Structural Strategy: Propose 3 specific equation-level modifications to deepen this discovery.\nOutput ONLY the summary and strategy blocks."}
                        ],
                        "temperature": 0.4,
                        "max_tokens": 1024
                    }
                    try:
                        r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=300)
                        meta_hints = r.json()["choices"][0]["message"]["content"]
                        os.makedirs(f"models/qwen36_27b_{d}", exist_ok=True)
                        with open(f"models/qwen36_27b_{d}/meta_hints.txt", "w") as f:
                            f.write(meta_hints)
                        print(f"  > Senior Director Analysis cached.")
                    except Exception as e:
                        print(f"  > Meta-Contemplation failed: {e}")
                        
            cmd = [".venv/bin/python", f"qwen36_{d}_orchestrator.py"]
            # To avoid the terminal buffering blowing up memory, we pipe directly to a log chunk file
            os.makedirs(f"scheduler_logs/run_{run_idx}", exist_ok=True)
            log_out = f"scheduler_logs/run_{run_idx}/{d}_master_exec.out"
            
            with open(log_out, "w") as out_f:
                process = subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT)
            
            # Post-run cleanup: isolate CSV and generated models
            run_history_file = f"qwen36_27b_{d}_run_history.csv"
            safe_dest = f"scheduler_logs/run_{run_idx}/"
            
            if os.path.exists(run_history_file):
                shutil.move(run_history_file, safe_dest + run_history_file)
                
            models_dir = f"models/qwen36_27b_{d}"
            if os.path.exists(models_dir):
                safe_model_dest = f"scheduler_logs/run_{run_idx}/models_qwen36_27b_{d}"
                if os.path.exists(safe_model_dest):
                    shutil.rmtree(safe_model_dest)
                shutil.move(models_dir, safe_model_dest)
                
            print(f"> Completed {d} and sandboxed data for Run {run_idx}.")
            
    print("All 10 Independent Runs Complete.")

if __name__ == "__main__":
    main()

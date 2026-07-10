import os, csv

DOMAINS = ["real_sunspots", "real_nile", "real_macro", "real_theophylline", "real_co2"]

for d in DOMAINS:
    best_final = float('inf')
    best_run = None
    
    for r in range(1, 11):
        path = f"scheduler_logs/run_{r}/qwen36_27b_{d}_run_history.csv"
        if not os.path.exists(path): continue
        
        final_loss = None
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                final_loss = float(row["Baseline_Loss"])
        
        if final_loss is not None and final_loss < best_final:
            best_final = final_loss
            best_run = r
            
    print(f"Domain: {d} | Best Run: Run {best_run} | Final Loss: {best_final:.6f}")
    if best_run is not None:
        best_logic_path = f"scheduler_logs/run_{best_run}/models_qwen36_27b_{d}/best_logic.py"
        if os.path.exists(best_logic_path):
            with open(best_logic_path, "r") as f:
                lines = f.readlines()
            # print dynamics function
            dyn_code = []
            capture = False
            for line in lines:
                if "def dynamics" in line:
                    capture = True
                if capture:
                    dyn_code.append(line)
                    if "return " in line:
                        capture = False
            print("Implementation in best_logic.py:")
            print("".join(dyn_code))
            print("-" * 60)

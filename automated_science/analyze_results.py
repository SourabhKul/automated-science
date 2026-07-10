import os, csv, glob

DOMAINS = [
    "ecology", "oncology", "econ", "battery", "pkpd", "synbio", "climate", "cardio", "bz_chem", "epidemiology",
    "neuro", "fluid", "astro", "chem_kinetics", "immune", "pop_genetics", "thermo", "materials", "agriculture", "social",
    "real_sunspots", "real_nile", "real_macro", "real_theophylline", "real_co2"
]

print("Overall Domain Performance Across 10 Runs:")
print(f"{'Domain':<15} | {'Runs w/ Improv.':<18} | {'Best Starting Loss':<20} | {'Best Final Loss':<18} | {'Total ACCEPTED':<15} | {'Total INF/ERR':<15}")
print("-" * 110)

for d in DOMAINS:
    runs_improving = 0
    total_acc = 0
    total_err = 0
    best_start = float('inf')
    best_final = float('inf')
    
    for r in range(1, 11):
        path = f"scheduler_logs/run_{r}/qwen36_27b_{d}_run_history.csv"
        if not os.path.exists(path): continue
        
        start_loss = None
        final_loss = None
        run_acc = 0
        
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loss = float(row["Baseline_Loss"])
                if start_loss is None: start_loss = loss
                final_loss = loss
                
                if row["Action"] == "ACCEPTED": 
                    run_acc += 1
                    total_acc += 1
                if row["Action"] == "ERROR_NO_CODE" or float(row["Proposed_Loss"]) == float('inf'):
                    total_err += 1
                    
        if start_loss is not None:
            best_start = min(best_start, start_loss)
            best_final = min(best_final, final_loss)
            if run_acc > 0:
                runs_improving += 1
                
    print(f"{d:<15} | {f'{runs_improving}/10':<18} | {best_start:<20.4f} | {best_final:<18.4f} | {total_acc:<15} | {total_err:<15}")


import os, glob, csv, re
from flask import Flask, jsonify, render_template

app = Flask(__name__)

DOMAINS = ["ecology", "oncology", "econ", "battery", "pkpd", "synbio", "climate", "cardio", "bz_chem", "epidemiology"]

def parse_logs():
    data = {
        "current_run": 1,
        "current_domain": "Waiting...",
        "domains_status": {d: "pending" for d in DOMAINS},
        "histories": {}, 
        "latest_log": ""
    }
    
    # Read overall batch out
    if os.path.exists('batch_execution_circuit.out'):
        with open('batch_execution_circuit.out', 'r') as f:
            lines = f.readlines()
            for line in lines:
                m1 = re.search(r"INITIATING SERIAL RUN (\d+)", line)
                if m1: data["current_run"] = int(m1.group(1))
                m2 = re.search(r"> Launching (.*?) pipeline...", line)
                if m2: data["current_domain"] = m2.group(1)
                
    run_dir = f"scheduler_logs/run_{data['current_run']}"
    
    for d in DOMAINS:
        # Determine status: "done", "running", "pending"
        history_file = f"{run_dir}/gemma4_26b_{d}_run_history.csv"
        master_out = f"scheduler_logs/run_{data['current_run']}/{d}_master_exec.out"
        
        if os.path.exists(history_file):
            data["domains_status"][d] = "done"
            # Attempt to parse history charting
            history = []
            try:
                with open(history_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        history.append({"iter": row["Iteration"], "loss": float(row["Baseline_Loss"]), "action": row["Action"]})
            except: pass
            if history: data["histories"][d] = history

        elif data["current_domain"] == d:
            data["domains_status"][d] = "running"
            if os.path.exists(master_out):
                with open(master_out, 'r') as f:
                    log_tail = f.readlines()[-30:] # last 30 lines
                    data["latest_log"] = "".join(log_tail)
            else:
                data["latest_log"] = "Starting up engine..."
        else:
            idx_current = DOMAINS.index(data["current_domain"]) if data["current_domain"] in DOMAINS else -1
            idx_this = DOMAINS.index(d)
            if idx_this < idx_current:
                data["domains_status"][d] = "done"
            else:
                data["domains_status"][d] = "pending"
                
    return data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify(parse_logs())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)

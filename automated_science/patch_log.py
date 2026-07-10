import os
DOMAINS = ["ecology", "oncology", "econ", "battery", "pkpd", "synbio", "climate", "cardio", "bz_chem", "epidemiology"]

def patch():
    for d in DOMAINS:
        path = f"gemma4_{d}_orchestrator.py"
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            
            with open(path, "w") as f:
                for line in lines:
                    f.write(line)
                    if "for iteration in range(1, 101):" in line:
                        f.write("        print(f'\\n>> STARTING ITERATION {iteration}/100. Prompting LLM (this may take minutes)...', flush=True)\\n")
                    if "reply = call_llm(prompt)" in line:
                        f.write("        print('>> LLM Generated reply. Evaluating via ABC-SMC...', flush=True)\\n")

if __name__ == "__main__":
    patch()
    print("Patch complete.")

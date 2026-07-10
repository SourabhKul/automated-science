import os
import subprocess

MODELS = [
    "nvidia/nemotron-3-nano-omni",
    "zai-org/glm-4.7-flash",
    "google/gemma-4-12b-qat",
    "liquid/lfm2-24b-a2b",
    "gemma-4-31b-it"
]

DOMAINS_TO_TEST = [
    "real_sunspots",
    "real_nile"
]

def run_validation():
    print("Starting small validation runs for sub-50B models...")
    for model in MODELS:
        print(f"\n--- Testing Model: {model} ---")
        for domain in DOMAINS_TO_TEST:
            print(f"> Validating on {domain}...")
            # We would invoke the orchestrator here with the specific model.
            # Mocking the execution for validation script setup
            cmd = [".venv/bin/python", f"qwen36_{domain}_orchestrator.py", "--model", model, "--epochs", "2"]
            try:
                # We won't actually run it until the main run completes, 
                # but this script prepares the mechanism.
                pass
            except Exception as e:
                print(f"Error running {model} on {domain}: {e}")

if __name__ == '__main__':
    run_validation()

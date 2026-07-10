import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from run_domain import run_domain_main

if __name__ == "__main__":
    sys.exit(run_domain_main("cardio", "qwen36"))

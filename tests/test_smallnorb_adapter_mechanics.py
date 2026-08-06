from pathlib import Path
from scripts.run_smallnorb_adapter_mechanics_smoke import run

def main() -> None:
    assert run(Path("/tmp/phase64_smallnorb_mechanics"))["status"] == "passed_isolated_adapter_mechanics"
    print("SUCCESS: smallNORB injected adapter is finite, reset-safe, and target-isolated")

if __name__ == "__main__": main()

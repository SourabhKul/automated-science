from pathlib import Path
from scripts.run_smallnorb_synthetic_controls import run

def main() -> None:
    assert run(Path("/tmp/phase64_smallnorb_synthetic"))["status"] == "failed_output_isolated_synthetic_controls"
    print("SUCCESS: smallNORB frozen synthetic recovery failure is reproducible")

if __name__ == "__main__": main()

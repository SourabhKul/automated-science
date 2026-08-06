from pathlib import Path

from scripts.run_uci_higgs_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase62_higgs_synthetic"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    print("SUCCESS: HIGGS output-isolated synthetic recovery and specificity controls pass")


if __name__ == "__main__":
    main()

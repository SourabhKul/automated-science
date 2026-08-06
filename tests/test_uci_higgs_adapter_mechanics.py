from pathlib import Path

from scripts.run_uci_higgs_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase62_higgs_mechanics"))
    assert result["status"] == "passed_isolated_adapter_mechanics"
    print("SUCCESS: HIGGS injected adapter is finite, reset-safe, and target-isolated")


if __name__ == "__main__":
    main()

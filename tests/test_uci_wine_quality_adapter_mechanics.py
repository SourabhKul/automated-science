from pathlib import Path

from scripts.run_uci_wine_quality_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase73_uci_wine_quality_adapter_test"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: Wine Quality injected adapter is raw-measurement-only, reset-safe, and isolated")


if __name__ == "__main__":
    main()

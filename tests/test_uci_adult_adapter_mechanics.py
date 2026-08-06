from pathlib import Path

from scripts.run_uci_adult_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase69_uci_adult_adapter_test"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: Adult injected adapter is raw-token-only, reset-safe, and isolated")


if __name__ == "__main__":
    main()

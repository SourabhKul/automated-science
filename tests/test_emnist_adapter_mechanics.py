from pathlib import Path

from scripts.run_emnist_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase65_emnist_adapter"))
    assert result["status"] == "passed_isolated_adapter_mechanics"
    assert all(result["controls"].values())
    print("SUCCESS: EMNIST injected adapter is source-unit, reset-safe, and evaluator-isolated")


if __name__ == "__main__":
    main()

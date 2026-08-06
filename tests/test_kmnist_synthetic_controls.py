from pathlib import Path

from scripts.run_kmnist_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase70_kmnist_synthetic_test"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert result["uses_observed_values"] is False
    assert result["uses_observed_labels"] is False
    print("SUCCESS: KMNIST artificial raw-grid controls passed")


if __name__ == "__main__":
    main()

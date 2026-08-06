from pathlib import Path

from scripts.run_svhn_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase59_svhn_synthetic"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    print("SUCCESS: SVHN output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

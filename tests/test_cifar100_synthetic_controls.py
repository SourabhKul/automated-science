from pathlib import Path

from scripts.run_cifar100_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase60_cifar100_synthetic"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    print("SUCCESS: CIFAR-100 output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

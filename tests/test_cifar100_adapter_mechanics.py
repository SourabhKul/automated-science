from pathlib import Path

from scripts.run_cifar100_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase60_cifar100_adapter"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: CIFAR-100 injected adapter is fine-label-only, reset-safe, and metadata-isolated")


if __name__ == "__main__":
    main()

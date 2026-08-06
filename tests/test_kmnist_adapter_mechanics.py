from pathlib import Path

from scripts.run_kmnist_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase70_kmnist_adapter_test"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: KMNIST injected raw-grid adapter is reset-safe and isolated")


if __name__ == "__main__":
    main()

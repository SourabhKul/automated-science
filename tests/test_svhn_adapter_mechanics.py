from pathlib import Path

from scripts.run_svhn_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase59_svhn_adapter"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: SVHN injected adapter is raw-pixel-only, reset-safe, and source-metadata-isolated")


if __name__ == "__main__":
    main()

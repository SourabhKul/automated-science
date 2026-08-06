from pathlib import Path

from scripts.run_stl10_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase67_stl10_adapter_test"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: STL-10 artificial adapter is isolated and reset-safe")


if __name__ == "__main__":
    main()

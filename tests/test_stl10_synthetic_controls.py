from pathlib import Path

from scripts.run_stl10_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase67_stl10_synthetic"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert all(result["controls"].values())
    print("SUCCESS: STL-10 output-isolated artificial controls passed")


if __name__ == "__main__":
    main()

from pathlib import Path

from scripts.run_emnist_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase65_emnist_synthetic"))
    assert result["status"] == "failed_output_isolated_synthetic_controls"
    assert result["controls"]["planted_recovery"]
    assert not result["controls"]["label_image_pairing"]
    print("SUCCESS: EMNIST synthetic test preserves the frozen label-pairing closure")


if __name__ == "__main__":
    main()

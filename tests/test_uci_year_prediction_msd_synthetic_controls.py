from pathlib import Path

from scripts.run_uci_year_prediction_msd_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase61_yearpredictionmsd_synthetic"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    print("SUCCESS: YearPredictionMSD output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

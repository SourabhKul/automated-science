from pathlib import Path

from scripts.run_uci_adult_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase69_uci_adult_synthetic_test"))
    assert result["status"] == "closed_negative_synthetic_control_failure"
    assert result["uses_observed_values"] is False
    assert result["uses_observed_labels"] is False
    assert result["controls"]["label_row_pairing_drop"] < result["thresholds"]["label_pairing_drop"]
    print("SUCCESS: Adult synthetic gate deterministically records its fixed pairing failure")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import tempfile

from scripts.run_cmapss_fd002_real_source_train_baselines import run_gate


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = run_gate(output_dir=Path(directory))
    assert result["official_test_rul_accessed"] is False
    assert result["official_test_history_accessed"] is False
    assert result["counts"] == {"train": 180, "selection": 40, "holdout": 40}
    assert result["selection"]["settings_sensor_ridge"]["aggregate"]["rmse"] >= 0.0
    assert result["decision"] in {
        "negative_closed_selection_gate_failed_no_holdout_or_official_test_score",
        "passed_source_train_selection_gate_holdout_scored_once",
    }
    if result["selection_passed"]:
        assert "holdout" in result
    else:
        assert "holdout" not in result
    print("SUCCESS: C-MAPSS real source-train gate preserves official-test isolation and bounded scoring")


if __name__ == "__main__":
    main()

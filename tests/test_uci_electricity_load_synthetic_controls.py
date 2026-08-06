from __future__ import annotations

from pathlib import Path
import tempfile

from scripts.run_uci_electricity_load_synthetic_controls import run


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = run(output_dir=Path(directory))
    assert result["status"] == "failed_output_isolated_native_grid_synthetic_controls"
    assert result["uses_measured_source_load_values"] is False
    assert result["opens_raw_load_member"] is False
    assert result["split_counts"] == {"train": 240, "selection": 65, "external": 65}
    assert result["selected_lag_absolute_error"] <= 0.02
    assert result["selection_nrmse"] <= 0.05
    assert result["control_degradation"]["client_history_pairing"] < 2.0
    assert result["control_degradation"]["quarter_week_time_order"] < 2.0
    assert result["control_degradation"]["calendar_ablation"] >= 1.25
    assert result["checks"]["planted_recovery"] is True
    assert result["checks"]["calendar_ablation"] is True
    assert result["checks"]["client_history_pairing"] is False
    assert result["checks"]["quarter_week_time_order"] is False
    print("SUCCESS: electricity-load synthetic controls deterministically record the frozen specificity-gate failure")


if __name__ == "__main__":
    main()

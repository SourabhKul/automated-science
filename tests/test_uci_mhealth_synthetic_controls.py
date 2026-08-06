from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    result = json.loads(Path("artifacts/evaluations/phase40_uci_mhealth_synthetic_controls_20260725/result.json").read_text())
    assert result["status"] == "failed_output_isolated_synthetic_controls"
    assert result["planted_selection_macro_f1"] >= 0.95
    assert result["label_window_pairing_drop"] < 0.40
    assert result["time_order_drop"] >= 0.15
    assert result["channel_pairing_drop"] < 0.15
    assert result["accelerometer_ablation_drop"] >= 0.05
    assert result["gyro_ablation_drop"] >= 0.05
    assert result["reset_invariant_under_window_reordering"] is True
    assert result["target_isolation_sentinel_rejected"] is True
    assert result["subject_isolation_sentinel_rejected"] is True
    assert result["probabilities_finite_and_bounded"] is True
    print("SUCCESS: MHEALTH synthetic-control failure is deterministic and preserved")


if __name__ == "__main__":
    main()

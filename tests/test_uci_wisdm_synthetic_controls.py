from __future__ import annotations

from pathlib import Path

from scripts.run_uci_wisdm_synthetic_controls import run


def main() -> None:
    result = run(output_dir=Path("/tmp/phase50_wisdm_synthetic_test"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert result["planted"]["macro_f1"] >= 0.95
    controls = result["controls"]
    assert controls["label_segment_pairing_drop"] >= 0.4
    assert controls["time_order_16_sample_shift_drop"] >= 0.1
    assert controls["axis_pairing_drop"] >= 0.1
    assert max(controls["axis_ablation_drops"]) >= 0.05
    print("SUCCESS: WISDM output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

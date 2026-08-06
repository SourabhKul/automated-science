from __future__ import annotations

from pathlib import Path

from scripts.run_uci_smartphone_har_synthetic_controls import run


def main() -> None:
    result = run(output_dir=Path("artifacts/evaluations/phase37_uci_smartphone_har_synthetic_controls_20260724"))
    assert result["status"] == "failed_output_isolated_synthetic_controls"
    assert result["checks"]["planted_recovery"] is False
    assert result["checks"]["label_window_pairing"] is False
    assert result["checks"]["gyroscope_channel_pairing"] is False
    assert result["checks"]["prefix_time_order"] is True
    assert result["checks"]["per_record_reset"] is True
    assert result["uses_measured_source_signal_values"] is False
    assert result["uses_measured_source_label_values"] is False
    assert result["external_artificial_records_scored"] is False
    print("SUCCESS: Smartphone HAR synthetic control failure is deterministic and preserved")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from scripts.run_uci_statlog_landsat_synthetic_controls import run


def main() -> None:
    result = run(output_dir=Path("/tmp/phase47_landsat_synthetic_test"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert result["planted"]["macro_f1"] >= 0.95
    assert result["controls"]["pixel_order_pairing_drop"] >= 0.10
    assert result["controls"]["band_pairing_drop"] >= 0.10
    print("SUCCESS: Statlog Landsat output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

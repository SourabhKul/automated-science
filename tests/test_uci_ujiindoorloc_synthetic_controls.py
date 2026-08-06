from __future__ import annotations

from pathlib import Path

from scripts.run_uci_ujiindoorloc_synthetic_controls import run


def main() -> None:
    result = run(output_dir=Path("/tmp/phase49_ujiindoorloc_synthetic_test"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert result["planted"]["macro_f1"] >= 0.95
    assert result["controls"]["wap_block_pairing_drop"] >= 0.10
    assert result["controls"]["sentinel_mask_pairing_drop"] >= 0.10
    print("SUCCESS: UJIIndoorLoc output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

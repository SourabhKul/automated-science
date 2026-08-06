from __future__ import annotations

from scripts.run_uci_ujiindoorloc_source_gate import audit


def main() -> None:
    result = audit()
    assert result["status"] == "passed_official_source_gate_only"
    assert result["schema"]["wap_columns"] == 520
    assert result["frozen_file_split"]["external"].endswith("validationData.csv")
    print("SUCCESS: UJIIndoorLoc official raw schema and validation-file boundary are fixed")


if __name__ == "__main__":
    main()

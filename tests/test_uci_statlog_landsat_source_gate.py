from __future__ import annotations

from scripts.run_uci_statlog_landsat_source_gate import audit


def main() -> None:
    result = audit()
    assert result["status"] == "passed_official_source_gate_only"
    assert result["schema"]["feature_count"] == 36
    assert result["frozen_file_split"]["external"] == "sat.tst"
    print("SUCCESS: Statlog Landsat official source schema and source-file split are fixed")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from scripts.run_uci_occupancy_detection_source_gate import ARCHIVE, run


def main() -> None:
    result = run(ARCHIVE, Path("artifacts/evaluations/phase39_uci_occupancy_detection_session_tracking_archive_gate_20260725"))
    assert result["status"] == "blocked_official_source_gate"
    assert result["checks"]["official_member_inventory"] is True
    assert all(result["checks"]["fixed_row_counts"].values())
    assert all(result["checks"]["literal_declared_headers"].values())
    assert not any(result["checks"]["raw_row_width_matches_declared_schema"].values())
    assert result["source_session_split"] is None
    assert result["isolation"]["measurement_transformation_or_fitting"] is False
    print("SUCCESS: UCI Occupancy Detection raw-schema mismatch blocks the fixed source gate")


if __name__ == "__main__":
    main()

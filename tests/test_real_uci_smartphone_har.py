from __future__ import annotations

import json
from pathlib import Path

from scripts.run_uci_smartphone_har_source_gate import ARCHIVE, SPLIT_PATH, run


def main() -> None:
    result = run(ARCHIVE, Path("artifacts/evaluations/phase37_uci_smartphone_har_activity_tracking_archive_gate_20260724"))
    assert result["status"] == "passed_official_source_gate_only"
    assert all(value if isinstance(value, bool) else all(value.values()) for value in result["checks"].values())
    assert result["source_schema"]["total_windows"] == 10299
    assert len(result["source_schema"]["subject_ids"]) == 30
    assert result["source_subject_split"] == json.loads(SPLIT_PATH.read_text())
    assert {name: len(values) for name, values in result["source_subject_split"].items()} == {"train": 15, "selection": 6, "external": 9}
    assert result["isolation"]["feature_matrix_opened"] is False
    print("SUCCESS: UCI Smartphone HAR official source gate, subject split, and isolation contract passed")


if __name__ == "__main__":
    main()

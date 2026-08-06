from __future__ import annotations

import json
from pathlib import Path

from scripts.run_uci_mhealth_source_gate import ARCHIVE, SPLIT_PATH, run


def main() -> None:
    result = run(ARCHIVE, Path("artifacts/evaluations/phase40_uci_mhealth_subject_activity_tracking_archive_gate_20260725"))
    assert result["status"] == "passed_official_source_gate_only"
    assert all(value if isinstance(value, bool) else all(value.values()) for value in result["checks"].values())
    assert sum(item["rows"] for item in result["subject_ledgers"].values()) == 1215745
    assert result["source_subject_split"] == json.loads(SPLIT_PATH.read_text())
    assert result["isolation"]["signal_or_label_transformation_or_fitting"] is False
    print("SUCCESS: MHEALTH official source gate, subject split, and isolation contract passed")


if __name__ == "__main__":
    main()

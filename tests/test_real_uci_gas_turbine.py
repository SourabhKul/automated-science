from __future__ import annotations

import json
from pathlib import Path

from scripts.run_uci_gas_turbine_source_gate import ARCHIVE, SPLIT_PATH, run


def main() -> None:
    result = run(ARCHIVE, Path("artifacts/evaluations/phase38_uci_gas_turbine_nox_year_holdout_archive_gate_20260724"))
    assert result["status"] == "passed_official_source_gate_only"
    assert all(result["checks"].values())
    assert sum(item["rows"] for item in result["annual_members"].values()) == 36733
    assert all(item["malformed_rows"] == 0 and item["nonfinite_cells"] == 0 for item in result["annual_members"].values())
    assert result["source_year_split"] == json.loads(SPLIT_PATH.read_text())
    assert result["isolation"]["measurement_transformation_or_fitting"] is False
    print("SUCCESS: UCI gas-turbine official source gate, year split, and measurement isolation passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the artificial-only Phase 61 adapter mechanics smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_year_prediction_msd import FEATURES, YearPredictionRecord, predict_independent, record_from_payload


OUT = Path("artifacts/evaluations/phase61_uci_year_prediction_msd_adapter_mechanics_smoke_20260801")


def rejected(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    values = np.linspace(-1.0, 1.0, FEATURES)
    record = YearPredictionRecord(values)
    states: list[float] = []
    predictions = predict_independent([record] * 3, lambda state, _values: states.append(state) or 2000.0)
    reordered = predict_independent([record] * 3, lambda _state, _values: 2000.0)
    result = {
        "phase": 61,
        "status": "passed_isolated_adapter_mechanics",
        "uses_observed_values": False,
        "uses_observed_targets": False,
        "controls": {
            "finite_90_value_candidate": bool(record.values.shape == (FEATURES,) and np.all(np.isfinite(record.values))),
            "zero_reset": states == [0.0, 0.0, 0.0],
            "reorder_invariant": bool(np.array_equal(predictions, reordered[::-1])),
            "target_rejected": rejected({"values": values, "target": 2000}),
            "file_rejected": rejected({"values": values, "file": "external"}),
            "split_rejected": rejected({"values": values, "split": "test"}),
            "row_rejected": rejected({"values": values, "row": 0}),
            "metadata_rejected": rejected({"values": values, "source_order": 0}),
            "duplicate_group_rejected": rejected({"values": values, "duplicate_group": "one"}),
            "finite_bounded_predictions": bool(np.all(np.isfinite(predictions)) and np.all((1922 <= predictions) & (predictions <= 2011))),
        },
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not all(result["controls"].values()):
        result["status"] = "failed_isolated_adapter_mechanics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

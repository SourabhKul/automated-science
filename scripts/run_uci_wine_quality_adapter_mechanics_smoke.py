#!/usr/bin/env python3
"""Run the injected-only Phase 73 Wine Quality adapter mechanics smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_wine_quality import WineMeasurement, measurement_from_payload, predict_independent

OUT = Path("artifacts/evaluations/phase73_uci_wine_quality_adapter_mechanics_smoke_20260806")


def rejected(payload: dict[str, object]) -> bool:
    try:
        measurement_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    values = tuple(float(index + 1) for index in range(11))
    records = [WineMeasurement(values), WineMeasurement(values[:-1] + (0.0,)), WineMeasurement((0.0,) + values[1:])]
    states: list[float] = []
    predictions = predict_independent(records, lambda state, row: states.append(state) or 5.0)
    reordered = predict_independent(records[::-1], lambda state, row: 5.0)
    controls = {
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(predictions, reordered[::-1])),
        "target_rejected": rejected({"values": values, "quality": 6}),
        "file_rejected": rejected({"values": values, "file": "winequality-white.csv"}),
        "color_rejected": rejected({"values": values, "color": "white"}),
        "split_rejected": rejected({"values": values, "split": "external"}),
        "row_rejected": rejected({"values": values, "row": 0}),
        "source_order_rejected": rejected({"values": values, "source_order": 0}),
        "duplicate_group_rejected": rejected({"values": values, "duplicate_group": 0}),
        "finite_bounded_predictions": bool(np.all(np.isfinite(predictions)) and np.all((0.0 <= predictions) & (predictions <= 10.0))),
    }
    result = {
        "phase": 73,
        "status": "passed_adapter_mechanics_only" if all(controls.values()) else "closed_negative_adapter_mechanics_failure",
        "uses_observed_values": False,
        "uses_observed_quality": False,
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

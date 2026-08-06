#!/usr/bin/env python3
"""Run the artificial-only Phase 62 HIGGS adapter mechanics smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_higgs import FEATURES, HiggsRecord, predict_independent, record_from_payload


OUT = Path("artifacts/evaluations/phase62_uci_higgs_adapter_mechanics_smoke_20260801")


def rejected(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    values = np.linspace(-1.0, 1.0, FEATURES)
    record = HiggsRecord(values)
    states: list[float] = []
    probabilities = predict_independent([record] * 3, lambda state, _values: states.append(state) or 0.75)
    reordered = predict_independent([record] * 3, lambda _state, _values: 0.75)
    result = {
        "phase": 62,
        "status": "passed_isolated_adapter_mechanics",
        "uses_observed_values": False,
        "uses_observed_targets": False,
        "controls": {
            "finite_28_value_candidate": bool(record.values.shape == (FEATURES,) and np.all(np.isfinite(record.values))),
            "zero_reset": states == [0.0, 0.0, 0.0],
            "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
            "target_rejected": rejected({"values": values, "target": 1}),
            "file_rejected": rejected({"values": values, "file": "source_tail"}),
            "split_rejected": rejected({"values": values, "split": "external"}),
            "row_rejected": rejected({"values": values, "row": 0}),
            "source_order_rejected": rejected({"values": values, "source_order": 0}),
            "metadata_rejected": rejected({"values": values, "metadata": {}}),
            "duplicate_group_rejected": rejected({"values": values, "duplicate_group": "one"}),
            "finite_bounded_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all((0.0 <= probabilities) & (probabilities <= 1.0))),
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

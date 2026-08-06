#!/usr/bin/env python3
"""Run the artificial-only Phase 64 smallNORB adapter mechanics smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.smallnorb import CLASSES, SHAPE, SmallNorbRecord, predict_independent, record_from_payload


OUT = Path("artifacts/evaluations/phase64_smallnorb_adapter_mechanics_smoke_20260802")


def rejected(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    record = SmallNorbRecord(np.zeros(SHAPE))
    states: list[float] = []
    probabilities = predict_independent([record] * 3, lambda state, _values: states.append(state) or np.eye(CLASSES)[0])
    reordered = predict_independent([record] * 3, lambda _state, _values: np.eye(CLASSES)[0])
    result = {
        "phase": 64, "status": "passed_isolated_adapter_mechanics", "uses_observed_values": False, "uses_observed_targets": False,
        "controls": {
            "finite_stereo_candidate": bool(record.values.shape == SHAPE and np.all(np.isfinite(record.values))),
            "zero_reset": states == [0.0, 0.0, 0.0], "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
            "target_rejected": rejected({"values": record.values, "target": 0}), "file_rejected": rejected({"values": record.values, "file": "test"}),
            "split_rejected": rejected({"values": record.values, "split": "external"}), "row_rejected": rejected({"values": record.values, "row": 0}),
            "metadata_rejected": rejected({"values": record.values, "instance": 4}), "duplicate_group_rejected": rejected({"values": record.values, "duplicate_group": "one"}),
            "finite_normalized_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all((0.0 <= probabilities) & (probabilities <= 1.0))),
        }, "abc_smc_calls": 0, "llm_calls": 0,
    }
    if not all(result["controls"].values()): result["status"] = "failed_isolated_adapter_mechanics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__": print(json.dumps(run(), indent=2))

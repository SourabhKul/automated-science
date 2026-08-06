#!/usr/bin/env python3
"""Run the injected-only Phase 69 UCI Adult adapter mechanics smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_adult import AdultRow, predict_independent, row_from_payload

OUT = Path("artifacts/evaluations/phase69_uci_adult_adapter_mechanics_smoke_20260803")


def rejected(payload: dict[str, object]) -> bool:
    try:
        row_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    fields = tuple(f"token_{index}" for index in range(14))
    rows = [AdultRow(fields), AdultRow(fields[:-1] + ("?",)), AdultRow(("9",) + fields[1:])]
    states: list[float] = []
    probabilities = predict_independent(rows, lambda state, row: states.append(state) or np.array([0.5, 0.5]))
    reordered = predict_independent(rows[::-1], lambda state, row: np.array([0.5, 0.5]))
    controls = {
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejected({"fields": fields, "label": ">50K"}),
        "file_rejected": rejected({"fields": fields, "file": "adult.test"}),
        "split_rejected": rejected({"fields": fields, "split": "external"}),
        "row_rejected": rejected({"fields": fields, "row": 0}),
        "demographic_rejected": rejected({"fields": fields, "demographic": "sex"}),
        "duplicate_group_rejected": rejected({"fields": fields, "duplicate_group": 0}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    result = {"phase": 69, "status": "passed_adapter_mechanics_only" if all(controls.values()) else "closed_negative_adapter_mechanics_failure", "uses_observed_values": False, "uses_observed_labels": False, "controls": controls, "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

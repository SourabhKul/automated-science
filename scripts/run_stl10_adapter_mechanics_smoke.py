#!/usr/bin/env python3
"""Run the injected-only Phase 67 STL-10 adapter mechanics smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.stl10 import Stl10Image, image_from_payload, predict_independent


OUT = Path("artifacts/evaluations/phase67_stl10_adapter_mechanics_smoke_20260802")


def rejected(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    generator = np.random.default_rng(2067001)
    images = [Stl10Image(generator.uniform(0, 255, (3, 96, 96))) for _ in range(3)]
    states: list[float] = []
    uniform = np.full(10, 0.1)
    probabilities = predict_independent(images, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent(images[::-1], lambda state, values: uniform)
    values = images[0].values
    controls = {
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejected({"values": values, "label": 0}),
        "file_rejected": rejected({"values": values, "file": "test_X.bin"}),
        "split_rejected": rejected({"values": values, "split": "external"}),
        "row_rejected": rejected({"values": values, "row": 0}),
        "fold_rejected": rejected({"values": values, "fold": 0}),
        "class_name_rejected": rejected({"values": values, "class_name": "airplane"}),
        "metadata_rejected": rejected({"values": values, "source": "creator"}),
        "duplicate_group_rejected": rejected({"values": values, "duplicate_group": 0}),
        "unlabeled_rejected": rejected({"values": values, "unlabeled": True}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    result = {
        "phase": 67,
        "status": "passed_adapter_mechanics_only" if all(controls.values()) else "closed_negative_adapter_mechanics_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_unlabeled_values": False,
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

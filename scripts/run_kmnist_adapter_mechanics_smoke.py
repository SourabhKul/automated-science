#!/usr/bin/env python3
"""Run the injected-only Phase 70 KMNIST adapter mechanics smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.kmnist import KmnistImage, SHAPE, image_from_payload, predict_independent

OUT = Path("artifacts/evaluations/phase70_kmnist_adapter_mechanics_smoke_20260803")


def rejected(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    rng = np.random.default_rng(2070001)
    images = [KmnistImage(rng.integers(0, 256, SHAPE, dtype=np.uint8)) for _ in range(4)]
    states: list[float] = []
    uniform = np.full(10, 0.1)
    probabilities = predict_independent(images, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent(images[::-1], lambda state, values: uniform)
    values = images[0].values
    controls = {
        "zero_reset": states == [0.0] * len(images),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejected({"values": values, "label": 0}),
        "file_rejected": rejected({"values": values, "file": "t10k-images-idx3-ubyte.gz"}),
        "split_rejected": rejected({"values": values, "split": "external"}),
        "row_rejected": rejected({"values": values, "row": 0}),
        "class_map_rejected": rejected({"values": values, "class_map": "hiragana"}),
        "duplicate_group_rejected": rejected({"values": values, "duplicate_group": 0}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    result = {"phase": 70, "status": "passed_adapter_mechanics_only" if all(controls.values()) else "closed_negative_adapter_mechanics_failure", "uses_observed_values": False, "uses_observed_labels": False, "controls": controls, "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

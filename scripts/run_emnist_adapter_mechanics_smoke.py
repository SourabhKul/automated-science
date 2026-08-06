#!/usr/bin/env python3
"""Run Phase 65's isolated artificial EMNIST adapter mechanics smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.emnist import EmnistImage, SHAPE, image_from_payload, predict_independent


OUTPUT_DIR = Path("artifacts/evaluations/phase65_nist_emnist_balanced_adapter_mechanics_smoke_20260802")


def rejects_payload(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    rng = np.random.default_rng(2065001)
    images = [EmnistImage(rng.uniform(0, 255, SHAPE)) for _ in range(4)]
    states: list[float] = []
    uniform = np.full(len(range(47)), 1.0 / 47.0)
    probabilities = predict_independent(images, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent(images[::-1], lambda state, values: uniform)
    values = images[0].values
    controls = {
        "native_28x28_source_units": values.shape == SHAPE and float(values.min()) >= 0 and float(values.max()) <= 255,
        "zero_reset": states == [0.0] * len(images),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejects_payload({"values": values, "label": 0}),
        "file_rejected": rejects_payload({"values": values, "file": "emnist-balanced-test"}),
        "split_rejected": rejects_payload({"values": values, "split": "external"}),
        "row_rejected": rejects_payload({"values": values, "row": 0}),
        "mapping_rejected": rejects_payload({"values": values, "mapping": "0 48"}),
        "metadata_rejected": rejects_payload({"values": values, "shape": SHAPE}),
        "duplicate_group_rejected": rejects_payload({"values": values, "duplicate_group": "source-train-duplicate-0"}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    result = {
        "phase": 65,
        "status": "passed_isolated_adapter_mechanics" if all(controls.values()) else "closed_negative_adapter_mechanics_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

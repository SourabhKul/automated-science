#!/usr/bin/env python3
"""Run output-isolated artificial controls for Phase 64 smallNORB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.smallnorb import CLASSES, SHAPE, SmallNorbRecord, predict_independent, record_from_payload


OUT = Path("artifacts/evaluations/phase64_smallnorb_synthetic_controls_20260802")
SEED = 2064001
PAIR_CODES = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2))


def rejected(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def artificial_data(rows_per_class: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    values = rng.normal(12.0, 1.0, size=(rows_per_class * CLASSES, *SHAPE)).astype(np.float32)
    labels = np.repeat(np.arange(CLASSES, dtype=np.uint8), rows_per_class)
    for index, label in enumerate(labels):
        left, right = PAIR_CODES[int(label)]
        values[index, 0, :24, :24] += 70.0 + 45.0 * left
        values[index, 1, :24, :24] += 70.0 + 45.0 * right
    return values, labels


def score(model: object, values: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    probabilities = model.predict_proba(values.reshape(len(values), -1))
    if not np.all(np.isfinite(probabilities)) or not np.all((0.0 <= probabilities) & (probabilities <= 1.0)) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("smallNORB synthetic model emitted invalid probabilities")
    predicted = np.argmax(probabilities, axis=1)
    return float(f1_score(labels, predicted, average="macro")), probabilities


def run(output_dir: Path = OUT) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    train_values, train_labels = artificial_data(90, rng)
    selection_values, selection_labels = artificial_data(50, rng)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED))
    model.fit(train_values.reshape(len(train_values), -1), train_labels)
    planted_f1, probabilities = score(model, selection_values, selection_labels)
    label_paired_f1, _ = score(model, selection_values, rng.permutation(selection_labels))
    camera_paired = selection_values.copy()
    camera_paired[:, 1] = camera_paired[rng.permutation(len(camera_paired)), 1]
    camera_paired_f1, _ = score(model, camera_paired, selection_labels)
    block_paired = selection_values.copy()
    permutation = rng.permutation(len(block_paired))
    block_paired[:, :, :24, :24] = block_paired[permutation, :, :24, :24]
    block_paired_f1, _ = score(model, block_paired, selection_labels)
    ablated = selection_values.copy()
    ablated[:, :, :24, :24] = 0.0
    ablated_f1, _ = score(model, ablated, selection_labels)
    record = SmallNorbRecord(np.zeros(SHAPE))
    states: list[float] = []
    reset = predict_independent([record] * 3, lambda state, _values: states.append(state) or np.eye(CLASSES)[0])
    reordered = predict_independent([record] * 3, lambda _state, _values: np.eye(CLASSES)[0])
    result = {
        "phase": 64, "status": "passed_output_isolated_synthetic_controls", "uses_observed_values": False, "uses_observed_targets": False,
        "metrics": {"planted_macro_f1": planted_f1, "label_record_pairing_degradation": planted_f1 - label_paired_f1, "camera_pairing_degradation": planted_f1 - camera_paired_f1, "spatial_block_pairing_degradation": planted_f1 - block_paired_f1, "block_ablation_degradation": planted_f1 - ablated_f1},
        "thresholds": {"planted_macro_f1_minimum": 0.95, "label_pairing_degradation_minimum": 0.4, "camera_pairing_degradation_minimum": 0.15, "spatial_block_pairing_degradation_minimum": 0.2, "block_ablation_degradation_minimum": 0.05},
        "controls": {
            "planted_recovery": planted_f1 >= 0.95, "label_record_pairing": planted_f1 - label_paired_f1 >= 0.4,
            "camera_pairing": planted_f1 - camera_paired_f1 >= 0.15, "spatial_block_pairing": planted_f1 - block_paired_f1 >= 0.2,
            "block_ablation": planted_f1 - ablated_f1 >= 0.05, "zero_reset": states == [0.0, 0.0, 0.0],
            "reorder_invariant": bool(np.array_equal(reset, reordered[::-1])), "target_isolation": rejected({"values": record.values, "target": 0}),
            "file_isolation": rejected({"values": record.values, "file": "test"}), "split_isolation": rejected({"values": record.values, "split": "external"}),
            "row_isolation": rejected({"values": record.values, "row": 0}), "metadata_isolation": rejected({"values": record.values, "instance": 4}),
            "duplicate_isolation": rejected({"values": record.values, "duplicate_group": "artificial"}),
            "finite_normalized_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all((0.0 <= probabilities) & (probabilities <= 1.0))),
        }, "abc_smc_calls": 0, "llm_calls": 0,
    }
    if not all(result["controls"].values()): result["status"] = "failed_output_isolated_synthetic_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__": print(json.dumps(run(), indent=2))

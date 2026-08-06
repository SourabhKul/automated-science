#!/usr/bin/env python3
"""Output-isolated artificial raw-grid controls for Phase 70 KMNIST."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.kmnist import KmnistImage, image_from_payload, predict_independent

OUT = Path("artifacts/evaluations/phase70_kmnist_synthetic_controls_20260803")
SEED = 2070001
BLOCKS = tuple((slice(row * 7, (row + 1) * 7), slice(column * 7, (column + 1) * 7)) for row in range(4) for column in range(4))


class Model:
    def fit(self, images: np.ndarray, labels: np.ndarray) -> "Model":
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, random_state=SEED)
        self.model.fit(self.scaler.fit_transform(images.reshape(len(images), -1)), labels)
        return self

    def score(self, images: np.ndarray, labels: np.ndarray) -> float:
        return float(f1_score(labels, self.model.predict(self.scaler.transform(images.reshape(len(images), -1))), labels=range(10), average="macro", zero_division=0))


def artificial_images(offset: int) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for label in range(10):
        for repeat in range(10):
            grid = np.random.default_rng(SEED + offset * 10000 + label * 100 + repeat).normal(18, 3, (28, 28))
            row, column = divmod(label, 4)
            grid[row * 7:(row + 1) * 7, column * 7:(column + 1) * 7] += 180
            images.append(np.clip(grid, 0, 255))
            labels.append(label)
    return np.asarray(images), np.asarray(labels)


def rejected(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    train_images, train_labels = artificial_images(0)
    selection_images, selection_labels = artificial_images(1)
    model = Model().fit(train_images, train_labels)
    planted = model.score(selection_images, selection_labels)
    label_paired = Model().fit(train_images, np.random.default_rng(SEED + 1).permutation(train_labels)).score(selection_images, selection_labels)
    permutation = np.roll(np.arange(28).reshape(4, 7), 1, axis=0).ravel()
    block_paired = model.score(selection_images[:, permutation], selection_labels)
    ablations = []
    for rows, columns in BLOCKS:
        ablated = selection_images.copy()
        ablated[:, rows, columns] = 0
        ablations.append(planted - model.score(ablated, selection_labels))
    images = [KmnistImage(values) for values in selection_images[:8]]
    states: list[float] = []
    uniform = np.full(10, 0.1)
    probabilities = predict_independent(images, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent(images[::-1], lambda state, values: uniform)
    values = images[0].values
    controls = {
        "label_image_pairing_drop": planted - label_paired,
        "spatial_block_pairing_drop": planted - block_paired,
        "spatial_block_ablation_drops": ablations,
        "zero_reset": states == [0.0] * len(images),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejected({"values": values, "label": 0}),
        "file_rejected": rejected({"values": values, "file": "t10k-images-idx3-ubyte.gz"}),
        "split_rejected": rejected({"values": values, "split": "external"}),
        "row_rejected": rejected({"values": values, "row": 0}),
        "class_map_rejected": rejected({"values": values, "class_map": "hiragana"}),
        "duplicate_group_rejected": rejected({"values": values, "duplicate_group": 0}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    invariants = ("zero_reset", "reorder_invariant", "target_rejected", "file_rejected", "split_rejected", "row_rejected", "class_map_rejected", "duplicate_group_rejected", "finite_probabilities")
    passed = planted >= 0.95 and controls["label_image_pairing_drop"] >= 0.40 and controls["spatial_block_pairing_drop"] >= 0.20 and max(ablations) >= 0.05 and all(controls[name] for name in invariants)
    result = {"phase": 70, "status": "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure", "uses_observed_values": False, "uses_observed_labels": False, "uses_source_statistics_or_duplicate_ledger": False, "planted": {"macro_f1": planted}, "controls": controls, "thresholds": {"planted_macro_f1": 0.95, "label_pairing_drop": 0.40, "spatial_block_pairing_drop": 0.20, "max_ablation_drop": 0.05}, "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

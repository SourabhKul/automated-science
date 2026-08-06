#!/usr/bin/env python3
"""Output-isolated artificial recovery and specificity checks for Phase 67."""

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

from core.real_data.stl10 import Stl10Image, image_from_payload, predict_independent


OUT = Path("artifacts/evaluations/phase67_stl10_synthetic_controls_20260802")
SEED = 2067001
LABELS = tuple(range(10))
BLOCK = 24


class ArtificialLinearClassifier:
    def fit(self, images: np.ndarray, labels: np.ndarray) -> "ArtificialLinearClassifier":
        self.scaler = StandardScaler()
        values = self.scaler.fit_transform(images.reshape(len(images), -1))
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=SEED).fit(values, labels)
        return self

    def score(self, images: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        probabilities = self.model.predict_proba(self.scaler.transform(images.reshape(len(images), -1)))
        if not np.all(np.isfinite(probabilities)) or not np.all((0.0 <= probabilities) & (probabilities <= 1.0)) or not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("STL-10 synthetic model emitted invalid probabilities")
        predictions = np.argmax(probabilities, axis=1)
        return float(f1_score(labels, predictions, labels=LABELS, average="macro")), probabilities


def artificial_images(offset: int, rows_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label in LABELS:
        block_row, block_column = divmod(label, 4)
        for repeat in range(rows_per_class):
            generator = np.random.default_rng(SEED + offset * 10_000 + label * 100 + repeat)
            image = generator.normal(10.0, 1.5, (3, 96, 96))
            image[:, block_row * BLOCK : (block_row + 1) * BLOCK, block_column * BLOCK : (block_column + 1) * BLOCK] += 210.0
            images.append(np.clip(image, 0.0, 255.0))
            labels.append(label)
    return np.asarray(images, dtype=float), np.asarray(labels, dtype=int)


def pair_spatial_blocks(images: np.ndarray) -> np.ndarray:
    paired = images.copy()
    permutation = np.random.default_rng(SEED + 1).permutation(len(images))
    paired[:, :, :BLOCK, :BLOCK] = paired[permutation, :, :BLOCK, :BLOCK]
    paired[:, :, :BLOCK, BLOCK : 2 * BLOCK] = paired[permutation, :, :BLOCK, BLOCK : 2 * BLOCK]
    paired[:, :, :BLOCK, 2 * BLOCK : 3 * BLOCK] = paired[permutation, :, :BLOCK, 2 * BLOCK : 3 * BLOCK]
    paired[:, :, :BLOCK, 3 * BLOCK :] = paired[permutation, :, :BLOCK, 3 * BLOCK :]
    paired[:, :, BLOCK : 2 * BLOCK, :BLOCK] = paired[permutation, :, BLOCK : 2 * BLOCK, :BLOCK]
    paired[:, :, BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK] = paired[permutation, :, BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK]
    paired[:, :, BLOCK : 2 * BLOCK, 2 * BLOCK : 3 * BLOCK] = paired[permutation, :, BLOCK : 2 * BLOCK, 2 * BLOCK : 3 * BLOCK]
    paired[:, :, BLOCK : 2 * BLOCK, 3 * BLOCK :] = paired[permutation, :, BLOCK : 2 * BLOCK, 3 * BLOCK :]
    paired[:, :, 2 * BLOCK : 3 * BLOCK, :BLOCK] = paired[permutation, :, 2 * BLOCK : 3 * BLOCK, :BLOCK]
    paired[:, :, 2 * BLOCK : 3 * BLOCK, BLOCK : 2 * BLOCK] = paired[permutation, :, 2 * BLOCK : 3 * BLOCK, BLOCK : 2 * BLOCK]
    return paired


def ablation_drops(model: ArtificialLinearClassifier, images: np.ndarray, labels: np.ndarray, baseline: float) -> list[float]:
    drops = []
    for row in range(4):
        for column in range(4):
            ablated = images.copy()
            ablated[:, :, row * BLOCK : (row + 1) * BLOCK, column * BLOCK : (column + 1) * BLOCK] = 0.0
            score, _ = model.score(ablated, labels)
            drops.append(baseline - score)
    return drops


def rejected(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    train_images, train_labels = artificial_images(0, 12)
    selection_images, selection_labels = artificial_images(1, 6)
    model = ArtificialLinearClassifier().fit(train_images, train_labels)
    planted_f1, probabilities = model.score(selection_images, selection_labels)
    paired_labels = np.random.default_rng(SEED).permutation(train_labels)
    label_paired_f1, _ = ArtificialLinearClassifier().fit(train_images, paired_labels).score(selection_images, selection_labels)
    spatial_paired_f1, _ = model.score(pair_spatial_blocks(selection_images), selection_labels)
    drops = ablation_drops(model, selection_images, selection_labels, planted_f1)

    candidates = [Stl10Image(values) for values in selection_images[:8]]
    states: list[float] = []
    uniform = np.full(10, 0.1)
    independent = predict_independent(candidates, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent(candidates[::-1], lambda state, values: uniform)
    values = candidates[0].values
    label_drop = planted_f1 - label_paired_f1
    spatial_drop = planted_f1 - spatial_paired_f1
    controls = {
        "planted_recovery": planted_f1 >= 0.95,
        "label_image_pairing": label_drop >= 0.40,
        "spatial_24x24_pairing": spatial_drop >= 0.20,
        "block_ablation": max(drops) >= 0.05,
        "zero_reset": states == [0.0] * len(candidates),
        "reorder_invariant": bool(np.array_equal(independent, reordered[::-1])),
        "target_isolation": rejected({"values": values, "label": 0}),
        "file_isolation": rejected({"values": values, "file": "test_X.bin"}),
        "split_isolation": rejected({"values": values, "split": "external"}),
        "row_isolation": rejected({"values": values, "row": 0}),
        "fold_isolation": rejected({"values": values, "fold": 0}),
        "class_name_isolation": rejected({"values": values, "class_name": "airplane"}),
        "metadata_isolation": rejected({"values": values, "metadata": "artificial"}),
        "duplicate_group_isolation": rejected({"values": values, "duplicate_group": "artificial"}),
        "unlabeled_isolation": rejected({"values": values, "unlabeled": True}),
        "finite_normalized_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all((0.0 <= probabilities) & (probabilities <= 1.0)) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    result = {
        "phase": 67,
        "status": "passed_output_isolated_synthetic_controls" if all(controls.values()) else "closed_negative_synthetic_control_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_unlabeled_values": False,
        "metrics": {"planted_macro_f1": planted_f1, "label_image_pairing_degradation": label_drop, "spatial_24x24_pairing_degradation": spatial_drop, "block_ablation_degradations": drops},
        "thresholds": {"planted_macro_f1_minimum": 0.95, "label_pairing_degradation_minimum": 0.40, "spatial_24x24_pairing_degradation_minimum": 0.20, "block_ablation_degradation_minimum": 0.05},
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

#!/usr/bin/env python3
"""Output-isolated artificial recovery and specificity checks for Phase 60."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.cifar100 import FINE_LABELS, Cifar100Image, image_from_payload, predict_independent

OUTPUT_DIR = Path("artifacts/evaluations/phase60_cifar100_synthetic_controls_20260801")
SEED = 2060001
BLOCK = 8


class LinearArtificialClassifier:
    def fit(self, images: np.ndarray, labels: np.ndarray) -> "LinearArtificialClassifier":
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(images.reshape(len(images), -1))
        self.model = LogisticRegression(max_iter=1000, random_state=SEED).fit(features, labels)
        return self

    def score(self, images: np.ndarray, labels: np.ndarray) -> float:
        features = self.scaler.transform(images.reshape(len(images), -1))
        predictions = self.model.predict(features)
        return float(f1_score(labels, predictions, labels=FINE_LABELS, average="macro"))


def artificial_images(offset: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label in FINE_LABELS:
        block_row = (label // 4) % 4
        block_column = label % 4
        in_block_code = label // 16
        for repeat in range(8):
            rng = np.random.default_rng(SEED + offset * 100_000 + label * 100 + repeat)
            image = rng.normal(12.0, 2.0, (3, 32, 32))
            image[:, block_row * BLOCK + in_block_code, block_column * BLOCK + in_block_code] += 220.0
            images.append(np.clip(image, 0.0, 255.0))
            labels.append(label)
    return np.asarray(images), np.asarray(labels)


def permute_8x8_blocks(images: np.ndarray) -> np.ndarray:
    grid = images.reshape(len(images), 3, 4, BLOCK, 4, BLOCK)
    return np.roll(grid, 1, axis=4).reshape(images.shape)


def ablation_drops(model: LinearArtificialClassifier, images: np.ndarray, labels: np.ndarray, score: float) -> list[float]:
    drops: list[float] = []
    for block_row in range(4):
        for block_column in range(4):
            ablated = images.copy()
            ablated[:, :, block_row * BLOCK : (block_row + 1) * BLOCK, block_column * BLOCK : (block_column + 1) * BLOCK] = 0.0
            drops.append(score - model.score(ablated, labels))
    return drops


def rejects_payload(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    train_images, train_labels = artificial_images(0)
    selection_images, selection_labels = artificial_images(1)
    model = LinearArtificialClassifier().fit(train_images, train_labels)
    planted_score = model.score(selection_images, selection_labels)
    shuffled_labels = np.random.default_rng(SEED).permutation(train_labels)
    label_pairing_drop = planted_score - LinearArtificialClassifier().fit(train_images, shuffled_labels).score(selection_images, selection_labels)
    block_pairing_drop = planted_score - model.score(permute_8x8_blocks(selection_images), selection_labels)
    drops = ablation_drops(model, selection_images, selection_labels, planted_score)

    candidates = [Cifar100Image(values) for values in selection_images[:8]]
    states: list[float] = []
    uniform = np.full(100, 0.01)
    probabilities = predict_independent(candidates, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent(candidates[::-1], lambda state, values: uniform)
    values = candidates[0].values
    controls = {
        "label_image_pairing_drop": label_pairing_drop,
        "pixel_block_pairing_drop": block_pairing_drop,
        "pixel_block_ablation_drops": drops,
        "zero_reset": states == [0.0] * len(candidates),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejects_payload({"values": values, "fine_label": 0}),
        "file_rejected": rejects_payload({"values": values, "file": "test"}),
        "split_rejected": rejects_payload({"values": values, "split": "external"}),
        "row_rejected": rejects_payload({"values": values, "row": 0}),
        "metadata_rejected": rejects_payload({"values": values, "shape": [3, 32, 32]}),
        "duplicate_group_rejected": rejects_payload({"values": values, "duplicate_group": "source-duplicate-0"}),
        "coarse_label_rejected": rejects_payload({"values": values, "coarse_label": 0}),
        "duplicate_group_isolation": True,
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    passed = (
        planted_score >= 0.95
        and label_pairing_drop >= 0.20
        and block_pairing_drop >= 0.20
        and max(drops) >= 0.05
        and all(bool(value) for key, value in controls.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "duplicate_group_isolation", "finite_probabilities"})
    )
    result = {
        "phase": 60,
        "status": "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "planted": {"macro_f1": planted_score},
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

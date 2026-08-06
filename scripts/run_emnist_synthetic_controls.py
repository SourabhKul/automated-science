#!/usr/bin/env python3
"""Run Phase 65's fixed output-isolated artificial EMNIST controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.emnist import EmnistImage, LABELS, SHAPE, image_from_payload, predict_independent


OUTPUT_DIR = Path("artifacts/evaluations/phase65_nist_emnist_balanced_synthetic_controls_20260802")
SEED = 2065001
BLOCKS = ((slice(0, 7), slice(0, 7)), (slice(0, 7), slice(7, 14)), (slice(7, 14), slice(0, 7)), (slice(7, 14), slice(7, 14)))
BITS_BY_BLOCK = ((0, 1), (2, 3), (4,), (5,))


def macro_f1(expected: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in LABELS:
        true_positive = int(np.count_nonzero((expected == label) & (predicted == label)))
        false_positive = int(np.count_nonzero((expected != label) & (predicted == label)))
        false_negative = int(np.count_nonzero((expected == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def artificial_grid(label: int, rng: np.random.Generator) -> np.ndarray:
    bits = tuple((label >> bit) & 1 for bit in range(6))
    values = np.full(SHAPE, 20.0)
    for block, block_bits in zip(BLOCKS, BITS_BY_BLOCK, strict=True):
        code = sum(bits[bit] << offset for offset, bit in enumerate(block_bits))
        values[block] = 30.0 + 60.0 * code
    return np.clip(values + rng.normal(0.0, 1.0, SHAPE), 0.0, 255.0)


def make_partition(per_class: int, seed: int) -> tuple[list[EmnistImage], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray(LABELS, dtype=int), per_class)
    return [EmnistImage(artificial_grid(int(label), rng)) for label in labels], labels


def fit_centroids(images: list[EmnistImage], labels: np.ndarray) -> np.ndarray:
    values = np.asarray([image.values for image in images], dtype=float)
    return np.asarray([values[labels == label].mean(axis=0) for label in LABELS], dtype=float)


def probabilities(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distance = np.square(centroids - values).sum(axis=(1, 2))
    winner = int(np.argmin(distance))
    result = np.zeros(len(LABELS), dtype=float)
    result[winner] = 1.0
    return result


def score(train_images: list[EmnistImage], train_labels: np.ndarray, selection_images: list[EmnistImage], selection_labels: np.ndarray) -> tuple[float, np.ndarray]:
    centroids = fit_centroids(train_images, train_labels)
    probs = predict_independent(selection_images, lambda state, values: probabilities(values, centroids))
    return macro_f1(selection_labels, probs.argmax(axis=1)), probs


def block_permuted(images: list[EmnistImage], block_index: int, seed: int) -> list[EmnistImage]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    values = np.asarray([image.values for image in images], dtype=float)
    altered = values.copy()
    block = BLOCKS[block_index]
    altered[:, block[0], block[1]] = values[order, block[0], block[1]]
    return [EmnistImage(value) for value in altered]


def ablated(images: list[EmnistImage], block_index: int) -> list[EmnistImage]:
    result = []
    block = BLOCKS[block_index]
    for image in images:
        values = image.values.copy()
        values[block] = 0.0
        result.append(EmnistImage(values))
    return result


def rejects_payload(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    train_images, train_labels = make_partition(12, SEED)
    selection_images, selection_labels = make_partition(6, SEED + 1)
    baseline_f1, baseline_probabilities = score(train_images, train_labels, selection_images, selection_labels)
    paired_f1, _ = score(train_images, np.roll(train_labels, 1), selection_images, selection_labels)
    block_paired_f1, _ = score(block_permuted(train_images, 0, SEED + 2), train_labels, selection_images, selection_labels)
    ablation_f1 = [score(ablated(train_images, block), train_labels, ablated(selection_images, block), selection_labels)[0] for block in range(len(BLOCKS))]
    reordered_f1, reordered_probabilities = score(train_images[::-1], train_labels[::-1], selection_images[::-1], selection_labels[::-1])
    values = selection_images[0].values
    metrics = {
        "planted_macro_f1": baseline_f1,
        "label_image_pairing_degradation": baseline_f1 - paired_f1,
        "block_pairing_degradation": baseline_f1 - block_paired_f1,
        "block_ablation_degradations": [baseline_f1 - value for value in ablation_f1],
    }
    controls = {
        "planted_recovery": baseline_f1 >= 0.95,
        "label_image_pairing": metrics["label_image_pairing_degradation"] >= 0.40,
        "7x7_block_pairing": metrics["block_pairing_degradation"] >= 0.20,
        "every_7x7_block_ablation": all(value >= 0.05 for value in metrics["block_ablation_degradations"]),
        "independent_reset": bool(np.array_equal(baseline_probabilities, predict_independent(selection_images, lambda state, grid: probabilities(grid, fit_centroids(train_images, train_labels))))),
        "row_reorder_invariant": bool(np.isclose(baseline_f1, reordered_f1) and np.array_equal(baseline_probabilities, reordered_probabilities[::-1])),
        "target_rejected": rejects_payload({"values": values, "label": 0}),
        "file_rejected": rejects_payload({"values": values, "file": "emnist-balanced-test"}),
        "split_rejected": rejects_payload({"values": values, "split": "external"}),
        "row_rejected": rejects_payload({"values": values, "row": 0}),
        "mapping_rejected": rejects_payload({"values": values, "mapping": "0 48"}),
        "metadata_rejected": rejects_payload({"values": values, "metadata": {"source": "NIST"}}),
        "duplicate_group_rejected": rejects_payload({"values": values, "duplicate_group": "synthetic-0"}),
        "all_grid_finite_source_bounds": all(np.all(np.isfinite(image.values)) and np.all((image.values >= 0) & (image.values <= 255)) for image in train_images + selection_images),
        "finite_probability_metric_bounds": bool(np.all(np.isfinite(baseline_probabilities)) and np.all(baseline_probabilities >= 0) and np.allclose(baseline_probabilities.sum(axis=1), 1.0) and all(np.isfinite(value) for value in [baseline_f1, paired_f1, block_paired_f1, *ablation_f1])),
    }
    result = {
        "phase": 65,
        "status": "passed_output_isolated_synthetic_controls" if all(controls.values()) else "failed_output_isolated_synthetic_controls",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_source_statistics": False,
        "metrics": metrics,
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

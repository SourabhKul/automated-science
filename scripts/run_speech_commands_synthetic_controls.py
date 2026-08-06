#!/usr/bin/env python3
"""Output-isolated artificial recovery and specificity checks for Phase 68."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.speech_commands import (
    CLASS_COUNT,
    PCM_BOUNDS,
    SAMPLES,
    SpeechCommandsWaveform,
    predict_independent,
    waveform_from_payload,
)


OUT = Path("artifacts/evaluations/phase68_tensorflow_speech_commands_v0_02_raw_waveform_synthetic_controls_20260803")
SEED = 2_068_001
LABELS = tuple(range(CLASS_COUNT))
BLOCK = 800
BLOCKS = SAMPLES // BLOCK
MARKER = 64


class ArtificialCentroidClassifier:
    def fit(self, waveforms: np.ndarray, labels: np.ndarray) -> "ArtificialCentroidClassifier":
        self.centroids = np.asarray([waveforms[labels == label].mean(axis=0) for label in LABELS], dtype=float)
        return self

    def score(self, waveforms: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        distances = np.mean((waveforms[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        scale = max(float(np.median(distances)), 1.0)
        shifted = -(distances - distances.min(axis=1, keepdims=True)) / scale
        weights = np.exp(shifted)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0)
        ):
            raise ValueError("Speech Commands synthetic model emitted invalid probabilities")
        predictions = np.argmax(probabilities, axis=1)
        return float(f1_score(labels, predictions, labels=LABELS, average="macro")), probabilities


def artificial_waveforms(offset: int, rows_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for label in LABELS:
        block = label % BLOCKS
        slot = label // BLOCKS
        start = block * BLOCK + slot * 80
        for repeat in range(rows_per_class):
            generator = np.random.default_rng(SEED + offset * 100_000 + label * 100 + repeat)
            waveform = generator.normal(0.0, 120.0, SAMPLES)
            waveform[start : start + MARKER] += 20_000.0
            waveforms.append(np.clip(np.rint(waveform), PCM_BOUNDS[0], PCM_BOUNDS[1]))
            labels.append(label)
    return np.asarray(waveforms, dtype=float), np.asarray(labels, dtype=int)


def pair_waveform_blocks(waveforms: np.ndarray) -> np.ndarray:
    paired = waveforms.copy()
    for block in range(BLOCKS):
        permutation = np.random.default_rng(SEED + block + 1).permutation(len(waveforms))
        begin = block * BLOCK
        paired[:, begin : begin + BLOCK] = waveforms[permutation, begin : begin + BLOCK]
    return paired


def ablation_drops(
    model: ArtificialCentroidClassifier,
    waveforms: np.ndarray,
    labels: np.ndarray,
    baseline: float,
) -> list[float]:
    drops = []
    for block in range(BLOCKS):
        ablated = waveforms.copy()
        begin = block * BLOCK
        ablated[:, begin : begin + BLOCK] = 0.0
        score, _ = model.score(ablated, labels)
        drops.append(baseline - score)
    return drops


def rejected(payload: dict[str, object]) -> bool:
    try:
        waveform_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    train_waveforms, train_labels = artificial_waveforms(0, 10)
    selection_waveforms, selection_labels = artificial_waveforms(1, 5)
    model = ArtificialCentroidClassifier().fit(train_waveforms, train_labels)
    planted_f1, probabilities = model.score(selection_waveforms, selection_labels)
    paired_labels = np.random.default_rng(SEED).permutation(train_labels)
    label_paired_f1, _ = ArtificialCentroidClassifier().fit(train_waveforms, paired_labels).score(
        selection_waveforms, selection_labels
    )
    block_paired_f1, _ = model.score(pair_waveform_blocks(selection_waveforms), selection_labels)
    time_shifted_f1, _ = model.score(np.roll(selection_waveforms, BLOCK, axis=1), selection_labels)
    drops = ablation_drops(model, selection_waveforms, selection_labels, planted_f1)

    candidates = [SpeechCommandsWaveform(values) for values in selection_waveforms[:8]]
    states: list[float] = []
    uniform = np.full(CLASS_COUNT, 1.0 / CLASS_COUNT)
    independent = predict_independent(candidates, lambda state, samples: states.append(state) or uniform)
    reordered = predict_independent(candidates[::-1], lambda state, samples: uniform)
    samples = candidates[0].samples
    label_drop = planted_f1 - label_paired_f1
    block_pairing_drop = planted_f1 - block_paired_f1
    time_order_drop = planted_f1 - time_shifted_f1
    controls = {
        "planted_recovery": planted_f1 >= 0.95,
        "label_window_pairing": label_drop >= 0.40,
        "waveform_800_block_pairing": block_pairing_drop >= 0.20,
        "circular_800_time_order": time_order_drop >= 0.20,
        "block_ablation": max(drops) >= 0.05,
        "zero_reset": states == [0.0] * len(candidates),
        "reorder_invariant": bool(np.array_equal(independent, reordered[::-1])),
        "target_isolation": rejected({"samples": samples, "label": 0}),
        "file_isolation": rejected({"samples": samples, "file": "testing_list.txt"}),
        "split_isolation": rejected({"samples": samples, "split": "external"}),
        "row_isolation": rejected({"samples": samples, "row": 0}),
        "path_isolation": rejected({"samples": samples, "path": "yes/0123abcd_nohash_0.wav"}),
        "speaker_isolation": rejected({"samples": samples, "speaker_hash": "0123abcd"}),
        "utterance_isolation": rejected({"samples": samples, "utterance_index": 0}),
        "suffix_isolation": rejected({"samples": samples, "suffix": samples}),
        "duplicate_group_isolation": rejected({"samples": samples, "duplicate_group": "artificial"}),
        "metadata_isolation": rejected({"samples": samples, "metadata": "locked_creator_list"}),
        "finite_normalized_probabilities": bool(
            np.all(np.isfinite(probabilities))
            and np.all((0.0 <= probabilities) & (probabilities <= 1.0))
            and np.allclose(probabilities.sum(axis=1), 1.0)
        ),
    }
    result = {
        "phase": 68,
        "status": "passed_output_isolated_synthetic_controls"
        if all(controls.values())
        else "closed_negative_synthetic_control_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_source_statistics": False,
        "metrics": {
            "planted_macro_f1": planted_f1,
            "label_window_pairing_degradation": label_drop,
            "waveform_800_block_pairing_degradation": block_pairing_drop,
            "circular_800_time_order_degradation": time_order_drop,
            "block_ablation_degradations": drops,
        },
        "thresholds": {
            "planted_macro_f1_minimum": 0.95,
            "label_pairing_degradation_minimum": 0.40,
            "waveform_800_block_pairing_degradation_minimum": 0.20,
            "circular_800_time_order_degradation_minimum": 0.20,
            "block_ablation_degradation_minimum": 0.05,
        },
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

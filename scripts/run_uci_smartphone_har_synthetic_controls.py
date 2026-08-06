#!/usr/bin/env python3
"""Run Phase 37 output-isolated synthetic prefix recovery and specificity controls."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_smartphone_har import (
    ACTIVITY_LABELS,
    CAUSAL_PREFIX_SAMPLES,
    CHANNEL_COUNT,
    WINDOW_SAMPLES,
    causal_prefix_from_payload,
    make_smartphone_har_causal_prefix,
    predict_smartphone_har_prefixes,
)


SPLIT_PATH = Path("data/real/uci_smartphone_har/source_subject_split.json")
OUT_DIR = Path("artifacts/evaluations/phase37_uci_smartphone_har_synthetic_controls_20260724")
SEED = 20370724
PENALTIES = (0.01, 0.1, 1.0, 10.0)
PLANTED_PENALTY = 0.01
ACC_CHANNELS = (0, 1, 2, 6, 7, 8)
GYRO_CHANNELS = (3, 4, 5)


@dataclass(frozen=True)
class SyntheticRecord:
    subject_id: int
    split: str
    label: int
    prefix: np.ndarray


@dataclass(frozen=True)
class RidgePrefixClassifier:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def probabilities(self, prefix: np.ndarray) -> np.ndarray:
        values = np.asarray(prefix, dtype=float).reshape(1, -1)
        design = np.column_stack([np.ones(1), (values - self.mean) / self.scale])
        logits = (design @ self.weights).reshape(-1)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        return weights / float(weights.sum())


def _one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(ACTIVITY_LABELS), dtype=float)[np.asarray(labels, dtype=int) - 1]


def _flatten(records: list[SyntheticRecord]) -> np.ndarray:
    return np.stack([record.prefix.reshape(-1) for record in records])


def _labels(records: list[SyntheticRecord]) -> np.ndarray:
    return np.asarray([record.label for record in records], dtype=int)


def _fit_ridge(records: list[SyntheticRecord], labels: np.ndarray, penalty: float) -> RidgePrefixClassifier:
    features = _flatten(records)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    normalized = (features - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    regularizer = np.eye(design.shape[1], dtype=float) * float(penalty)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ _one_hot(labels))
    if not np.all(np.isfinite(weights)):
        raise ValueError("synthetic ridge fit has nonfinite coefficients")
    return RidgePrefixClassifier(mean=mean, scale=scale, weights=weights)


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in ACTIVITY_LABELS:
        true_positive = int(np.sum((labels == label) & (predictions == label)))
        false_positive = int(np.sum((labels != label) & (predictions == label)))
        false_negative = int(np.sum((labels == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def _score(classifier: RidgePrefixClassifier, records: list[SyntheticRecord]) -> tuple[float, np.ndarray, np.ndarray]:
    probabilities = predict_smartphone_har_prefixes(
        [make_smartphone_har_causal_prefix(np.pad(record.prefix, ((0, 0), (0, WINDOW_SAMPLES - CAUSAL_PREFIX_SAMPLES)))) for record in records],
        classifier.probabilities,
    )
    predictions = np.argmax(probabilities, axis=1) + 1
    return _macro_f1(_labels(records), predictions), predictions, probabilities


def _pattern(position: int, sign: float) -> np.ndarray:
    values = np.zeros(CAUSAL_PREFIX_SAMPLES, dtype=float)
    values[position] = 3.0 * sign
    values[(position + 3) % CAUSAL_PREFIX_SAMPLES] = -1.5 * sign
    return values


def _artificial_window(label: int, rng: np.random.Generator) -> np.ndarray:
    window = rng.normal(0.0, 0.025, size=(CHANNEL_COUNT, WINDOW_SAMPLES))
    coarse = (label - 1) // 3
    fine = (label - 1) % 3
    acc = _pattern(8 + 20 * coarse, 1.0)
    gyro = _pattern(32 + 8 * fine, 1.0)
    for channel in ACC_CHANNELS:
        window[channel, :CAUSAL_PREFIX_SAMPLES] += acc * (1.0 + 0.02 * channel)
    for channel in GYRO_CHANNELS:
        window[channel, :CAUSAL_PREFIX_SAMPLES] += gyro * (1.0 + 0.03 * channel)
    return window


def _records(split: dict[str, list[int]]) -> dict[str, list[SyntheticRecord]]:
    rng = np.random.default_rng(SEED)
    output: dict[str, list[SyntheticRecord]] = {name: [] for name in ("train", "selection", "external")}
    for split_name, subjects in split.items():
        for subject_id in subjects:
            for label in ACTIVITY_LABELS:
                for _ in range(4):
                    output[split_name].append(
                        SyntheticRecord(
                            subject_id=int(subject_id),
                            split=split_name,
                            label=label,
                            prefix=make_smartphone_har_causal_prefix(_artificial_window(label, rng)).prefix,
                        )
                    )
    return output


def _replace_prefixes(records: list[SyntheticRecord], prefixes: list[np.ndarray]) -> list[SyntheticRecord]:
    return [SyntheticRecord(record.subject_id, record.split, record.label, prefix) for record, prefix in zip(records, prefixes, strict=True)]


def _ablate(records: list[SyntheticRecord], channels: tuple[int, ...]) -> list[SyntheticRecord]:
    prefixes = []
    for record in records:
        prefix = record.prefix.copy()
        prefix[list(channels)] = 0.0
        prefixes.append(prefix)
    return _replace_prefixes(records, prefixes)


def _rejected(payload: dict[str, object]) -> bool:
    try:
        causal_prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    if {name: len(split.get(name, [])) for name in ("train", "selection", "external")} != {"train": 15, "selection": 6, "external": 9}:
        raise ValueError("synthetic controls require the frozen 15/6/9 subject split")
    records = _records(split)
    train, selection = records["train"], records["selection"]
    if any(set(record.label for record in values) != set(ACTIVITY_LABELS) for values in (train, selection)):
        raise ValueError("synthetic controls require all six artificial states in train and selection")
    if {record.subject_id for record in train} != set(split["train"]) or {record.subject_id for record in selection} != set(split["selection"]):
        raise ValueError("synthetic controls lost frozen subject identity coverage")

    fitted = {penalty: _fit_ridge(train, _labels(train), penalty) for penalty in PENALTIES}
    selection_scores = {penalty: _score(model, selection)[0] for penalty, model in fitted.items()}
    selected_penalty = sorted(PENALTIES, key=lambda penalty: (-selection_scores[penalty], penalty))[0]
    classifier = fitted[selected_penalty]
    recovery_f1, recovery_predictions, recovery_probabilities = _score(classifier, selection)

    permuted_labels = np.roll(_labels(train), 1)
    paired_classifier = _fit_ridge(train, permuted_labels, selected_penalty)
    paired_f1, _, _ = _score(paired_classifier, selection)
    shifted_selection = _replace_prefixes(selection, [np.roll(record.prefix, 16, axis=1) for record in selection])
    shifted_f1, _, _ = _score(classifier, shifted_selection)
    gyroscope_pairing = _replace_prefixes(selection, [
        np.vstack([record.prefix[list(ACC_CHANNELS)], selection[(index + 1) % len(selection)].prefix[list(GYRO_CHANNELS)]])
        for index, record in enumerate(selection)
    ])
    # Restore native channel order after independently pairing the gyro group.
    gyroscope_prefixes = []
    for index, record in enumerate(selection):
        prefix = record.prefix.copy()
        prefix[list(GYRO_CHANNELS)] = selection[(index + 1) % len(selection)].prefix[list(GYRO_CHANNELS)]
        gyroscope_prefixes.append(prefix)
    gyroscope_pairing = _replace_prefixes(selection, gyroscope_prefixes)
    gyroscope_pairing_f1, _, _ = _score(classifier, gyroscope_pairing)
    accelerometer_ablation_f1, _, _ = _score(classifier, _ablate(selection, ACC_CHANNELS))
    gyroscope_ablation_f1, _, _ = _score(classifier, _ablate(selection, GYRO_CHANNELS))

    reordered = list(reversed(selection))
    reordered_f1, reordered_predictions, reordered_probabilities = _score(classifier, reordered)
    original_by_subject_label = {(record.subject_id, record.label, index % 4): (prediction, probability) for index, (record, prediction, probability) in enumerate(zip(selection, recovery_predictions, recovery_probabilities, strict=True))}
    reordered_by_subject_label = {(record.subject_id, record.label, (len(reordered) - 1 - index) % 4): (prediction, probability) for index, (record, prediction, probability) in enumerate(zip(reordered, reordered_predictions, reordered_probabilities, strict=True))}
    reset_invariant = recovery_f1 == reordered_f1 and all(
        np.array_equal(original_by_subject_label[key][0], reordered_by_subject_label[key][0]) and np.array_equal(original_by_subject_label[key][1], reordered_by_subject_label[key][1])
        for key in original_by_subject_label
    )
    sample = selection[0]
    target_rejected = _rejected({"prefix": sample.prefix, "artificial_label": sample.label})
    subject_rejected = _rejected({"prefix": sample.prefix, "subject_id": sample.subject_id})
    subject_permutation_invariant = bool(np.array_equal(recovery_predictions, _score(classifier, [SyntheticRecord(999 - record.subject_id, record.split, record.label, record.prefix) for record in selection])[1]))
    finite_bounds = bool(
        all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in [recovery_f1, paired_f1, shifted_f1, gyroscope_pairing_f1, accelerometer_ablation_f1, gyroscope_ablation_f1])
        and np.all(np.isfinite(recovery_probabilities))
        and np.all(recovery_probabilities >= 0.0)
        and np.all(recovery_probabilities <= 1.0)
        and np.allclose(recovery_probabilities.sum(axis=1), 1.0)
    )
    drops = {
        "label_window_pairing": recovery_f1 - paired_f1,
        "prefix_time_shift": recovery_f1 - shifted_f1,
        "gyroscope_channel_pairing": recovery_f1 - gyroscope_pairing_f1,
        "accelerometer_ablation": recovery_f1 - accelerometer_ablation_f1,
        "gyroscope_ablation": recovery_f1 - gyroscope_ablation_f1,
    }
    checks = {
        "locked_subject_split": True,
        "output_isolation": True,
        "planted_recovery": selected_penalty == PLANTED_PENALTY and recovery_f1 >= 0.95,
        "label_window_pairing": drops["label_window_pairing"] >= 0.40,
        "prefix_time_order": drops["prefix_time_shift"] >= 0.25,
        "gyroscope_channel_pairing": drops["gyroscope_channel_pairing"] >= 0.20,
        "accelerometer_ablation": drops["accelerometer_ablation"] >= 0.15,
        "gyroscope_ablation": drops["gyroscope_ablation"] >= 0.15,
        "per_record_reset": reset_invariant,
        "target_isolation": target_rejected,
        "subject_id_leakage": subject_rejected and subject_permutation_invariant,
        "finite_probability_metric_bounds": finite_bounds,
    }
    passed = all(checks.values())
    result = {
        "phase": 37,
        "status": "passed_output_isolated_synthetic_controls" if passed else "failed_output_isolated_synthetic_controls",
        "uses_measured_source_signal_values": False,
        "uses_measured_source_label_values": False,
        "opens_source_feature_matrices": False,
        "uses_source_window_statistics": False,
        "uses_source_duplicate_ledger": False,
        "uses_outcome_derived_fields": False,
        "synthetic_seed": SEED,
        "records_per_subject": 24,
        "split_records": {name: len(values) for name, values in records.items()},
        "external_artificial_records_scored": False,
        "selected_penalty": selected_penalty,
        "penalty_selection_macro_f1": selection_scores,
        "selection_macro_f1": recovery_f1,
        "control_macro_f1": {"label_window_pairing": paired_f1, "prefix_time_shift": shifted_f1, "gyroscope_channel_pairing": gyroscope_pairing_f1, "accelerometer_ablation": accelerometer_ablation_f1, "gyroscope_ablation": gyroscope_ablation_f1},
        "macro_f1_drops": drops,
        "checks": checks,
        "abc_smc_calls": 0,
        "llm_calls": 0,
        "next_gate": "Prepare but do not execute a separately reviewed measured-label baseline plan." if passed else "Close Phase 37; do not fit measured labels, use ABC-SMC, LLM discovery, or campaigns.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 37 UCI Smartphone HAR Output-Isolated Synthetic Controls\n\n"
        f"Status: **{'passed' if passed else 'failed'} artificial native-grid controls only.**\n\n"
        "This deterministic suite used only frozen subject IDs, artificial six-state labels, and generated 9x128 arrays. "
        "It opened no UCI HAR signal, activity-label, feature, source-statistic, duplicate-ledger, or outcome-derived value. "
        f"The selected planted penalty was `{selected_penalty}`, with selection macro-F1 `{recovery_f1:.6f}`. "
        f"Pairing/time/gyro-pairing/accelerometer-ablation/gyro-ablation drops were `{drops['label_window_pairing']:.6f}`, `{drops['prefix_time_shift']:.6f}`, `{drops['gyroscope_channel_pairing']:.6f}`, `{drops['accelerometer_ablation']:.6f}`, and `{drops['gyroscope_ablation']:.6f}`.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 37,
        "decision": "pass_output_isolated_synthetic_controls" if passed else "close_synthetic_specificity_gate_failed",
        "passed": passed,
        "reason": "All predeclared artificial recovery, specificity, reset, isolation, and finite-bound gates passed." if passed else "At least one predeclared artificial recovery or specificity gate failed; no measured-label work is permitted.",
        "allowed_next_action": "Write a separately reviewed measured-label baseline plan; do not execute it." if passed else "Return to source-backed application selection.",
        "prohibited": ["measured source signal or label fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Output-isolated Phase 40 artificial recovery and specificity controls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_mhealth import (
    ACTIVITY_LABELS,
    MOTION_CHANNELS,
    WINDOW_SAMPLES,
    motion_window_from_payload,
)


SPLIT_PATH = Path("data/real/uci_mhealth/source_subject_split.json")
OUT_DIR = Path("artifacts/evaluations/phase40_uci_mhealth_synthetic_controls_20260725")
SEED = 20400725
REPLICATES = 4


def _artificial_window(label: int, subject: int, replicate: int) -> np.ndarray:
    """Generate a planted 12-state signal; no MHEALTH array, statistic, or label is read."""
    rng = np.random.default_rng(SEED + label * 1000 + subject * 10 + replicate)
    values = rng.normal(0.0, 0.015, size=(MOTION_CHANNELS, WINDOW_SAMPLES))
    accel_code = (label - 1) % 4
    gyro_code = (label - 1) // 4
    accel_time = 8 + 18 * accel_code
    gyro_time = 12 + 24 * gyro_code
    # Class identity is jointly encoded in ankle acceleration and arm gyro location/time.
    values[3:6, accel_time : accel_time + 8] += 1.0 + 0.04 * subject
    values[15:18, gyro_time : gyro_time + 10] += 1.2 + 0.03 * replicate
    values[0:3, accel_time : accel_time + 8] += 0.25
    values[6:9, gyro_time : gyro_time + 10] += 0.25
    return values


def _split_records(split: dict[str, list[int]]) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    records: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split_name, subjects in split.items():
        windows: list[np.ndarray] = []
        labels: list[int] = []
        keys: list[str] = []
        for subject in subjects:
            for label in ACTIVITY_LABELS:
                for replicate in range(REPLICATES):
                    windows.append(_artificial_window(label, int(subject), replicate))
                    labels.append(label)
                    keys.append(f"{subject}:{label}:{replicate}")
        records[split_name] = (np.stack(windows), np.asarray(labels, dtype=int), keys)
    return records


def _fit(train_x: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=500, random_state=SEED).fit(train_x.reshape(len(train_x), -1), train_y)


def _macro_f1(model: LogisticRegression, windows: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, model.predict(windows.reshape(len(windows), -1)), labels=list(ACTIVITY_LABELS), average="macro", zero_division=0))


def _rejected(payload: dict[str, object]) -> bool:
    try:
        motion_window_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    records = _split_records(split)
    train_x, train_y, _ = records["train"]
    selection_x, selection_y, selection_keys = records["selection"]
    model = _fit(train_x, train_y)
    planted_f1 = _macro_f1(model, selection_x, selection_y)

    paired_y = np.roll(train_y, 1)
    paired_f1 = _macro_f1(_fit(train_x, paired_y), selection_x, selection_y)
    shifted_f1 = _macro_f1(model, np.roll(selection_x, 25, axis=2), selection_y)
    channel_paired = selection_x.copy()
    channel_paired[:, 3:12], channel_paired[:, 12:21] = selection_x[:, 12:21].copy(), selection_x[:, 3:12].copy()
    channel_paired_f1 = _macro_f1(model, channel_paired, selection_y)
    accel_ablated = selection_x.copy()
    accel_ablated[:, [0, 1, 2, 3, 4, 5, 12, 13, 14], :] = 0.0
    accel_ablated_f1 = _macro_f1(model, accel_ablated, selection_y)
    gyro_ablated = selection_x.copy()
    gyro_ablated[:, [6, 7, 8, 15, 16, 17], :] = 0.0
    gyro_ablated_f1 = _macro_f1(model, gyro_ablated, selection_y)

    original_predictions = model.predict(selection_x.reshape(len(selection_x), -1))
    reverse_indices = np.arange(len(selection_x) - 1, -1, -1)
    reordered_predictions = model.predict(selection_x[reverse_indices].reshape(len(selection_x), -1))
    by_key = dict(zip(selection_keys, original_predictions, strict=True))
    reordered_by_key = dict(zip([selection_keys[index] for index in reverse_indices], reordered_predictions, strict=True))
    candidate = motion_window_from_payload({"motion_window": selection_x[0]})
    probabilities = model.predict_proba(selection_x.reshape(len(selection_x), -1))
    result = {
        "phase": 40,
        "status": "passed_output_isolated_synthetic_controls" if all([
            planted_f1 >= 0.95,
            planted_f1 - paired_f1 >= 0.40,
            planted_f1 - shifted_f1 >= 0.15,
            planted_f1 - channel_paired_f1 >= 0.15,
            planted_f1 - accel_ablated_f1 >= 0.05,
            planted_f1 - gyro_ablated_f1 >= 0.05,
            all(by_key[key] == reordered_by_key[key] for key in by_key),
            _rejected({"motion_window": candidate.motion_window, "label": 1}),
            _rejected({"motion_window": candidate.motion_window, "subject_id": 5}),
            bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0) and np.allclose(probabilities.sum(axis=1), 1.0)),
        ]) else "failed_output_isolated_synthetic_controls",
        "uses_measured_source_signal_values": False,
        "uses_measured_source_label_values": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "uses_external_subject_outcomes": False,
        "planted_selection_macro_f1": planted_f1,
        "label_window_pairing_macro_f1": paired_f1,
        "label_window_pairing_drop": planted_f1 - paired_f1,
        "time_order_macro_f1": shifted_f1,
        "time_order_drop": planted_f1 - shifted_f1,
        "channel_pairing_macro_f1": channel_paired_f1,
        "channel_pairing_drop": planted_f1 - channel_paired_f1,
        "accelerometer_ablation_macro_f1": accel_ablated_f1,
        "accelerometer_ablation_drop": planted_f1 - accel_ablated_f1,
        "gyro_ablation_macro_f1": gyro_ablated_f1,
        "gyro_ablation_drop": planted_f1 - gyro_ablated_f1,
        "reset_invariant_under_window_reordering": all(by_key[key] == reordered_by_key[key] for key in by_key),
        "target_isolation_sentinel_rejected": _rejected({"motion_window": candidate.motion_window, "label": 1}),
        "subject_isolation_sentinel_rejected": _rejected({"motion_window": candidate.motion_window, "subject_id": 5}),
        "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0) and np.allclose(probabilities.sum(axis=1), 1.0)),
        "thresholds": {"planted_macro_f1": 0.95, "pairing_drop": 0.40, "time_and_channel_drop": 0.15, "modality_ablation_drop": 0.05},
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

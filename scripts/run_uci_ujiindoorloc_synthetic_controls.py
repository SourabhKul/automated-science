#!/usr/bin/env python3
"""Run Phase 49 output-isolated artificial WLAN recovery controls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_ujiindoorloc import (
    CLASS_LABELS,
    FEATURE_COUNT,
    UJIIndoorLocWLAN,
    predict_independent_wlans,
    wlan_from_payload,
)

SEED = 2049001
BLOCK_COUNT = 8
BLOCK_SIZE = FEATURE_COUNT // BLOCK_COUNT
OUT = Path("artifacts/evaluations/phase49_uci_ujiindoorloc_synthetic_controls_20260727")


class Model:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=SEED)

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "Model":
        self.model.fit(self.scaler.fit_transform(values), labels)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(values))


def artificial_records(repeats: int, partition_offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate source-unit vectors without inspecting any source WAP field."""
    values, labels = [], []
    for label in CLASS_LABELS:
        for repeat in range(repeats):
            rng = np.random.default_rng(SEED + 100_000 * partition_offset + 100 * label + repeat)
            record = np.full(FEATURE_COUNT, 100.0)
            # A planted building state is solely a source-unit WAP position and sentinel mask.
            record[label] = float(-42 - 8 * label + rng.integers(-2, 3))
            values.append(record)
            labels.append(label)
    return np.asarray(values), np.asarray(labels)


def metrics(model: Model, values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(values)
    if probabilities.shape != (len(labels), len(CLASS_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("UJIIndoorLoc synthetic probabilities are invalid")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("UJIIndoorLoc synthetic probabilities are out of bounds")
    prediction = np.asarray(CLASS_LABELS)[np.argmax(probabilities, axis=1)]
    return {
        "macro_f1": float(f1_score(labels, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(labels, probabilities, labels=CLASS_LABELS)),
    }


def reject(payload: dict[str, object]) -> bool:
    try:
        wlan_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    train_values, train_labels = artificial_records(20, 0)
    selection_values, selection_labels = artificial_records(20, 1)
    model = Model().fit(train_values, train_labels)
    planted = metrics(model, selection_values, selection_labels)
    paired = metrics(
        Model().fit(train_values, np.random.default_rng(SEED).permutation(train_labels)),
        selection_values,
        selection_labels,
    )
    block_permutation = np.roll(np.arange(BLOCK_COUNT), 1)
    block_paired_values = selection_values.reshape(-1, BLOCK_COUNT, BLOCK_SIZE)[:, block_permutation, :].reshape(-1, FEATURE_COUNT)
    block_paired = metrics(model, block_paired_values, selection_labels)
    shifted_sentinel_mask = np.roll(selection_values == 100.0, 1, axis=1)
    sentinel_paired_values = np.where(shifted_sentinel_mask, 100.0, -104.0)
    sentinel_paired = metrics(model, sentinel_paired_values, selection_labels)
    block_ablations = []
    for block in range(BLOCK_COUNT):
        ablated = selection_values.copy()
        ablated[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE] = 100.0
        block_ablations.append(planted["macro_f1"] - metrics(model, ablated, selection_labels)["macro_f1"])
    wlans = [UJIIndoorLocWLAN(value) for value in selection_values[:9]]
    states: list[float] = []
    uniform = np.full(len(CLASS_LABELS), 1.0 / len(CLASS_LABELS))
    probabilities = predict_independent_wlans(wlans, lambda state, _values: states.append(state) or uniform)
    reordered = predict_independent_wlans(list(reversed(wlans)), lambda _state, _values: uniform)
    candidate = wlans[0]
    controls = {
        "label_record_pairing_drop": planted["macro_f1"] - paired["macro_f1"],
        "wap_block_pairing_drop": planted["macro_f1"] - block_paired["macro_f1"],
        "sentinel_mask_pairing_drop": planted["macro_f1"] - sentinel_paired["macro_f1"],
        "block_ablation_drops": block_ablations,
        "per_record_zero_reset": states == [0.0] * len(wlans),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_isolation_sentinel_rejected": reject({"waps": candidate.waps, "label": 0}),
        "file_isolation_sentinel_rejected": reject({"waps": candidate.waps, "source_file": "UJIndoorLoc/trainingData.csv"}),
        "split_isolation_sentinel_rejected": reject({"waps": candidate.waps, "split": "source_train"}),
        "metadata_isolation_sentinel_rejected": reject({"waps": candidate.waps, "longitude": 0.0, "floor": 0, "timestamp": 1}),
        "probabilities_finite_and_bounded": bool(
            np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1.0)
        ),
    }
    passed = all(
        [
            planted["macro_f1"] >= 0.95,
            controls["label_record_pairing_drop"] >= 0.40,
            controls["wap_block_pairing_drop"] >= 0.10,
            controls["sentinel_mask_pairing_drop"] >= 0.10,
            max(controls["block_ablation_drops"]) >= 0.05,
            *[
                bool(value)
                for key, value in controls.items()
                if key.endswith("rejected") or key in {"per_record_zero_reset", "reorder_invariant", "probabilities_finite_and_bounded"}
            ],
        ]
    )
    result = {
        "phase": 49,
        "status": "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure",
        "uses_observed_waps": False,
        "uses_observed_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "locked_source_files_as_metadata": ["UJIndoorLoc/trainingData.csv", "UJIndoorLoc/validationData.csv"],
        "planted": planted,
        "controls": controls,
        "thresholds": {"planted_macro_f1": 0.95, "label_pairing_drop": 0.40, "wap_block_pairing_drop": 0.10, "sentinel_pairing_drop": 0.10, "ablation_drop": 0.05},
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

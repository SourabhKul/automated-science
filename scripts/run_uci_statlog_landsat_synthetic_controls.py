#!/usr/bin/env python3
"""Run Phase 47 output-isolated artificial Landsat recovery controls."""
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

from core.real_data.uci_statlog_landsat import (
    BANDS,
    CLASS_LABELS,
    PIXELS,
    LandsatNeighborhood,
    neighborhood_from_payload,
    predict_independent_neighborhoods,
)


SEED = 2047001
OUT = Path("artifacts/evaluations/phase47_uci_statlog_landsat_synthetic_controls_20260727")
PIXEL_PERMUTATION = np.roll(np.arange(PIXELS).reshape(3, 3), 1, axis=1).ravel()
BAND_PERMUTATION = np.roll(np.arange(BANDS), 1)


class Model:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "Model":
        self.model.fit(self.scaler.fit_transform(values.reshape(len(values), -1)), labels)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(values.reshape(len(values), -1)))


def artificial_records(repeats: int, partition_offset: int) -> tuple[np.ndarray, np.ndarray]:
    values, labels = [], []
    for label_index, label in enumerate(CLASS_LABELS):
        for repeat in range(repeats):
            rng = np.random.default_rng(SEED + 100_000 * partition_offset + 100 * label_index + repeat)
            record = rng.normal(0, 0.03, size=(PIXELS, BANDS))
            pixel, band = divmod(label_index, BANDS)
            record[pixel, band] += 4
            record[(pixel + 4) % PIXELS, (band + 1) % BANDS] -= 2
            values.append(record)
            labels.append(label)
    return np.asarray(values), np.asarray(labels)


def metrics(model: Model, values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(values)
    if probabilities.shape != (len(labels), len(CLASS_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Landsat synthetic probabilities are invalid")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Landsat synthetic probabilities are out of bounds")
    prediction = np.asarray(CLASS_LABELS)[np.argmax(probabilities, axis=1)]
    return {
        "macro_f1": float(f1_score(labels, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(labels, probabilities, labels=CLASS_LABELS)),
    }


def reject(payload: dict[str, object]) -> bool:
    try:
        neighborhood_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    train_values, train_labels = artificial_records(4, 0)
    selection_values, selection_labels = artificial_records(4, 1)
    model = Model().fit(train_values, train_labels)
    planted = metrics(model, selection_values, selection_labels)
    paired = metrics(Model().fit(train_values, np.random.default_rng(SEED).permutation(train_labels)), selection_values, selection_labels)
    pixel_paired = metrics(model, selection_values[:, PIXEL_PERMUTATION, :], selection_labels)
    band_paired = metrics(model, selection_values[:, :, BAND_PERMUTATION], selection_labels)
    pixel_ablations, band_ablations = [], []
    for pixel in range(PIXELS):
        values = selection_values.copy()
        values[:, pixel, :] = 0
        pixel_ablations.append(planted["macro_f1"] - metrics(model, values, selection_labels)["macro_f1"])
    for band in range(BANDS):
        values = selection_values.copy()
        values[:, :, band] = 0
        band_ablations.append(planted["macro_f1"] - metrics(model, values, selection_labels)["macro_f1"])
    neighborhoods = [LandsatNeighborhood(value) for value in selection_values[:8]]
    states: list[float] = []
    uniform = np.full(len(CLASS_LABELS), 1 / len(CLASS_LABELS))
    probabilities = predict_independent_neighborhoods(neighborhoods, lambda state, value: states.append(state) or uniform)
    reordered = predict_independent_neighborhoods(list(reversed(neighborhoods)), lambda state, value: uniform)
    candidate = neighborhoods[0]
    controls = {
        "label_record_pairing_drop": planted["macro_f1"] - paired["macro_f1"],
        "pixel_order_pairing_drop": planted["macro_f1"] - pixel_paired["macro_f1"],
        "band_pairing_drop": planted["macro_f1"] - band_paired["macro_f1"],
        "pixel_ablation_drops": pixel_ablations,
        "band_ablation_drops": band_ablations,
        "per_record_zero_reset": states == [0.0] * len(neighborhoods),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_isolation_sentinel_rejected": reject({"values": candidate.values, "label": 1}),
        "file_isolation_sentinel_rejected": reject({"values": candidate.values, "source_file": "sat.trn"}),
        "split_isolation_sentinel_rejected": reject({"values": candidate.values, "split": "train"}),
        "row_isolation_sentinel_rejected": reject({"values": candidate.values, "row_index": 0}),
        "location_isolation_sentinel_rejected": reject({"values": candidate.values, "location": [0, 0]}),
        "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1)),
    }
    passed = all([
        planted["macro_f1"] >= 0.95,
        controls["label_record_pairing_drop"] >= 0.40,
        controls["pixel_order_pairing_drop"] >= 0.10,
        controls["band_pairing_drop"] >= 0.10,
        max(controls["pixel_ablation_drops"] + controls["band_ablation_drops"]) >= 0.03,
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key in {"per_record_zero_reset", "reorder_invariant", "probabilities_finite_and_bounded"}],
    ])
    result = {
        "phase": 47,
        "status": "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "planted": planted,
        "controls": controls,
        "thresholds": {"planted_macro_f1": 0.95, "label_pairing_drop": 0.40, "pixel_order_drop": 0.10, "band_pairing_drop": 0.10, "ablation_drop": 0.03},
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

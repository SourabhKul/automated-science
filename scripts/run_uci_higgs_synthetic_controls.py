#!/usr/bin/env python3
"""Run output-isolated artificial controls for Phase 62 UCI HIGGS."""

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

from core.real_data.uci_higgs import FEATURES, HiggsRecord, predict_independent, record_from_payload


OUT = Path("artifacts/evaluations/phase62_uci_higgs_synthetic_controls_20260801")
SEED = 2062001
LOW = slice(0, 21)
HIGH = slice(21, 28)


def rejected(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def artificial_data(rows: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    values = rng.normal(size=(rows, FEATURES))
    score = 2.5 * values[:, 0] - 2.0 * values[:, 4] + 2.8 * values[:, 21] - 2.3 * values[:, 25]
    return values, (score >= 0.0).astype(int)


def macro_f1(model: object, values: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    probabilities = model.predict_proba(values)
    if not np.all(np.isfinite(probabilities)) or not np.all((0.0 <= probabilities) & (probabilities <= 1.0)):
        raise ValueError("synthetic HIGGS model emitted invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("synthetic HIGGS probabilities are not normalized")
    prediction = model.classes_[np.argmax(probabilities, axis=1)]
    return float(f1_score(labels, prediction, average="macro")), probabilities


def run(output_dir: Path = OUT) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    train_values, train_labels = artificial_data(1_200, rng)
    selection_values, selection_labels = artificial_data(800, rng)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=SEED))
    model.fit(train_values, train_labels)
    planted_f1, probabilities = macro_f1(model, selection_values, selection_labels)

    label_permutation = rng.permutation(selection_labels)
    label_paired_f1, _ = macro_f1(model, selection_values, label_permutation)
    high_paired = selection_values.copy()
    high_paired[:, HIGH] = high_paired[rng.permutation(len(high_paired)), HIGH]
    high_paired_f1, _ = macro_f1(model, high_paired, selection_labels)
    low_ablated = selection_values.copy()
    low_ablated[:, LOW] = 0.0
    low_ablated_f1, _ = macro_f1(model, low_ablated, selection_labels)
    high_ablated = selection_values.copy()
    high_ablated[:, HIGH] = 0.0
    high_ablated_f1, _ = macro_f1(model, high_ablated, selection_labels)

    artificial_record = HiggsRecord(np.linspace(-1.0, 1.0, FEATURES))
    reset_states: list[float] = []
    reset_probabilities = predict_independent([artificial_record] * 3, lambda state, _values: reset_states.append(state) or 0.5)
    reordered = predict_independent([artificial_record] * 3, lambda _state, _values: 0.5)
    ablation_degradation = max(planted_f1 - low_ablated_f1, planted_f1 - high_ablated_f1)
    result = {
        "phase": 62,
        "status": "passed_output_isolated_synthetic_controls",
        "uses_observed_values": False,
        "uses_observed_targets": False,
        "metrics": {
            "planted_macro_f1": planted_f1,
            "label_record_pairing_degradation": planted_f1 - label_paired_f1,
            "low_high_block_pairing_degradation": planted_f1 - high_paired_f1,
            "low_block_ablation_degradation": planted_f1 - low_ablated_f1,
            "high_block_ablation_degradation": planted_f1 - high_ablated_f1,
            "maximum_block_ablation_degradation": ablation_degradation,
        },
        "thresholds": {
            "planted_macro_f1_minimum": 0.95,
            "label_record_pairing_degradation_minimum": 0.4,
            "low_high_block_pairing_degradation_minimum": 0.2,
            "block_ablation_degradation_minimum": 0.05,
        },
        "controls": {
            "planted_recovery": planted_f1 >= 0.95,
            "label_record_pairing": planted_f1 - label_paired_f1 >= 0.4,
            "low_high_block_pairing": planted_f1 - high_paired_f1 >= 0.2,
            "block_ablation": ablation_degradation >= 0.05,
            "zero_reset": reset_states == [0.0, 0.0, 0.0],
            "reorder_invariant": bool(np.array_equal(reset_probabilities, reordered[::-1])),
            "target_isolation": rejected({"values": artificial_record.values, "target": 1}),
            "file_isolation": rejected({"values": artificial_record.values, "file": "source_tail"}),
            "split_isolation": rejected({"values": artificial_record.values, "split": "external"}),
            "row_isolation": rejected({"values": artificial_record.values, "row": 1}),
            "source_order_isolation": rejected({"values": artificial_record.values, "source_order": 1}),
            "metadata_isolation": rejected({"values": artificial_record.values, "metadata": {}}),
            "duplicate_group_isolation": rejected({"values": artificial_record.values, "duplicate_group": "artificial"}),
            "finite_normalized_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all((0.0 <= probabilities) & (probabilities <= 1.0))),
        },
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not all(result["controls"].values()):
        result["status"] = "failed_output_isolated_synthetic_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

#!/usr/bin/env python3
"""Output-isolated artificial raw-token controls for Phase 69 UCI Adult."""

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

from core.real_data.uci_adult import AdultRow, predict_independent, row_from_payload

SEED = 2069001
OUT = Path("artifacts/evaluations/phase69_uci_adult_synthetic_controls_20260803")
NUMERIC_FIELDS = (0, 2, 4, 10, 11, 12)
CATEGORICAL_FIELDS = tuple(index for index in range(14) if index not in NUMERIC_FIELDS)


def artificial_records(count_per_class: int, seed: int) -> tuple[list[AdultRow], np.ndarray]:
    """Make balanced artificial rows where both token blocks are necessary."""
    rng = np.random.default_rng(seed)
    rows: list[AdultRow] = []
    labels: list[int] = []
    for label in (0, 1):
        for _ in range(count_per_class):
            if label == 1:
                category_bit, numeric_bit = 1, 1
            else:
                category_bit, numeric_bit = ((0, 0), (0, 1), (1, 0))[rng.integers(3)]
            fields = [f"cat_{index}_fixed" for index in range(14)]
            for index in NUMERIC_FIELDS:
                fields[index] = "0"
            fields[1] = f"workclass_bit_{category_bit}"
            fields[0] = str(numeric_bit)
            rows.append(AdultRow(tuple(fields)))
            labels.append(label)
    order = rng.permutation(len(rows))
    return [rows[index] for index in order], np.asarray(labels, dtype=int)[order]


class TokenModel:
    def fit(self, rows: list[AdultRow], labels: np.ndarray) -> "TokenModel":
        self.categories = {
            index: {value: offset for offset, value in enumerate(sorted({row.fields[index] for row in rows}))}
            for index in CATEGORICAL_FIELDS
        }
        self.offsets: dict[int, int] = {}
        offset = len(NUMERIC_FIELDS)
        for index in CATEGORICAL_FIELDS:
            self.offsets[index] = offset
            offset += len(self.categories[index])
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        self.model.fit(self.scaler.fit_transform(self.encode(rows)), labels)
        return self

    def encode(self, rows: list[AdultRow]) -> np.ndarray:
        width = len(NUMERIC_FIELDS) + sum(len(self.categories[index]) for index in CATEGORICAL_FIELDS)
        matrix = np.zeros((len(rows), width), dtype=float)
        for row_index, row in enumerate(rows):
            for numeric_index, field_index in enumerate(NUMERIC_FIELDS):
                matrix[row_index, numeric_index] = float(row.fields[field_index])
            for field_index in CATEGORICAL_FIELDS:
                category_index = self.categories[field_index].get(row.fields[field_index])
                if category_index is not None:
                    matrix[row_index, self.offsets[field_index] + category_index] = 1.0
        return matrix

    def score(self, rows: list[AdultRow], labels: np.ndarray) -> float:
        return float(f1_score(labels, self.model.predict(self.scaler.transform(self.encode(rows))), average="macro", zero_division=0))


def permute_fields(rows: list[AdultRow], fields: tuple[int, ...], seed: int) -> list[AdultRow]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    result = []
    for index, row in enumerate(rows):
        values = list(row.fields)
        for field in fields:
            values[field] = rows[order[index]].fields[field]
        result.append(AdultRow(tuple(values)))
    return result


def ablate_fields(rows: list[AdultRow], fields: tuple[int, ...]) -> list[AdultRow]:
    result = []
    for row in rows:
        values = list(row.fields)
        for field in fields:
            values[field] = "0" if field in NUMERIC_FIELDS else "missing_artificial_token"
        result.append(AdultRow(tuple(values)))
    return result


def rejected(payload: dict[str, object]) -> bool:
    try:
        row_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    train_rows, train_labels = artificial_records(120, SEED)
    selection_rows, selection_labels = artificial_records(60, SEED + 1)
    model = TokenModel().fit(train_rows, train_labels)
    planted = model.score(selection_rows, selection_labels)
    paired = TokenModel().fit(train_rows, np.random.default_rng(SEED + 2).permutation(train_labels)).score(selection_rows, selection_labels)
    categorical_paired = model.score(permute_fields(selection_rows, CATEGORICAL_FIELDS, SEED + 3), selection_labels)
    numeric_paired = model.score(permute_fields(selection_rows, NUMERIC_FIELDS, SEED + 4), selection_labels)
    categorical_ablation = planted - model.score(ablate_fields(selection_rows, CATEGORICAL_FIELDS), selection_labels)
    numeric_ablation = planted - model.score(ablate_fields(selection_rows, NUMERIC_FIELDS), selection_labels)

    rows = selection_rows[:8]
    states: list[float] = []
    probabilities = predict_independent(rows, lambda state, fields: states.append(state) or np.array([0.5, 0.5]))
    reordered = predict_independent(rows[::-1], lambda state, fields: np.array([0.5, 0.5]))
    controls = {
        "label_row_pairing_drop": planted - paired,
        "categorical_block_pairing_drop": planted - categorical_paired,
        "numeric_block_pairing_drop": planted - numeric_paired,
        "categorical_block_ablation_drop": categorical_ablation,
        "numeric_block_ablation_drop": numeric_ablation,
        "zero_reset": states == [0.0] * len(rows),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejected({"fields": rows[0].fields, "label": ">50K"}),
        "file_rejected": rejected({"fields": rows[0].fields, "file": "adult.test"}),
        "split_rejected": rejected({"fields": rows[0].fields, "split": "external"}),
        "row_rejected": rejected({"fields": rows[0].fields, "row": 0}),
        "demographic_rejected": rejected({"fields": rows[0].fields, "demographic": "sex"}),
        "duplicate_group_rejected": rejected({"fields": rows[0].fields, "duplicate_group": 0}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    invariants = ("zero_reset", "reorder_invariant", "target_rejected", "file_rejected", "split_rejected", "row_rejected", "demographic_rejected", "duplicate_group_rejected", "finite_probabilities")
    passed = (
        planted >= 0.95
        and controls["label_row_pairing_drop"] >= 0.40
        and controls["categorical_block_pairing_drop"] >= 0.20
        and controls["numeric_block_pairing_drop"] >= 0.20
        and max(categorical_ablation, numeric_ablation) >= 0.05
        and all(controls[name] for name in invariants)
    )
    result = {
        "phase": 69,
        "status": "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "artificial_contract": {"field_count": 14, "categorical_fields": list(CATEGORICAL_FIELDS), "numeric_fields": list(NUMERIC_FIELDS), "labels": [0, 1]},
        "planted": {"selection_macro_f1": planted},
        "controls": controls,
        "thresholds": {"planted_macro_f1": 0.95, "label_pairing_drop": 0.40, "categorical_pairing_drop": 0.20, "numeric_pairing_drop": 0.20, "max_ablation_drop": 0.05},
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

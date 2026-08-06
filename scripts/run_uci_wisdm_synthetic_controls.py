#!/usr/bin/env python3
"""Run Phase 50 output-isolated artificial WISDM recovery and controls."""
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

from core.real_data.uci_wisdm import (
    ACTIVITY_LABELS,
    FULL_SEGMENT_SAMPLES,
    PREFIX_SAMPLES,
    SOURCE_AXES,
    WISDMInjectedRecord,
    predict_independent_prefixes,
    prefix_from_payload,
)


SEED = 2050001
SPLIT = Path("data/real/uci_wisdm/source_subject_split.json")
OUT = Path("artifacts/evaluations/phase50_uci_wisdm_synthetic_controls_20260728")


class LinearPrefixModel:
    """Fixed linear artificial recovery model over flattened raw prefixes."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=SEED)

    def fit(self, prefixes: np.ndarray, labels: np.ndarray) -> "LinearPrefixModel":
        self.model.fit(self.scaler.fit_transform(prefixes.reshape(len(prefixes), -1)), labels)
        if tuple(self.model.classes_) != ACTIVITY_LABELS:
            raise ValueError("artificial training must retain all 18 fixed states")
        return self

    def predict_proba(self, prefixes: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(prefixes.reshape(len(prefixes), -1)))


def _artificial_records(participants: list[int], partition: str, repeats: int, offset: int) -> list[WISDMInjectedRecord]:
    records: list[WISDMInjectedRecord] = []
    for participant in participants:
        for state, label in enumerate(ACTIVITY_LABELS):
            for repeat in range(repeats):
                rng = np.random.default_rng(SEED + 100000 * offset + 1000 * participant + 31 * state + repeat)
                prefix = rng.normal(0.0, 0.02, size=(PREFIX_SAMPLES, SOURCE_AXES))
                start, axis = 3 * state, state % SOURCE_AXES
                prefix[start : start + 3, axis] += 8.0 + 0.01 * participant
                prefix[start + 3 : start + 5, axis] -= 4.0
                suffix = rng.normal(0.0, 1.0, size=(FULL_SEGMENT_SAMPLES - PREFIX_SAMPLES, SOURCE_AXES))
                records.append(
                    WISDMInjectedRecord(
                        subject_id=int(participant),
                        partition=partition,
                        member_path=f"artificial/raw/phone/accel/data_{participant}_accel_phone.txt",
                        artificial_prefix=prefix,
                        artificial_suffix=suffix,
                        artificial_label=label,
                        artificial_timestamps=np.arange(FULL_SEGMENT_SAMPLES, dtype=np.int64) + (state + repeat * 100) * FULL_SEGMENT_SAMPLES,
                    )
                )
    return records


def _surface(records: list[WISDMInjectedRecord]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([record.candidate_input().prefix for record in records]), np.asarray([record.artificial_label for record in records])


def _metrics(model: LinearPrefixModel, prefixes: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(prefixes)
    if probabilities.shape != (len(labels), len(ACTIVITY_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("nonfinite or malformed synthetic probabilities")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("out-of-bounds synthetic probabilities")
    predictions = np.asarray(ACTIVITY_LABELS)[np.argmax(probabilities, axis=1)]
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=ACTIVITY_LABELS, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(labels, probabilities, labels=ACTIVITY_LABELS)),
    }


def _rejected(payload: dict[str, object]) -> bool:
    try:
        prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT, output_dir: Path = OUT) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    train_records = _artificial_records(split["train_subjects"], "train_subjects", repeats=3, offset=0)
    selection_records = _artificial_records(split["selection_subjects"], "selection_subjects", repeats=3, offset=1)
    train_x, train_y = _surface(train_records)
    selection_x, selection_y = _surface(selection_records)
    model = LinearPrefixModel().fit(train_x, train_y)
    planted = _metrics(model, selection_x, selection_y)
    paired = _metrics(LinearPrefixModel().fit(train_x, np.random.default_rng(SEED).permutation(train_y)), selection_x, selection_y)
    time_shifted = _metrics(model, np.roll(selection_x, 16, axis=1), selection_y)
    axis_permuted = _metrics(model, selection_x[:, :, (1, 2, 0)], selection_y)
    ablations = []
    for axis in range(SOURCE_AXES):
        ablated = selection_x.copy()
        ablated[:, :, axis] = 0.0
        ablations.append(planted["macro_f1"] - _metrics(model, ablated, selection_y)["macro_f1"])

    prefixes = [record.candidate_input() for record in selection_records[:18]]
    states: list[float] = []
    uniform = np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS))
    probabilities = predict_independent_prefixes(prefixes, lambda state, prefix: states.append(state) or uniform)
    reordered_probabilities = predict_independent_prefixes(list(reversed(prefixes)), lambda state, prefix: uniform)
    candidate, record = prefixes[0], selection_records[0]
    controls = {
        "label_segment_pairing_drop": planted["macro_f1"] - paired["macro_f1"],
        "time_order_16_sample_shift_drop": planted["macro_f1"] - time_shifted["macro_f1"],
        "axis_pairing_drop": planted["macro_f1"] - axis_permuted["macro_f1"],
        "axis_ablation_drops": ablations,
        "per_segment_zero_reset": states == [0.0] * len(prefixes),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered_probabilities[::-1])),
        "target_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "activity_label": record.artificial_label}),
        "participant_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "subject_id": record.subject_id}),
        "path_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "member_path": record.member_path}),
        "timestamp_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "timestamps": record.artificial_timestamps}),
        "suffix_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "suffix": record.artificial_suffix}),
        "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }
    thresholds = {
        "planted_macro_f1": 0.95,
        "label_pairing_drop": 0.4,
        "time_order_drop": 0.1,
        "axis_pairing_drop": 0.1,
        "axis_ablation_drop": 0.05,
    }
    passed = all([
        planted["macro_f1"] >= thresholds["planted_macro_f1"],
        controls["label_segment_pairing_drop"] >= thresholds["label_pairing_drop"],
        controls["time_order_16_sample_shift_drop"] >= thresholds["time_order_drop"],
        controls["axis_pairing_drop"] >= thresholds["axis_pairing_drop"],
        max(controls["axis_ablation_drops"]) >= thresholds["axis_ablation_drop"],
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key in {"per_segment_zero_reset", "reorder_invariant", "probabilities_finite_and_bounded"}],
    ])
    result = {
        "phase": 50,
        "status": "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure",
        "uses_observed_source_acceleration_values": False,
        "uses_observed_source_activity_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "planted": planted,
        "controls": controls,
        "thresholds": thresholds,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

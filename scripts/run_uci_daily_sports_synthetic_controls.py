#!/usr/bin/env python3
"""Run Phase 45 output-isolated artificial prefix recovery and controls."""
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

from core.real_data.uci_daily_sports import ACTIVITY_LABELS, PREFIX_SAMPLES, SOURCE_CHANNELS, DailySportsPrefix, prefix_from_payload


SEED = 2045001
SPLIT = Path("data/real/uci_daily_sports/source_subject_split.json")
OUT = Path("artifacts/evaluations/phase45_uci_daily_sports_synthetic_controls_20260727")
UNIT_BLOCKS = tuple(np.arange(SOURCE_CHANNELS).reshape(5, 9)[unit].ravel() for unit in range(5))


class LinearPrefixModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)

    def fit(self, prefixes: np.ndarray, labels: np.ndarray) -> "LinearPrefixModel":
        self.model.fit(self.scaler.fit_transform(prefixes.reshape(len(prefixes), -1)), labels)
        return self

    def predict_proba(self, prefixes: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(prefixes.reshape(len(prefixes), -1)))


def _artificial_records(participants: list[int], repeats: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    prefixes, labels = [], []
    for participant in participants:
        for label in ACTIVITY_LABELS:
            for repeat in range(repeats):
                rng = np.random.default_rng(SEED + 10000 * offset + 100 * participant + 3 * label + repeat)
                values = rng.normal(0.0, 0.03, size=(PREFIX_SAMPLES, SOURCE_CHANNELS))
                unit = (label - 1) % len(UNIT_BLOCKS)
                start = 3 * (label - 1)
                values[start : start + 4, UNIT_BLOCKS[unit]] += 4.0 + 0.05 * participant
                values[start + 4 : start + 6, UNIT_BLOCKS[unit][:3]] -= 2.0
                prefixes.append(values)
                labels.append(label)
    return np.asarray(prefixes), np.asarray(labels)


def _metrics(model: LinearPrefixModel, prefixes: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(prefixes)
    if probabilities.shape != (len(labels), len(ACTIVITY_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("nonfinite or malformed synthetic probabilities")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9):
        raise ValueError("out-of-bounds synthetic probabilities")
    prediction = np.asarray(ACTIVITY_LABELS)[np.argmax(probabilities, axis=1)]
    return {
        "macro_f1": float(f1_score(labels, prediction, labels=ACTIVITY_LABELS, average="macro", zero_division=0)),
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
    train_x, train_y = _artificial_records(split["train"], repeats=5, offset=0)
    selection_x, selection_y = _artificial_records(split["selection"], repeats=5, offset=1)
    model = LinearPrefixModel().fit(train_x, train_y)
    planted = _metrics(model, selection_x, selection_y)
    paired = _metrics(LinearPrefixModel().fit(train_x, np.random.default_rng(SEED).permutation(train_y)), selection_x, selection_y)
    time_shifted = _metrics(model, np.roll(selection_x, 16, axis=1), selection_y)
    permutation = np.concatenate((UNIT_BLOCKS[-1], *UNIT_BLOCKS[:-1]))
    unit_permuted = _metrics(model, selection_x[:, :, permutation], selection_y)
    ablations = []
    for block in UNIT_BLOCKS:
        ablated = selection_x.copy()
        ablated[:, :, block] = 0
        ablations.append(planted["macro_f1"] - _metrics(model, ablated, selection_y)["macro_f1"])
    prefixes = [DailySportsPrefix(values) for values in selection_x[:8]]
    states: list[float] = []
    uniform = np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS))
    from core.real_data.uci_daily_sports import predict_independent_prefixes

    probabilities = predict_independent_prefixes(prefixes, lambda state, prefix: states.append(state) or uniform)
    reverse_probabilities = predict_independent_prefixes(list(reversed(prefixes)), lambda state, prefix: uniform)
    candidate = prefixes[0]
    result = {
        "phase": 45,
        "status": "passed_output_isolated_synthetic_controls",
        "uses_observed_source_signals": False,
        "uses_observed_source_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "planted": planted,
        "controls": {
            "label_segment_pairing_drop": planted["macro_f1"] - paired["macro_f1"],
            "time_order_16_sample_shift_drop": planted["macro_f1"] - time_shifted["macro_f1"],
            "unit_block_pairing_drop": planted["macro_f1"] - unit_permuted["macro_f1"],
            "unit_ablation_drops": ablations,
            "per_segment_zero_reset": states == [0.0] * len(prefixes),
            "reorder_invariant": bool(np.array_equal(probabilities, reverse_probabilities[::-1])),
            "target_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "label": 1}),
            "participant_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "participant_id": 1}),
            "file_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "member_path": "data/a01/p1/s01.txt"}),
            "suffix_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "suffix": np.zeros((61, 45))}),
            "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1)),
        },
        "thresholds": {"planted_macro_f1": 0.95, "label_pairing_drop": 0.4, "time_order_drop": 0.1, "unit_block_drop": 0.1, "unit_ablation_drop": 0.05},
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    controls = result["controls"]
    passed = all([
        planted["macro_f1"] >= 0.95,
        controls["label_segment_pairing_drop"] >= 0.4,
        controls["time_order_16_sample_shift_drop"] >= 0.1,
        controls["unit_block_pairing_drop"] >= 0.1,
        max(controls["unit_ablation_drops"]) >= 0.05,
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key in {"per_segment_zero_reset", "reorder_invariant", "probabilities_finite_and_bounded"}],
    ])
    result["status"] = "passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

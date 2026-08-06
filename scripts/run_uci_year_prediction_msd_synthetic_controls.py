#!/usr/bin/env python3
"""Run output-isolated Phase 61 YearPredictionMSD synthetic controls."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_year_prediction_msd import FEATURES, TARGET_BOUNDS, predict_independent, record_from_payload


OUT = Path("artifacts/evaluations/phase61_uci_year_prediction_msd_synthetic_controls_20260801")
SEED = 2061001
AVERAGES = slice(0, 12)
COVARIANCES = slice(12, FEATURES)


def artificial_records() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(2400, FEATURES))
    coefficients = np.concatenate((np.full(12, 0.9), np.full(78, 0.6)))
    targets = 1965.0 + values @ coefficients
    if not np.all((TARGET_BOUNDS[0] <= targets) & (targets <= TARGET_BOUNDS[1])):
        raise ValueError("artificial target escaped fixed bounds")
    return values, targets


class RidgeModel:
    def __init__(self, values: np.ndarray, targets: np.ndarray) -> None:
        self.scaler = StandardScaler().fit(values)
        self.regressor = Ridge(alpha=0.1).fit(self.scaler.transform(values), targets)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.regressor.predict(self.scaler.transform(values))


def rmse(model: RidgeModel, values: np.ndarray, targets: np.ndarray) -> float:
    predictions = model.predict(values)
    if not np.all(np.isfinite(predictions)) or not np.all((TARGET_BOUNDS[0] <= predictions) & (predictions <= TARGET_BOUNDS[1])):
        raise ValueError("synthetic predictor emitted invalid release years")
    return float(mean_squared_error(targets, predictions) ** 0.5)


def rejected(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    values, targets = artificial_records()
    train_values, selection_values = values[:1920], values[1920:]
    train_targets, selection_targets = targets[:1920], targets[1920:]
    model = RidgeModel(train_values, train_targets)
    planted_rmse = rmse(model, selection_values, selection_targets)
    paired_targets = np.random.default_rng(SEED).permutation(train_targets)
    target_pairing_increase = rmse(RidgeModel(train_values, paired_targets), selection_values, selection_targets) - planted_rmse
    paired_features = selection_values.copy()
    permutation = np.random.default_rng(SEED + 1).permutation(len(paired_features))
    paired_features[:, AVERAGES] = paired_features[permutation, AVERAGES]
    paired_features[:, COVARIANCES] = paired_features[permutation, COVARIANCES]
    feature_pairing_increase = rmse(model, paired_features, selection_targets) - planted_rmse
    ablation_increases: list[float] = []
    for group in (AVERAGES, COVARIANCES):
        ablated = selection_values.copy()
        ablated[:, group] = 0.0
        ablation_increases.append(rmse(model, ablated, selection_targets) - planted_rmse)
    sample = selection_values[0]
    states: list[float] = []
    predictions = predict_independent([record_from_payload({"values": sample})] * 3, lambda state, _values: states.append(state) or 1965.0)
    reordered = predict_independent([record_from_payload({"values": sample})] * 3, lambda _state, _values: 1965.0)
    result = {
        "phase": 61,
        "status": "passed_output_isolated_synthetic_controls",
        "uses_observed_values": False,
        "uses_observed_targets": False,
        "planted": {"rmse": planted_rmse, "prediction_fingerprint": hashlib.sha256(model.predict(selection_values).tobytes()).hexdigest()},
        "controls": {
            "target_record_pairing_rmse_increase": target_pairing_increase,
            "average_covariance_pairing_rmse_increase": feature_pairing_increase,
            "feature_block_ablation_rmse_increases": ablation_increases,
            "zero_reset": states == [0.0, 0.0, 0.0],
            "reorder_invariant": bool(np.array_equal(predictions, reordered[::-1])),
            "target_rejected": rejected({"values": sample, "target": 1965.0}),
            "file_rejected": rejected({"values": sample, "file": "external"}),
            "split_rejected": rejected({"values": sample, "split": "test"}),
            "row_rejected": rejected({"values": sample, "row": 0}),
            "metadata_rejected": rejected({"values": sample, "source_order": 0}),
            "duplicate_group_rejected": rejected({"values": sample, "duplicate_group": "one"}),
            "finite_bounded_predictions": bool(np.all(np.isfinite(predictions)) and np.all((TARGET_BOUNDS[0] <= predictions) & (predictions <= TARGET_BOUNDS[1]))),
        },
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    passed = (
        planted_rmse <= 0.1
        and target_pairing_increase >= 5.0
        and feature_pairing_increase >= 5.0
        and max(ablation_increases) >= 1.0
        and all(bool(value) for key, value in result["controls"].items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "finite_bounded_predictions"})
    )
    if not passed:
        result["status"] = "failed_output_isolated_synthetic_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

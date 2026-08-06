#!/usr/bin/env python3
"""Execute the fixed Phase 61 YearPredictionMSD observed baseline gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path
import zipfile

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_year_prediction_msd import FEATURES, TARGET_BOUNDS, predict_independent, record_from_payload
from scripts.run_uci_year_prediction_msd_source_gate import EXTERNAL_ROWS, MEMBER, SOURCE_ROWS, TOTAL_ROWS, parse_line


ARCHIVE = Path("data/real/uci_year_prediction_msd/raw/yearpredictionmsd.zip")
OUT = Path("artifacts/evaluations/phase61_uci_year_prediction_msd_real_baselines_20260801")
SEED = 2061001
MODELS = ("train_median", "ridge_0.1", "ridge_1", "ridge_10", "hist_gradient_boosting")
EXPECTED_LEDGER = {"source_complete_duplicates": 193, "external_complete_duplicates": 21, "cross_file_complete_duplicates": 0, "source_candidate_duplicates": 195, "external_candidate_duplicates": 21, "cross_file_candidate_duplicates": 0}


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    targets: np.ndarray
    fingerprints: tuple[str, ...]


def load_records() -> tuple[Records, Records, dict[str, int]]:
    values = np.empty((TOTAL_ROWS, FEATURES), dtype=np.float32)
    targets = np.empty(TOTAL_ROWS, dtype=np.int16)
    complete: list[str] = []
    candidates: list[str] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        if tuple(archive.namelist()) != (MEMBER,):
            raise ValueError("YearPredictionMSD ZIP inventory changed")
        with archive.open(MEMBER) as handle:
            for index, raw in enumerate(handle):
                parsed = parse_line(raw, index + 1)
                values[index] = parsed[1:]
                targets[index] = int(parsed[0])
                complete.append(hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest())
                candidates.append(hashlib.sha256(parsed[1:].astype("<f8", copy=False).tobytes()).hexdigest())
    if len(targets) != TOTAL_ROWS or not np.all(np.isfinite(values)) or not np.all((TARGET_BOUNDS[0] <= targets) & (targets <= TARGET_BOUNDS[1])):
        raise ValueError("YearPredictionMSD source contract changed")
    source = Records(values[:SOURCE_ROWS], targets[:SOURCE_ROWS], tuple(candidates[:SOURCE_ROWS]))
    external = Records(values[SOURCE_ROWS:], targets[SOURCE_ROWS:], tuple(candidates[SOURCE_ROWS:]))
    source_complete, external_complete = set(complete[:SOURCE_ROWS]), set(complete[SOURCE_ROWS:])
    source_candidates, external_candidates = set(source.fingerprints), set(external.fingerprints)
    ledger = {
        "source_complete_duplicates": SOURCE_ROWS - len(source_complete),
        "external_complete_duplicates": EXTERNAL_ROWS - len(external_complete),
        "cross_file_complete_duplicates": len(source_complete & external_complete),
        "source_candidate_duplicates": SOURCE_ROWS - len(source_candidates),
        "external_candidate_duplicates": EXTERNAL_ROWS - len(external_candidates),
        "cross_file_candidate_duplicates": len(source_candidates & external_candidates),
    }
    if ledger != EXPECTED_LEDGER:
        raise ValueError("YearPredictionMSD frozen duplicate ledger changed")
    return source, external, ledger


def grouped_split(records: Records) -> tuple[Records, Records, dict[str, int]]:
    groups: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(records.fingerprints):
        groups.setdefault(fingerprint, []).append(index)
    group_indices = list(groups.values())
    labels = np.asarray([records.targets[indices[0]] for indices in group_indices], dtype=int)
    if any(not np.all(records.targets[indices] == labels[group_index]) for group_index, indices in enumerate(group_indices)):
        raise ValueError("inconsistent release years in candidate-only duplicate group")
    train_groups, selection_groups = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(labels)), labels))
    train_indices = np.asarray([index for group in train_groups for index in group_indices[group]], dtype=int)
    selection_indices = np.asarray([index for group in selection_groups for index in group_indices[group]], dtype=int)
    if set(records.fingerprints[index] for index in train_indices) & set(records.fingerprints[index] for index in selection_indices):
        raise ValueError("candidate duplicate group crossed the frozen source-train split")

    def subset(indices: np.ndarray) -> Records:
        return Records(records.values[indices], records.targets[indices], tuple(records.fingerprints[index] for index in indices))

    return subset(train_indices), subset(selection_indices), {"source_unique_candidates": len(group_indices), "source_candidate_duplicate_groups": sum(len(group) > 1 for group in group_indices)}


class Model:
    def __init__(self, name: str, baseline: float | None = None, scaler: StandardScaler | None = None, estimator: object | None = None) -> None:
        self.name = name
        self.baseline = baseline
        self.scaler = scaler
        self.estimator = estimator

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.baseline is not None:
            return np.full(len(values), self.baseline)
        assert self.estimator is not None
        transformed = self.scaler.transform(values) if self.scaler is not None else values
        return np.asarray(self.estimator.predict(transformed), dtype=float)


def fit(name: str, values: np.ndarray, targets: np.ndarray, seed: int = SEED) -> Model:
    if name == "train_median":
        return Model(name, baseline=float(np.median(targets)))
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    if name.startswith("ridge_"):
        return Model(name, scaler=scaler, estimator=Ridge(alpha=float(name.removeprefix("ridge_"))).fit(transformed, targets))
    return Model(name, estimator=HistGradientBoostingRegressor(max_depth=3, max_iter=100, l2_regularization=1.0, random_state=seed).fit(values, targets))


def metric(model: Model, records: Records) -> dict[str, object]:
    predictions = model.predict(records.values)
    if predictions.shape != records.targets.shape or not np.all(np.isfinite(predictions)) or not np.all((TARGET_BOUNDS[0] <= predictions) & (predictions <= TARGET_BOUNDS[1])):
        raise ValueError("YearPredictionMSD model emitted invalid bounded predictions")
    return {"rmse": float(mean_squared_error(records.targets, predictions) ** 0.5), "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest()}


def rejects(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def controls(name: str, train: Records, selection: Records, model: Model, score: float, ledger: dict[str, int]) -> dict[str, object]:
    paired_targets = np.random.default_rng(SEED).permutation(train.targets)
    target_pairing_increase = metric(fit(name, train.values, paired_targets), selection)["rmse"] - score
    paired = selection.values.copy()
    permutation = np.random.default_rng(SEED + 1).permutation(len(paired))
    paired[:, :12] = paired[permutation, :12]
    paired[:, 12:] = paired[permutation, 12:]
    block_pairing_increase = metric(model, Records(paired, selection.targets, selection.fingerprints))["rmse"] - score
    ablation_increases: list[float] = []
    for group in (slice(0, 12), slice(12, FEATURES)):
        ablated = selection.values.copy()
        ablated[:, group] = 0.0
        ablation_increases.append(metric(model, Records(ablated, selection.targets, selection.fingerprints))["rmse"] - score)
    sample = selection.values[0]
    states: list[float] = []
    predictions = predict_independent([record_from_payload({"values": sample})] * 3, lambda state, _values: states.append(state) or 1965.0)
    reordered = predict_independent([record_from_payload({"values": sample})] * 3, lambda _state, _values: 1965.0)
    return {
        "target_record_pairing_rmse_increase": target_pairing_increase,
        "average_covariance_pairing_rmse_increase": block_pairing_increase,
        "feature_block_ablation_rmse_increases": ablation_increases,
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(predictions, reordered[::-1])),
        "target_rejected": rejects({"values": sample, "target": 1965}),
        "file_rejected": rejects({"values": sample, "file": "external"}),
        "split_rejected": rejects({"values": sample, "split": "test"}),
        "row_rejected": rejects({"values": sample, "row": 0}),
        "metadata_rejected": rejects({"values": sample, "source_order": 0}),
        "duplicate_group_rejected": rejects({"values": sample, "duplicate_group": "one"}),
        "duplicate_ledger_integrity": ledger == EXPECTED_LEDGER,
        "finite_bounded_predictions": True,
    }


def run(output_dir: Path = OUT) -> dict[str, object]:
    source, external, ledger = load_records()
    try:
        train, selection, group_ledger = grouped_split(source)
    except ValueError as error:
        result: dict[str, object] = {"phase": 61, "status": "closed_negative_duplicate_group_gate", "reason": str(error), "source_revalidation": {"source_rows": len(source.targets), "external_rows": len(external.targets), **ledger}, "models_fitted": [], "external_outcome_score_count": 0, "abc_smc_calls": 0, "llm_calls": 0}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    candidates = {name: metric(fit(name, train.values, train.targets), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (float(candidates[name]["rmse"]), MODELS.index(name)))
    model = fit(selected, train.values, train.targets)
    score = float(candidates[selected]["rmse"])
    selection_controls = controls(selected, train, selection, model, score, ledger)
    passed = selection_controls["target_record_pairing_rmse_increase"] >= 5.0 and selection_controls["average_covariance_pairing_rmse_increase"] >= 0.5 and max(selection_controls["feature_block_ablation_rmse_increases"]) >= 0.25 and all(bool(value) for key, value in selection_controls.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "duplicate_ledger_integrity", "finite_bounded_predictions"})
    result = {"phase": 61, "source_revalidation": {"source_rows": len(source.targets), "external_rows": len(external.targets), "train_rows": len(train.targets), "selection_rows": len(selection.targets), **ledger, **group_ledger}, "models": candidates, "selected_model": selected, "selection_controls": selection_controls, "selection_passed": passed, "external_outcome_score_count": 0, "abc_smc_calls": 0, "llm_calls": 0}
    if not passed:
        result.update(status="closed_negative_selection_gate", reason="fixed selection control failed; artist-heldout source tail was not scored")
    else:
        seed_metrics: list[dict[str, object]] = []
        fingerprints: list[str] = []
        for seed in range(SEED, SEED + 8):
            item = metric(fit(selected, source.values, source.targets, seed), external)
            item["seed"] = seed
            seed_metrics.append(item)
            fingerprints.append(str(item["prediction_fingerprint"]))
            if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
                break
        result.update(status="passed_observed_baseline_and_external_stability", external_outcome_score_count=1, external_seed_metrics=seed_metrics, executed_seed_count=len(seed_metrics), adaptive_distinct_yield_stopped=len(seed_metrics) < 8)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

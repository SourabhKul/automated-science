#!/usr/bin/env python3
"""Execute the fixed Phase 49 UJIIndoorLoc observed baseline."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import warnings
from zipfile import ZipFile

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_ujiindoorloc import CLASS_LABELS, FEATURE_COUNT, UJIIndoorLocWLAN, predict_independent_wlans, wlan_from_payload
from scripts.run_uci_ujiindoorloc_source_gate import ARCHIVE, HEADER, MEMBERS, ROWS, TRAIN, VALIDATION

OUT = Path("artifacts/evaluations/phase49_uci_ujiindoorloc_real_baselines_20260728")
SEED = 2049001
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
BLOCK_COUNT = 8
BLOCK_SIZE = FEATURE_COUNT // BLOCK_COUNT


@dataclass(frozen=True)
class Rows:
    x: np.ndarray
    y: np.ndarray
    complete_keys: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    candidate_label_keys: tuple[str, ...]

    def subset(self, index: np.ndarray) -> "Rows":
        return Rows(
            self.x[index],
            self.y[index],
            tuple(self.complete_keys[i] for i in index),
            tuple(self.candidate_keys[i] for i in index),
            tuple(self.candidate_label_keys[i] for i in index),
        )


class Model:
    def __init__(self, name: str, scaler=None, estimator=None, prior=None):
        self.name, self.scaler, self.estimator, self.prior = name, scaler, estimator, prior

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.prior is not None:
            return np.broadcast_to(self.prior, (len(values), len(CLASS_LABELS))).copy()
        transformed = self.scaler.transform(values)
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(transformed)
        distances = -np.sum((transformed[:, None, :] - self.estimator.centroids_[None, :, :]) ** 2, axis=2)
        distances -= distances.max(axis=1, keepdims=True)
        weights = np.exp(distances)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(CLASS_LABELS)[np.argmax(self.predict_proba(values), axis=1)]


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _read(archive: ZipFile, member: str) -> Rows:
    reader = csv.reader(io.TextIOWrapper(io.BytesIO(archive.read(member)), encoding="utf-8", newline=""))
    if tuple(next(reader)) != HEADER:
        raise ValueError(f"{member} header differs from locked contract")
    values, labels, complete_keys, candidate_keys, candidate_label_keys = [], [], [], [], []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(HEADER):
            raise ValueError(f"{member} row {row_number} has wrong field count")
        try:
            waps = [int(value) for value in row[:FEATURE_COUNT]]
            longitude, latitude = float(row[FEATURE_COUNT]), float(row[FEATURE_COUNT + 1])
            metadata_ints = [int(value) for value in row[FEATURE_COUNT + 2 :]]
        except ValueError as exc:
            raise ValueError(f"{member} row {row_number} violates raw numeric schema") from exc
        if not all(-104 <= value <= 100 and (value == 100 or value <= 0) for value in waps):
            raise ValueError(f"{member} row {row_number} violates native WAP range or sentinel contract")
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            raise ValueError(f"{member} row {row_number} has nonfinite coordinate")
        building = metadata_ints[1]
        if building not in CLASS_LABELS:
            raise ValueError(f"{member} row {row_number} violates building support")
        candidate = _digest(row[:FEATURE_COUNT])
        values.append(waps)
        labels.append(building)
        complete_keys.append(_digest(row))
        candidate_keys.append(candidate)
        candidate_label_keys.append(_digest(row[:FEATURE_COUNT] + [str(building)]))
    x, y = np.asarray(values, dtype=float), np.asarray(labels, dtype=int)
    if x.shape != (ROWS[member], FEATURE_COUNT) or y.shape != (ROWS[member],) or set(y) != set(CLASS_LABELS):
        raise ValueError(f"{member} violates fixed count, shape, or building support")
    return Rows(x, y, tuple(complete_keys), tuple(candidate_keys), tuple(candidate_label_keys))


def load_locked_rows(archive_path: Path = ARCHIVE) -> tuple[Rows, Rows]:
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != MEMBERS:
            raise ValueError("UJIIndoorLoc archive inventory changed")
        source, external = _read(archive, TRAIN), _read(archive, VALIDATION)
    return source, external


def _fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        prior = np.asarray([np.count_nonzero(labels == label) for label in CLASS_LABELS], dtype=float)
        return Model(name, prior=prior / prior.sum())
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    if name.startswith("logreg_"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator = LogisticRegression(C=float(name.split("_")[1]), max_iter=5000, solver="lbfgs", random_state=seed).fit(transformed, labels)
        return Model(name, scaler, estimator)
    if name == "nearest_centroid":
        return Model(name, scaler, NearestCentroid().fit(transformed, labels))
    raise ValueError("unlocked model")


def _metrics(model: Model, rows: Rows) -> dict[str, object]:
    probabilities = model.predict_proba(rows.x)
    if (
        probabilities.shape != (len(rows.y), len(CLASS_LABELS))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    ):
        raise ValueError("UJIIndoorLoc probabilities are invalid")
    prediction = model.predict(rows.x)
    result: dict[str, object] = {
        "macro_f1": float(f1_score(rows.y, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(rows.y, prediction)),
        "multiclass_log_loss": float(log_loss(rows.y, probabilities, labels=CLASS_LABELS)),
        "prediction_fingerprint": hashlib.sha256(prediction.tobytes()).hexdigest(),
    }
    if not all(np.isfinite(value) for key, value in result.items() if key != "prediction_fingerprint"):
        raise ValueError("UJIIndoorLoc metric is nonfinite")
    return result


def _duplicate_ledger(rows: Rows) -> dict[str, object]:
    candidate_labels: dict[str, set[int]] = {}
    for key, label in zip(rows.candidate_keys, rows.y):
        candidate_labels.setdefault(key, set()).add(int(label))
    return {
        "complete_row_duplicates": len(rows.y) - len(set(rows.complete_keys)),
        "candidate_duplicates": len(rows.y) - len(set(rows.candidate_keys)),
        "candidate_label_duplicates": len(rows.y) - len(set(rows.candidate_label_keys)),
        "candidate_only_inconsistent_label_groups": sum(len(labels) > 1 for labels in candidate_labels.values()),
    }


def _group_split(source: Rows) -> tuple[Rows, Rows, dict[str, object]]:
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(source.candidate_label_keys):
        groups.setdefault(key, []).append(index)
    group_keys = tuple(sorted(groups))
    group_labels = np.asarray([source.y[groups[key][0]] for key in group_keys])
    train_group_index, selection_group_index = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros((len(group_keys), 1)), group_labels)
    )
    train_index = np.asarray(sorted(index for group_index in train_group_index for index in groups[group_keys[group_index]]))
    selection_index = np.asarray(sorted(index for group_index in selection_group_index for index in groups[group_keys[group_index]]))
    train, selection = source.subset(train_index), source.subset(selection_index)
    crossing = len(set(train.candidate_label_keys) & set(selection.candidate_label_keys))
    if crossing:
        raise ValueError("candidate-label duplicate group crosses source-train selection partition")
    return train, selection, {
        "candidate_label_group_count": len(group_keys),
        "train_group_count": len(train_group_index),
        "selection_group_count": len(selection_group_index),
        "train_rows": len(train.y),
        "selection_rows": len(selection.y),
        "cross_partition_candidate_label_groups": crossing,
    }


def _reject(payload: dict[str, object]) -> bool:
    try:
        wlan_from_payload(payload)
    except ValueError:
        return True
    return False


def _controls(name: str, train: Rows, selection: Rows, model: Model, macro_f1: float) -> dict[str, object]:
    paired = _metrics(_fit(name, train.x, np.random.default_rng(SEED).permutation(train.y)), selection)["macro_f1"]
    permutation = np.roll(np.arange(BLOCK_COUNT), 1)
    block_values = selection.x.reshape(-1, BLOCK_COUNT, BLOCK_SIZE)[:, permutation, :].reshape(-1, FEATURE_COUNT)
    block_paired = _metrics(model, Rows(block_values, selection.y, selection.complete_keys, selection.candidate_keys, selection.candidate_label_keys))["macro_f1"]
    sentinel_values = np.where(np.roll(selection.x == 100.0, 1, axis=1), 100.0, -104.0)
    sentinel_paired = _metrics(model, Rows(sentinel_values, selection.y, selection.complete_keys, selection.candidate_keys, selection.candidate_label_keys))["macro_f1"]
    ablations = []
    for block in range(BLOCK_COUNT):
        values = selection.x.copy()
        values[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE] = 100.0
        ablations.append(macro_f1 - _metrics(model, Rows(values, selection.y, selection.complete_keys, selection.candidate_keys, selection.candidate_label_keys))["macro_f1"])
    sample = [UJIIndoorLocWLAN(values) for values in selection.x[:12]]
    uniform = np.full(len(CLASS_LABELS), 1.0 / len(CLASS_LABELS))
    states: list[float] = []
    reset_probabilities = predict_independent_wlans(sample, lambda state, _values: states.append(state) or uniform)
    reordered_probabilities = predict_independent_wlans(list(reversed(sample)), lambda _state, _values: uniform)
    candidate = selection.x[0]
    return {
        "label_record_pairing_drop": macro_f1 - paired,
        "wap_block_pairing_drop": macro_f1 - block_paired,
        "sentinel_mask_pairing_drop": macro_f1 - sentinel_paired,
        "wap_block_ablation_drops": ablations,
        "per_record_zero_reset": states == [0.0] * len(sample),
        "row_reorder_invariant": bool(np.array_equal(reset_probabilities, reordered_probabilities[::-1])) and bool(np.array_equal(model.predict(selection.x), model.predict(selection.x[::-1])[::-1])),
        "target_isolation_sentinel_rejected": _reject({"waps": candidate, "label": 0}),
        "file_isolation_sentinel_rejected": _reject({"waps": candidate, "source_file": TRAIN}),
        "split_isolation_sentinel_rejected": _reject({"waps": candidate, "split": "selection"}),
        "metadata_isolation_sentinel_rejected": _reject({"waps": candidate, "longitude": 0.0, "floor": 0, "user": 1, "phone": 1, "timestamp": 1}),
        "probabilities_finite_and_bounded": True,
    }


def _write(result: dict[str, object], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT) -> dict[str, object]:
    source, external = load_locked_rows(archive_path)
    source_ledger, external_ledger = _duplicate_ledger(source), _duplicate_ledger(external)
    candidate_label_crossing = len(set(source.candidate_label_keys) & set(external.candidate_label_keys))
    candidate_only_crossing = len(set(source.candidate_keys) & set(external.candidate_keys))
    revalidation = {
        "source_train_rows": len(source.y),
        "external_rows": len(external.y),
        "source_train_duplicate_ledger": source_ledger,
        "external_duplicate_ledger": external_ledger,
        "source_external_candidate_label_duplicate_groups": candidate_label_crossing,
        "source_external_candidate_only_groups": candidate_only_crossing,
    }
    if candidate_label_crossing:
        return _write({
            "phase": 49,
            "status": "closed_negative_revalidation_gate",
            "reason": "candidate-label duplicate group crosses source train and validation files",
            "source_revalidation": revalidation,
            "external_outcome_score_count": 0,
            "abc_smc_calls": 0,
            "llm_calls": 0,
        }, output_dir)
    train, selection, group_ledger = _group_split(source)
    revalidation["selection_group_ledger"] = group_ledger
    candidates = {name: _metrics(_fit(name, train.x, train.y), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-candidates[name]["macro_f1"], MODELS.index(name)))
    model = _fit(selected, train.x, train.y)
    selection_f1 = candidates[selected]["macro_f1"]
    controls = _controls(selected, train, selection, model, selection_f1)
    selection_passed = all([
        selection_f1 >= 0.80,
        selection_f1 - candidates["majority"]["macro_f1"] >= 0.45,
        controls["label_record_pairing_drop"] >= 0.40,
        controls["wap_block_pairing_drop"] >= 0.05,
        controls["sentinel_mask_pairing_drop"] >= 0.05,
        max(controls["wap_block_ablation_drops"]) >= 0.02,
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key in {"per_record_zero_reset", "row_reorder_invariant", "probabilities_finite_and_bounded"}],
    ])
    result: dict[str, object] = {
        "phase": 49,
        "source_revalidation": revalidation,
        "models": candidates,
        "selected_model": selected,
        "selection_controls": controls,
        "selection_passed": selection_passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not selection_passed:
        result.update({"status": "closed_negative_selection_gate", "reason": "fixed selection or specificity gate failed; validation file was not scored"})
        return _write(result, output_dir)
    metrics, fingerprints = [], []
    for seed in range(SEED, SEED + 8):
        full = _fit(selected, source.x, source.y, seed)
        metric = _metrics(full, external)
        metric["seed"] = seed
        metrics.append(metric)
        fingerprints.append((metric["prediction_fingerprint"], hashlib.sha256(full.predict(source.x).tobytes()).hexdigest()))
        if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
            break
    primary = metrics[0]
    external_passed = primary["macro_f1"] >= 0.75 and primary["macro_f1"] >= 0.85 * selection_f1
    result.update({
        "status": "passed_observed_baseline_and_external_stability" if external_passed else "closed_negative_external_stability",
        "external_outcome_score_count": 1,
        "external_seed_metrics": metrics,
        "executed_seed_count": len(metrics),
        "adaptive_distinct_yield_stopped": len(metrics) < 8,
        "external_passed": external_passed,
        "reason": "fixed one-operation validation stability completed" if external_passed else "fixed validation metric stop failed",
    })
    return _write(result, output_dir)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

#!/usr/bin/env python3
"""Execute the fixed Phase 47 Statlog Landsat observed baseline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_statlog_landsat import CLASS_LABELS, neighborhood_from_payload
from scripts.run_uci_statlog_landsat_source_gate import ARCHIVE, FILES, MEMBERS, parse_rows


OUT = Path("artifacts/evaluations/phase47_uci_statlog_landsat_real_baselines_20260727")
SEED = 2047001
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
PIXEL_PERMUTATION = np.roll(np.arange(9).reshape(3, 3), 1, axis=1).ravel()
BAND_PERMUTATION = np.roll(np.arange(4), 1)


@dataclass(frozen=True)
class Rows:
    x: np.ndarray
    y: np.ndarray
    keys: tuple[str, ...]


class Model:
    def __init__(self, name: str, scaler=None, estimator=None, prior=None):
        self.name, self.scaler, self.estimator, self.prior = name, scaler, estimator, prior

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.prior is not None:
            return np.broadcast_to(self.prior, (len(values), len(CLASS_LABELS))).copy()
        transformed = self.scaler.transform(values.reshape(len(values), -1))
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(transformed)
        distance = -np.sum((transformed[:, None, :] - self.estimator.centroids_[None, :, :]) ** 2, axis=2)
        distance -= distance.max(axis=1, keepdims=True)
        weights = np.exp(distance)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(CLASS_LABELS)[np.argmax(self.predict_proba(values), axis=1)]


def _read(archive: ZipFile, member: str) -> Rows:
    matrix, keys = parse_rows(archive.read(member), member)
    return Rows(matrix[:, :-1].reshape(len(matrix), 9, 4).astype(float), matrix[:, -1].astype(int), keys)


def load_locked_rows(archive_path: Path = ARCHIVE) -> tuple[Rows, Rows]:
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != MEMBERS:
            raise ValueError("Statlog Landsat inventory changed")
        source, external = _read(archive, "sat.trn"), _read(archive, "sat.tst")
    if set(source.keys) & set(external.keys):
        raise ValueError("complete row duplicate crosses source-file boundary")
    return source, external


def _fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        prior = np.asarray([np.count_nonzero(labels == label) for label in CLASS_LABELS], dtype=float)
        return Model(name, prior=prior / prior.sum())
    scaler = StandardScaler().fit(values.reshape(len(values), -1))
    transformed = scaler.transform(values.reshape(len(values), -1))
    if name.startswith("logreg_"):
        return Model(name, scaler, LogisticRegression(C=float(name.split("_")[1]), max_iter=2000, solver="lbfgs", random_state=seed).fit(transformed, labels))
    if name == "nearest_centroid":
        return Model(name, scaler, NearestCentroid().fit(transformed, labels))
    raise ValueError("unlocked model")


def _metrics(model: Model, rows: Rows) -> dict[str, object]:
    probabilities = model.predict_proba(rows.x)
    if probabilities.shape != (len(rows.y), len(CLASS_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Statlog Landsat probabilities are invalid")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Statlog Landsat probabilities are out of bounds")
    prediction = model.predict(rows.x)
    result: dict[str, object] = {
        "macro_f1": float(f1_score(rows.y, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(rows.y, prediction)),
        "multiclass_log_loss": float(log_loss(rows.y, probabilities, labels=CLASS_LABELS)),
        "prediction_fingerprint": hashlib.sha256(prediction.tobytes()).hexdigest(),
    }
    if not all(np.isfinite(value) for key, value in result.items() if key != "prediction_fingerprint"):
        raise ValueError("Statlog Landsat metric is nonfinite")
    return result


def _reject(payload: dict[str, object]) -> bool:
    try:
        neighborhood_from_payload(payload)
    except ValueError:
        return True
    return False


def _controls(name: str, train: Rows, selection: Rows, model: Model, macro_f1: float) -> dict[str, object]:
    paired = _metrics(_fit(name, train.x, np.random.default_rng(SEED).permutation(train.y)), selection)["macro_f1"]
    pixel_paired = _metrics(model, Rows(selection.x[:, PIXEL_PERMUTATION, :], selection.y, selection.keys))["macro_f1"]
    band_paired = _metrics(model, Rows(selection.x[:, :, BAND_PERMUTATION], selection.y, selection.keys))["macro_f1"]
    pixel_ablations, band_ablations = [], []
    for pixel in range(9):
        values = selection.x.copy()
        values[:, pixel, :] = 0
        pixel_ablations.append(macro_f1 - _metrics(model, Rows(values, selection.y, selection.keys))["macro_f1"])
    for band in range(4):
        values = selection.x.copy()
        values[:, :, band] = 0
        band_ablations.append(macro_f1 - _metrics(model, Rows(values, selection.y, selection.keys))["macro_f1"])
    reverse = np.arange(len(selection.y) - 1, -1, -1)
    candidate = selection.x[0]
    return {
        "label_record_pairing_drop": macro_f1 - paired,
        "pixel_order_pairing_drop": macro_f1 - pixel_paired,
        "band_pairing_drop": macro_f1 - band_paired,
        "pixel_ablation_drops": pixel_ablations,
        "band_ablation_drops": band_ablations,
        "row_reorder_invariant": bool(np.array_equal(model.predict(selection.x), model.predict(selection.x[reverse])[::-1])),
        "target_isolation_sentinel_rejected": _reject({"values": candidate, "label": 1}),
        "file_isolation_sentinel_rejected": _reject({"values": candidate, "source_file": "sat.trn"}),
        "split_isolation_sentinel_rejected": _reject({"values": candidate, "split": "train"}),
        "row_isolation_sentinel_rejected": _reject({"values": candidate, "row_index": 0}),
        "location_isolation_sentinel_rejected": _reject({"values": candidate, "location": [0, 0]}),
    }


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT) -> dict[str, object]:
    source, external = load_locked_rows(archive_path)
    train_index, selection_index = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(source.x, source.y))
    train = Rows(source.x[train_index], source.y[train_index], tuple(source.keys[index] for index in train_index))
    selection = Rows(source.x[selection_index], source.y[selection_index], tuple(source.keys[index] for index in selection_index))
    if set(train.keys) & set(selection.keys):
        raise ValueError("complete row duplicate crosses source-train selection partition")
    candidates = {name: _metrics(_fit(name, train.x, train.y), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-candidates[name]["macro_f1"], MODELS.index(name)))
    model = _fit(selected, train.x, train.y)
    selection_f1 = candidates[selected]["macro_f1"]
    controls = _controls(selected, train, selection, model, selection_f1)
    selection_passed = all([
        selection_f1 >= 0.80,
        selection_f1 - candidates["majority"]["macro_f1"] >= 0.70,
        controls["label_record_pairing_drop"] >= 0.40,
        controls["pixel_order_pairing_drop"] >= 0.10,
        controls["band_pairing_drop"] >= 0.10,
        max(controls["pixel_ablation_drops"] + controls["band_ablation_drops"]) >= 0.03,
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key == "row_reorder_invariant"],
    ])
    result: dict[str, object] = {
        "phase": 47,
        "source_revalidation": {"source_train_rows": len(source.y), "external_rows": len(external.y), "train_rows": len(train.y), "selection_rows": len(selection.y), "cross_partition_complete_duplicates": 0},
        "models": candidates,
        "selected_model": selected,
        "selection_controls": controls,
        "selection_passed": selection_passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not selection_passed:
        result.update({"status": "closed_negative_selection_gate", "reason": "fixed selection or control gate failed; sat.tst was not scored"})
    else:
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
            "reason": "fixed one-operation source-test stability completed" if external_passed else "fixed source-test metric stop failed",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

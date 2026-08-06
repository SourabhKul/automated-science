#!/usr/bin/env python3
"""Execute the frozen Phase 44 Optical Digits observed baseline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

from core.real_data.uci_optdigits import grid_from_payload

ARCHIVE = Path("data/real/uci_optdigits/raw/uci_optdigits_80.zip")
OUT = Path("artifacts/evaluations/phase44_uci_optdigits_real_baselines_20260726")
SEED = 2044001
TRAIN, EXTERNAL = "optdigits.tra", "optdigits.tes"
ROWS = {TRAIN: 3823, EXTERNAL: 1797}
LABELS = tuple(range(10))
BLOCKS = tuple(np.arange(64).reshape(8, 8)[row : row + 4, column : column + 4].ravel() for row in (0, 4) for column in (0, 4))
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
INVENTORY = (
    "optdigits-orig.cv.Z",
    "optdigits-orig.names",
    "optdigits-orig.tra.Z",
    "optdigits-orig.wdep.Z",
    "optdigits-orig.windep.Z",
    "optdigits.names",
    EXTERNAL,
    TRAIN,
    "readme.txt",
)


@dataclass(frozen=True)
class Rows:
    x: np.ndarray
    y: np.ndarray
    keys: tuple[str, ...]


class Model:
    def __init__(self, name: str, scaler=None, estimator=None, prior=None):
        self.name, self.scaler, self.estimator, self.prior = name, scaler, estimator, prior

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.prior is not None:
            return np.broadcast_to(self.prior, (len(x), len(LABELS))).copy()
        z = self.scaler.transform(x)
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(z)
        distance = -np.sum((z[:, None, :] - self.estimator.centroids_[None, :, :]) ** 2, axis=2)
        distance -= distance.max(axis=1, keepdims=True)
        weights = np.exp(distance)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(LABELS)[np.argmax(self.predict_proba(x), axis=1)]


def _read(archive: ZipFile, member: str) -> Rows:
    values, labels, keys = [], [], []
    for index, line in enumerate(archive.read(member).decode("ascii").splitlines()):
        raw = line.strip()
        row = np.fromstring(raw, dtype=float, sep=",")
        if (
            not raw
            or row.shape != (65,)
            or not np.all(np.isfinite(row))
            or not np.all(row == np.floor(row))
            or np.any(row[:-1] < 0)
            or np.any(row[:-1] > 16)
            or int(row[-1]) not in LABELS
        ):
            raise ValueError(f"{member} violates locked schema at row {index}")
        values.append(row[:-1])
        labels.append(int(row[-1]))
        keys.append(hashlib.sha256(raw.encode()).hexdigest())
    x, y = np.asarray(values), np.asarray(labels)
    if x.shape != (ROWS[member], 64) or y.shape != (ROWS[member],) or set(y) != set(LABELS):
        raise ValueError(f"{member} violates locked count/class support")
    return Rows(x, y, tuple(keys))


def load_locked_rows(archive_path: Path = ARCHIVE) -> tuple[Rows, Rows]:
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != INVENTORY:
            raise ValueError("Optical Digits inventory changed")
        train, external = _read(archive, TRAIN), _read(archive, EXTERNAL)
    if set(train.keys) & set(external.keys):
        raise ValueError("complete duplicate crosses writer-file boundary")
    return train, external


def _fit(name: str, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        prior = np.asarray([np.count_nonzero(y == label) for label in LABELS], dtype=float)
        return Model(name, prior=prior / prior.sum())
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    if name.startswith("logreg_"):
        return Model(name, scaler, LogisticRegression(C=float(name.split("_")[1]), max_iter=2000, solver="lbfgs", random_state=seed).fit(z, y))
    if name == "nearest_centroid":
        return Model(name, scaler, NearestCentroid().fit(z, y))
    raise ValueError("unlocked model")


def _metrics(model: Model, rows: Rows) -> dict[str, object]:
    probabilities = model.predict_proba(rows.x)
    if (
        probabilities.shape != (len(rows.y), len(LABELS))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9)
    ):
        raise ValueError("invalid probabilities")
    prediction = model.predict(rows.x)
    result = {
        "macro_f1": float(f1_score(rows.y, prediction, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(rows.y, prediction)),
        "multiclass_log_loss": float(log_loss(rows.y, probabilities, labels=LABELS)),
        "prediction_fingerprint": hashlib.sha256(prediction.tobytes()).hexdigest(),
    }
    if not all(np.isfinite(value) for key, value in result.items() if key != "prediction_fingerprint"):
        raise ValueError("nonfinite metric")
    return result


def _reject(payload: dict[str, object]) -> bool:
    try:
        grid_from_payload(payload)
    except ValueError:
        return True
    return False


def _controls(name: str, train: Rows, selection: Rows, model: Model, macro_f1: float) -> dict[str, object]:
    paired = _metrics(_fit(name, train.x, np.random.default_rng(SEED).permutation(train.y)), selection)["macro_f1"]
    permutation = np.concatenate((BLOCKS[-1], *BLOCKS[:-1]))
    block = _metrics(model, Rows(selection.x[:, permutation], selection.y, selection.keys))["macro_f1"]
    ablations = []
    for indices in BLOCKS:
        x = selection.x.copy()
        x[:, indices] = 0
        ablations.append(macro_f1 - _metrics(model, Rows(x, selection.y, selection.keys))["macro_f1"])
    reverse = np.arange(len(selection.y) - 1, -1, -1)
    candidate = selection.x[0]
    return {
        "label_pairing_drop": macro_f1 - paired,
        "pixel_block_permutation_drop": macro_f1 - block,
        "pixel_block_ablation_drops": ablations,
        "row_reorder_invariant": bool(np.array_equal(model.predict(selection.x), model.predict(selection.x[reverse])[::-1])),
        "target_isolation_sentinel_rejected": _reject({"pixels": candidate, "label": 0}),
        "file_isolation_sentinel_rejected": _reject({"pixels": candidate, "source_file": EXTERNAL}),
        "split_isolation_sentinel_rejected": _reject({"pixels": candidate, "selection_assignment": True}),
        "writer_isolation_sentinel_rejected": _reject({"pixels": candidate, "writer_identity": "synthetic"}),
    }


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT) -> dict[str, object]:
    source, external = load_locked_rows(archive_path)
    train_index, selection_index = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(source.x, source.y))
    train = Rows(source.x[train_index], source.y[train_index], tuple(source.keys[index] for index in train_index))
    selection = Rows(source.x[selection_index], source.y[selection_index], tuple(source.keys[index] for index in selection_index))
    if set(train.keys) & set(selection.keys):
        raise ValueError("complete duplicate crosses source-train selection partition")
    candidates = {name: _metrics(_fit(name, train.x, train.y), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-candidates[name]["macro_f1"], MODELS.index(name)))
    model = _fit(selected, train.x, train.y)
    controls = _controls(selected, train, selection, model, candidates[selected]["macro_f1"])
    macro_f1 = candidates[selected]["macro_f1"]
    passed = all(
        [
            macro_f1 >= 0.9,
            macro_f1 - candidates["majority"]["macro_f1"] >= 0.8,
            controls["label_pairing_drop"] >= 0.5,
            controls["pixel_block_permutation_drop"] >= 0.1,
            max(controls["pixel_block_ablation_drops"]) >= 0.05,
            *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key == "row_reorder_invariant"],
        ]
    )
    result: dict[str, object] = {
        "phase": 44,
        "source_revalidation": {
            "source_train_rows": len(source.y),
            "external_rows": len(external.y),
            "train_rows": len(train.y),
            "selection_rows": len(selection.y),
            "cross_writer_file_complete_duplicates": 0,
        },
        "models": candidates,
        "selected_model": selected,
        "selection_controls": controls,
        "selection_passed": passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not passed:
        result.update({"status": "closed_negative_selection_gate", "reason": "fixed selection or control gate failed; optdigits.tes was not scored"})
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
        external_pass = primary["macro_f1"] >= 0.85 and primary["macro_f1"] >= 0.85 * macro_f1
        result.update(
            {
                "status": "passed_observed_baseline_and_external_stability" if external_pass else "closed_negative_external_stability",
                "external_outcome_score_count": 1,
                "external_seed_metrics": metrics,
                "executed_seed_count": len(metrics),
                "adaptive_distinct_yield_stopped": len(metrics) < 8,
                "external_passed": external_pass,
                "reason": "fixed one-operation external stability completed" if external_pass else "fixed external metric stop failed",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

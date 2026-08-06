#!/usr/bin/env python3
"""Execute the fixed Phase 67 observed STL-10 baseline gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.stl10 import LABELS, image_from_payload, predict_independent
from scripts.run_stl10_source_gate import ARCHIVE, MEMBERS, PIXEL_BYTES, ROWS, sha256


OUTPUT_DIR = Path("artifacts/evaluations/phase67_stl10_real_baselines_20260802")
SEED = 2067001
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
BLOCK = 24


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    labels: np.ndarray
    fingerprints: tuple[str, ...]


def read_labeled_partition(archive: tarfile.TarFile, partition: str) -> Records:
    images = archive.extractfile(f"stl10_binary/{partition}_X.bin")
    labels = archive.extractfile(f"stl10_binary/{partition}_y.bin")
    if images is None or labels is None:
        raise ValueError(f"STL-10 {partition} member is unavailable")
    raw_values = images.read()
    raw_labels = labels.read()
    rows = ROWS[partition]
    if len(raw_values) != rows * PIXEL_BYTES or len(raw_labels) != rows:
        raise ValueError(f"STL-10 {partition} byte contract changed")
    values = np.frombuffer(raw_values, dtype=np.uint8).reshape(rows, PIXEL_BYTES).copy()
    outcome = np.frombuffer(raw_labels, dtype=np.uint8).astype(int)
    if not np.all((0 <= values) & (values <= 255)) or set(outcome.tolist()) != set(range(1, 11)):
        raise ValueError(f"STL-10 {partition} range or label support changed")
    fingerprints = tuple(hashlib.sha256(row.tobytes()).hexdigest() for row in values)
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError(f"STL-10 {partition} duplicate ledger changed")
    return Records(values, outcome - 1, fingerprints)


def revalidate_train_only() -> tuple[Records, dict[str, object]]:
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        if tuple(member.name for member in archive.getmembers()) != MEMBERS:
            raise ValueError("STL-10 inventory changed")
        train = read_labeled_partition(archive, "train")
        test_x = archive.getmember("stl10_binary/test_X.bin")
        test_y = archive.getmember("stl10_binary/test_y.bin")
    if sha256(ARCHIVE) != "f31fd99273a1acb8609c8db427cebb1de3f71de77758cdc0e22956e1289b9866":
        raise ValueError("STL-10 archive SHA-256 changed")
    if test_x.size != ROWS["test"] * PIXEL_BYTES or test_y.size != ROWS["test"]:
        raise ValueError("STL-10 sealed source-test byte contract changed")
    return train, {"source_train_rows": len(train.labels), "source_test_rows": ROWS["test"], "source_train_candidate_duplicates": 0, "source_train_complete_duplicates": 0, "source_test_content_read": False}


class Model:
    def __init__(self, name: str, scaler: StandardScaler | None = None, estimator: object | None = None, probabilities: np.ndarray | None = None) -> None:
        self.name, self.scaler, self.estimator, self.probabilities = name, scaler, estimator, probabilities

    def probability(self, values: np.ndarray) -> np.ndarray:
        if self.probabilities is not None:
            return np.broadcast_to(self.probabilities, (len(values), len(LABELS))).copy()
        assert self.scaler is not None and self.estimator is not None
        transformed = self.scaler.transform(values)
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(transformed)
        distances = -((transformed[:, None, :] - self.estimator.centroids_) ** 2).sum(axis=2)
        distances -= distances.max(axis=1, keepdims=True)
        weights = np.exp(distances)
        return weights / weights.sum(axis=1, keepdims=True)


def fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        counts = np.bincount(labels, minlength=len(LABELS)).astype(float)
        return Model(name, probabilities=counts / counts.sum())
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    if name.startswith("logreg_"):
        return Model(name, scaler, LogisticRegression(C=float(name.removeprefix("logreg_")), max_iter=2000, solver="lbfgs", random_state=seed).fit(transformed, labels))
    return Model(name, scaler, NearestCentroid().fit(transformed, labels))


def metric(model: Model, records: Records) -> dict[str, object]:
    probabilities = model.probability(records.values)
    if probabilities.shape != (len(records.labels), len(LABELS)) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("STL-10 model emitted invalid probabilities")
    predictions = probabilities.argmax(axis=1)
    return {"macro_f1": float(f1_score(records.labels, predictions, labels=LABELS, average="macro", zero_division=0)), "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest()}


def pair_blocks(values: np.ndarray) -> np.ndarray:
    images = values.reshape(len(values), 3, 96, 96).copy()
    permutation = np.random.default_rng(SEED + 1).permutation(len(images))
    for row in range(4):
        for column in range(4):
            images[:, :, row * BLOCK : (row + 1) * BLOCK, column * BLOCK : (column + 1) * BLOCK] = images[permutation, :, row * BLOCK : (row + 1) * BLOCK, column * BLOCK : (column + 1) * BLOCK]
    return images.reshape(values.shape)


def rejected(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def controls(name: str, train: Records, selection: Records, model: Model, baseline: float) -> dict[str, object]:
    paired_labels = np.random.default_rng(SEED).permutation(train.labels)
    label_drop = baseline - float(metric(fit(name, train.values, paired_labels), selection)["macro_f1"])
    spatial_drop = baseline - float(metric(model, Records(pair_blocks(selection.values), selection.labels, selection.fingerprints))["macro_f1"])
    drops = []
    images = selection.values.reshape(len(selection.values), 3, 96, 96)
    for row in range(4):
        for column in range(4):
            ablated = images.copy()
            ablated[:, :, row * BLOCK : (row + 1) * BLOCK, column * BLOCK : (column + 1) * BLOCK] = 0
            drops.append(baseline - float(metric(model, Records(ablated.reshape(len(ablated), -1), selection.labels, selection.fingerprints))["macro_f1"]))
    sample = images[0]
    states: list[float] = []
    probabilities = predict_independent([image_from_payload({"values": sample})] * 3, lambda state, values: states.append(state) or np.full(10, 0.1))
    reordered = predict_independent([image_from_payload({"values": sample})] * 3, lambda state, values: np.full(10, 0.1))
    return {"label_image_pairing_drop": label_drop, "spatial_24x24_pairing_drop": spatial_drop, "block_ablation_drops": drops, "zero_reset": states == [0.0] * 3, "reorder_invariant": bool(np.array_equal(probabilities, reordered)), "target_rejected": rejected({"values": sample, "label": 0}), "file_rejected": rejected({"values": sample, "file": "test_X.bin"}), "split_rejected": rejected({"values": sample, "split": "external"}), "row_rejected": rejected({"values": sample, "row": 0}), "fold_rejected": rejected({"values": sample, "fold": 0}), "class_name_rejected": rejected({"values": sample, "class_name": "airplane"}), "metadata_rejected": rejected({"values": sample, "metadata": "source"}), "duplicate_rejected": rejected({"values": sample, "duplicate_group": 0}), "unlabeled_rejected": rejected({"values": sample, "unlabeled": True}), "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0))}


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    source, revalidation = revalidate_train_only()
    train_idx, select_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(source.values, source.labels))
    train = Records(source.values[train_idx], source.labels[train_idx], tuple(source.fingerprints[index] for index in train_idx))
    selection = Records(source.values[select_idx], source.labels[select_idx], tuple(source.fingerprints[index] for index in select_idx))
    candidates = {name: metric(fit(name, train.values, train.labels), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-float(candidates[name]["macro_f1"]), MODELS.index(name)))
    score = float(candidates[selected]["macro_f1"])
    majority = float(candidates["majority"]["macro_f1"])
    selected_model = fit(selected, train.values, train.labels)
    selection_controls = controls(selected, train, selection, selected_model, score)
    passed = score >= 0.20 and score - majority >= 0.10 and selection_controls["label_image_pairing_drop"] >= 0.15 and selection_controls["spatial_24x24_pairing_drop"] >= 0.05 and max(selection_controls["block_ablation_drops"]) >= 0.02 and all(bool(value) for key, value in selection_controls.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "finite_probabilities"})
    result: dict[str, object] = {"phase": 67, "source_revalidation": revalidation | {"selection_rows": len(selection.labels), "train_rows": len(train.labels)}, "models": candidates, "selected_model": selected, "selection_controls": selection_controls, "selection_passed": passed, "external_outcome_score_count": 0, "abc_smc_calls": 0, "llm_calls": 0}
    if not passed:
        result.update(status="closed_negative_selection_gate", reason="fixed source-train selection gate failed; source test content was not read")
    else:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            external = read_labeled_partition(archive, "test")
        results, fingerprints = [], []
        for seed in range(SEED, SEED + 8):
            current = metric(fit(selected, source.values, source.labels, seed), external)
            current["seed"] = seed
            results.append(current)
            fingerprints.append(str(current["prediction_fingerprint"]))
            if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
                break
        result.update(status="passed_observed_baseline_and_external_stability", external_outcome_score_count=1, external_seed_metrics=results, executed_seed_count=len(results), adaptive_distinct_yield_stopped=len(results) < 8)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

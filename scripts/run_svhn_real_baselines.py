#!/usr/bin/env python3
"""Execute the fixed Phase 59 duplicate-aware observed SVHN baseline gate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.svhn import NORMALIZED_LABELS, image_from_payload, normalized_label, predict_independent

RAW = Path("data/real/svhn/raw")
OUTPUT_DIR = Path("artifacts/evaluations/phase59_svhn_real_baselines_20260801")
SEED = 2059001
FILES = {"train_32x32.mat": 73_257, "test_32x32.mat": 26_032}
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
BLOCK = 8


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    labels: np.ndarray
    fingerprints: tuple[str, ...]


def read_matrix(name: str) -> Records:
    expected_records = FILES[name]
    content = loadmat(RAW / name, variable_names=("X", "y"))
    if set(content) != {"__header__", "__version__", "__globals__", "X", "y"}:
        raise ValueError(f"{name} raw variable contract changed")
    images = np.asarray(content["X"])
    raw_labels = np.asarray(content["y"])
    if images.shape != (32, 32, 3, expected_records) or images.dtype != np.uint8:
        raise ValueError(f"{name} image contract changed")
    if raw_labels.shape not in {(expected_records,), (expected_records, 1)} or not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError(f"{name} label contract changed")
    raw_labels = raw_labels.reshape(-1)
    if set(raw_labels) != set(range(1, 11)) or not np.all(np.isfinite(images)):
        raise ValueError(f"{name} raw support or finite contract changed")
    labels = np.asarray([normalized_label(int(label)) for label in raw_labels], dtype=int)
    values = np.transpose(images, (3, 2, 0, 1)).reshape(expected_records, -1)
    fingerprints = tuple(hashlib.sha256(images[:, :, :, index].tobytes()).hexdigest() for index in range(expected_records))
    return Records(values, labels, fingerprints)


def duplicate_groups(records: Records) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(records.fingerprints):
        grouped[fingerprint].append(index)
    group_ids = np.empty(len(records.labels), dtype=int)
    groups: dict[int, np.ndarray] = {}
    for group_id, indices in enumerate(grouped.values()):
        members = np.asarray(indices, dtype=int)
        if len(set(records.labels[members])) != 1:
            raise ValueError("complete-image duplicate group has inconsistent mapped labels")
        group_ids[members] = group_id
        groups[group_id] = members
    return group_ids, groups


def group_aware_split(records: Records) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    group_ids, groups = duplicate_groups(records)
    group_labels = np.asarray([records.labels[groups[group_id][0]] for group_id in range(len(groups))])
    train_groups, selection_groups = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros((len(groups), 1)), group_labels)
    )
    train_ids = np.concatenate([groups[int(group_id)] for group_id in train_groups])
    selection_ids = np.concatenate([groups[int(group_id)] for group_id in selection_groups])
    if set(group_ids[train_ids]) & set(group_ids[selection_ids]):
        raise ValueError("duplicate group crossed the frozen source-train split")
    return train_ids, selection_ids, {
        "source_duplicate_groups": len(groups),
        "source_complete_duplicates": len(records.fingerprints) - len(set(records.fingerprints)),
        "groups_crossing_selection_split": 0,
    }


def subrecords(records: Records, indices: np.ndarray) -> Records:
    return Records(records.values[indices], records.labels[indices], tuple(records.fingerprints[index] for index in indices))


class Model:
    def __init__(self, name: str, scaler: StandardScaler | None = None, estimator: object | None = None, probabilities: np.ndarray | None = None) -> None:
        self.name = name
        self.scaler = scaler
        self.estimator = estimator
        self.probabilities = probabilities

    def probability(self, values: np.ndarray) -> np.ndarray:
        if self.probabilities is not None:
            return np.broadcast_to(self.probabilities, (len(values), len(NORMALIZED_LABELS))).copy()
        assert self.scaler is not None and self.estimator is not None
        transformed = self.scaler.transform(values)
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(transformed)
        distances = -((transformed[:, None, :] - self.estimator.centroids_) ** 2).sum(axis=2)
        distances -= distances.max(axis=1, keepdims=True)
        weights = np.exp(distances)
        return weights / weights.sum(axis=1, keepdims=True)

    def prediction(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(NORMALIZED_LABELS)[self.probability(values).argmax(axis=1)]


def fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        probabilities = np.array([(labels == label).sum() for label in NORMALIZED_LABELS], dtype=float)
        return Model(name, probabilities=probabilities / probabilities.sum())
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    if name.startswith("logreg_"):
        estimator = LogisticRegression(C=float(name.removeprefix("logreg_")), max_iter=2000, solver="lbfgs", random_state=seed).fit(transformed, labels)
        return Model(name, scaler, estimator)
    return Model(name, scaler, NearestCentroid().fit(transformed, labels))


def metric(model: Model, records: Records) -> dict[str, object]:
    probabilities = model.probability(records.values)
    if probabilities.shape != (len(records.labels), len(NORMALIZED_LABELS)) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("SVHN model emitted invalid probabilities")
    predictions = model.prediction(records.values)
    return {
        "macro_f1": float(f1_score(records.labels, predictions, labels=NORMALIZED_LABELS, average="macro", zero_division=0)),
        "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }


def permute_8x8_blocks(values: np.ndarray) -> np.ndarray:
    images = values.reshape(len(values), 3, 32, 32)
    grid = images.reshape(len(images), 3, 4, BLOCK, 4, BLOCK)
    return np.roll(grid, 1, axis=4).reshape(values.shape)


def rejects_payload(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def controls(name: str, train: Records, selection: Records, model: Model, score: float, duplicate_ledger: dict[str, int]) -> dict[str, object]:
    shuffled_labels = np.random.default_rng(SEED).permutation(train.labels)
    label_pairing_drop = score - float(metric(fit(name, train.values, shuffled_labels), selection)["macro_f1"])
    block_pairing_drop = score - float(metric(model, Records(permute_8x8_blocks(selection.values), selection.labels, selection.fingerprints))["macro_f1"])
    ablation_drops: list[float] = []
    selection_images = selection.values.reshape(len(selection.values), 3, 32, 32)
    for block_row in range(4):
        for block_column in range(4):
            ablated = selection_images.copy()
            ablated[:, :, block_row * BLOCK : (block_row + 1) * BLOCK, block_column * BLOCK : (block_column + 1) * BLOCK] = 0
            ablated_records = Records(ablated.reshape(len(ablated), -1), selection.labels, selection.fingerprints)
            ablation_drops.append(score - float(metric(model, ablated_records)["macro_f1"]))
    sample = selection.values[0].reshape(3, 32, 32)
    sequence = [sample] * 3
    states: list[float] = []
    probabilities = predict_independent([image_from_payload({"values": image}) for image in sequence], lambda state, image: states.append(state) or np.full(10, 0.1))
    reordered = predict_independent([image_from_payload({"values": image}) for image in sequence[::-1]], lambda state, image: np.full(10, 0.1))
    return {
        "raw_label_mapping": normalized_label(10) == 0 and normalized_label(1) == 1,
        "label_image_pairing_drop": label_pairing_drop,
        "pixel_block_pairing_drop": block_pairing_drop,
        "pixel_block_ablation_drops": ablation_drops,
        "zero_reset": states == [0.0] * len(sequence),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejects_payload({"values": sample, "label": 10}),
        "file_rejected": rejects_payload({"values": sample, "file": "test_32x32.mat"}),
        "split_rejected": rejects_payload({"values": sample, "split": "external"}),
        "row_rejected": rejects_payload({"values": sample, "row": 0}),
        "metadata_rejected": rejects_payload({"values": sample, "shape": [3, 32, 32]}),
        "duplicate_group_rejected": rejects_payload({"values": sample, "duplicate_group": "source-duplicate-0"}),
        "duplicate_ledger_integrity": duplicate_ledger["source_complete_duplicates"] == 1 and duplicate_ledger["external_complete_duplicates"] == 0 and duplicate_ledger["cross_file_complete_duplicates"] == 0 and duplicate_ledger["groups_crossing_selection_split"] == 0,
        "finite_probabilities": True,
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    source = read_matrix("train_32x32.mat")
    external = read_matrix("test_32x32.mat")
    train_indices, selection_indices, duplicate_ledger = group_aware_split(source)
    duplicate_ledger.update(
        external_complete_duplicates=len(external.fingerprints) - len(set(external.fingerprints)),
        cross_file_complete_duplicates=len(set(source.fingerprints) & set(external.fingerprints)),
    )
    train = subrecords(source, train_indices)
    selection = subrecords(source, selection_indices)
    candidates = {name: metric(fit(name, train.values, train.labels), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-float(candidates[name]["macro_f1"]), MODELS.index(name)))
    model = fit(selected, train.values, train.labels)
    selection_score = float(candidates[selected]["macro_f1"])
    selection_controls = controls(selected, train, selection, model, selection_score, duplicate_ledger)
    selection_passed = (
        selection_controls["label_image_pairing_drop"] >= 0.20
        and selection_controls["pixel_block_pairing_drop"] >= 0.20
        and max(selection_controls["pixel_block_ablation_drops"]) >= 0.05
        and all(bool(value) for key, value in selection_controls.items() if key.endswith("rejected") or key in {"raw_label_mapping", "zero_reset", "reorder_invariant", "duplicate_ledger_integrity", "finite_probabilities"})
    )
    result: dict[str, object] = {
        "phase": 59,
        "source_revalidation": {"source_rows": len(source.labels), "external_rows": len(external.labels), "train_rows": len(train.labels), "selection_rows": len(selection.labels), **duplicate_ledger},
        "models": candidates,
        "selected_model": selected,
        "selection_controls": selection_controls,
        "selection_passed": selection_passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not selection_passed:
        result.update(status="closed_negative_selection_gate", reason="fixed selection or specificity control failed; test_32x32.mat was not scored")
    else:
        external_metrics: list[dict[str, object]] = []
        fingerprints: list[str] = []
        for seed in range(SEED, SEED + 8):
            external_metric = metric(fit(selected, source.values, source.labels, seed), external)
            external_metric["seed"] = seed
            external_metrics.append(external_metric)
            fingerprints.append(str(external_metric["prediction_fingerprint"]))
            if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
                break
        result.update(status="passed_observed_baseline_and_external_stability", external_outcome_score_count=1, external_seed_metrics=external_metrics, executed_seed_count=len(external_metrics), adaptive_distinct_yield_stopped=len(external_metrics) < 8)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

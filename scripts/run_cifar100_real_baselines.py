#!/usr/bin/env python3
"""Execute the fixed Phase 60 observed CIFAR-100 baseline gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.cifar100 import FINE_LABELS, image_from_payload, predict_independent


ARCHIVE = Path("data/real/cifar100/raw/cifar-100-python.tar.gz")
OUTPUT_DIR = Path("artifacts/evaluations/phase60_cifar100_real_baselines_20260801")
SEED = 2060001
BLOCK = 8
MEMBERS = ("cifar-100-python", "cifar-100-python/file.txt~", "cifar-100-python/train", "cifar-100-python/test", "cifar-100-python/meta")
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
EXPECTED_DUPLICATES = {"source_complete_duplicates": 14, "external_complete_duplicates": 2, "cross_file_complete_duplicates": 10}


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    labels: np.ndarray
    fingerprints: tuple[str, ...]


def member_name(member: str) -> str:
    return f"cifar-100-python/{member}"


def read_member(archive: tarfile.TarFile, member: str, expected_rows: int) -> Records:
    extracted = archive.extractfile(member_name(member))
    if extracted is None:
        raise ValueError(f"missing fixed CIFAR-100 member {member}")
    payload = pickle.load(extracted, encoding="bytes")
    expected_keys = {b"filenames", b"batch_label", b"fine_labels", b"coarse_labels", b"data"}
    if set(payload) != expected_keys:
        raise ValueError(f"CIFAR-100 {member} schema changed")
    values = np.asarray(payload[b"data"])
    labels = np.asarray(payload[b"fine_labels"], dtype=int)
    if values.shape != (expected_rows, 3072) or values.dtype != np.uint8 or labels.shape != (expected_rows,):
        raise ValueError(f"CIFAR-100 {member} source surface changed")
    if not np.all((values >= 0) & (values <= 255)) or set(labels) != set(FINE_LABELS):
        raise ValueError(f"CIFAR-100 {member} raw range or fine-label support changed")
    fingerprints = tuple(hashlib.sha256(row.tobytes()).hexdigest() for row in values)
    return Records(values, labels, fingerprints)


def load_records() -> tuple[Records, Records, dict[str, int]]:
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        inventory = tuple(member.name for member in archive.getmembers())
        if inventory != MEMBERS:
            raise ValueError("CIFAR-100 archive inventory changed")
        source = read_member(archive, "train", 50_000)
        external = read_member(archive, "test", 10_000)
        meta_handle = archive.extractfile(member_name("meta"))
        if meta_handle is None:
            raise ValueError("CIFAR-100 meta member is missing")
        fine_names = pickle.load(meta_handle, encoding="bytes").get(b"fine_label_names")
    if not isinstance(fine_names, list) or len(fine_names) != len(FINE_LABELS) or len(set(fine_names)) != len(FINE_LABELS):
        raise ValueError("CIFAR-100 fine-label metadata changed")
    ledger = {
        "source_complete_duplicates": len(source.fingerprints) - len(set(source.fingerprints)),
        "external_complete_duplicates": len(external.fingerprints) - len(set(external.fingerprints)),
        "cross_file_complete_duplicates": len(set(source.fingerprints) & set(external.fingerprints)),
    }
    if ledger != EXPECTED_DUPLICATES:
        raise ValueError("CIFAR-100 frozen duplicate ledger changed")
    return source, external, ledger


def grouped_split(records: Records) -> tuple[Records, Records, dict[str, int]]:
    groups: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(records.fingerprints):
        groups.setdefault(fingerprint, []).append(index)
    group_indices = list(groups.values())
    group_labels = np.asarray([records.labels[indices[0]] for indices in group_indices], dtype=int)
    if any(not np.all(records.labels[indices] == group_labels[group_index]) for group_index, indices in enumerate(group_indices)):
        raise ValueError("inconsistent fine labels in complete-image duplicate group")
    group_train, group_selection = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(group_labels)), group_labels)
    )
    train_indices = np.asarray([index for group_index in group_train for index in group_indices[group_index]], dtype=int)
    selection_indices = np.asarray([index for group_index in group_selection for index in group_indices[group_index]], dtype=int)
    if set(records.fingerprints[train_index] for train_index in train_indices) & set(records.fingerprints[selection_index] for selection_index in selection_indices):
        raise ValueError("duplicate group crossed the frozen source-train split")

    def subset(indices: np.ndarray) -> Records:
        return Records(records.values[indices], records.labels[indices], tuple(records.fingerprints[index] for index in indices))

    return subset(train_indices), subset(selection_indices), {
        "source_unique_complete_images": len(group_indices),
        "source_duplicate_groups": sum(len(indices) > 1 for indices in group_indices),
    }


class Model:
    def __init__(self, name: str, scaler: StandardScaler | None = None, estimator: object | None = None, probabilities: np.ndarray | None = None) -> None:
        self.name = name
        self.scaler = scaler
        self.estimator = estimator
        self.probabilities = probabilities

    def probability(self, values: np.ndarray) -> np.ndarray:
        if self.probabilities is not None:
            return np.broadcast_to(self.probabilities, (len(values), len(FINE_LABELS))).copy()
        assert self.scaler is not None and self.estimator is not None
        transformed = self.scaler.transform(values)
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(transformed)
        centroids = self.estimator.centroids_
        distances = 2.0 * transformed @ centroids.T - (centroids**2).sum(axis=1)
        distances -= distances.max(axis=1, keepdims=True)
        weights = np.exp(distances)
        return weights / weights.sum(axis=1, keepdims=True)

    def prediction(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(FINE_LABELS)[self.probability(values).argmax(axis=1)]


def fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        probabilities = np.array([(labels == label).sum() for label in FINE_LABELS], dtype=float)
        return Model(name, probabilities=probabilities / probabilities.sum())
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    if name.startswith("logreg_"):
        estimator = LogisticRegression(C=float(name.removeprefix("logreg_")), max_iter=2000, solver="lbfgs", random_state=seed).fit(transformed, labels)
        return Model(name, scaler, estimator)
    return Model(name, scaler, NearestCentroid().fit(transformed, labels))


def metric(model: Model, records: Records) -> dict[str, object]:
    probabilities = model.probability(records.values)
    if probabilities.shape != (len(records.labels), len(FINE_LABELS)) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("CIFAR-100 model emitted invalid probabilities")
    predictions = np.asarray(FINE_LABELS)[probabilities.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(records.labels, predictions, labels=FINE_LABELS, average="macro", zero_division=0)),
        "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }


def permute_8x8_blocks(values: np.ndarray) -> np.ndarray:
    images = values.reshape(len(values), 3, 4, BLOCK, 4, BLOCK)
    return np.roll(images, 1, axis=4).reshape(values.shape)


def rejects_payload(payload: dict[str, object]) -> bool:
    try:
        image_from_payload(payload)
    except ValueError:
        return True
    return False


def controls(name: str, train: Records, selection: Records, model: Model, score: float, ledger: dict[str, int]) -> dict[str, object]:
    paired_labels = np.random.default_rng(SEED).permutation(train.labels)
    label_pairing_drop = score - metric(fit(name, train.values, paired_labels), selection)["macro_f1"]
    block_pairing_drop = score - metric(model, Records(permute_8x8_blocks(selection.values), selection.labels, selection.fingerprints))["macro_f1"]
    ablation_drops: list[float] = []
    images = selection.values.reshape(len(selection.values), 3, 32, 32)
    for block_row in range(4):
        for block_column in range(4):
            ablated = images.copy()
            ablated[:, :, block_row * BLOCK : (block_row + 1) * BLOCK, block_column * BLOCK : (block_column + 1) * BLOCK] = 0
            ablation_drops.append(score - metric(model, Records(ablated.reshape(len(ablated), -1), selection.labels, selection.fingerprints))["macro_f1"])
    sample = selection.values[0].reshape(3, 32, 32)
    states: list[float] = []
    probability = np.full(len(FINE_LABELS), 1.0 / len(FINE_LABELS))
    predictions = predict_independent([image_from_payload({"values": sample})] * 3, lambda state, _values: states.append(state) or probability)
    reordered = predict_independent([image_from_payload({"values": sample})] * 3, lambda _state, _values: probability)
    return {
        "label_image_pairing_drop": label_pairing_drop,
        "pixel_block_pairing_drop": block_pairing_drop,
        "pixel_block_ablation_drops": ablation_drops,
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(predictions, reordered[::-1])),
        "target_rejected": rejects_payload({"values": sample, "target": 0}),
        "file_rejected": rejects_payload({"values": sample, "file": "test"}),
        "split_rejected": rejects_payload({"values": sample, "split": "external"}),
        "row_rejected": rejects_payload({"values": sample, "row": 0}),
        "metadata_rejected": rejects_payload({"values": sample, "filename": "example.png"}),
        "duplicate_group_rejected": rejects_payload({"values": sample, "duplicate_group": "group-1"}),
        "coarse_label_rejected": rejects_payload({"values": sample, "coarse_label": 0}),
        "duplicate_ledger_integrity": ledger == EXPECTED_DUPLICATES,
        "finite_probabilities": True,
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    source, external, ledger = load_records()
    try:
        train, selection, group_ledger = grouped_split(source)
    except ValueError as error:
        result: dict[str, object] = {
            "phase": 60,
            "status": "closed_negative_duplicate_group_gate",
            "reason": str(error),
            "source_revalidation": {"source_rows": len(source.labels), "external_rows": len(external.labels), **ledger},
            "models_fitted": [],
            "external_outcome_score_count": 0,
            "abc_smc_calls": 0,
            "llm_calls": 0,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    candidates = {name: metric(fit(name, train.values, train.labels), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-float(candidates[name]["macro_f1"]), MODELS.index(name)))
    model = fit(selected, train.values, train.labels)
    selection_score = float(candidates[selected]["macro_f1"])
    selection_controls = controls(selected, train, selection, model, selection_score, ledger)
    selection_passed = (
        selection_controls["label_image_pairing_drop"] >= 0.20
        and selection_controls["pixel_block_pairing_drop"] >= 0.20
        and max(selection_controls["pixel_block_ablation_drops"]) >= 0.05
        and all(bool(value) for key, value in selection_controls.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "duplicate_ledger_integrity", "finite_probabilities"})
    )
    result = {
        "phase": 60,
        "source_revalidation": {"source_rows": len(source.labels), "external_rows": len(external.labels), "train_rows": len(train.labels), "selection_rows": len(selection.labels), **ledger, **group_ledger},
        "models": candidates,
        "selected_model": selected,
        "selection_controls": selection_controls,
        "selection_passed": selection_passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not selection_passed:
        result.update(status="closed_negative_selection_gate", reason="fixed selection control failed; test was not scored")
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
        result.update(
            status="passed_observed_baseline_and_external_stability",
            external_outcome_score_count=1,
            external_seed_metrics=external_metrics,
            executed_seed_count=len(external_metrics),
            adaptive_distinct_yield_stopped=len(external_metrics) < 8,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

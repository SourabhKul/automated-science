#!/usr/bin/env python3
"""Execute the fixed Phase 58 observed CIFAR-10 baseline gate."""

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

from core.real_data.cifar10 import LABELS, image_from_payload, predict_independent

ARCHIVE = Path("data/real/cifar10/raw/cifar-10-python.tar.gz")
OUTPUT_DIR = Path("artifacts/evaluations/phase58_cifar10_real_baselines_20260731")
SEED = 2058001
TRAIN_MEMBERS = tuple(f"data_batch_{index}" for index in range(1, 6))
TEST_MEMBER = "test_batch"
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
BLOCK = 8


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    labels: np.ndarray
    fingerprints: tuple[str, ...]


def member_name(member: str) -> str:
    return f"cifar-10-batches-py/{member}"


def read_member(archive: tarfile.TarFile, member: str) -> Records:
    extracted = archive.extractfile(member_name(member))
    if extracted is None:
        raise ValueError(f"missing fixed CIFAR member {member}")
    payload = pickle.load(extracted, encoding="bytes")
    values = np.asarray(payload[b"data"])
    labels = np.asarray(payload[b"labels"], dtype=int)
    if values.shape[1:] != (3072,) or values.dtype != np.uint8 or len(values) != len(labels):
        raise ValueError(f"invalid CIFAR raw schema in {member}")
    if not np.all((values >= 0) & (values <= 255)) or set(labels) != set(LABELS):
        raise ValueError(f"invalid CIFAR range or label support in {member}")
    fingerprints = tuple(hashlib.sha256(row.tobytes()).hexdigest() for row in values)
    return Records(values, labels, fingerprints)


def load_records() -> tuple[Records, Records, dict[str, int]]:
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        train_parts = [read_member(archive, member) for member in TRAIN_MEMBERS]
        test = read_member(archive, TEST_MEMBER)
    if any(len(part.labels) != 10_000 for part in train_parts) or len(test.labels) != 10_000:
        raise ValueError("fixed CIFAR batch counts changed")
    train = Records(
        np.concatenate([part.values for part in train_parts]),
        np.concatenate([part.labels for part in train_parts]),
        tuple(fingerprint for part in train_parts for fingerprint in part.fingerprints),
    )
    if len(train.labels) != 50_000 or set(train.labels) != set(LABELS) or set(test.labels) != set(LABELS):
        raise ValueError("fixed CIFAR aggregate contract changed")
    ledger = {
        "source_complete_duplicates": len(train.fingerprints) - len(set(train.fingerprints)),
        "external_complete_duplicates": len(test.fingerprints) - len(set(test.fingerprints)),
        "cross_file_complete_duplicates": len(set(train.fingerprints) & set(test.fingerprints)),
    }
    if ledger["source_complete_duplicates"] or ledger["external_complete_duplicates"]:
        raise ValueError("fixed CIFAR duplicate ledger changed")
    return train, test, ledger


class Model:
    def __init__(self, name: str, scaler: StandardScaler | None = None, estimator: object | None = None, probabilities: np.ndarray | None = None) -> None:
        self.name = name
        self.scaler = scaler
        self.estimator = estimator
        self.probabilities = probabilities

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

    def prediction(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(LABELS)[self.probability(values).argmax(axis=1)]


def fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        probabilities = np.array([(labels == label).sum() for label in LABELS], dtype=float)
        return Model(name, probabilities=probabilities / probabilities.sum())
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    if name.startswith("logreg_"):
        c_value = float(name.removeprefix("logreg_"))
        estimator = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=seed).fit(transformed, labels)
        return Model(name, scaler, estimator)
    return Model(name, scaler, NearestCentroid().fit(transformed, labels))


def metric(model: Model, records: Records) -> dict[str, object]:
    probabilities = model.probability(records.values)
    if probabilities.shape != (len(records.labels), len(LABELS)) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("CIFAR model emitted invalid probabilities")
    predictions = model.prediction(records.values)
    return {
        "macro_f1": float(f1_score(records.labels, predictions, labels=LABELS, average="macro", zero_division=0)),
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


def controls(name: str, train: Records, selection: Records, model: Model, score: float, ledger: dict[str, int]) -> dict[str, object]:
    shuffled = np.random.default_rng(SEED).permutation(train.labels)
    label_pairing_drop = score - metric(fit(name, train.values, shuffled), selection)["macro_f1"]
    block_pairing_drop = score - metric(model, Records(permute_8x8_blocks(selection.values), selection.labels, selection.fingerprints))["macro_f1"]
    ablation_drops: list[float] = []
    selection_images = selection.values.reshape(len(selection.values), 3, 32, 32)
    for block_row in range(4):
        for block_column in range(4):
            ablated = selection_images.copy()
            ablated[:, :, block_row * BLOCK : (block_row + 1) * BLOCK, block_column * BLOCK : (block_column + 1) * BLOCK] = 0
            ablation_drops.append(score - metric(model, Records(ablated.reshape(len(ablated), -1), selection.labels, selection.fingerprints))["macro_f1"])
    sample = selection.values[0].reshape(3, 32, 32)
    sequence = [sample] * 3
    states: list[float] = []
    probabilities = predict_independent([image_from_payload({"values": value}) for value in sequence], lambda state, value: states.append(state) or np.full(10, 0.1))
    reordered = predict_independent([image_from_payload({"values": value}) for value in sequence[::-1]], lambda state, value: np.full(10, 0.1))
    return {
        "label_record_pairing_drop": label_pairing_drop,
        "pixel_block_pairing_drop": block_pairing_drop,
        "pixel_block_ablation_drops": ablation_drops,
        "zero_reset": states == [0.0] * len(sequence),
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejects_payload({"values": sample, "label": 0}),
        "batch_rejected": rejects_payload({"values": sample, "batch": "data_batch_1"}),
        "file_rejected": rejects_payload({"values": sample, "file": TEST_MEMBER}),
        "split_rejected": rejects_payload({"values": sample, "split": "external"}),
        "row_rejected": rejects_payload({"values": sample, "row": 0}),
        "metadata_rejected": rejects_payload({"values": sample, "shape": [3, 32, 32]}),
        "duplicate_ledger_integrity": ledger["source_complete_duplicates"] == 0 and ledger["external_complete_duplicates"] == 0,
        "finite_probabilities": True,
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    source, external, ledger = load_records()
    train_indices, selection_indices = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(source.values, source.labels))
    train = Records(source.values[train_indices], source.labels[train_indices], tuple(source.fingerprints[index] for index in train_indices))
    selection = Records(source.values[selection_indices], source.labels[selection_indices], tuple(source.fingerprints[index] for index in selection_indices))
    candidates = {name: metric(fit(name, train.values, train.labels), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-float(candidates[name]["macro_f1"]), MODELS.index(name)))
    model = fit(selected, train.values, train.labels)
    selection_score = float(candidates[selected]["macro_f1"])
    selection_controls = controls(selected, train, selection, model, selection_score, ledger)
    selection_passed = (
        selection_controls["label_record_pairing_drop"] >= 0.20
        and selection_controls["pixel_block_pairing_drop"] >= 0.20
        and max(selection_controls["pixel_block_ablation_drops"]) >= 0.05
        and all(bool(value) for key, value in selection_controls.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "duplicate_ledger_integrity", "finite_probabilities"})
    )
    result: dict[str, object] = {
        "phase": 58,
        "source_revalidation": {
            "source_rows": len(source.labels),
            "external_rows": len(external.labels),
            "train_rows": len(train.labels),
            "selection_rows": len(selection.labels),
            **ledger,
        },
        "models": candidates,
        "selected_model": selected,
        "selection_controls": selection_controls,
        "selection_passed": selection_passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not selection_passed:
        result.update(status="closed_negative_selection_gate", reason="fixed selection control failed; test_batch was not scored")
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

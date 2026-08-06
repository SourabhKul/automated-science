#!/usr/bin/env python3
"""Execute the fixed Phase 46 source-speaker observed MFCC baseline."""
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
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_spoken_arabic_digit import (
    DIGIT_LABELS,
    MFCC_COEFFICIENTS,
    PREFIX_FRAMES,
    prefix_from_payload,
)
from scripts.run_uci_spoken_arabic_digit_source_gate import (
    ARCHIVE,
    FILES,
    TEST,
    TRAIN,
    _source_speaker,
    parse_blocks,
)


SPLIT = Path("data/real/uci_spoken_arabic_digit/source_speaker_split.json")
OUT = Path("artifacts/evaluations/phase46_uci_spoken_arabic_digit_real_baselines_20260727")
SEED = 2046001
BLOCKS = (np.arange(0, 3), np.arange(3, 6), np.arange(6, 9), np.arange(9, 13))
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")


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
            return np.broadcast_to(self.prior, (len(x), len(DIGIT_LABELS))).copy()
        values = self.scaler.transform(x.reshape(len(x), -1))
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(values)
        distance = -np.sum((values[:, None, :] - self.estimator.centroids_[None, :, :]) ** 2, axis=2)
        distance -= distance.max(axis=1, keepdims=True)
        weights = np.exp(distance)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(DIGIT_LABELS)[np.argmax(self.predict_proba(x), axis=1)]


def _rows(blocks: list[np.ndarray], source: str) -> dict[int, Rows]:
    values: dict[int, list[np.ndarray]] = {}
    labels: dict[int, list[int]] = {}
    keys: dict[int, list[str]] = {}
    per_digit = 660 if source == TRAIN else 220
    for digit in DIGIT_LABELS:
        for within_digit, block in enumerate(blocks[digit * per_digit : (digit + 1) * per_digit]):
            speaker = _source_speaker(within_digit) if source == TRAIN else within_digit // 10 + 1
            if block.shape[1] != MFCC_COEFFICIENTS or block.shape[0] < PREFIX_FRAMES:
                raise ValueError("Spoken Arabic Digit locked block contract changed")
            values.setdefault(speaker, []).append(block[:PREFIX_FRAMES])
            labels.setdefault(speaker, []).append(digit)
            keys.setdefault(speaker, []).append(hashlib.sha256(block.tobytes()).hexdigest())
    result = {speaker: Rows(np.asarray(values[speaker]), np.asarray(labels[speaker]), tuple(keys[speaker])) for speaker in values}
    if set(result) != set(range(1, 67 if source == TRAIN else 23)):
        raise ValueError("Spoken Arabic Digit speaker coverage changed")
    if any(rows.x.shape != (100, PREFIX_FRAMES, MFCC_COEFFICIENTS) or set(rows.y) != set(DIGIT_LABELS) for rows in result.values()):
        raise ValueError("Spoken Arabic Digit digit/repetition coverage changed")
    return result


def load_locked_rows(archive_path: Path = ARCHIVE) -> tuple[dict[int, Rows], dict[int, Rows]]:
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != FILES:
            raise ValueError("Spoken Arabic Digit archive inventory changed")
        train = _rows(parse_blocks(archive.read(TRAIN), TRAIN), TRAIN)
        external = _rows(parse_blocks(archive.read(TEST), TEST), TEST)
    return train, external


def _combine(rows: dict[int, Rows], speakers: list[int]) -> Rows:
    return Rows(
        np.concatenate([rows[speaker].x for speaker in speakers]),
        np.concatenate([rows[speaker].y for speaker in speakers]),
        tuple(key for speaker in speakers for key in rows[speaker].keys),
    )


def _fit(name: str, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        prior = np.asarray([np.count_nonzero(y == label) for label in DIGIT_LABELS], dtype=float)
        return Model(name, prior=prior / prior.sum())
    scaler = StandardScaler().fit(x.reshape(len(x), -1))
    values = scaler.transform(x.reshape(len(x), -1))
    if name.startswith("logreg_"):
        return Model(name, scaler, LogisticRegression(C=float(name.split("_")[1]), max_iter=2000, solver="lbfgs", random_state=seed).fit(values, y))
    if name == "nearest_centroid":
        return Model(name, scaler, NearestCentroid().fit(values, y))
    raise ValueError("unlocked model")


def _metrics(model: Model, rows: Rows) -> dict[str, object]:
    probabilities = model.predict_proba(rows.x)
    if probabilities.shape != (len(rows.y), len(DIGIT_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Spoken Arabic Digit probabilities are invalid")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Spoken Arabic Digit probabilities are out of bounds")
    prediction = model.predict(rows.x)
    result: dict[str, object] = {
        "macro_f1": float(f1_score(rows.y, prediction, labels=DIGIT_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(rows.y, prediction)),
        "multiclass_log_loss": float(log_loss(rows.y, probabilities, labels=DIGIT_LABELS)),
        "prediction_fingerprint": hashlib.sha256(prediction.tobytes()).hexdigest(),
    }
    if not all(np.isfinite(value) for key, value in result.items() if key != "prediction_fingerprint"):
        raise ValueError("Spoken Arabic Digit metric is nonfinite")
    return result


def _reject(payload: dict[str, object]) -> bool:
    try:
        prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def _controls(name: str, train: Rows, selection: Rows, model: Model, macro_f1: float) -> dict[str, object]:
    paired = _metrics(_fit(name, train.x, np.random.default_rng(SEED).permutation(train.y)), selection)["macro_f1"]
    reversed_frames = _metrics(model, Rows(selection.x[:, ::-1], selection.y, selection.keys))["macro_f1"]
    permutation = np.concatenate((BLOCKS[-1], *BLOCKS[:-1]))
    paired_coefficients = _metrics(model, Rows(selection.x[:, :, permutation], selection.y, selection.keys))["macro_f1"]
    ablations = []
    for block in BLOCKS:
        values = selection.x.copy()
        values[:, :, block] = 0
        ablations.append(macro_f1 - _metrics(model, Rows(values, selection.y, selection.keys))["macro_f1"])
    reverse = np.arange(len(selection.y) - 1, -1, -1)
    candidate = selection.x[0]
    return {
        "label_utterance_pairing_drop": macro_f1 - paired,
        "four_frame_reversal_drop": macro_f1 - reversed_frames,
        "coefficient_block_pairing_drop": macro_f1 - paired_coefficients,
        "coefficient_block_ablation_drops": ablations,
        "row_reorder_invariant": bool(np.array_equal(model.predict(selection.x), model.predict(selection.x[reverse])[::-1])),
        "target_isolation_sentinel_rejected": _reject({"frames": candidate, "label": 0}),
        "speaker_isolation_sentinel_rejected": _reject({"frames": candidate, "speaker_id": 1}),
        "file_isolation_sentinel_rejected": _reject({"frames": candidate, "source_file": TRAIN}),
        "block_isolation_sentinel_rejected": _reject({"frames": candidate, "block_index": 0}),
        "suffix_isolation_sentinel_rejected": _reject({"frames": candidate, "suffix": np.zeros((1, MFCC_COEFFICIENTS))}),
    }


def run(archive_path: Path = ARCHIVE, split_path: Path = SPLIT, output_dir: Path = OUT) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    expected = {
        "train_male_speakers": list(range(1, 23)),
        "train_female_speakers": list(range(34, 56)),
        "selection_male_speakers": list(range(23, 34)),
        "selection_female_speakers": list(range(56, 67)),
        "external_official_test_speakers": list(range(1, 23)),
    }
    if split != expected:
        raise ValueError("Spoken Arabic Digit frozen speaker split changed")
    source, external_source = load_locked_rows(archive_path)
    train_speakers = split["train_male_speakers"] + split["train_female_speakers"]
    selection_speakers = split["selection_male_speakers"] + split["selection_female_speakers"]
    train, selection = _combine(source, train_speakers), _combine(source, selection_speakers)
    external = _combine(external_source, split["external_official_test_speakers"])
    all_fingerprints = {"train": set(train.keys), "selection": set(selection.keys), "external": set(external.keys)}
    crossings = sum(len(all_fingerprints[left] & all_fingerprints[right]) for left, right in (("train", "selection"), ("train", "external"), ("selection", "external")))
    if crossings:
        raise ValueError("complete utterance duplicate crosses frozen partition")
    candidates = {name: _metrics(_fit(name, train.x, train.y), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-candidates[name]["macro_f1"], MODELS.index(name)))
    model = _fit(selected, train.x, train.y)
    selection_f1 = candidates[selected]["macro_f1"]
    controls = _controls(selected, train, selection, model, selection_f1)
    selection_passed = all([
        selection_f1 >= 0.60,
        selection_f1 - candidates["majority"]["macro_f1"] >= 0.50,
        controls["label_utterance_pairing_drop"] >= 0.30,
        controls["four_frame_reversal_drop"] >= 0.05,
        controls["coefficient_block_pairing_drop"] >= 0.05,
        max(controls["coefficient_block_ablation_drops"]) >= 0.03,
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key == "row_reorder_invariant"],
    ])
    result: dict[str, object] = {
        "phase": 46,
        "source_revalidation": {
            "source_train_blocks": 6600,
            "external_blocks": 2200,
            "train_blocks": len(train.y),
            "selection_blocks": len(selection.y),
            "cross_partition_complete_duplicates": crossings,
        },
        "models": candidates,
        "selected_model": selected,
        "selection_controls": controls,
        "selection_passed": selection_passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not selection_passed:
        result.update({"status": "closed_negative_selection_gate", "reason": "fixed selection or control gate failed; official test was not scored"})
    else:
        full_train = _combine(source, train_speakers + selection_speakers)
        metrics, fingerprints = [], []
        for seed in range(SEED, SEED + 8):
            full_model = _fit(selected, full_train.x, full_train.y, seed)
            metric = _metrics(full_model, external)
            metric["seed"] = seed
            metrics.append(metric)
            fingerprints.append((metric["prediction_fingerprint"], hashlib.sha256(full_model.predict(full_train.x).tobytes()).hexdigest()))
            if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
                break
        primary = metrics[0]
        external_passed = primary["macro_f1"] >= 0.55 and primary["macro_f1"] >= 0.85 * selection_f1
        result.update({
            "status": "passed_observed_baseline_and_external_stability" if external_passed else "closed_negative_external_stability",
            "external_outcome_score_count": 1,
            "external_seed_metrics": metrics,
            "executed_seed_count": len(metrics),
            "adaptive_distinct_yield_stopped": len(metrics) < 8,
            "external_passed": external_passed,
            "reason": "fixed one-operation official-test stability completed" if external_passed else "fixed official-test metric stop failed",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

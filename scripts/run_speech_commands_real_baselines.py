#!/usr/bin/env python3
"""Execute the fixed Phase 68 observed Speech Commands baseline gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import wave

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.speech_commands import CLASS_COUNT, PCM_BOUNDS, SAMPLES, predict_independent, waveform_from_payload
from scripts.run_speech_commands_source_gate import (
    ARCHIVE,
    ARCHIVE_PREFIX,
    BACKGROUND_DIRECTORY,
    WAV_FILENAME,
    member_path,
    parse_list,
)


OUT = Path("artifacts/evaluations/phase68_tensorflow_speech_commands_v0_02_raw_waveform_real_baselines_20260803")
SEED = 2_068_001
LABELS = tuple(range(CLASS_COUNT))
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")
BLOCK = 800


class GateFailure(ValueError):
    """A fixed scientific contract failed."""


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    labels: np.ndarray
    groups: np.ndarray


class Model:
    def __init__(self, name: str, scaler: StandardScaler | None = None, estimator: object | None = None, probabilities: np.ndarray | None = None) -> None:
        self.name, self.scaler, self.estimator, self.probabilities = name, scaler, estimator, probabilities

    def probability(self, values: np.ndarray) -> np.ndarray:
        if self.probabilities is not None:
            return np.broadcast_to(self.probabilities, (len(values), CLASS_COUNT)).copy()
        assert self.scaler is not None and self.estimator is not None
        transformed = self.scaler.transform(values.astype(np.float32, copy=False))
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(transformed)
        centroids = self.estimator
        distances = (
            np.square(transformed).sum(axis=1, keepdims=True)
            + np.square(centroids).sum(axis=1)[None, :]
            - 2.0 * transformed @ centroids.T
        )
        shifted = -(distances - distances.min(axis=1, keepdims=True)) / max(float(np.median(distances)), 1.0)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=1, keepdims=True)


def fit(name: str, values: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        counts = np.bincount(labels, minlength=CLASS_COUNT).astype(float)
        return Model(name, probabilities=counts / counts.sum())
    scaler = StandardScaler().fit(values.astype(np.float32, copy=False))
    transformed = scaler.transform(values.astype(np.float32, copy=False))
    if name.startswith("logreg_"):
        estimator = LogisticRegression(
            C=float(name.removeprefix("logreg_")),
            max_iter=500,
            solver="saga",
            random_state=seed,
        ).fit(transformed, labels)
        return Model(name, scaler, estimator)
    return Model(name, scaler, np.asarray([transformed[labels == label].mean(axis=0) for label in LABELS]))


def metric(model: Model, records: Records) -> dict[str, object]:
    probabilities = model.probability(records.values)
    if (
        probabilities.shape != (len(records.labels), CLASS_COUNT)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise GateFailure("model emitted invalid 35-class probabilities")
    predictions = probabilities.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(records.labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }


def label_for_member(name: str, label_index: dict[str, int]) -> tuple[str, int]:
    relative = name.removeprefix(ARCHIVE_PREFIX)
    parts = relative.split("/")
    if len(parts) != 2 or parts[0] not in label_index or WAV_FILENAME.fullmatch(parts[1]) is None:
        raise GateFailure(f"invalid supervised member grammar: {name}")
    return relative, label_index[parts[0]]


def revalidate() -> tuple[dict[int, Records], dict[str, object]]:
    labels = json.loads(Path("data/real/tensorflow_speech_commands/manifest.json").read_text())["word_directories"]
    label_index = {label: index for index, label in enumerate(labels)}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = archive.getmembers()
        validation_raw = archive.extractfile(member_path("validation_list.txt"))
        testing_raw = archive.extractfile(member_path("testing_list.txt"))
        if validation_raw is None or testing_raw is None:
            raise GateFailure("creator list member unavailable during revalidation")
        validation_paths = set(parse_list(validation_raw.read(), "validation_list.txt"))
        testing_paths = set(parse_list(testing_raw.read(), "testing_list.txt"))
        if validation_paths & testing_paths:
            raise GateFailure("creator list overlap changed during revalidation")
        candidates = [
            member
            for member in members
            if member.name.startswith(ARCHIVE_PREFIX)
            and member.name.endswith(".wav")
            and member.name.removeprefix(ARCHIVE_PREFIX).split("/", 1)[0] != BACKGROUND_DIRECTORY
        ]
        values = np.empty((len(candidates), SAMPLES), dtype=np.int16)
        outcomes = np.empty(len(candidates), dtype=np.int16)
        partitions = np.empty(len(candidates), dtype=np.int8)
        groups: list[bytes] = []
        rejection_counts: dict[str, int] = {}
        group_rows: dict[bytes, list[int]] = {}
        for row, member in enumerate(sorted(candidates, key=lambda item: item.offset_data)):
            relative, label = label_for_member(member.name, label_index)
            partition = 1 if relative in validation_paths else 2 if relative in testing_paths else 0
            extracted = archive.extractfile(member)
            if extracted is None:
                raise GateFailure(f"unreadable supervised member: {relative}")
            payload = extracted.read()
            try:
                with wave.open(io.BytesIO(payload), "rb") as audio:
                    header = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype(), audio.getnframes())
                    raw = audio.readframes(SAMPLES)
            except (wave.Error, EOFError) as exc:
                raise GateFailure(f"malformed WAV: {relative}") from exc
            if header[:4] != (1, 2, 16_000, "NONE") or header[4] < SAMPLES or len(raw) != 2 * SAMPLES:
                rejection_counts["source_contract"] = rejection_counts.get("source_contract", 0) + 1
                raise GateFailure(f"source WAV is ineligible: {relative}")
            prefix = np.frombuffer(raw, dtype="<i2")
            if prefix.shape != (SAMPLES,) or np.any(prefix < PCM_BOUNDS[0]) or np.any(prefix > PCM_BOUNDS[1]):
                raise GateFailure(f"candidate prefix bounds changed: {relative}")
            digest = hashlib.sha256(raw).digest()
            values[row], outcomes[row], partitions[row] = prefix, label, partition
            groups.append(digest)
            group_rows.setdefault(digest, []).append(row)

    inconsistent = 0
    crossing = 0
    for rows in group_rows.values():
        if len({int(outcomes[row]) for row in rows}) != 1:
            inconsistent += 1
        if len({int(partitions[row]) for row in rows}) != 1:
            crossing += 1
    if inconsistent:
        raise GateFailure(f"{inconsistent} candidate-prefix duplicate groups have inconsistent labels")
    if crossing:
        raise GateFailure(f"{crossing} candidate-prefix duplicate groups cross creator fitted/evaluation boundaries")
    if set(outcomes[partitions == 0]) != set(LABELS) or set(outcomes[partitions == 1]) != set(LABELS) or set(outcomes[partitions == 2]) != set(LABELS):
        raise GateFailure("one creator partition lacks full 35-class support")

    records: dict[int, Records] = {}
    for partition in (0, 1, 2):
        mask = partitions == partition
        unique = {digest: index for index, digest in enumerate(sorted(set(groups[row] for row in np.flatnonzero(mask))))}
        records[partition] = Records(values[mask], outcomes[mask], np.asarray([unique[groups[row]] for row in np.flatnonzero(mask)], dtype=int))
    ledger = {
        "supervised_rows": len(values),
        "source_train_rows": int(np.count_nonzero(partitions == 0)),
        "creator_validation_rows": int(np.count_nonzero(partitions == 1)),
        "creator_test_rows": int(np.count_nonzero(partitions == 2)),
        "candidate_prefix_duplicate_groups": sum(len(rows) > 1 for rows in group_rows.values()),
        "candidate_prefix_cross_boundary_groups": crossing,
        "label_inconsistent_duplicate_groups": inconsistent,
        "rejection_counts": rejection_counts,
        "creator_test_scored": False,
    }
    return records, ledger


def grouped_stratified_split(records: Records) -> tuple[np.ndarray, np.ndarray]:
    selected_groups: set[int] = set()
    generator = np.random.default_rng(SEED)
    for label in LABELS:
        groups = np.unique(records.groups[records.labels == label])
        generator.shuffle(groups)
        target = 0.2 * np.count_nonzero(records.labels == label)
        count = 0
        for group in groups:
            selected_groups.add(int(group))
            count += np.count_nonzero(records.groups == group)
            if count >= target:
                break
    selection = np.flatnonzero(np.isin(records.groups, list(selected_groups)))
    train = np.flatnonzero(~np.isin(records.groups, list(selected_groups)))
    if not len(selection) or not len(train) or set(records.labels[selection]) != set(LABELS) or set(records.labels[train]) != set(LABELS):
        raise GateFailure("fixed duplicate-group-aware source-train split lacks class support")
    if set(records.groups[selection]) & set(records.groups[train]):
        raise GateFailure("source-train duplicate group crossed frozen internal split")
    return train, selection


def pair_blocks(values: np.ndarray) -> np.ndarray:
    paired = values.copy()
    for block in range(SAMPLES // BLOCK):
        permutation = np.random.default_rng(SEED + block + 1).permutation(len(values))
        begin = block * BLOCK
        paired[:, begin : begin + BLOCK] = values[permutation, begin : begin + BLOCK]
    return paired


def rejected(payload: dict[str, object]) -> bool:
    try:
        waveform_from_payload(payload)
    except ValueError:
        return True
    return False


def controls(name: str, train: Records, selection: Records, model: Model, baseline: float) -> dict[str, object]:
    paired_labels = np.random.default_rng(SEED).permutation(train.labels)
    label_drop = baseline - float(metric(fit(name, train.values, paired_labels), selection)["macro_f1"])
    paired = Records(pair_blocks(selection.values), selection.labels, selection.groups)
    block_drop = baseline - float(metric(model, paired)["macro_f1"])
    shifted = Records(np.roll(selection.values, BLOCK, axis=1), selection.labels, selection.groups)
    time_drop = baseline - float(metric(model, shifted)["macro_f1"])
    ablations = []
    for block in range(SAMPLES // BLOCK):
        ablated = selection.values.copy()
        begin = block * BLOCK
        ablated[:, begin : begin + BLOCK] = 0
        ablations.append(baseline - float(metric(model, Records(ablated, selection.labels, selection.groups))["macro_f1"]))
    sample = selection.values[0]
    states: list[float] = []
    uniform = np.full(CLASS_COUNT, 1.0 / CLASS_COUNT)
    probabilities = predict_independent([waveform_from_payload({"samples": sample})] * 3, lambda state, values: states.append(state) or uniform)
    reordered = predict_independent([waveform_from_payload({"samples": sample})] * 3, lambda state, values: uniform)
    return {
        "label_window_pairing_drop": label_drop,
        "waveform_800_block_pairing_drop": block_drop,
        "circular_800_time_order_drop": time_drop,
        "block_ablation_drops": ablations,
        "zero_reset": states == [0.0] * 3,
        "reorder_invariant": bool(np.array_equal(probabilities, reordered)),
        "target_rejected": rejected({"samples": sample, "label": 0}),
        "file_rejected": rejected({"samples": sample, "file": "testing_list.txt"}),
        "split_rejected": rejected({"samples": sample, "split": "external"}),
        "row_rejected": rejected({"samples": sample, "row": 0}),
        "path_rejected": rejected({"samples": sample, "path": "yes/0123abcd_nohash_0.wav"}),
        "speaker_rejected": rejected({"samples": sample, "speaker_hash": "0123abcd"}),
        "utterance_rejected": rejected({"samples": sample, "utterance_index": 0}),
        "suffix_rejected": rejected({"samples": sample, "suffix": sample}),
        "duplicate_rejected": rejected({"samples": sample, "duplicate_group": 0}),
        "finite_probabilities": bool(np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)),
    }


def execute() -> dict[str, object]:
    partitions, ledger = revalidate()
    source = partitions[0]
    train_idx, selection_idx = grouped_stratified_split(source)
    train = Records(source.values[train_idx], source.labels[train_idx], source.groups[train_idx])
    selection = Records(source.values[selection_idx], source.labels[selection_idx], source.groups[selection_idx])
    candidates = {name: metric(fit(name, train.values, train.labels), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-float(candidates[name]["macro_f1"]), MODELS.index(name)))
    selection_f1 = float(candidates[selected]["macro_f1"])
    majority_f1 = float(candidates["majority"]["macro_f1"])
    selected_model = fit(selected, train.values, train.labels)
    control = controls(selected, train, selection, selected_model, selection_f1)
    passed = (
        selection_f1 >= 0.20
        and selection_f1 - majority_f1 >= 0.10
        and control["label_window_pairing_drop"] >= 0.15
        and control["waveform_800_block_pairing_drop"] >= 0.05
        and control["circular_800_time_order_drop"] >= 0.03
        and max(control["block_ablation_drops"]) >= 0.02
        and all(bool(value) for key, value in control.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "finite_probabilities"})
    )
    result: dict[str, object] = {
        "phase": 68, "source_revalidation": ledger | {"train_rows": len(train.labels), "selection_rows": len(selection.labels)},
        "models": candidates, "selected_model": selected, "selection_controls": control,
        "selection_passed": passed, "creator_validation_score_count": 0, "creator_test_score_count": 0,
        "abc_smc_calls": 0, "llm_calls": 0,
    }
    if not passed:
        return result | {"status": "closed_negative_selection_gate", "reason": "fixed internal selection gate failed; creator validation/test were not scored"}
    validation = partitions[1]
    validation_metric = metric(fit(selected, source.values, source.labels), validation)
    validation_passed = float(validation_metric["macro_f1"]) >= 0.20 and float(validation_metric["macro_f1"]) >= 0.5 * selection_f1
    result.update(creator_validation_score_count=1, creator_validation=validation_metric, creator_validation_passed=validation_passed)
    if not validation_passed:
        return result | {"status": "closed_negative_creator_validation_gate", "reason": "fixed creator validation stability gate failed; creator test was not scored"}
    combined = Records(np.concatenate([source.values, validation.values]), np.concatenate([source.labels, validation.labels]), np.concatenate([source.groups, validation.groups]))
    external_metrics, fingerprints = [], []
    for seed in range(SEED, SEED + 8):
        current = metric(fit(selected, combined.values, combined.labels, seed), partitions[2])
        current["seed"] = seed
        external_metrics.append(current)
        fingerprints.append(str(current["prediction_fingerprint"]))
        if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
            break
    return result | {
        "status": "passed_observed_baseline_and_creator_test_stability",
        "creator_test_score_count": 1, "creator_test_seed_metrics": external_metrics,
        "executed_seed_count": len(external_metrics), "adaptive_distinct_yield_stopped": len(external_metrics) < 8,
    }


def run(output_dir: Path = OUT) -> dict[str, object]:
    try:
        result = execute()
    except GateFailure as exc:
        result = {"phase": 68, "status": "closed_negative_revalidation_gate", "reason": str(exc), "creator_validation_score_count": 0, "creator_test_score_count": 0, "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

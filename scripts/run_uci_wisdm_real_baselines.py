#!/usr/bin/env python3
"""Execute the fixed Phase 50 WISDM participant-heldout observed baseline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
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

from core.real_data.uci_wisdm import ACTIVITY_LABELS, FULL_SEGMENT_SAMPLES, PREFIX_SAMPLES, SOURCE_AXES, prefix_from_payload
from scripts.run_uci_wisdm_source_gate import ARCHIVE, PHONE_ACCEL, _nested_archive, _stream_members


SPLIT = Path("data/real/uci_wisdm/source_subject_split.json")
OUT = Path("artifacts/evaluations/phase50_uci_wisdm_real_baselines_20260728")
SEED = 2050001
MODELS = ("majority", "logreg_0.01", "logreg_0.1", "logreg_1", "nearest_centroid")


@dataclass(frozen=True)
class Rows:
    x: np.ndarray
    y: np.ndarray
    keys: tuple[str, ...]


class Model:
    def __init__(self, name: str, scaler: StandardScaler | None = None, estimator: object | None = None, prior: np.ndarray | None = None):
        self.name, self.scaler, self.estimator, self.prior = name, scaler, estimator, prior

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.prior is not None:
            return np.broadcast_to(self.prior, (len(x), len(ACTIVITY_LABELS))).copy()
        values = self.scaler.transform(x.reshape(len(x), -1))
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(values)
        probabilities = np.empty((len(values), len(ACTIVITY_LABELS)), dtype=float)
        for start in range(0, len(values), 4096):
            batch = values[start : start + 4096]
            distances = -np.sum((batch[:, None, :] - self.estimator.centroids_[None, :, :]) ** 2, axis=2)
            distances -= distances.max(axis=1, keepdims=True)
            weights = np.exp(distances)
            probabilities[start : start + len(batch)] = weights / weights.sum(axis=1, keepdims=True)
        return probabilities


def _flush_run(values: list[np.ndarray], raw_rows: list[str], label: str, prefixes: list[np.ndarray], labels: list[str], keys: list[str]) -> int:
    complete = len(values) // FULL_SEGMENT_SAMPLES
    for segment in range(complete):
        start = segment * FULL_SEGMENT_SAMPLES
        full = np.asarray(values[start : start + FULL_SEGMENT_SAMPLES], dtype=float)
        if full.shape != (FULL_SEGMENT_SAMPLES, SOURCE_AXES) or not np.all(np.isfinite(full)):
            raise ValueError("WISDM complete segment violates the locked raw surface")
        prefixes.append(full[:PREFIX_SAMPLES])
        labels.append(label)
        keys.append(hashlib.sha256("\n".join(raw_rows[start : start + FULL_SEGMENT_SAMPLES]).encode("utf-8")).hexdigest())
    return complete


def _parse_member(archive: ZipFile, subject: int, member: str) -> tuple[Rows, dict[str, int]]:
    prefixes: list[np.ndarray] = []
    labels: list[str] = []
    keys: list[str] = []
    rejections = {"malformed_rows": 0, "nonfinite_axes": 0, "nonincreasing_same_activity_transitions": 0, "activity_changes": 0, "incomplete_records": 0}
    run_values: list[np.ndarray] = []
    run_rows: list[str] = []
    previous_activity: str | None = None
    previous_timestamp: int | None = None
    raw_rows = 0
    for line_number, line in enumerate(io.TextIOWrapper(archive.open(member), encoding="utf-8", newline=""), start=1):
        raw = line.strip()
        if not raw.endswith(";"):
            rejections["malformed_rows"] += 1
            raise ValueError(f"{member} line {line_number} violates the fixed semicolon grammar")
        fields = raw[:-1].split(",")
        if len(fields) != 6:
            rejections["malformed_rows"] += 1
            raise ValueError(f"{member} line {line_number} has the wrong field count")
        try:
            parsed_subject, activity, timestamp = int(fields[0]), fields[1], int(fields[2])
            axes = np.asarray([float(value) for value in fields[3:]], dtype=float)
        except ValueError as exc:
            rejections["malformed_rows"] += 1
            raise ValueError(f"{member} line {line_number} is not numeric under the locked schema") from exc
        if parsed_subject != subject or activity not in ACTIVITY_LABELS or timestamp <= 0:
            raise ValueError(f"{member} line {line_number} violates fixed subject/activity/timestamp ownership")
        if not np.all(np.isfinite(axes)):
            rejections["nonfinite_axes"] += 1
            raise ValueError(f"{member} line {line_number} has a nonfinite axis")
        continuing = previous_activity == activity and previous_timestamp is not None and timestamp > previous_timestamp
        if not continuing and previous_activity is not None:
            if activity == previous_activity:
                rejections["nonincreasing_same_activity_transitions"] += 1
            else:
                rejections["activity_changes"] += 1
            rejections["incomplete_records"] += len(run_values) % FULL_SEGMENT_SAMPLES
            _flush_run(run_values, run_rows, previous_activity, prefixes, labels, keys)
            run_values, run_rows = [], []
        run_values.append(axes)
        run_rows.append(raw)
        previous_activity, previous_timestamp = activity, timestamp
        raw_rows += 1
    if previous_activity is None:
        raise ValueError(f"{member} is empty")
    rejections["incomplete_records"] += len(run_values) % FULL_SEGMENT_SAMPLES
    _flush_run(run_values, run_rows, previous_activity, prefixes, labels, keys)
    rows = Rows(np.asarray(prefixes, dtype=float), np.asarray(labels), tuple(keys))
    if rows.x.ndim != 3 or rows.x.shape[1:] != (PREFIX_SAMPLES, SOURCE_AXES) or not np.all(np.isfinite(rows.x)):
        raise ValueError(f"{member} failed locked candidate construction")
    if not set(rows.y).issubset(ACTIVITY_LABELS) or not len(rows.y):
        raise ValueError(f"{member} lost fixed activity support")
    return rows, {"raw_rows": raw_rows, **rejections, "complete_segments": len(rows.y), "within_member_complete_duplicates": len(rows.keys) - len(set(rows.keys))}


def load_locked_rows(archive_path: Path = ARCHIVE) -> tuple[dict[int, Rows], dict[str, object]]:
    rows: dict[int, Rows] = {}
    ledger: dict[str, object] = {"members": {}, "all_raw_members": 0}
    with ZipFile(archive_path) as outer:
        inner = _nested_archive(outer)
        members = _stream_members(inner, PHONE_ACCEL)
        for subject, member in sorted(members.items()):
            subject_rows, subject_ledger = _parse_member(inner, subject, member)
            rows[subject] = subject_rows
            ledger["members"][str(subject)] = subject_ledger
    if tuple(sorted(rows)) != tuple(range(1600, 1651)):
        raise ValueError("WISDM source member set changed")
    ledger["all_raw_members"] = len(rows)
    return rows, ledger


def _combine(rows: dict[int, Rows], subjects: list[int]) -> Rows:
    return Rows(
        np.concatenate([rows[subject].x for subject in subjects]),
        np.concatenate([rows[subject].y for subject in subjects]),
        tuple(key for subject in subjects for key in rows[subject].keys),
    )


def _fit(name: str, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> Model:
    if set(y) != set(ACTIVITY_LABELS):
        raise ValueError("WISDM fitting partition lacks fixed 18-state label support")
    if name == "majority":
        prior = np.asarray([np.count_nonzero(y == label) for label in ACTIVITY_LABELS], dtype=float)
        return Model(name, prior=prior / prior.sum())
    scaler = StandardScaler().fit(x.reshape(len(x), -1))
    values = scaler.transform(x.reshape(len(x), -1))
    if name.startswith("logreg_"):
        estimator = LogisticRegression(C=float(name.split("_")[1]), max_iter=5000, solver="lbfgs", random_state=seed).fit(values, y)
        if tuple(estimator.classes_) != ACTIVITY_LABELS:
            raise ValueError("WISDM logistic model lacks fixed 18-state support")
        return Model(name, scaler, estimator)
    if name == "nearest_centroid":
        return Model(name, scaler, NearestCentroid().fit(values, y))
    raise ValueError("unlocked model")


def _metrics(model: Model, rows: Rows) -> dict[str, object]:
    probabilities = model.predict_proba(rows.x)
    if probabilities.shape != (len(rows.y), len(ACTIVITY_LABELS)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("WISDM model emitted invalid probabilities")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("WISDM model emitted out-of-bounds probabilities")
    predictions = np.asarray(ACTIVITY_LABELS)[np.argmax(probabilities, axis=1)]
    result: dict[str, object] = {
        "macro_f1": float(f1_score(rows.y, predictions, labels=ACTIVITY_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(rows.y, predictions)),
        "multiclass_log_loss": float(log_loss(rows.y, probabilities, labels=ACTIVITY_LABELS)),
        "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }
    if not all(np.isfinite(value) for key, value in result.items() if key != "prediction_fingerprint"):
        raise ValueError("WISDM metric is nonfinite")
    return result


def _reject(payload: dict[str, object]) -> bool:
    try:
        prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def _controls(name: str, train: Rows, selection: Rows, model: Model, macro_f1: float) -> dict[str, object]:
    paired = _metrics(_fit(name, train.x, np.random.default_rng(SEED).permutation(train.y)), selection)["macro_f1"]
    shifted = _metrics(model, Rows(np.roll(selection.x, 16, axis=1), selection.y, selection.keys))["macro_f1"]
    axis_paired = _metrics(model, Rows(selection.x[:, :, (1, 2, 0)], selection.y, selection.keys))["macro_f1"]
    ablations = []
    for axis in range(SOURCE_AXES):
        values = selection.x.copy()
        values[:, :, axis] = 0.0
        ablations.append(macro_f1 - _metrics(model, Rows(values, selection.y, selection.keys))["macro_f1"])
    reverse = np.arange(len(selection.y) - 1, -1, -1)
    candidate = selection.x[0]
    return {
        "label_segment_pairing_drop": macro_f1 - paired,
        "time_order_16_sample_shift_drop": macro_f1 - shifted,
        "axis_pairing_drop": macro_f1 - axis_paired,
        "axis_ablation_drops": ablations,
        "row_reorder_invariant": bool(np.array_equal(np.asarray(ACTIVITY_LABELS)[np.argmax(model.predict_proba(selection.x), axis=1)], np.asarray(ACTIVITY_LABELS)[np.argmax(model.predict_proba(selection.x[reverse]), axis=1)][::-1])),
        "target_isolation_sentinel_rejected": _reject({"prefix": candidate, "activity_label": selection.y[0]}),
        "participant_isolation_sentinel_rejected": _reject({"prefix": candidate, "subject_id": 1631}),
        "path_isolation_sentinel_rejected": _reject({"prefix": candidate, "member_path": "raw/phone/accel/data_1631_accel_phone.txt"}),
        "timestamp_isolation_sentinel_rejected": _reject({"prefix": candidate, "timestamps": np.arange(PREFIX_SAMPLES)}),
        "suffix_isolation_sentinel_rejected": _reject({"prefix": candidate, "suffix": np.zeros((FULL_SEGMENT_SAMPLES - PREFIX_SAMPLES, SOURCE_AXES))}),
        "split_isolation_sentinel_rejected": _reject({"prefix": candidate, "split": "selection_subjects"}),
        "probabilities_finite_and_bounded": True,
    }


def run(archive_path: Path = ARCHIVE, split_path: Path = SPLIT, output_dir: Path = OUT) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    expected = {"train_subjects": list(range(1600, 1631)), "selection_subjects": list(range(1631, 1641)), "external_subjects": list(range(1641, 1651)), "source_stream": "phone_accelerometer", "segment_samples": 128, "candidate_prefix_samples": 64}
    if split != expected:
        raise ValueError("WISDM frozen participant split changed")
    partitions, ledger = load_locked_rows(archive_path)
    train = _combine(partitions, split["train_subjects"])
    selection = _combine(partitions, split["selection_subjects"])
    external = _combine(partitions, split["external_subjects"])
    all_keys = {"train": set(train.keys), "selection": set(selection.keys), "external": set(external.keys)}
    crossings = sum(len(all_keys[left] & all_keys[right]) for left, right in (("train", "selection"), ("train", "external"), ("selection", "external")))
    candidates = {name: _metrics(_fit(name, train.x, train.y), selection) for name in MODELS}
    selected = min(MODELS, key=lambda name: (-candidates[name]["macro_f1"], MODELS.index(name)))
    model = _fit(selected, train.x, train.y)
    selection_f1 = candidates[selected]["macro_f1"]
    controls = _controls(selected, train, selection, model, selection_f1)
    selection_passed = all([
        crossings == 0,
        selection_f1 >= 0.45,
        selection_f1 - candidates["majority"]["macro_f1"] >= 0.39,
        controls["label_segment_pairing_drop"] >= 0.40,
        controls["time_order_16_sample_shift_drop"] >= 0.05,
        controls["axis_pairing_drop"] >= 0.05,
        max(controls["axis_ablation_drops"]) >= 0.02,
        *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key in {"row_reorder_invariant", "probabilities_finite_and_bounded"}],
    ])
    result: dict[str, object] = {
        "phase": 50,
        "source_revalidation": {
            **ledger,
            "train_segments": len(train.y),
            "selection_segments": len(selection.y),
            "external_segments_revalidated_not_scored": len(external.y),
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
        result.update({"status": "closed_negative_selection_gate", "reason": "fixed selection or control gate failed; external participants were not scored"})
    else:
        full_train = _combine(partitions, split["train_subjects"] + split["selection_subjects"])
        metrics, fingerprints = [], []
        for seed in range(SEED, SEED + 8):
            full_model = _fit(selected, full_train.x, full_train.y, seed)
            metric = _metrics(full_model, external)
            metric["seed"] = seed
            metrics.append(metric)
            fingerprints.append((metric["prediction_fingerprint"], hashlib.sha256(np.asarray(ACTIVITY_LABELS)[np.argmax(full_model.predict_proba(full_train.x), axis=1)].tobytes()).hexdigest()))
            if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
                break
        primary = metrics[0]
        external_passed = primary["macro_f1"] >= 0.40 and primary["macro_f1"] >= 0.80 * selection_f1
        result.update({
            "status": "passed_observed_baseline_and_external_stability" if external_passed else "closed_negative_external_stability",
            "external_outcome_score_count": 1,
            "external_seed_metrics": metrics,
            "executed_seed_count": len(metrics),
            "adaptive_distinct_yield_stopped": len(metrics) < 8,
            "external_passed": external_passed,
            "reason": "fixed one-operation external stability completed" if external_passed else "fixed external metric stop failed",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

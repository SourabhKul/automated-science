#!/usr/bin/env python3
"""Execute the fixed Phase 62 UCI HIGGS observed baseline gate."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import sys
from pathlib import Path
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_higgs import FEATURES, predict_independent, record_from_payload
from scripts.run_uci_higgs_source_gate import EXTERNAL_ROWS, MEMBER, SOURCE_ROWS, TOTAL_ROWS, parse_line


ARCHIVE = Path("data/real/uci_higgs/raw/higgs.zip")
OUT = Path("artifacts/evaluations/phase62_uci_higgs_real_baselines_20260801")
SEED = 2062001
MODELS = ("train_majority", "logistic_0.01", "logistic_0.1", "logistic_1", "nearest_centroid")
EXPECTED_LEDGER = {
    "source_complete_duplicates": 254_218,
    "external_complete_duplicates": 602,
    "cross_file_complete_duplicates": 23_878,
    "source_candidate_duplicates": 254_218,
    "external_candidate_duplicates": 602,
    "cross_file_candidate_duplicates": 23_878,
}


@dataclass(frozen=True)
class Records:
    values: np.ndarray
    targets: np.ndarray
    group_ids: np.ndarray | None = None


class DuplicateGroupFailure(ValueError):
    pass


def digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def load_records() -> tuple[Records, Records, dict[str, int]]:
    source_values = np.empty((SOURCE_ROWS, FEATURES), dtype=np.float32)
    external_values = np.empty((EXTERNAL_ROWS, FEATURES), dtype=np.float32)
    source_targets = np.empty(SOURCE_ROWS, dtype=np.uint8)
    external_targets = np.empty(EXTERNAL_ROWS, dtype=np.uint8)
    source_group_ids = np.empty(SOURCE_ROWS, dtype=np.int32)
    source_complete: set[bytes] = set()
    external_complete: set[bytes] = set()
    candidate_info: dict[bytes, tuple[int, int]] = {}
    external_candidates: set[bytes] = set()
    source_complete_duplicates = external_complete_duplicates = 0
    source_candidate_duplicates = external_candidate_duplicates = 0
    inconsistent_candidates = 0
    source_group_count = 0
    row_count = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        if tuple(archive.namelist()) != (MEMBER,):
            raise ValueError("HIGGS ZIP inventory changed")
        with archive.open(MEMBER) as compressed:
            with gzip.GzipFile(fileobj=compressed) as handle:
                for row_count, raw in enumerate(handle, start=1):
                    parsed = parse_line(raw, row_count)
                    target = int(parsed[0])
                    feature_values = parsed[1:].astype(np.float32, copy=False)
                    complete = digest(raw.rstrip(b"\r\n"))
                    candidate = digest(parsed[1:].astype("<f8", copy=False).tobytes())
                    if row_count <= SOURCE_ROWS:
                        index = row_count - 1
                        source_values[index] = feature_values
                        source_targets[index] = target
                        source_complete_duplicates += complete in source_complete
                        source_complete.add(complete)
                        prior = candidate_info.get(candidate)
                        if prior is None:
                            candidate_info[candidate] = (target, source_group_count)
                            source_group_ids[index] = source_group_count
                            source_group_count += 1
                        else:
                            source_candidate_duplicates += 1
                            if prior[0] != target:
                                inconsistent_candidates += 1
                            source_group_ids[index] = prior[1]
                    else:
                        index = row_count - SOURCE_ROWS - 1
                        external_values[index] = feature_values
                        external_targets[index] = target
                        external_complete_duplicates += complete in external_complete
                        external_complete.add(complete)
                        if candidate in external_candidates:
                            external_candidate_duplicates += 1
                        external_candidates.add(candidate)
                        prior = candidate_info.get(candidate)
                        if prior is None:
                            candidate_info[candidate] = (target, -1)
                        elif prior[0] != target:
                            inconsistent_candidates += 1
    if row_count != TOTAL_ROWS:
        raise ValueError(f"HIGGS source row count changed: expected {TOTAL_ROWS}, got {row_count}")
    if not np.all(np.isfinite(source_values)) or not np.all(np.isfinite(external_values)):
        raise ValueError("HIGGS source values became nonfinite")
    if not np.all((source_targets == 0) | (source_targets == 1)) or not np.all((external_targets == 0) | (external_targets == 1)):
        raise ValueError("HIGGS binary class support changed")
    ledger = {
        "source_complete_duplicates": source_complete_duplicates,
        "external_complete_duplicates": external_complete_duplicates,
        "cross_file_complete_duplicates": len(source_complete & external_complete),
        "source_candidate_duplicates": source_candidate_duplicates,
        "external_candidate_duplicates": external_candidate_duplicates,
        "cross_file_candidate_duplicates": sum(candidate in candidate_info and candidate_info[candidate][1] >= 0 for candidate in external_candidates),
    }
    if ledger != EXPECTED_LEDGER:
        raise ValueError(f"HIGGS frozen duplicate ledger changed: {ledger}")
    if inconsistent_candidates:
        raise DuplicateGroupFailure(f"{inconsistent_candidates} candidate-only duplicate records have inconsistent binary labels")
    return Records(source_values, source_targets, source_group_ids), Records(external_values, external_targets), ledger


def grouped_split(records: Records) -> tuple[Records, Records, dict[str, int]]:
    if records.group_ids is None:
        raise ValueError("HIGGS source records lack duplicate-group identities")
    group_count = int(records.group_ids.max()) + 1
    group_labels = np.empty(group_count, dtype=np.uint8)
    seen = np.zeros(group_count, dtype=bool)
    for index, group in enumerate(records.group_ids):
        if not seen[group]:
            group_labels[group] = records.targets[index]
            seen[group] = True
        elif group_labels[group] != records.targets[index]:
            raise DuplicateGroupFailure("candidate-only duplicate group has inconsistent binary labels")
    train_groups, selection_groups = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(group_count), group_labels)
    )
    selection_group_mask = np.zeros(group_count, dtype=bool)
    selection_group_mask[selection_groups] = True
    selection_rows = selection_group_mask[records.group_ids]
    train_rows = ~selection_rows
    if np.any(selection_group_mask[records.group_ids[train_rows]]):
        raise ValueError("candidate duplicate group crossed the frozen source-train split")
    return (
        Records(records.values[train_rows], records.targets[train_rows]),
        Records(records.values[selection_rows], records.targets[selection_rows]),
        {"source_unique_candidates": group_count, "source_candidate_duplicate_groups": int(np.count_nonzero(np.bincount(records.group_ids) > 1))},
    )


class Model:
    def __init__(self, name: str, majority: int | None = None, scaler: StandardScaler | None = None, estimator: object | None = None) -> None:
        self.name = name
        self.majority = majority
        self.scaler = scaler
        self.estimator = estimator

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        if self.majority is not None:
            probabilities = np.zeros((len(values), 2), dtype=float)
            probabilities[:, self.majority] = 1.0
            return probabilities
        assert self.estimator is not None
        transformed = self.scaler.transform(values) if self.scaler is not None else values
        if hasattr(self.estimator, "predict_proba"):
            probabilities = np.asarray(self.estimator.predict_proba(transformed), dtype=float)
        else:
            predictions = np.asarray(self.estimator.predict(transformed), dtype=int)
            probabilities = np.zeros((len(values), 2), dtype=float)
            probabilities[np.arange(len(values)), predictions] = 1.0
        if probabilities.shape != (len(values), 2):
            raise ValueError("HIGGS model returned an invalid probability shape")
        return probabilities


def fit(name: str, values: np.ndarray, targets: np.ndarray, seed: int = SEED) -> Model:
    if name == "train_majority":
        return Model(name, majority=int(np.bincount(targets, minlength=2).argmax()))
    if name == "nearest_centroid":
        return Model(name, estimator=NearestCentroid().fit(values, targets))
    scaler = StandardScaler().fit(values)
    transformed = scaler.transform(values)
    c_value = float(name.removeprefix("logistic_"))
    estimator = LogisticRegression(C=c_value, solver="saga", max_iter=100, tol=1e-4, random_state=seed).fit(transformed, targets)
    return Model(name, scaler=scaler, estimator=estimator)


def metric(model: Model, records: Records) -> dict[str, object]:
    probabilities = model.probabilities(records.values)
    if not np.all(np.isfinite(probabilities)) or not np.all((0.0 <= probabilities) & (probabilities <= 1.0)) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("HIGGS model emitted nonfinite or unnormalized probabilities")
    predictions = np.argmax(probabilities, axis=1).astype(np.uint8)
    if not np.all((predictions == 0) | (predictions == 1)):
        raise ValueError("HIGGS model emitted an invalid binary prediction")
    return {
        "macro_f1": float(f1_score(records.targets, predictions, average="macro")),
        "prediction_fingerprint": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }


def rejects(payload: dict[str, object]) -> bool:
    try:
        record_from_payload(payload)
    except ValueError:
        return True
    return False


def controls(name: str, train: Records, selection: Records, model: Model, score: float, ledger: dict[str, int]) -> dict[str, object]:
    paired_targets = np.random.default_rng(SEED).permutation(train.targets)
    label_pairing_degradation = score - float(metric(fit(name, train.values, paired_targets), selection)["macro_f1"])
    paired_values = selection.values.copy()
    paired_values[:, 21:] = paired_values[np.random.default_rng(SEED + 1).permutation(len(paired_values)), 21:]
    feature_pairing_degradation = score - float(metric(model, Records(paired_values, selection.targets))["macro_f1"])
    ablations: list[float] = []
    for fields in (slice(0, 21), slice(21, FEATURES)):
        ablated = selection.values.copy()
        ablated[:, fields] = 0.0
        ablations.append(score - float(metric(model, Records(ablated, selection.targets))["macro_f1"]))
    sample = selection.values[0]
    states: list[float] = []
    probabilities = predict_independent([record_from_payload({"values": sample})] * 3, lambda state, _values: states.append(state) or 0.5)
    reordered = predict_independent([record_from_payload({"values": sample})] * 3, lambda _state, _values: 0.5)
    return {
        "label_record_pairing_macro_f1_degradation": label_pairing_degradation,
        "low_high_block_pairing_macro_f1_degradation": feature_pairing_degradation,
        "feature_block_ablation_macro_f1_degradations": ablations,
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejects({"values": sample, "target": 1}),
        "file_rejected": rejects({"values": sample, "file": "source_tail"}),
        "split_rejected": rejects({"values": sample, "split": "external"}),
        "row_rejected": rejects({"values": sample, "row": 0}),
        "source_order_rejected": rejects({"values": sample, "source_order": 0}),
        "metadata_rejected": rejects({"values": sample, "metadata": {}}),
        "duplicate_group_rejected": rejects({"values": sample, "duplicate_group": "one"}),
        "duplicate_ledger_integrity": ledger == EXPECTED_LEDGER,
        "finite_normalized_probabilities": True,
    }


def write_result(output_dir: Path, result: dict[str, object]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run(output_dir: Path = OUT) -> dict[str, object]:
    try:
        source, external, ledger = load_records()
        train, selection, group_ledger = grouped_split(source)
    except DuplicateGroupFailure as error:
        return write_result(output_dir, {"phase": 62, "status": "closed_negative_duplicate_group_gate", "reason": str(error), "models_fitted": [], "external_outcome_score_count": 0, "abc_smc_calls": 0, "llm_calls": 0})
    except ValueError as error:
        return write_result(output_dir, {"phase": 62, "status": "closed_negative_source_revalidation_gate", "reason": str(error), "models_fitted": [], "external_outcome_score_count": 0, "abc_smc_calls": 0, "llm_calls": 0})
    models = {name: fit(name, train.values, train.targets) for name in MODELS}
    candidate_metrics = {name: metric(model, selection) for name, model in models.items()}
    selected = max(MODELS, key=lambda name: (float(candidate_metrics[name]["macro_f1"]), -MODELS.index(name)))
    selection_score = float(candidate_metrics[selected]["macro_f1"])
    selection_controls = controls(selected, train, selection, models[selected], selection_score, ledger)
    majority_score = float(candidate_metrics["train_majority"]["macro_f1"])
    passed = (
        selection_score >= 0.60
        and selection_score - majority_score >= 0.10
        and float(selection_controls["label_record_pairing_macro_f1_degradation"]) >= 0.40
        and float(selection_controls["low_high_block_pairing_macro_f1_degradation"]) >= 0.10
        and max(selection_controls["feature_block_ablation_macro_f1_degradations"]) >= 0.03
        and all(bool(value) for key, value in selection_controls.items() if key.endswith("rejected") or key in {"zero_reset", "reorder_invariant", "duplicate_ledger_integrity", "finite_normalized_probabilities"})
    )
    result: dict[str, object] = {
        "phase": 62,
        "source_revalidation": {"source_rows": len(source.targets), "external_rows": len(external.targets), "train_rows": len(train.targets), "selection_rows": len(selection.targets), **ledger, **group_ledger},
        "models": candidate_metrics,
        "selected_model": selected,
        "selection_controls": selection_controls,
        "selection_passed": passed,
        "external_outcome_score_count": 0,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not passed:
        result.update(status="closed_negative_selection_gate", reason="fixed selection metric or specificity control failed; final source tail was not scored")
        return write_result(output_dir, result)
    seed_metrics: list[dict[str, object]] = []
    fingerprints: list[str] = []
    for seed in range(SEED, SEED + 8):
        full_model = fit(selected, source.values, source.targets, seed)
        item = metric(full_model, external)
        item["seed"] = seed
        seed_metrics.append(item)
        fingerprints.append(str(item["prediction_fingerprint"]))
        if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
            break
    result.update(status="passed_observed_baseline_and_external_stability", external_outcome_score_count=1, external_seed_metrics=seed_metrics, executed_seed_count=len(seed_metrics), adaptive_distinct_yield_stopped=len(seed_metrics) < 8)
    return write_result(output_dir, result)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

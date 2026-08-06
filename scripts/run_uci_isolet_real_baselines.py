#!/usr/bin/env python3
"""Execute the fixed Phase 41 ISOLET observed baseline and one external operation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.real_data.uci_isolet import CLASS_LABELS, FEATURE_COUNT, vector_from_payload

ARCHIVE = Path("data/real/uci_isolet/raw/uci_isolet_54.zip")
SPLIT = Path("data/real/uci_isolet/source_file_split.json")
OUT_DIR = Path("artifacts/evaluations/phase41_uci_isolet_real_baselines_20260725")
TRAIN_MEMBER = "isolet1+2+3+4.data.Z"
EXTERNAL_MEMBER = "isolet5.data.Z"
ROWS = {TRAIN_MEMBER: 6238, EXTERNAL_MEMBER: 1559}
SEED = 2041054
CS = (0.01, 0.1, 1.0)
BLOCKS = tuple(np.array_split(np.arange(FEATURE_COUNT), 4))

@dataclass(frozen=True)
class Rows:
    x: np.ndarray
    y: np.ndarray
    keys: tuple[str, ...]

class Model:
    def __init__(self, name: str, scaler: StandardScaler | None, estimator: object | None, probabilities: np.ndarray | None = None):
        self.name, self.scaler, self.estimator, self.probabilities = name, scaler, estimator, probabilities
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if self.probabilities is not None:
            return np.broadcast_to(self.probabilities, (len(values), len(CLASS_LABELS))).copy()
        scaled = self.scaler.transform(values)
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.predict_proba(scaled)
        distances = np.sum((scaled[:, None, :] - self.estimator.centroids_[None, :, :]) ** 2, axis=2)
        logits = -distances
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(CLASS_LABELS)[np.argmax(self.predict_proba(x), axis=1)]

def _decompress(archive: ZipFile, member: str) -> bytes:
    with TemporaryDirectory() as directory:
        path = Path(directory) / member
        path.write_bytes(archive.read(member))
        process = subprocess.run(["uncompress", "-c", str(path)], capture_output=True, check=False)
    if process.returncode:
        raise ValueError(f"could not decompress {member}")
    return process.stdout

def _read_member(archive: ZipFile, member: str) -> Rows:
    lines = _decompress(archive, member).decode("ascii").splitlines()
    values, labels, keys = [], [], []
    for index, line in enumerate(lines):
        raw = line.strip()
        cells = np.fromstring(raw, sep=",", dtype=float)
        if not raw or cells.shape != (FEATURE_COUNT + 1,) or not np.all(np.isfinite(cells)) or not cells[-1].is_integer() or int(cells[-1]) not in CLASS_LABELS:
            raise ValueError(f"{member} violates the locked finite 618-field schema at row {index}")
        values.append(cells[:-1]); labels.append(int(cells[-1])); keys.append(hashlib.sha256(raw.encode("ascii")).hexdigest())
    x, y = np.asarray(values), np.asarray(labels)
    if x.shape != (ROWS[member], FEATURE_COUNT) or y.shape != (ROWS[member],):
        raise ValueError(f"{member} row count violates the fixed contract")
    if set(y) != set(CLASS_LABELS):
        raise ValueError(f"{member} lacks a locked class")
    return Rows(x, y, tuple(keys))

def load_locked_rows(archive_path: Path = ARCHIVE) -> tuple[Rows, Rows]:
    with ZipFile(archive_path) as archive:
        if set(archive.namelist()) != {"Index", "isolet.info", "isolet.names", TRAIN_MEMBER, EXTERNAL_MEMBER}:
            raise ValueError("ISOLET archive inventory changed")
        train, external = _read_member(archive, TRAIN_MEMBER), _read_member(archive, EXTERNAL_MEMBER)
    if set(train.keys) & set(external.keys):
        raise ValueError("complete raw fingerprint crosses the frozen source-file partition")
    return train, external

def _fit(name: str, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> Model:
    if name == "majority":
        counts = np.asarray([np.count_nonzero(y == label) for label in CLASS_LABELS], dtype=float)
        return Model(name, None, None, counts / counts.sum())
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    if name.startswith("logreg_"):
        c = float(name.split("_")[1])
        return Model(name, scaler, LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed).fit(scaled, y))
    if name == "nearest_centroid":
        return Model(name, scaler, NearestCentroid().fit(scaled, y))
    raise ValueError("unlocked ISOLET model")

def _validate_probabilities(probs: np.ndarray) -> None:
    if probs.shape[1] != len(CLASS_LABELS) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1, atol=1e-9):
        raise ValueError("ISOLET probabilities violate finite [0,1] simplex semantics")

def _metrics(model: Model, rows: Rows) -> dict[str, object]:
    probs = model.predict_proba(rows.x); _validate_probabilities(probs)
    prediction = model.predict(rows.x)
    if np.any(~np.isin(prediction, CLASS_LABELS)):
        raise ValueError("ISOLET prediction violates label bounds")
    result = {"macro_f1": float(f1_score(rows.y, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(rows.y, prediction)), "multiclass_log_loss": float(log_loss(rows.y, probs, labels=CLASS_LABELS)), "per_label_f1": {str(label): float(f1_score(rows.y, prediction, labels=[label], average="macro", zero_division=0)) for label in CLASS_LABELS}, "prediction_fingerprint": hashlib.sha256(prediction.tobytes()).hexdigest()}
    if not all(np.isfinite(value) for key, value in result.items() if key != "per_label_f1" and key != "prediction_fingerprint"):
        raise ValueError("ISOLET metric is non-finite")
    return result

def _sentinel_rejected(payload: dict[str, object]) -> bool:
    try: vector_from_payload(payload)
    except ValueError: return True
    return False

def _fit_selected(name: str, rows: Rows, y: np.ndarray | None = None, seed: int = SEED) -> Model:
    return _fit(name, rows.x, rows.y if y is None else y, seed)

def _selection_controls(name: str, train: Rows, select: Rows, selected: Model, baseline_f1: float) -> dict[str, object]:
    paired = _metrics(_fit_selected(name, train, np.random.default_rng(SEED).permutation(train.y)), select)["macro_f1"]
    permutation = np.concatenate((BLOCKS[3], BLOCKS[0], BLOCKS[1], BLOCKS[2]))
    block_rows = Rows(select.x[:, permutation], select.y, select.keys)
    block = _metrics(selected, block_rows)["macro_f1"]
    ablations = []
    for indices in BLOCKS:
        x = select.x.copy(); x[:, indices] = 0.0
        ablations.append(baseline_f1 - _metrics(selected, Rows(x, select.y, select.keys))["macro_f1"])
    reverse = np.arange(len(select.y) - 1, -1, -1)
    reordered = selected.predict(select.x[reverse])[::-1]
    ordinary = selected.predict(select.x)
    candidate = select.x[0]
    return {"label_pairing_drop": baseline_f1 - paired, "block_pairing_drop": baseline_f1 - block, "block_ablation_drops": ablations, "row_reorder_invariant": bool(np.array_equal(ordinary, reordered)), "target_isolation_sentinel_rejected": _sentinel_rejected({"features": candidate, "label": 1}), "file_isolation_sentinel_rejected": _sentinel_rejected({"features": candidate, "source_file": EXTERNAL_MEMBER}), "split_isolation_sentinel_rejected": _sentinel_rejected({"features": candidate, "selection_assignment": True}), "speaker_isolation_sentinel_rejected": _sentinel_rejected({"features": candidate, "speaker_identity": "synthetic"})}

def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT_DIR) -> dict[str, object]:
    payload = json.loads(SPLIT.read_text())
    if payload != {"source_train": TRAIN_MEMBER, "external": EXTERNAL_MEMBER, "selection_design": "stratified_80_20_seed_2041054_inside_source_train"}:
        raise ValueError("frozen ISOLET source-file split changed")
    source, external_quarantined = load_locked_rows(archive_path)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=SEED)
    train_index, select_index = next(splitter.split(source.x, source.y))
    train = Rows(source.x[train_index], source.y[train_index], tuple(source.keys[i] for i in train_index)); select = Rows(source.x[select_index], source.y[select_index], tuple(source.keys[i] for i in select_index))
    if set(train.keys) & set(select.keys): raise ValueError("complete raw fingerprint crosses train/selection")
    models = ["majority", *(f"logreg_{c:g}" for c in CS), "nearest_centroid"]
    candidates = {name: _metrics(_fit_selected(name, train), select) for name in models}
    ranking = sorted(models, key=lambda name: (-candidates[name]["macro_f1"], models.index(name)))
    chosen = ranking[0]; selected = _fit_selected(chosen, train); chosen_metrics = candidates[chosen]; majority = candidates["majority"]["macro_f1"]
    controls = _selection_controls(chosen, train, select, selected, chosen_metrics["macro_f1"])
    selection_pass = all([chosen_metrics["macro_f1"] >= .75, chosen_metrics["macro_f1"] - majority >= .45, controls["label_pairing_drop"] >= .30, controls["block_pairing_drop"] >= .10, max(controls["block_ablation_drops"]) >= .05, *[bool(value) for key, value in controls.items() if key.endswith("rejected") or key == "row_reorder_invariant"]])
    result: dict[str, object] = {"phase": 41, "source_revalidation": {"source_train_rows": len(source.y), "external_rows": len(external_quarantined.y), "cross_source_file_complete_duplicates": 0, "selection_rows": len(select.y), "train_rows": len(train.y)}, "external_outcome_score_count": 0, "models": candidates, "selected_model": chosen, "selection_controls": controls, "selection_passed": selection_pass, "abc_smc_calls": 0, "llm_calls": 0}
    if not selection_pass:
        result.update({"status": "closed_negative_selection_gate", "reason": "A fixed observed selection quality or specificity gate failed; isolet5 was not scored."})
    else:
        seed_metrics = []; fingerprints = []
        for seed in range(SEED, SEED + 8):
            full = _fit_selected(chosen, source, seed=seed); metric = _metrics(full, external_quarantined); metric["seed"] = seed; seed_metrics.append(metric); fingerprints.append((metric["prediction_fingerprint"], hashlib.sha256(full.predict(source.x).tobytes()).hexdigest()))
            if len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1: break
        primary = seed_metrics[0]
        external_pass = primary["macro_f1"] >= .65 and primary["macro_f1"] >= .75 * chosen_metrics["macro_f1"]
        result.update({"external_outcome_score_count": 1, "external_seed_metrics": seed_metrics, "executed_seed_count": len(seed_metrics), "adaptive_distinct_yield_stopped": len(seed_metrics) < 8, "status": "passed_observed_baseline_and_external_stability" if external_pass else "closed_negative_external_stability", "external_passed": external_pass, "reason": "Fixed external operation completed without model changes." if external_pass else "Fixed external macro-F1 or selection-ratio stop failed; no model change is allowed."})
    output_dir.mkdir(parents=True, exist_ok=True); (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result

def main() -> int:
    print(json.dumps(run(), indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())

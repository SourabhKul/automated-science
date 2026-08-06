"""Isolated VitalDB conditional-prediction adapter primitives.

This module deliberately does not participate in the autonomous dense-domain
sandbox.  It supports only the Phase 26 claim boundary: causal conditional
prediction of recorded MAP from a case's initial MAP and recorded medication
histories.  It is not a causal drug-effect or clinical decision interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


REQUIRED_TRACKS = (
    "Solar8000/ART_MBP",
    "Orchestra/PPF20_RATE",
    "Orchestra/RFTN20_RATE",
    "Orchestra/PHEN_RATE",
)
INPUT_NAMES = ("propofol_rate", "remifentanil_rate", "phenylephrine_rate")
GRID_SECONDS = 10
MIN_CONTIGUOUS_SAMPLES = 60
EXPECTED_SPLIT_COUNTS = {"train": 24, "selection": 8, "external": 8}
STABLE_ALPHA_GRID = tuple(float(value) for value in np.arange(0.0, 1.0, 0.05))
MAP_PREDICTION_BOUNDS = (20.0, 250.0)


@dataclass(frozen=True)
class VitalDBRecord:
    """A finite, contiguous MAP/medication segment from one subject's case."""

    record_id: str
    case_id: int
    subject_id: int
    split: str
    department: str
    map_mmhg: np.ndarray
    input_rates_ml_per_hr: np.ndarray
    grid_seconds: int = GRID_SECONDS

    def __post_init__(self) -> None:
        map_mmhg = np.asarray(self.map_mmhg, dtype=float)
        inputs = np.asarray(self.input_rates_ml_per_hr, dtype=float)
        if self.split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"unknown VitalDB split: {self.split}")
        if map_mmhg.ndim != 1 or map_mmhg.size < 2 or not np.all(np.isfinite(map_mmhg)):
            raise ValueError("VitalDB MAP must be a finite one-dimensional segment with at least two samples")
        if inputs.shape != (map_mmhg.size, len(INPUT_NAMES)) or not np.all(np.isfinite(inputs)):
            raise ValueError("VitalDB input rates must be finite with shape (samples, 3)")
        if int(self.grid_seconds) <= 0:
            raise ValueError("VitalDB grid_seconds must be positive")
        object.__setattr__(self, "map_mmhg", map_mmhg.copy())
        object.__setattr__(self, "input_rates_ml_per_hr", inputs.copy())


@dataclass(frozen=True)
class VitalDBInputRecord:
    """Causal prediction surface without future MAP targets."""

    record_id: str
    case_id: int
    split: str
    initial_map_mmhg: float
    input_rates_ml_per_hr: np.ndarray

    def __post_init__(self) -> None:
        inputs = np.asarray(self.input_rates_ml_per_hr, dtype=float)
        if not np.isfinite(self.initial_map_mmhg) or inputs.ndim != 2 or inputs.shape[1:] != (len(INPUT_NAMES),):
            raise ValueError("VitalDB input record requires a finite initial MAP and an (n, 3) input matrix")
        if inputs.shape[0] < 2 or not np.all(np.isfinite(inputs)):
            raise ValueError("VitalDB input record requires at least two finite input rows")
        object.__setattr__(self, "input_rates_ml_per_hr", inputs.copy())


@dataclass(frozen=True)
class VitalDBMedicationGrid:
    """A source medication grid intentionally lacking any measured MAP field."""

    record_id: str
    case_id: int
    subject_id: int
    split: str
    department: str
    input_rates_ml_per_hr: np.ndarray

    def __post_init__(self) -> None:
        inputs = np.asarray(self.input_rates_ml_per_hr, dtype=float)
        if self.split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"unknown VitalDB split: {self.split}")
        if inputs.ndim != 2 or inputs.shape[0] < 2 or inputs.shape[1] != len(INPUT_NAMES) or not np.all(np.isfinite(inputs)):
            raise ValueError("VitalDB medication grid requires at least two finite rows with three rate columns")
        object.__setattr__(self, "input_rates_ml_per_hr", inputs.copy())


@dataclass(frozen=True)
class TrainOnlyInputScaler:
    center: np.ndarray
    scale: np.ndarray
    fitted_case_ids: tuple[int, ...]

    def transform(self, input_rates_ml_per_hr: np.ndarray) -> np.ndarray:
        values = np.asarray(input_rates_ml_per_hr, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(INPUT_NAMES) or not np.all(np.isfinite(values)):
            raise ValueError("VitalDB scaler requires finite input rows with three medication-rate columns")
        return (values - self.center) / self.scale


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    start = best_start = best_end = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            if index == 0 or not mask[index - 1]:
                start = index
            if index + 1 - start > best_end - best_start:
                best_start, best_end = start, index + 1
    return best_start, best_end


def _validate_locked_split(entries: list[dict[str, Any]]) -> None:
    counts = {split: 0 for split in EXPECTED_SPLIT_COUNTS}
    case_ids: set[int] = set()
    subject_ids: set[int] = set()
    for entry in entries:
        split = entry.get("split")
        if split not in counts:
            raise ValueError(f"candidate split contains unknown split: {split}")
        case_id, subject_id = int(entry["caseid"]), int(entry["subjectid"])
        if case_id in case_ids or subject_id in subject_ids:
            raise ValueError("VitalDB candidate split must be case- and subject-disjoint")
        if tuple(entry.get("required_tracks", ())) != REQUIRED_TRACKS:
            raise ValueError("VitalDB candidate split has an unexpected required-track contract")
        if int(entry.get("coverage_grid_seconds", -1)) != GRID_SECONDS:
            raise ValueError("VitalDB candidate split has an unexpected grid interval")
        case_ids.add(case_id)
        subject_ids.add(subject_id)
        counts[split] += 1
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"VitalDB candidate split counts must be {EXPECTED_SPLIT_COUNTS}, found {counts}")


def load_vitaldb_records(
    candidate_split_path: str | Path,
    case_reader: Callable[[int, tuple[str, ...], int], np.ndarray],
    *,
    min_contiguous_samples: int = MIN_CONTIGUOUS_SAMPLES,
) -> dict[str, list[VitalDBRecord]]:
    """Read the locked records without interpolation or cross-case pooling.

    ``case_reader`` is injected so the adapter can be tested without network
    access.  In a later, separately authorized real adapter pass it may wrap
    ``vitaldb.load_case(caseid, REQUIRED_TRACKS, interval=10)``.
    """
    if min_contiguous_samples < 2:
        raise ValueError("min_contiguous_samples must be at least two")
    payload = json.loads(Path(candidate_split_path).read_text())
    entries = list(payload.get("selected", []))
    _validate_locked_split(entries)
    records = {split: [] for split in EXPECTED_SPLIT_COUNTS}
    for entry in entries:
        values = np.asarray(case_reader(int(entry["caseid"]), REQUIRED_TRACKS, GRID_SECONDS), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(REQUIRED_TRACKS):
            raise ValueError(f"case {entry['caseid']} reader must return an (n, 4) track matrix")
        start, end = _longest_true_run(np.all(np.isfinite(values), axis=1))
        if end - start < min_contiguous_samples:
            raise ValueError(
                f"case {entry['caseid']} has only {end - start} contiguous jointly finite samples; "
                f"need {min_contiguous_samples} without interpolation"
            )
        segment = values[start:end]
        record = VitalDBRecord(
            record_id=f"case_{int(entry['caseid'])}",
            case_id=int(entry["caseid"]),
            subject_id=int(entry["subjectid"]),
            split=str(entry["split"]),
            department=str(entry["department"]),
            map_mmhg=segment[:, 0],
            input_rates_ml_per_hr=segment[:, 1:],
        )
        records[record.split].append(record)
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if len(records[split]) != expected:
            raise ValueError(f"VitalDB {split} loader count mismatch")
    return records


def assess_vitaldb_adapter_contract(
    candidate_split_path: str | Path,
    case_reader: Callable[[int, tuple[str, ...], int], np.ndarray],
    *,
    min_contiguous_samples: int = MIN_CONTIGUOUS_SAMPLES,
) -> dict[str, Any]:
    """Assess every locked case without constructing records or fitting a model.

    The returned rows contain only source-quality accounting, never trace
    values.  Unlike ``load_vitaldb_records``, it continues after an individual
    reader failure so a rejected locked cohort has a complete case ledger.
    """
    if min_contiguous_samples < 2:
        raise ValueError("min_contiguous_samples must be at least two")
    payload = json.loads(Path(candidate_split_path).read_text())
    entries = list(payload.get("selected", []))
    try:
        _validate_locked_split(entries)
    except Exception as exc:
        return {
            "passed": False,
            "integrity_passed": False,
            "failure_modes": [f"split_integrity: {type(exc).__name__}: {exc}"],
            "required_tracks": list(REQUIRED_TRACKS),
            "grid_seconds": GRID_SECONDS,
            "minimum_contiguous_samples": min_contiguous_samples,
            "rows": [],
        }

    rows = []
    for entry in entries:
        row: dict[str, Any] = {
            "case_id": int(entry["caseid"]),
            "subject_id": int(entry["subjectid"]),
            "split": str(entry["split"]),
            "department": str(entry["department"]),
            "native_shape": None,
            "native_samples": None,
            "joint_finite_samples": None,
            "longest_contiguous_samples": 0,
            "longest_contiguous_start": None,
            "longest_contiguous_end": None,
            "eligible": False,
            "rejection_reason": None,
        }
        try:
            values = np.asarray(case_reader(row["case_id"], REQUIRED_TRACKS, GRID_SECONDS), dtype=float)
            row["native_shape"] = list(values.shape)
            if values.ndim != 2 or values.shape[1] != len(REQUIRED_TRACKS):
                row["rejection_reason"] = "invalid_track_matrix_shape"
            else:
                row["native_samples"] = int(values.shape[0])
                joint_finite = np.all(np.isfinite(values), axis=1)
                start, end = _longest_true_run(joint_finite)
                row["joint_finite_samples"] = int(np.sum(joint_finite))
                row["longest_contiguous_samples"] = int(end - start)
                row["longest_contiguous_start"] = int(start)
                row["longest_contiguous_end"] = int(end)
                if end - start < min_contiguous_samples:
                    row["rejection_reason"] = "insufficient_jointly_finite_contiguous_samples"
                else:
                    row["eligible"] = True
        except Exception as exc:
            row["rejection_reason"] = f"reader_error: {type(exc).__name__}: {exc}"
        rows.append(row)
    rejected = [row for row in rows if not row["eligible"]]
    return {
        "passed": not rejected,
        "integrity_passed": True,
        "failure_modes": sorted({str(row["rejection_reason"]) for row in rejected if row["rejection_reason"]}),
        "required_tracks": list(REQUIRED_TRACKS),
        "grid_seconds": GRID_SECONDS,
        "minimum_contiguous_samples": min_contiguous_samples,
        "rows": rows,
    }


def load_vitaldb_medication_grids(
    candidate_split_path: str | Path,
    case_reader: Callable[[int, tuple[str, ...], int], np.ndarray],
    *,
    min_contiguous_samples: int = MIN_CONTIGUOUS_SAMPLES,
) -> dict[str, list[VitalDBMedicationGrid]]:
    """Load native medication grids while discarding measured MAP before return.

    MAP is used solely to identify the predeclared jointly finite interval. The
    returned records do not retain it, preventing synthetic generators and
    fitters from reading measured outcomes.
    """
    if min_contiguous_samples < 2:
        raise ValueError("min_contiguous_samples must be at least two")
    payload = json.loads(Path(candidate_split_path).read_text())
    entries = list(payload.get("selected", []))
    _validate_locked_split(entries)
    grids = {split: [] for split in EXPECTED_SPLIT_COUNTS}
    for entry in entries:
        values = np.asarray(case_reader(int(entry["caseid"]), REQUIRED_TRACKS, GRID_SECONDS), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(REQUIRED_TRACKS):
            raise ValueError(f"case {entry['caseid']} reader must return an (n, 4) track matrix")
        start, end = _longest_true_run(np.all(np.isfinite(values), axis=1))
        if end - start < min_contiguous_samples:
            raise ValueError(f"case {entry['caseid']} fails the locked contiguous-window contract")
        grid = VitalDBMedicationGrid(
            record_id=f"case_{int(entry['caseid'])}",
            case_id=int(entry["caseid"]),
            subject_id=int(entry["subjectid"]),
            split=str(entry["split"]),
            department=str(entry["department"]),
            input_rates_ml_per_hr=values[start:end, 1:],
        )
        grids[grid.split].append(grid)
    return grids


def as_input_record(record: VitalDBRecord) -> VitalDBInputRecord:
    return VitalDBInputRecord(
        record_id=record.record_id,
        case_id=record.case_id,
        split=record.split,
        initial_map_mmhg=float(record.map_mmhg[0]),
        input_rates_ml_per_hr=record.input_rates_ml_per_hr,
    )


def fit_train_only_input_scaler(records: list[VitalDBRecord | VitalDBMedicationGrid]) -> TrainOnlyInputScaler:
    if not records or any(record.split != "train" for record in records):
        raise ValueError("VitalDB input scaling must use non-empty train records only")
    values = np.concatenate([record.input_rates_ml_per_hr for record in records], axis=0)
    center = np.median(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), 1e-8)
    return TrainOnlyInputScaler(center=center, scale=scale, fitted_case_ids=tuple(sorted(record.case_id for record in records)))


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)[1:]
    predicted = np.asarray(predicted, dtype=float)[1:]
    if observed.shape != predicted.shape or observed.size == 0 or not np.all(np.isfinite(predicted)):
        return {"mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "late_nrmse": float("nan")}
    residual = predicted - observed
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    late = residual[int(0.75 * len(residual)):]
    mse = float(np.mean(np.square(residual)))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "nrmse": float(np.sqrt(mse) / scale), "late_nrmse": float(np.sqrt(np.mean(np.square(late))) / scale)}


def fit_causal_linear_baseline(train_records: list[VitalDBRecord], scaler: TrainOnlyInputScaler) -> dict[str, Any]:
    """Fit one pooled, one-step conditional predictor using train cases only."""
    if not train_records or any(record.split != "train" for record in train_records):
        raise ValueError("VitalDB baseline fitting must use train cases only")
    features, targets = [], []
    for record in train_records:
        inputs = scaler.transform(record.input_rates_ml_per_hr)
        features.append(np.column_stack([record.map_mmhg[:-1], inputs[1:], np.ones(len(record.map_mmhg) - 1)]))
        targets.append(record.map_mmhg[1:])
    x, y = np.vstack(features), np.concatenate(targets)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"coefficients": coefficients, "input_center": scaler.center.copy(), "input_scale": scaler.scale.copy(), "fit_case_ids": scaler.fitted_case_ids}


def predict_causal_linear_baseline(input_record: VitalDBInputRecord, model: dict[str, Any]) -> np.ndarray:
    """Free-run one case with only its initial MAP and causal recorded inputs."""
    coefficients = np.asarray(model["coefficients"], dtype=float)
    center, scale = np.asarray(model["input_center"], dtype=float), np.asarray(model["input_scale"], dtype=float)
    if coefficients.shape != (5,) or center.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0.0):
        raise ValueError("invalid VitalDB causal-linear baseline model")
    inputs = (input_record.input_rates_ml_per_hr - center) / scale
    prediction = np.empty(len(inputs), dtype=float)
    prediction[0] = input_record.initial_map_mmhg
    for index in range(1, len(prediction)):
        prediction[index] = float(coefficients @ np.concatenate([[prediction[index - 1]], inputs[index], [1.0]]))
    return prediction


def score_vitaldb_records(records: list[VitalDBRecord], predictor: Callable[[VitalDBInputRecord], np.ndarray]) -> dict[str, Any]:
    """Score output targets only after a predictor has received input-only records."""
    rows = []
    for record in records:
        metrics = _metrics(record.map_mmhg, predictor(as_input_record(record)))
        rows.append({"record_id": record.record_id, "case_id": record.case_id, "split": record.split, "finite": bool(all(np.isfinite(value) for value in metrics.values())), **metrics})
    if not rows or not all(row["finite"] for row in rows):
        raise ValueError("non-finite VitalDB held-out metric")
    keys = ("mse", "rmse", "nrmse", "late_nrmse")
    return {"per_record": rows, "aggregate": {f"median_{key}": float(np.median([row[key] for row in rows])) for key in keys}}


def persistence_prediction(input_record: VitalDBInputRecord) -> np.ndarray:
    """Free-run persistence from the declared observed case initial MAP."""
    return np.full(len(input_record.input_rates_ml_per_hr), input_record.initial_map_mmhg, dtype=float)


def fit_stable_vitaldb_linear(
    train_records: list[VitalDBRecord],
    scaler: TrainOnlyInputScaler | None,
    *,
    input_columns: tuple[int, ...],
) -> dict[str, Any]:
    """Fit a fixed-grid stable autoregression or conditional linear model on train only."""
    if not train_records or any(record.split != "train" for record in train_records):
        raise ValueError("VitalDB stable fitting must use train records only")
    if input_columns and scaler is None:
        raise ValueError("VitalDB conditional fitting requires a train-only input scaler")
    best: dict[str, Any] | None = None
    for alpha in STABLE_ALPHA_GRID:
        features, targets = [], []
        for record in train_records:
            if input_columns:
                assert scaler is not None
                inputs = scaler.transform(record.input_rates_ml_per_hr)[:, input_columns]
                features.append(np.column_stack([inputs[1:], np.ones(len(record.map_mmhg) - 1)]))
            else:
                features.append(np.ones((len(record.map_mmhg) - 1, 1)))
            targets.append(record.map_mmhg[1:] - alpha * record.map_mmhg[:-1])
        x, y = np.vstack(features), np.concatenate(targets)
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        train_mse = float(np.mean(np.square(x @ coefficients - y)))
        candidate = {"alpha": alpha, "coefficients": coefficients[:-1], "intercept": float(coefficients[-1]), "input_columns": input_columns, "train_mse": train_mse}
        if best is None or candidate["train_mse"] < best["train_mse"]:
            best = candidate
    assert best is not None
    if scaler is not None:
        best["input_center"] = scaler.center.copy()
        best["input_scale"] = scaler.scale.copy()
        best["fit_case_ids"] = scaler.fitted_case_ids
    else:
        best["fit_case_ids"] = tuple(sorted(record.case_id for record in train_records))
    return best


def predict_stable_vitaldb_linear(input_record: VitalDBInputRecord, model: dict[str, Any]) -> np.ndarray:
    """Free-run a stable fitted model with no access to future MAP."""
    alpha = float(model["alpha"])
    columns = tuple(int(value) for value in model["input_columns"])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if alpha not in STABLE_ALPHA_GRID or coefficients.shape != (len(columns),):
        raise ValueError("invalid VitalDB stable baseline model")
    if columns:
        center = np.asarray(model["input_center"], dtype=float)
        scale = np.asarray(model["input_scale"], dtype=float)
        if center.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0.0):
            raise ValueError("invalid VitalDB input scaling in stable baseline model")
        inputs = ((input_record.input_rates_ml_per_hr - center) / scale)[:, columns]
    else:
        inputs = np.empty((len(input_record.input_rates_ml_per_hr), 0), dtype=float)
    prediction = np.empty(len(inputs), dtype=float)
    prediction[0] = input_record.initial_map_mmhg
    for index in range(1, len(prediction)):
        prediction[index] = alpha * prediction[index - 1] + float(coefficients @ inputs[index]) + float(model["intercept"])
    return prediction


def score_bounded_vitaldb_records(
    records: list[VitalDBRecord],
    predictor: Callable[[VitalDBInputRecord], np.ndarray],
    *,
    bounds: tuple[float, float] = MAP_PREDICTION_BOUNDS,
) -> dict[str, Any]:
    """Score a free-run predictor and fail hard on non-finite or out-of-bound MAP."""
    low, high = bounds
    rows = []
    for record in records:
        prediction = np.asarray(predictor(as_input_record(record)), dtype=float)
        reason = None
        if prediction.shape != record.map_mmhg.shape:
            reason = "shape_mismatch"
        elif not np.all(np.isfinite(prediction)):
            reason = "nonfinite_prediction"
        elif float(np.min(prediction)) < low or float(np.max(prediction)) > high:
            reason = "prediction_bound_exceeded"
        metrics = _metrics(record.map_mmhg, prediction) if reason is None else {"mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "late_nrmse": float("nan")}
        rows.append({"record_id": record.record_id, "case_id": record.case_id, "split": record.split, "department": record.department, "finite": reason is None and all(np.isfinite(value) for value in metrics.values()), "min_prediction": float(np.min(prediction)) if prediction.size and np.all(np.isfinite(prediction)) else None, "max_prediction": float(np.max(prediction)) if prediction.size and np.all(np.isfinite(prediction)) else None, "failure_reason": reason, **metrics})
    passed = bool(rows) and all(row["finite"] for row in rows)
    result: dict[str, Any] = {"passed": passed, "bounds_mmhg": [low, high], "per_record": rows}
    if passed:
        keys = ("mse", "rmse", "nrmse", "late_nrmse")
        result["aggregate"] = {f"median_{key}": float(np.median([row[key] for row in rows])) for key in keys}
        result["by_department"] = {
            department: {f"median_{key}": float(np.median([row[key] for row in rows if row["department"] == department])) for key in keys}
            for department in sorted({row["department"] for row in rows})
        }
    else:
        result["aggregate"] = None
        result["by_department"] = None
    return result


def evaluate_vitaldb_baselines(data: dict[str, list[VitalDBRecord]]) -> dict[str, Any]:
    """Fit once on train, report selection and untouched external case metrics."""
    if set(data) != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError("VitalDB baseline evaluation requires train, selection, and external splits")
    scaler = fit_train_only_input_scaler(data["train"])
    model = fit_causal_linear_baseline(data["train"], scaler)
    predictor = lambda item: predict_causal_linear_baseline(item, model)
    return {
        "initial_state_policy": "observed_case_initial_map_only",
        "input_access": "current_and_past_recorded_medication_rates_only",
        "train_fit_case_ids": list(model["fit_case_ids"]),
        "causal_linear": {
            "parameters": {"coefficients": model["coefficients"].tolist(), "input_center": model["input_center"].tolist(), "input_scale": model["input_scale"].tolist()},
            "selection": score_vitaldb_records(data["selection"], predictor),
            "external": score_vitaldb_records(data["external"], predictor),
        },
    }

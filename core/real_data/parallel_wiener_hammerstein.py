from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


AMPLITUDES = (0.1, 0.325, 0.55, 0.775, 1.0)
ESTIMATION_REALIZATIONS_PER_LEVEL = 20
SAMPLES_PER_RECORD = 32768


@dataclass(frozen=True)
class ParallelWHRecord:
    record_id: str
    source_split: str
    regime: str
    amplitude: float | None
    realization_index: int | None
    input_u: np.ndarray
    output_y: np.ndarray
    source_member: str


@dataclass(frozen=True)
class ParallelWHInputRecord:
    """Published input metadata without a measured-output field.

    The synthetic-control path uses this type so its generator and fitter
    cannot accidentally consume the source's recorded output values.
    """

    record_id: str
    source_split: str
    regime: str
    amplitude: float | None
    realization_index: int | None
    input_u: np.ndarray
    source_member: str


def _nonempty_csv_rows(archive: zipfile.ZipFile, member: str) -> list[list[str]]:
    with archive.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
        rows = [row for row in reader if any(value.strip() for value in row)]
    if len(rows) != SAMPLES_PER_RECORD + 1:
        raise ValueError(f"{member} must contain one header plus {SAMPLES_PER_RECORD} data rows, found {len(rows)}")
    return rows


def _finite_column(rows: list[list[str]], index: int, member: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[index]) for row in rows[1:]], dtype=float)
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid numeric column {index} in {member}") from error
    if values.shape != (SAMPLES_PER_RECORD,) or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite or malformed numeric column {index} in {member}")
    return values


def _estimation_records(archive: zipfile.ZipFile, level: int) -> list[ParallelWHRecord]:
    member = f"ParWHFiles/ParWHData_Estimation_Level{level}.csv"
    rows = _nonempty_csv_rows(archive, member)
    if rows[0][:4] != ["amplitude", "fs", "lines", "u"] or len(rows[0]) < 43:
        raise ValueError(f"unexpected estimation header in {member}: {rows[0][:4]}")
    amplitude = float(rows[1][0])
    sample_rate = float(rows[1][1])
    if not np.isclose(amplitude, AMPLITUDES[level - 1]) or sample_rate != 78125.0:
        raise ValueError(f"unexpected amplitude or sample rate in {member}")
    records = []
    for realization in range(ESTIMATION_REALIZATIONS_PER_LEVEL):
        records.append(
            ParallelWHRecord(
                record_id=f"estimation_level{level}_realization{realization + 1:02d}",
                source_split="estimation",
                regime="fixed_amplitude_multisine",
                amplitude=amplitude,
                realization_index=realization + 1,
                input_u=_finite_column(rows, 3 + realization, member),
                output_y=_finite_column(rows, 23 + realization, member),
                source_member=member,
            )
        )
    return records


def _validation_record(archive: zipfile.ZipFile, level: int) -> ParallelWHRecord:
    member = f"ParWHFiles/ParWHData_Validation_Level{level}.csv"
    rows = _nonempty_csv_rows(archive, member)
    if rows[0][:5] != ["amplitude", "fs", "lines", "u", "y"]:
        raise ValueError(f"unexpected fixed-amplitude validation header in {member}: {rows[0][:5]}")
    amplitude = float(rows[1][0])
    sample_rate = float(rows[1][1])
    if not np.isclose(amplitude, AMPLITUDES[level - 1]) or sample_rate != 78125.0:
        raise ValueError(f"unexpected amplitude or sample rate in {member}")
    return ParallelWHRecord(
        record_id=f"validation_level{level}",
        source_split="official_validation",
        regime="fixed_amplitude_multisine",
        amplitude=amplitude,
        realization_index=None,
        input_u=_finite_column(rows, 3, member),
        output_y=_finite_column(rows, 4, member),
        source_member=member,
    )


def _arrow_record(archive: zipfile.ZipFile) -> ParallelWHRecord:
    member = "ParWHFiles/ParWHData_ValidationArrow.csv"
    rows = _nonempty_csv_rows(archive, member)
    if rows[0][:4] != ["fs", "lines", "u", "y"] or float(rows[1][0]) != 78125.0:
        raise ValueError(f"unexpected growing-amplitude validation header in {member}: {rows[0][:4]}")
    return ParallelWHRecord(
        record_id="validation_growing_amplitude",
        source_split="official_validation",
        regime="growing_amplitude_gaussian_noise",
        amplitude=None,
        realization_index=None,
        input_u=_finite_column(rows, 2, member),
        output_y=_finite_column(rows, 3, member),
        source_member=member,
    )


def load_parallel_wiener_hammerstein(raw_archive: str | Path) -> list[ParallelWHRecord]:
    """Load all published records directly from the fixed V1 archive."""
    archive_path = Path(raw_archive)
    with zipfile.ZipFile(archive_path) as archive:
        records = [record for level in range(1, 6) for record in _estimation_records(archive, level)]
        records.extend(_validation_record(archive, level) for level in range(1, 6))
        records.append(_arrow_record(archive))
    if len(records) != 106:
        raise ValueError(f"expected 106 published Parallel WH records, found {len(records)}")
    return records


def _estimation_input_records(archive: zipfile.ZipFile, level: int) -> list[ParallelWHInputRecord]:
    member = f"ParWHFiles/ParWHData_Estimation_Level{level}.csv"
    rows = _nonempty_csv_rows(archive, member)
    if rows[0][:4] != ["amplitude", "fs", "lines", "u"] or len(rows[0]) < 43:
        raise ValueError(f"unexpected estimation header in {member}: {rows[0][:4]}")
    amplitude = float(rows[1][0])
    if not np.isclose(amplitude, AMPLITUDES[level - 1]) or float(rows[1][1]) != 78125.0:
        raise ValueError(f"unexpected amplitude or sample rate in {member}")
    return [
        ParallelWHInputRecord(
            record_id=f"estimation_level{level}_realization{realization + 1:02d}",
            source_split="estimation",
            regime="fixed_amplitude_multisine",
            amplitude=amplitude,
            realization_index=realization + 1,
            input_u=_finite_column(rows, 3 + realization, member),
            source_member=member,
        )
        for realization in range(ESTIMATION_REALIZATIONS_PER_LEVEL)
    ]


def _validation_input_record(archive: zipfile.ZipFile, level: int) -> ParallelWHInputRecord:
    member = f"ParWHFiles/ParWHData_Validation_Level{level}.csv"
    rows = _nonempty_csv_rows(archive, member)
    if rows[0][:5] != ["amplitude", "fs", "lines", "u", "y"]:
        raise ValueError(f"unexpected fixed-amplitude validation header in {member}: {rows[0][:5]}")
    amplitude = float(rows[1][0])
    if not np.isclose(amplitude, AMPLITUDES[level - 1]) or float(rows[1][1]) != 78125.0:
        raise ValueError(f"unexpected amplitude or sample rate in {member}")
    return ParallelWHInputRecord(
        record_id=f"validation_level{level}",
        source_split="official_validation",
        regime="fixed_amplitude_multisine",
        amplitude=amplitude,
        realization_index=None,
        input_u=_finite_column(rows, 3, member),
        source_member=member,
    )


def _arrow_input_record(archive: zipfile.ZipFile) -> ParallelWHInputRecord:
    member = "ParWHFiles/ParWHData_ValidationArrow.csv"
    rows = _nonempty_csv_rows(archive, member)
    if rows[0][:4] != ["fs", "lines", "u", "y"] or float(rows[1][0]) != 78125.0:
        raise ValueError(f"unexpected growing-amplitude validation header in {member}: {rows[0][:4]}")
    return ParallelWHInputRecord(
        record_id="validation_growing_amplitude",
        source_split="official_validation",
        regime="growing_amplitude_gaussian_noise",
        amplitude=None,
        realization_index=None,
        input_u=_finite_column(rows, 2, member),
        source_member=member,
    )


def load_parallel_wiener_hammerstein_inputs(raw_archive: str | Path) -> list[ParallelWHInputRecord]:
    """Load only published U columns for output-isolated synthetic controls."""
    archive_path = Path(raw_archive)
    with zipfile.ZipFile(archive_path) as archive:
        records = [record for level in range(1, 6) for record in _estimation_input_records(archive, level)]
        records.extend(_validation_input_record(archive, level) for level in range(1, 6))
        records.append(_arrow_input_record(archive))
    if len(records) != 106:
        raise ValueError(f"expected 106 published Parallel WH input records, found {len(records)}")
    return records


def build_estimation_split(records: list[ParallelWHRecord]) -> dict[str, list[ParallelWHRecord]]:
    """Make the fixed 16/4 record split inside each published estimation level."""
    estimation = [record for record in records if record.source_split == "estimation"]
    official_validation = [record for record in records if record.source_split == "official_validation"]
    train: list[ParallelWHRecord] = []
    selection: list[ParallelWHRecord] = []
    for amplitude in AMPLITUDES:
        level = [record for record in estimation if np.isclose(record.amplitude, amplitude)]
        if [record.realization_index for record in level] != list(range(1, 21)):
            raise ValueError(f"estimation realization contract failed for amplitude {amplitude}")
        train.extend(level[:16])
        selection.extend(level[16:])
    if len(train) != 80 or len(selection) != 20 or len(official_validation) != 6:
        raise ValueError("Parallel WH record split contract failed")
    return {"train": train, "selection": selection, "official_validation": official_validation}


def build_estimation_input_split(records: list[ParallelWHInputRecord]) -> dict[str, list[ParallelWHInputRecord]]:
    """Apply the same record-level 16/4 protocol to input-only records."""
    estimation = [record for record in records if record.source_split == "estimation"]
    official_validation = [record for record in records if record.source_split == "official_validation"]
    train: list[ParallelWHInputRecord] = []
    selection: list[ParallelWHInputRecord] = []
    for amplitude in AMPLITUDES:
        level = [record for record in estimation if np.isclose(record.amplitude, amplitude)]
        if [record.realization_index for record in level] != list(range(1, 21)):
            raise ValueError(f"estimation realization contract failed for amplitude {amplitude}")
        train.extend(level[:16])
        selection.extend(level[16:])
    if len(train) != 80 or len(selection) != 20 or len(official_validation) != 6:
        raise ValueError("Parallel WH input split contract failed")
    return {"train": train, "selection": selection, "official_validation": official_validation}


def _metrics(observed: np.ndarray, predicted: np.ndarray, *, warmup: int = 1, arrow: bool = False) -> dict[str, float]:
    start = max(warmup, 500 if arrow else 0)
    observed = np.asarray(observed, dtype=float)[start:]
    predicted = np.asarray(predicted, dtype=float)[start:]
    if observed.shape != predicted.shape or observed.size == 0 or not np.all(np.isfinite(predicted)):
        return {"mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan")}
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    return {"mse": float(np.mean(np.square(residual))), "rmse": rmse, "nrmse": rmse / scale}


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.median([row[key] for row in rows])) for key in ("mse", "rmse", "nrmse")}


def _score(records: list[ParallelWHRecord], predictor: Any, *, warmup: int = 1) -> dict[str, Any]:
    rows = []
    for record in records:
        metrics = _metrics(
            record.output_y,
            predictor(record),
            warmup=warmup,
            arrow=record.regime == "growing_amplitude_gaussian_noise",
        )
        rows.append({"record_id": record.record_id, "regime": record.regime, "amplitude": record.amplitude, **metrics})
    if not all(np.isfinite(row["nrmse"]) for row in rows):
        raise ValueError("non-finite Parallel WH baseline metric")
    return {"per_record": rows, "aggregate": _aggregate(rows)}


def _persistence(record: ParallelWHRecord) -> np.ndarray:
    return np.full_like(record.output_y, record.output_y[0], dtype=float)


def _arx_features(y: np.ndarray, u: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    columns = [y[order - lag : len(y) - lag] for lag in range(1, order + 1)]
    columns.extend(u[order - lag : len(u) - lag] for lag in range(0, order + 1))
    columns.append(np.ones(len(y) - order))
    return np.column_stack(columns), y[order:]


def _fit_arx(records: list[ParallelWHRecord], *, order: int = 4, ridge: float = 1e-4) -> dict[str, Any]:
    dimension = 2 * order + 2
    xtx = np.zeros((dimension, dimension))
    xty = np.zeros(dimension)
    for record in records:
        features, target = _arx_features(record.output_y, record.input_u, order)
        xtx += features.T @ features
        xty += features.T @ target
    penalty = np.eye(dimension) * ridge
    penalty[-1, -1] = 0.0
    return {"order": order, "ridge": ridge, "coefficients": np.linalg.solve(xtx + penalty, xty)}


def _arx_predict(record: ParallelWHRecord, model: dict[str, Any]) -> np.ndarray:
    order = int(model["order"])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    prediction = record.output_y[:order].astype(float, copy=True).tolist()
    for index in range(order, len(record.output_y)):
        y_lags = np.asarray(prediction[index - order : index][::-1])
        u_lags = record.input_u[index - order : index + 1][::-1]
        prediction.append(float(np.dot(coefficients, np.concatenate([y_lags, u_lags, [1.0]]))))
    return np.asarray(prediction, dtype=float)


def _fit_stable_first_order(records: list[ParallelWHRecord], alpha: float = 0.995) -> dict[str, float]:
    xtx = np.zeros((2, 2))
    xty = np.zeros(2)
    for record in records:
        features = np.column_stack([record.input_u[1:], np.ones(len(record.input_u) - 1)])
        target = record.output_y[1:] - alpha * record.output_y[:-1]
        xtx += features.T @ features
        xty += features.T @ target
    beta, intercept = np.linalg.solve(xtx, xty)
    return {"alpha": alpha, "beta": float(beta), "intercept": float(intercept)}


def _stable_first_order_predict(record: ParallelWHRecord, model: dict[str, float]) -> np.ndarray:
    prediction = np.empty_like(record.output_y, dtype=float)
    prediction[0] = record.output_y[0]
    for index in range(1, len(prediction)):
        prediction[index] = model["alpha"] * prediction[index - 1] + model["beta"] * record.input_u[index] + model["intercept"]
    return prediction


def _filtered_input(values: np.ndarray, alpha: float) -> np.ndarray:
    state = 0.0
    result = np.empty_like(values, dtype=float)
    for index, value in enumerate(values):
        state = alpha * state + float(value)
        result[index] = state
    return result


def _fit_fixed_parallel_wh(records: list[ParallelWHRecord]) -> dict[str, float]:
    """Fit a fixed two-branch causal baseline, resetting both branches per record."""
    input_scale = max(float(np.std(np.concatenate([record.input_u for record in records]))), 1e-12)
    xtx = np.zeros((3, 3))
    xty = np.zeros(3)
    for record in records:
        linear_branch = _filtered_input(record.input_u, 0.90)
        nonlinear_branch = _filtered_input(np.tanh(record.input_u / input_scale), 0.995)
        features = np.column_stack([linear_branch, nonlinear_branch, np.ones(len(record.input_u))])
        xtx += features.T @ features
        xty += features.T @ record.output_y
    beta_linear, beta_nonlinear, intercept = np.linalg.solve(xtx, xty)
    return {
        "linear_alpha": 0.90,
        "nonlinear_alpha": 0.995,
        "input_scale": input_scale,
        "beta_linear": float(beta_linear),
        "beta_nonlinear": float(beta_nonlinear),
        "intercept": float(intercept),
    }


def _fixed_parallel_wh_predict(record: ParallelWHRecord, model: dict[str, float]) -> np.ndarray:
    linear_branch = _filtered_input(record.input_u, model["linear_alpha"])
    nonlinear_branch = _filtered_input(np.tanh(record.input_u / model["input_scale"]), model["nonlinear_alpha"])
    return model["beta_linear"] * linear_branch + model["beta_nonlinear"] * nonlinear_branch + model["intercept"]


def evaluate_parallel_wiener_hammerstein_baselines(split: dict[str, list[ParallelWHRecord]]) -> dict[str, Any]:
    """Fit on estimation-train only; report selection and untouched official validation."""
    train = split["train"]
    arx = _fit_arx(train)
    stable = _fit_stable_first_order(train)
    parallel_wh = _fit_fixed_parallel_wh(train)
    models = {
        "persistence": ({}, _persistence, 1),
        "regularized_arx": ({"order": arx["order"], "ridge": arx["ridge"]}, lambda record: _arx_predict(record, arx), arx["order"]),
        "stable_first_order": (stable, lambda record: _stable_first_order_predict(record, stable), 1),
        "fixed_parallel_wiener_hammerstein": (parallel_wh, lambda record: _fixed_parallel_wh_predict(record, parallel_wh), 1),
    }
    results = {}
    for name, (parameters, predictor, warmup) in models.items():
        results[name] = {
            "parameters": parameters,
            "selection": _score(split["selection"], predictor, warmup=warmup),
            "official_validation": _score(split["official_validation"], predictor, warmup=warmup),
        }
    return results

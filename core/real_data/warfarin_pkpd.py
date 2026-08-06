from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.evaluation import MaskedMetrics, masked_metrics


CANONICAL_COLUMNS = [
    "subject_id",
    "time_h",
    "dose_mg",
    "endpoint",
    "observed_value",
    "weight_kg",
    "age_y",
    "sex",
    "is_dose",
    "is_observation",
]

ENDPOINT_BY_DVID = {
    1: "concentration",
    2: "pca_response",
}

ENDPOINT_COLUMNS = ("concentration", "pca_response")


@dataclass(frozen=True)
class SubjectTargetBundle:
    subject_id: str
    time_points: np.ndarray
    observed: np.ndarray
    observation_mask: np.ndarray
    y0: np.ndarray
    covariates: dict[str, Any]


def _read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", comment="#")


def _column_map(columns: list[str]) -> dict[str, str]:
    by_lower = {column.lower(): column for column in columns}
    aliases = {
        "id": "id",
        "subject_id": "id",
        "time": "time",
        "time_h": "time",
        "amt": "amt",
        "dose_mg": "amt",
        "dv": "dv",
        "observed_value": "dv",
        "dvid": "dvid",
        "endpoint_id": "dvid",
        "wt": "wt",
        "weight_kg": "wt",
        "age": "age",
        "age_y": "age",
        "sex": "sex",
    }
    resolved: dict[str, str] = {}
    for source_name, canonical_name in aliases.items():
        if source_name in by_lower and canonical_name not in resolved:
            resolved[canonical_name] = by_lower[source_name]
    return resolved


def normalize_warfarin_table(raw_path: str | Path) -> pd.DataFrame:
    """Normalize Monolix/nlmixr-style Warfarin PK/PD tables.

    The adapter keeps dose rows and observation rows, but target builders use only
    finite observations. `dvid` values 1 and 2 are mapped to concentration and
    Prothrombin Complex/PCA response respectively.
    """
    raw = _read_table(raw_path)
    columns = _column_map(list(raw.columns))
    required = {"id", "time", "dv", "dvid", "wt", "age", "sex"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"missing required Warfarin columns: {missing}")

    normalized = pd.DataFrame()
    normalized["subject_id"] = raw[columns["id"]].astype(str)
    normalized["time_h"] = pd.to_numeric(raw[columns["time"]], errors="coerce")
    if "amt" in columns:
        normalized["dose_mg"] = pd.to_numeric(raw[columns["amt"]], errors="coerce")
    else:
        normalized["dose_mg"] = np.nan
    dvid = pd.to_numeric(raw[columns["dvid"]], errors="coerce").astype("Int64")
    normalized["endpoint"] = dvid.map(lambda value: ENDPOINT_BY_DVID.get(int(value), f"endpoint_{value}") if pd.notna(value) else None)
    normalized["observed_value"] = pd.to_numeric(raw[columns["dv"]], errors="coerce")
    normalized["weight_kg"] = pd.to_numeric(raw[columns["wt"]], errors="coerce")
    normalized["age_y"] = pd.to_numeric(raw[columns["age"]], errors="coerce")
    normalized["sex"] = raw[columns["sex"]].astype(str)
    normalized["is_dose"] = normalized["dose_mg"].fillna(0.0).astype(float) > 0.0
    normalized["is_observation"] = normalized["observed_value"].notna()

    normalized = normalized[CANONICAL_COLUMNS]
    normalized = normalized.dropna(subset=["subject_id", "time_h", "endpoint"])
    if normalized[normalized["is_observation"]].empty:
        raise ValueError("Warfarin table contains no finite observations")
    return normalized.sort_values(["subject_id", "time_h", "endpoint"]).reset_index(drop=True)


def build_dense_endpoint_target(
    normalized: pd.DataFrame,
    *,
    subject_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the current sandbox-compatible dense `[concentration, response]` target.

    This is an interim bridge until endpoint-masked sparse evaluation lands. It
    averages selected subjects, pivots endpoints by time, and interpolates missing
    endpoint observations along the time axis.
    """
    observed = normalized[normalized["is_observation"]].copy()
    if subject_id is not None:
        observed = observed[observed["subject_id"].astype(str) == str(subject_id)]
    if observed.empty:
        raise ValueError("no observations available for selected Warfarin subject")

    pivot = observed.pivot_table(
        index="time_h",
        columns="endpoint",
        values="observed_value",
        aggfunc="mean",
    ).sort_index()
    for endpoint in ENDPOINT_COLUMNS:
        if endpoint not in pivot:
            raise ValueError(f"missing required endpoint `{endpoint}`")
    dense = pivot[list(ENDPOINT_COLUMNS)].interpolate(
        method="linear",
        limit_direction="both",
    )
    if dense.isna().any().any():
        raise ValueError("unable to build dense Warfarin endpoint target")
    time_points = dense.index.to_numpy(dtype=float)
    target = dense.to_numpy(dtype=float)
    return time_points, target


def build_endpoint_target_with_mask(
    normalized: pd.DataFrame,
    *,
    subject_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense values plus an endpoint-observation mask.

    Missing endpoint values are linearly interpolated for solver-compatible target
    arrays, while the mask records which endpoint/time cells were truly observed.
    Fitting and held-out metrics should use the mask.
    """
    observed = normalized[normalized["is_observation"]].copy()
    if subject_id is not None:
        observed = observed[observed["subject_id"].astype(str) == str(subject_id)]
    if observed.empty:
        raise ValueError("no observations available for selected Warfarin subject")

    pivot = observed.pivot_table(
        index="time_h",
        columns="endpoint",
        values="observed_value",
        aggfunc="mean",
    ).sort_index()
    for endpoint in ENDPOINT_COLUMNS:
        if endpoint not in pivot:
            raise ValueError(f"missing required endpoint `{endpoint}`")
    endpoint_frame = pivot[list(ENDPOINT_COLUMNS)]
    mask = endpoint_frame.notna().to_numpy(dtype=bool)
    dense = endpoint_frame.interpolate(method="linear", limit_direction="both")
    if dense.isna().any().any():
        raise ValueError("unable to build dense Warfarin endpoint target")
    return (
        dense.index.to_numpy(dtype=float),
        dense.to_numpy(dtype=float),
        mask,
    )


def _first_observed_value(subject_observed: pd.DataFrame, endpoint: str, default: float) -> float:
    endpoint_rows = subject_observed[subject_observed["endpoint"] == endpoint].sort_values("time_h")
    if endpoint_rows.empty:
        return float(default)
    return float(endpoint_rows.iloc[0]["observed_value"])


def _subject_covariates(subject_frame: pd.DataFrame) -> dict[str, Any]:
    first = subject_frame.sort_values("time_h").iloc[0]
    dose_rows = subject_frame[subject_frame["is_dose"]].sort_values("time_h")
    dose_mg = None
    if not dose_rows.empty:
        dose_mg = float(dose_rows.iloc[0]["dose_mg"])
    return {
        "weight_kg": float(first["weight_kg"]) if pd.notna(first["weight_kg"]) else None,
        "age_y": float(first["age_y"]) if pd.notna(first["age_y"]) else None,
        "sex": str(first["sex"]) if pd.notna(first["sex"]) else None,
        "dose_mg": dose_mg,
    }


def build_subject_target_bundles(
    normalized: pd.DataFrame,
    subject_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
) -> list[SubjectTargetBundle]:
    """Build per-subject target bundles for grouped PK/PD evaluation.

    This is the first subject-holdout building block. It does not change the
    dense sandbox path; callers opt in by using these bundles directly.
    """
    subjects = (
        [str(value) for value in subject_ids]
        if subject_ids is not None
        else sorted(normalized["subject_id"].astype(str).unique().tolist())
    )
    if not subjects:
        raise ValueError("at least one subject_id is required")

    bundles: list[SubjectTargetBundle] = []
    normalized_subject_ids = normalized["subject_id"].astype(str)
    for subject_id in subjects:
        subject_frame = normalized[normalized_subject_ids == str(subject_id)].copy()
        if subject_frame.empty:
            raise ValueError(f"unknown Warfarin subject_id: {subject_id}")
        time_points, observed, observation_mask = build_endpoint_target_with_mask(
            subject_frame,
            subject_id=str(subject_id),
        )
        subject_observed = subject_frame[subject_frame["is_observation"]].copy()
        y0 = np.array(
            [
                _first_observed_value(subject_observed, "concentration", 0.0),
                _first_observed_value(subject_observed, "pca_response", 100.0),
            ],
            dtype=float,
        )
        bundles.append(
            SubjectTargetBundle(
                subject_id=str(subject_id),
                time_points=time_points,
                observed=observed,
                observation_mask=observation_mask,
                y0=y0,
                covariates=_subject_covariates(subject_frame),
            )
        )
    return bundles


def _metrics_dict(metrics: MaskedMetrics) -> dict[str, Any]:
    return {
        "mse": float(metrics.mse),
        "rmse": float(metrics.rmse),
        "observed_count": int(metrics.observed_count),
    }


def subject_prediction_metrics(
    bundle: SubjectTargetBundle,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Score one subject prediction using only observed endpoint cells."""
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != bundle.observed.shape:
        raise ValueError(f"prediction shape mismatch for subject {bundle.subject_id}: {predicted.shape} != {bundle.observed.shape}")

    aggregate = masked_metrics(bundle.observed, predicted, bundle.observation_mask)
    endpoint_metrics: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(ENDPOINT_COLUMNS):
        endpoint_mask = bundle.observation_mask[:, endpoint_index]
        if not np.any(endpoint_mask):
            endpoint_metrics[endpoint] = {
                "mse": None,
                "rmse": None,
                "observed_count": 0,
            }
            continue
        endpoint_metrics[endpoint] = _metrics_dict(
            masked_metrics(
                bundle.observed[:, endpoint_index],
                predicted[:, endpoint_index],
                endpoint_mask,
            )
        )

    return {
        "subject_id": bundle.subject_id,
        "aggregate": _metrics_dict(aggregate),
        "endpoint_metrics": endpoint_metrics,
    }


def _prediction_for_subject(
    predictions_by_subject: dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
    bundle_index: int,
    subject_id: str,
) -> np.ndarray:
    if isinstance(predictions_by_subject, dict):
        if subject_id not in predictions_by_subject:
            raise ValueError(f"missing prediction for subject {subject_id}")
        return predictions_by_subject[subject_id]
    if bundle_index >= len(predictions_by_subject):
        raise ValueError(f"missing prediction at bundle index {bundle_index}")
    return predictions_by_subject[bundle_index]


def evaluate_subject_predictions(
    bundles: list[SubjectTargetBundle],
    predictions_by_subject: dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """Score grouped subject predictions with observation-count weighted metrics."""
    if not bundles:
        raise ValueError("at least one subject bundle is required")

    per_subject = []
    observed_values = []
    predicted_values = []
    for index, bundle in enumerate(bundles):
        predicted = np.asarray(
            _prediction_for_subject(predictions_by_subject, index, bundle.subject_id),
            dtype=float,
        )
        if predicted.shape != bundle.observed.shape:
            raise ValueError(f"prediction shape mismatch for subject {bundle.subject_id}: {predicted.shape} != {bundle.observed.shape}")
        per_subject.append(subject_prediction_metrics(bundle, predicted))
        observed_values.append(bundle.observed[bundle.observation_mask])
        predicted_values.append(predicted[bundle.observation_mask])

    observed_flat = np.concatenate(observed_values)
    predicted_flat = np.concatenate(predicted_values)
    aggregate = masked_metrics(observed_flat, predicted_flat)
    return {
        "subject_count": len(bundles),
        "aggregate": _metrics_dict(aggregate),
        "per_subject": per_subject,
    }


def predict_subject_bundles(
    model: Any,
    params: np.ndarray,
    bundles: list[SubjectTargetBundle],
) -> dict[str, np.ndarray]:
    """Simulate one parameter vector against each subject bundle."""
    if not bundles:
        raise ValueError("at least one subject bundle is required")
    params = np.asarray(params, dtype=float)
    if params.ndim != 1:
        raise ValueError(f"params must be one-dimensional, got shape {params.shape}")

    predictions: dict[str, np.ndarray] = {}
    for bundle in bundles:
        predicted = np.asarray(model.simulate(params, bundle.time_points, bundle.y0), dtype=float)
        if predicted.shape != bundle.observed.shape:
            raise ValueError(f"prediction shape mismatch for subject {bundle.subject_id}: {predicted.shape} != {bundle.observed.shape}")
        predictions[bundle.subject_id] = predicted
    return predictions


def grouped_subject_distances(
    model: Any,
    params_batch: np.ndarray,
    bundles: list[SubjectTargetBundle],
) -> np.ndarray:
    """Compute observation-weighted masked RMSE for each parameter vector.

    This is the grouped train-subject distance primitive for Warfarin PK/PD.
    It is intentionally opt-in and does not affect the existing dense-domain
    `SBIEngine` path.
    """
    if not bundles:
        raise ValueError("at least one subject bundle is required")

    params_batch = np.asarray(params_batch, dtype=float)
    if params_batch.ndim == 1:
        params_batch = params_batch[np.newaxis, :]
    if params_batch.ndim != 2:
        raise ValueError(f"params_batch must be one- or two-dimensional, got shape {params_batch.shape}")

    distances = []
    for params in params_batch:
        squared_error_sum = 0.0
        observed_count = 0
        for bundle in bundles:
            predicted = np.asarray(model.simulate(params, bundle.time_points, bundle.y0), dtype=float)
            if predicted.shape != bundle.observed.shape:
                raise ValueError(f"prediction shape mismatch for subject {bundle.subject_id}: {predicted.shape} != {bundle.observed.shape}")
            predicted_observed = predicted[bundle.observation_mask]
            if not np.all(np.isfinite(predicted_observed)):
                squared_error_sum = float("inf")
                break
            residuals = predicted_observed - bundle.observed[bundle.observation_mask]
            squared_error_sum += float(np.sum(np.square(residuals)))
            observed_count += int(np.sum(bundle.observation_mask))
        if observed_count == 0:
            raise ValueError("subject bundles must include at least one observed endpoint value")
        distances.append(float(np.sqrt(squared_error_sum / observed_count)))
    return np.asarray(distances, dtype=float)


def grouped_subject_distance(
    model: Any,
    params: np.ndarray,
    bundles: list[SubjectTargetBundle],
) -> float:
    """Compute grouped masked RMSE for one parameter vector."""
    return float(grouped_subject_distances(model, params, bundles)[0])


def write_normalized_artifacts(
    raw_path: str | Path,
    *,
    normalized_csv: str | Path,
    provenance_json: str | Path,
    source_url: str,
) -> pd.DataFrame:
    normalized = normalize_warfarin_table(raw_path)
    normalized_path = Path(normalized_csv)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(normalized_path, index=False)

    provenance = {
        "source_url": source_url,
        "raw_path": str(raw_path),
        "normalized_csv": str(normalized_path),
        "schema_version": 1,
        "row_count": int(len(normalized)),
        "observation_count": int(normalized["is_observation"].sum()),
        "subject_count": int(normalized["subject_id"].astype(str).nunique()),
        "subjects": sorted(normalized["subject_id"].astype(str).unique().tolist()),
        "endpoints": sorted(normalized["endpoint"].dropna().unique().tolist()),
        "subject_summary": subject_summary(normalized),
        "notes": [
            "Dense target generation is a solver-compatible bridge; observation masks define sparse endpoint objectives.",
            "Scientific promotion should use downloaded source data and subject-level holdouts.",
        ],
    }
    Path(provenance_json).write_text(__import__("json").dumps(provenance, indent=2, sort_keys=True) + "\n")
    return normalized


def write_dense_target(
    normalized: pd.DataFrame,
    *,
    data_dir: str | Path = "data",
    domain: str = "real_warfarin_pkpd",
    subject_id: str | None = None,
) -> dict[str, Any]:
    time_points, target, mask = build_endpoint_target_with_mask(normalized, subject_id=subject_id)
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = out_dir / f"{domain}_ground_truth.npy"
    time_points_path = out_dir / f"{domain}_time_points.npy"
    mask_path = out_dir / f"{domain}_observation_mask.npy"
    np.save(ground_truth_path, target)
    np.save(time_points_path, time_points)
    np.save(mask_path, mask)
    return {
        "domain": domain,
        "subject_id": subject_id,
        "ground_truth_path": str(ground_truth_path),
        "time_points_path": str(time_points_path),
        "observation_mask_path": str(mask_path),
        "observed_values": int(np.sum(mask)),
        "time_points": int(len(time_points)),
        "state_count": int(target.shape[1]),
    }


def subject_summary(normalized: pd.DataFrame) -> list[dict[str, Any]]:
    observed = normalized[normalized["is_observation"]].copy()
    summaries = []
    for subject_id, subject_frame in observed.groupby("subject_id", sort=True):
        endpoints = sorted(subject_frame["endpoint"].dropna().unique().tolist())
        summaries.append(
            {
                "subject_id": str(subject_id),
                "observation_count": int(len(subject_frame)),
                "time_point_count": int(subject_frame["time_h"].nunique()),
                "endpoints": endpoints,
                "min_time_h": float(subject_frame["time_h"].min()),
                "max_time_h": float(subject_frame["time_h"].max()),
            }
        )
    return summaries


def build_subject_holdout_split(
    normalized: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    subjects = np.array(sorted(normalized["subject_id"].astype(str).unique().tolist()), dtype=object)
    if len(subjects) < 2:
        raise ValueError("subject holdout requires at least two subjects")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_fraction))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_subjects = sorted(str(value) for value in shuffled[:train_count])
    test_subjects = sorted(str(value) for value in shuffled[train_count:])
    return {
        "schema_version": 1,
        "status": "ok",
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_observation_count": int(normalized[
            normalized["subject_id"].astype(str).isin(train_subjects) & normalized["is_observation"]
        ].shape[0]),
        "test_observation_count": int(normalized[
            normalized["subject_id"].astype(str).isin(test_subjects) & normalized["is_observation"]
        ].shape[0]),
    }


def write_subject_holdout_split(
    normalized: pd.DataFrame,
    *,
    path: str | Path,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_subject_holdout_split(normalized, train_fraction=train_fraction, seed=seed)
    except ValueError as exc:
        payload = {
            "schema_version": 1,
            "status": "skipped",
            "reason": str(exc),
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "subject_count": int(normalized["subject_id"].astype(str).nunique()),
        }
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

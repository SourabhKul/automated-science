from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainTestMetrics:
    split_index: int
    train_mse: float
    test_mse: float
    train_rmse: float
    test_rmse: float


@dataclass(frozen=True)
class MaskedMetrics:
    mse: float
    rmse: float
    observed_count: int


def split_time_series(time_points, data, train_fraction: float = 0.8):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    n = len(time_points)
    if n < 3:
        raise ValueError("need at least three time points for train/test split")
    split_idx = int(n * train_fraction)
    split_idx = min(max(split_idx, 1), n - 1)
    return (
        time_points[:split_idx],
        data[:split_idx],
        time_points[split_idx:],
        data[split_idx:],
        split_idx,
    )


def mse(y_true, y_pred, mask=None) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")
    squared = np.square(y_true - y_pred)
    if mask is None:
        return float(np.mean(squared))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != y_true.shape:
        raise ValueError(f"mask shape mismatch: {mask.shape} != {y_true.shape}")
    if not np.any(mask):
        raise ValueError("mask must include at least one observed value")
    return float(np.mean(squared[mask]))


def masked_metrics(y_true, y_pred, mask=None) -> MaskedMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")
    if mask is None:
        observed_count = int(y_true.size)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y_true.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {y_true.shape}")
        observed_count = int(np.sum(mask))
    value = mse(y_true, y_pred, mask)
    return MaskedMetrics(
        mse=value,
        rmse=float(np.sqrt(value)),
        observed_count=observed_count,
    )


def train_test_metrics(observed, predicted, train_fraction: float = 0.8, mask=None) -> TrainTestMetrics:
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    if observed.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {observed.shape} != {predicted.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {observed.shape}")
    split_idx = int(len(observed) * train_fraction)
    split_idx = min(max(split_idx, 1), len(observed) - 1)
    train_mask = mask[:split_idx] if mask is not None else None
    test_mask = mask[split_idx:] if mask is not None else None
    train_mse = mse(observed[:split_idx], predicted[:split_idx], train_mask)
    test_mse = mse(observed[split_idx:], predicted[split_idx:], test_mask)
    return TrainTestMetrics(
        split_index=split_idx,
        train_mse=train_mse,
        test_mse=test_mse,
        train_rmse=float(np.sqrt(train_mse)),
        test_rmse=float(np.sqrt(test_mse)),
    )


def write_metrics(metrics: TrainTestMetrics, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n")
    return path

"""Artificial-only raw-measurement candidate surface for Phase 73."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


FEATURE_COUNT = 11
QUALITY_BOUNDS = (0.0, 10.0)


@dataclass(frozen=True)
class WineMeasurement:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if len(values) != FEATURE_COUNT or not np.all(np.isfinite(values)):
            raise ValueError("Wine candidate must contain exactly 11 finite raw measurements")
        object.__setattr__(self, "values", values)


def measurement_from_payload(payload: Mapping[str, object]) -> WineMeasurement:
    if set(payload) != {"values"}:
        raise ValueError("Wine payload may contain only values")
    return WineMeasurement(tuple(payload["values"]))


def predict_independent(
    records: list[WineMeasurement], predictor: Callable[[float, tuple[float, ...]], float]
) -> np.ndarray:
    predictions = []
    for record in records:
        value = float(predictor(0.0, record.values))
        if not np.isfinite(value) or not QUALITY_BOUNDS[0] <= value <= QUALITY_BOUNDS[1]:
            raise ValueError("invalid Wine quality prediction")
        predictions.append(value)
    return np.asarray(predictions, dtype=float)

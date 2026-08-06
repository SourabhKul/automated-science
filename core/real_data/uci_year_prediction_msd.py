"""Artificial-only candidate surface for Phase 61 UCI YearPredictionMSD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


FEATURES = 90
TARGET_BOUNDS = (1922.0, 2011.0)


@dataclass(frozen=True)
class YearPredictionRecord:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (FEATURES,) or not np.all(np.isfinite(values)):
            raise ValueError("YearPredictionMSD candidate must be a finite 90-value vector")
        object.__setattr__(self, "values", values.copy())


def record_from_payload(payload: Mapping[str, object]) -> YearPredictionRecord:
    if set(payload) != {"values"}:
        raise ValueError("YearPredictionMSD payload may contain only values")
    return YearPredictionRecord(np.asarray(payload["values"], dtype=float))


def predict_independent(records: list[YearPredictionRecord], predictor: Callable[[float, np.ndarray], float]) -> np.ndarray:
    output = np.empty(len(records), dtype=float)
    for index, record in enumerate(records):
        prediction = float(predictor(0.0, record.values.copy()))
        if not np.isfinite(prediction) or not TARGET_BOUNDS[0] <= prediction <= TARGET_BOUNDS[1]:
            raise ValueError("YearPredictionMSD predictor emitted an invalid release year")
        output[index] = prediction
    return output

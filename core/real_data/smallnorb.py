"""Artificial-only candidate surface for Phase 64 creator-published smallNORB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SHAPE = (2, 96, 96)
CLASSES = 5


@dataclass(frozen=True)
class SmallNorbRecord:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != SHAPE or not np.all(np.isfinite(values)):
            raise ValueError("smallNORB candidate must be a finite 2x96x96 stereo grid")
        object.__setattr__(self, "values", values.copy())


def record_from_payload(payload: Mapping[str, object]) -> SmallNorbRecord:
    if set(payload) != {"values"}:
        raise ValueError("smallNORB payload may contain only values")
    return SmallNorbRecord(np.asarray(payload["values"], dtype=float))


def predict_independent(records: list[SmallNorbRecord], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    output = np.empty((len(records), CLASSES), dtype=float)
    for index, record in enumerate(records):
        probabilities = np.asarray(predictor(0.0, record.values.copy()), dtype=float)
        if probabilities.shape != (CLASSES,) or not np.all(np.isfinite(probabilities)) or not np.all((0.0 <= probabilities) & (probabilities <= 1.0)) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("smallNORB predictor emitted invalid probabilities")
        output[index] = probabilities
    return output

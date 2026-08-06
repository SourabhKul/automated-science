"""Artificial-only candidate surface for Phase 67 creator-published STL-10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SHAPE = (3, 96, 96)
LABELS = tuple(range(10))


@dataclass(frozen=True)
class Stl10Image:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != SHAPE or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
            raise ValueError("STL-10 candidate must be finite 3x96x96 in 0..255")
        object.__setattr__(self, "values", values.copy())


def image_from_payload(payload: Mapping[str, object]) -> Stl10Image:
    if set(payload) != {"values"}:
        raise ValueError("STL-10 payload may contain only values")
    return Stl10Image(np.asarray(payload["values"], dtype=float))


def predict_independent(images: list[Stl10Image], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    probabilities = []
    for image in images:
        value = np.asarray(predictor(0.0, image.values.copy()), dtype=float)
        if value.shape != (10,) or not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1) or not np.isclose(value.sum(), 1.0):
            raise ValueError("invalid STL-10 probabilities")
        probabilities.append(value)
    return np.asarray(probabilities)

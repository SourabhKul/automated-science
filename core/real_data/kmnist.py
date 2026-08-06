"""Artificial-only raw-grid candidate mechanics for Phase 70 KMNIST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

SHAPE = (28, 28)
CLASS_COUNT = 10
FILES = ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")


@dataclass(frozen=True)
class KmnistImage:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != SHAPE or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
            raise ValueError("KMNIST image must be a finite 28x28 raw grid in 0..255")
        object.__setattr__(self, "values", values.copy())


def image_from_payload(payload: Mapping[str, object]) -> KmnistImage:
    if set(payload) != {"values"}:
        raise ValueError("KMNIST payload may contain only raw grid values")
    return KmnistImage(np.asarray(payload["values"], dtype=float))


def predict_independent(images: list[KmnistImage], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    probabilities = []
    for image in images:
        value = np.asarray(predictor(0.0, image.values.copy()), dtype=float)
        if value.shape != (CLASS_COUNT,) or not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1) or not np.isclose(value.sum(), 1.0):
            raise ValueError("invalid KMNIST probabilities")
        probabilities.append(value)
    return np.asarray(probabilities)

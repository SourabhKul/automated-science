"""Artificial-only Phase 65 EMNIST Balanced candidate mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SHAPE = (28, 28)
LABELS = tuple(range(47))


@dataclass(frozen=True)
class EmnistImage:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != SHAPE or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
            raise ValueError("EMNIST candidate must be finite 28x28 source-unit pixels in 0..255")
        object.__setattr__(self, "values", values.copy())


def image_from_payload(payload: Mapping[str, object]) -> EmnistImage:
    if set(payload) != {"values"}:
        raise ValueError("EMNIST payload may contain only raw candidate values")
    return EmnistImage(np.asarray(payload["values"], dtype=float))


def predict_independent(images: list[EmnistImage], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    output = np.empty((len(images), len(LABELS)), dtype=float)
    for index, image in enumerate(images):
        probabilities = np.asarray(predictor(0.0, image.values.copy()), dtype=float)
        if probabilities.shape != (len(LABELS),) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("EMNIST predictor emitted invalid probabilities")
        output[index] = probabilities
    return output

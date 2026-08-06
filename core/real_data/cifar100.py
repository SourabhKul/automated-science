"""Artificial-only candidate surface for Phase 60 CIFAR-100 fine labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SHAPE = (3, 32, 32)
FINE_LABELS = tuple(range(100))


@dataclass(frozen=True)
class Cifar100Image:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != SHAPE or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
            raise ValueError("CIFAR-100 candidate must be finite 3x32x32 in 0..255")
        object.__setattr__(self, "values", values.copy())


def image_from_payload(payload: Mapping[str, object]) -> Cifar100Image:
    if set(payload) != {"values"}:
        raise ValueError("CIFAR-100 payload may contain only values")
    return Cifar100Image(np.asarray(payload["values"], dtype=float))


def predict_independent(images: list[Cifar100Image], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    output = np.empty((len(images), len(FINE_LABELS)), dtype=float)
    for index, image in enumerate(images):
        probabilities = np.asarray(predictor(0.0, image.values.copy()), dtype=float)
        if probabilities.shape != (len(FINE_LABELS),) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("CIFAR-100 predictor emitted invalid probabilities")
        output[index] = probabilities
    return output

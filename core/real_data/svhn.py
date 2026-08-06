"""Artificial-only candidate surface for Phase 59 SVHN cropped digits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SHAPE = (3, 32, 32)
RAW_LABELS = tuple(range(1, 11))
NORMALIZED_LABELS = tuple(range(10))


@dataclass(frozen=True)
class SvhnImage:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != SHAPE or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
            raise ValueError("SVHN candidate must be finite 3x32x32 in 0..255")
        object.__setattr__(self, "values", values.copy())


def normalized_label(raw_label: int) -> int:
    if raw_label not in RAW_LABELS:
        raise ValueError("SVHN raw label must be in 1..10")
    return raw_label % 10


def image_from_payload(payload: Mapping[str, object]) -> SvhnImage:
    if set(payload) != {"values"}:
        raise ValueError("SVHN payload may contain only values")
    return SvhnImage(np.asarray(payload["values"], dtype=float))


def predict_independent(images: list[SvhnImage], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    output = np.empty((len(images), len(NORMALIZED_LABELS)), dtype=float)
    for index, image in enumerate(images):
        probabilities = np.asarray(predictor(0.0, image.values.copy()), dtype=float)
        if probabilities.shape != (len(NORMALIZED_LABELS),) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("SVHN predictor emitted invalid probabilities")
        output[index] = probabilities
    return output

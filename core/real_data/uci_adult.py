"""Artificial-only raw-token candidate surface for Phase 69 UCI Adult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

FIELD_COUNT = 14
CLASS_COUNT = 2


@dataclass(frozen=True)
class AdultRow:
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        fields = tuple(str(value) for value in self.fields)
        if len(fields) != FIELD_COUNT or any(not value for value in fields):
            raise ValueError("Adult candidate must contain exactly 14 nonempty raw tokens")
        object.__setattr__(self, "fields", fields)


def row_from_payload(payload: Mapping[str, object]) -> AdultRow:
    if set(payload) != {"fields"}:
        raise ValueError("Adult payload may contain only fields")
    return AdultRow(tuple(payload["fields"]))


def predict_independent(rows: list[AdultRow], predictor: Callable[[float, tuple[str, ...]], np.ndarray]) -> np.ndarray:
    probabilities = []
    for row in rows:
        value = np.asarray(predictor(0.0, row.fields), dtype=float)
        if value.shape != (CLASS_COUNT,) or not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1) or not np.isclose(value.sum(), 1.0):
            raise ValueError("invalid Adult probabilities")
        probabilities.append(value)
    return np.asarray(probabilities)

"""Artificial-only candidate surface for Phase 62 UCI HIGGS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


FEATURES = 28


@dataclass(frozen=True)
class HiggsRecord:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (FEATURES,) or not np.all(np.isfinite(values)):
            raise ValueError("HIGGS candidate must be a finite 28-value vector")
        object.__setattr__(self, "values", values.copy())


def record_from_payload(payload: Mapping[str, object]) -> HiggsRecord:
    if set(payload) != {"values"}:
        raise ValueError("HIGGS payload may contain only values")
    return HiggsRecord(np.asarray(payload["values"], dtype=float))


def predict_independent(records: list[HiggsRecord], predictor: Callable[[float, np.ndarray], float]) -> np.ndarray:
    output = np.empty(len(records), dtype=float)
    for index, record in enumerate(records):
        probability = float(predictor(0.0, record.values.copy()))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("HIGGS predictor emitted an invalid probability")
        output[index] = probability
    return output

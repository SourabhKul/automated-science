"""Isolated artificial candidate surface for Phase 54 Image Segmentation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np

FEATURES = 19
CLASS_LABELS = ("BRICKFACE", "CEMENT", "FOLIAGE", "GRASS", "PATH", "SKY", "WINDOW")
SOURCE_FILES = ("segmentation.data", "segmentation.test")

@dataclass(frozen=True)
class SegmentationRow:
    values: np.ndarray
    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (FEATURES,) or not np.all(np.isfinite(values)):
            raise ValueError("segmentation candidate must be finite shape (19,)")
        object.__setattr__(self, "values", values.copy())

@dataclass(frozen=True)
class InjectedSegmentationRecord:
    source_file: str
    split: str
    artificial_key: str
    artificial_values: np.ndarray
    artificial_label: str
    def __post_init__(self) -> None:
        if self.source_file not in SOURCE_FILES or self.split not in {"train", "selection", "external"} or not self.artificial_key or self.artificial_label not in CLASS_LABELS:
            raise ValueError("invalid injected segmentation metadata")
        SegmentationRow(self.artificial_values)
    def candidate_input(self) -> SegmentationRow:
        return SegmentationRow(self.artificial_values)

def row_from_payload(payload: Mapping[str, object]) -> SegmentationRow:
    if set(payload) != {"values"}:
        raise ValueError("segmentation payload may contain only values")
    return SegmentationRow(np.asarray(payload["values"], dtype=float))

def predict_independent_rows(rows: list[SegmentationRow], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    out = np.empty((len(rows), len(CLASS_LABELS)), dtype=float)
    for index, row in enumerate(rows):
        prob = np.asarray(predictor(0.0, row.values.copy()), dtype=float)
        if prob.shape != (len(CLASS_LABELS),) or not np.all(np.isfinite(prob)) or np.any(prob < 0) or np.any(prob > 1) or not np.isclose(prob.sum(), 1.0):
            raise ValueError("segmentation predictor emitted invalid probabilities")
        out[index] = prob
    return out

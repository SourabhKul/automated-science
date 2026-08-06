"""Isolated raw-neighborhood surfaces for Phase 47 Statlog Landsat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


PIXELS = 9
BANDS = 4
CLASS_LABELS = (1, 2, 3, 4, 5, 7)
SOURCE_FILES = ("sat.trn", "sat.tst")


@dataclass(frozen=True)
class LandsatNeighborhood:
    """Candidate-facing finite raw values with no provenance or target fields."""

    values: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values, dtype=float)
        if matrix.shape != (PIXELS, BANDS) or not np.all(np.isfinite(matrix)):
            raise ValueError("Landsat candidate must be finite shape (9, 4)")
        object.__setattr__(self, "values", matrix.copy())


@dataclass(frozen=True)
class LandsatInjectedRecord:
    """Mechanics-only record retaining artificial metadata outside the candidate."""

    source_file: str
    split: str
    artificial_row_key: str
    artificial_values: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        values = np.asarray(self.artificial_values, dtype=float)
        if self.source_file not in SOURCE_FILES or self.split not in {"train", "external"}:
            raise ValueError("Landsat injected record has invalid source-file assignment")
        if not self.artificial_row_key or self.artificial_label not in CLASS_LABELS:
            raise ValueError("Landsat injected record has invalid artificial metadata")
        if values.shape != (PIXELS, BANDS) or not np.all(np.isfinite(values)):
            raise ValueError("Landsat injected record must contain finite 9x4 values")
        object.__setattr__(self, "artificial_values", values.copy())

    def candidate_input(self) -> LandsatNeighborhood:
        return LandsatNeighborhood(self.artificial_values)


def neighborhood_from_payload(payload: Mapping[str, object]) -> LandsatNeighborhood:
    if set(payload) != {"values"}:
        raise ValueError("Landsat payload may contain only values; target, file, split, row, and location are forbidden")
    return LandsatNeighborhood(np.asarray(payload["values"], dtype=float))


def partition_injected_records(records: list[LandsatInjectedRecord]) -> dict[str, list[LandsatInjectedRecord]]:
    expected = {"train": "sat.trn", "external": "sat.tst"}
    partitioned = {name: [] for name in expected}
    seen_files: set[str] = set()
    for record in records:
        if record.source_file != expected[record.split] or record.source_file in seen_files:
            raise ValueError("Landsat injected record crosses frozen source-file split")
        seen_files.add(record.source_file)
        partitioned[record.split].append(record)
    if seen_files != set(SOURCE_FILES):
        raise ValueError("Landsat injected records omit or add a source file")
    return partitioned


def predict_independent_neighborhoods(
    neighborhoods: list[LandsatNeighborhood], predictor: Callable[[float, np.ndarray], np.ndarray]
) -> np.ndarray:
    probabilities = np.empty((len(neighborhoods), len(CLASS_LABELS)), dtype=float)
    for index, neighborhood in enumerate(neighborhoods):
        values = np.asarray(predictor(0.0, neighborhood.values.copy()), dtype=float)
        if values.shape != (len(CLASS_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("Landsat predictor emitted invalid class probabilities")
        if np.any(values < 0) or np.any(values > 1) or not np.isclose(float(values.sum()), 1, atol=1e-9):
            raise ValueError("Landsat predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

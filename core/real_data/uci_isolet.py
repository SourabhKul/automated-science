"""Isolated opaque-vector surfaces for the Phase 41 ISOLET benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


FEATURE_COUNT = 617
CLASS_LABELS = tuple(range(1, 27))
SOURCE_FILES = ("isolet1+2+3+4.data.Z", "isolet5.data.Z")


@dataclass(frozen=True)
class ISOLETRawVector:
    """Candidate-facing source-order vector with no label or source-file metadata."""

    features: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=float)
        if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError("ISOLET candidate vector must be a finite 617-feature raw vector")
        object.__setattr__(self, "features", values.copy())


@dataclass(frozen=True)
class ISOLETInjectedRecord:
    source_file: str
    artificial_row_key: str
    artificial_features: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        values = np.asarray(self.artificial_features, dtype=float)
        if self.source_file not in SOURCE_FILES or not self.artificial_row_key or self.artificial_label not in CLASS_LABELS:
            raise ValueError("ISOLET injected record has invalid source metadata or artificial label")
        if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError("ISOLET injected record must carry a finite 617-feature artificial vector")
        object.__setattr__(self, "artificial_features", values.copy())

    def candidate_input(self) -> ISOLETRawVector:
        return ISOLETRawVector(self.artificial_features)


def vector_from_payload(payload: Mapping[str, object]) -> ISOLETRawVector:
    if set(payload) != {"features"}:
        raise ValueError("ISOLET candidate payload may contain only features; target and source metadata are forbidden")
    return ISOLETRawVector(np.asarray(payload["features"], dtype=float))


def verify_injected_source_files(records: list[ISOLETInjectedRecord], split_path: str | Path) -> dict[str, list[ISOLETInjectedRecord]]:
    payload = json.loads(Path(split_path).read_text())
    expected = {str(payload.get("source_train")): "source_train", str(payload.get("external")): "external"}
    if set(expected) != set(SOURCE_FILES):
        raise ValueError("ISOLET frozen source-file split is invalid")
    partitioned = {"source_train": [], "external": []}
    observed: set[str] = set()
    for record in records:
        partition = expected.get(record.source_file)
        if partition is None or record.source_file in observed:
            raise ValueError("ISOLET mechanics expects one injected record for each frozen source file")
        observed.add(record.source_file)
        partitioned[partition].append(record)
    if observed != set(expected):
        raise ValueError("ISOLET injected records omit or add a frozen source file")
    return partitioned


def predict_independent_vectors(vectors: list[ISOLETRawVector], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    probabilities = np.empty((len(vectors), len(CLASS_LABELS)), dtype=float)
    for index, vector in enumerate(vectors):
        values = np.asarray(predictor(0.0, vector.features.copy()), dtype=float)
        if values.shape != (len(CLASS_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("ISOLET predictor emitted invalid class probabilities")
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError("ISOLET predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

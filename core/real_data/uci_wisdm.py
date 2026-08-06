"""Isolated raw-prefix surfaces for Phase 50 WISDM phone accelerometer."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


PREFIX_SAMPLES = 64
SOURCE_AXES = 3
FULL_SEGMENT_SAMPLES = 128
ACTIVITY_LABELS = tuple("ABCDEFGHIJKLMOPQRS")
SPLIT_COUNTS = {"train_subjects": 31, "selection_subjects": 10, "external_subjects": 10}


@dataclass(frozen=True)
class WISDMPhoneAccelPrefix:
    """Candidate-facing raw prefix without target, metadata, or suffix."""

    prefix: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.prefix, dtype=float)
        if values.shape != (PREFIX_SAMPLES, SOURCE_AXES) or not np.all(np.isfinite(values)):
            raise ValueError("WISDM candidate prefix must be finite shape (64, 3)")
        object.__setattr__(self, "prefix", values.copy())


@dataclass(frozen=True)
class WISDMInjectedRecord:
    """Artificial mechanics record whose identity and target stay evaluator-owned."""

    subject_id: int
    partition: str
    member_path: str
    artificial_prefix: np.ndarray
    artificial_suffix: np.ndarray
    artificial_label: str
    artificial_timestamps: np.ndarray

    def __post_init__(self) -> None:
        prefix = np.asarray(self.artificial_prefix, dtype=float)
        suffix = np.asarray(self.artificial_suffix, dtype=float)
        timestamps = np.asarray(self.artificial_timestamps, dtype=np.int64)
        if self.partition not in SPLIT_COUNTS or not isinstance(self.subject_id, int) or self.subject_id < 1600:
            raise ValueError("WISDM injected record has invalid participant or partition")
        if not self.member_path or self.artificial_label not in ACTIVITY_LABELS:
            raise ValueError("WISDM injected record has invalid artificial metadata")
        if prefix.shape != (PREFIX_SAMPLES, SOURCE_AXES) or suffix.shape != (FULL_SEGMENT_SAMPLES - PREFIX_SAMPLES, SOURCE_AXES):
            raise ValueError("WISDM injected record has invalid prefix or suffix shape")
        if timestamps.shape != (FULL_SEGMENT_SAMPLES,) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("WISDM injected record requires strictly increasing artificial timestamps")
        if not np.all(np.isfinite(prefix)) or not np.all(np.isfinite(suffix)):
            raise ValueError("WISDM injected record must be finite")
        object.__setattr__(self, "artificial_prefix", prefix.copy())
        object.__setattr__(self, "artificial_suffix", suffix.copy())
        object.__setattr__(self, "artificial_timestamps", timestamps.copy())

    def candidate_input(self) -> WISDMPhoneAccelPrefix:
        return WISDMPhoneAccelPrefix(self.artificial_prefix)


def prefix_from_payload(payload: Mapping[str, object]) -> WISDMPhoneAccelPrefix:
    """Admit only the literal causal prefix into candidate-facing evaluation."""
    if set(payload) != {"prefix"}:
        raise ValueError("WISDM payload may contain only prefix; target, identity, timestamps, suffix, path, and split fields are forbidden")
    return WISDMPhoneAccelPrefix(np.asarray(payload["prefix"], dtype=float))


def partition_injected_records(records: list[WISDMInjectedRecord], split_path: str | Path) -> dict[str, list[WISDMInjectedRecord]]:
    """Require exactly one artificial record for every frozen source participant."""
    payload = json.loads(Path(split_path).read_text())
    expected_by_partition = {name: [int(subject) for subject in payload.get(name, [])] for name in SPLIT_COUNTS}
    if {name: len(subjects) for name, subjects in expected_by_partition.items()} != SPLIT_COUNTS:
        raise ValueError("WISDM frozen split must contain 31/10/10 participants")
    expected = {subject: partition for partition, subjects in expected_by_partition.items() for subject in subjects}
    if len(expected) != sum(SPLIT_COUNTS.values()):
        raise ValueError("WISDM frozen split has duplicate participant IDs")
    partitioned = {name: [] for name in SPLIT_COUNTS}
    observed: set[int] = set()
    for record in records:
        if expected.get(record.subject_id) != record.partition:
            raise ValueError("WISDM injected record crosses frozen participant split")
        if record.subject_id in observed:
            raise ValueError("WISDM mechanics expects exactly one injected record per participant")
        observed.add(record.subject_id)
        partitioned[record.partition].append(record)
    if observed != set(expected):
        raise ValueError("WISDM injected records omit or add a locked participant")
    return partitioned


def predict_independent_prefixes(prefixes: list[WISDMPhoneAccelPrefix], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    """Use a fresh zero state for each segment and enforce finite probabilities."""
    probabilities = np.empty((len(prefixes), len(ACTIVITY_LABELS)), dtype=float)
    for index, prefix in enumerate(prefixes):
        values = np.asarray(predictor(0.0, prefix.prefix.copy()), dtype=float)
        if values.shape != (len(ACTIVITY_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("WISDM predictor emitted invalid class probabilities")
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError("WISDM predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

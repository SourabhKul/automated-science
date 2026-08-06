"""Isolated raw-prefix surfaces for Phase 45 Daily and Sports Activities."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


PREFIX_SAMPLES = 64
SOURCE_CHANNELS = 45
ACTIVITY_LABELS = tuple(range(1, 20))
SPLIT_COUNTS = {"train": 4, "selection": 2, "external": 2}


@dataclass(frozen=True)
class DailySportsPrefix:
    """Candidate-facing raw prefix without outcome, participant, path, or suffix."""

    prefix: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.prefix, dtype=float)
        if values.shape != (PREFIX_SAMPLES, SOURCE_CHANNELS) or not np.all(np.isfinite(values)):
            raise ValueError("Daily Sports candidate prefix must be finite shape (64, 45)")
        object.__setattr__(self, "prefix", values.copy())


@dataclass(frozen=True)
class DailySportsInjectedRecord:
    """Mechanics-only artificial record; its target and identity stay evaluator-owned."""

    participant_id: int
    split: str
    artificial_key: str
    artificial_prefix: np.ndarray
    artificial_suffix: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        prefix, suffix = np.asarray(self.artificial_prefix, dtype=float), np.asarray(self.artificial_suffix, dtype=float)
        if self.split not in SPLIT_COUNTS or not isinstance(self.participant_id, int) or self.participant_id <= 0:
            raise ValueError("Daily Sports injected record has invalid participant or split")
        if not self.artificial_key or self.artificial_label not in ACTIVITY_LABELS:
            raise ValueError("Daily Sports injected record has invalid artificial key or label")
        if prefix.shape != (PREFIX_SAMPLES, SOURCE_CHANNELS) or suffix.shape != (61, SOURCE_CHANNELS):
            raise ValueError("Daily Sports injected record has invalid prefix or suffix shape")
        if not np.all(np.isfinite(prefix)) or not np.all(np.isfinite(suffix)):
            raise ValueError("Daily Sports injected record must be finite")
        object.__setattr__(self, "artificial_prefix", prefix.copy())
        object.__setattr__(self, "artificial_suffix", suffix.copy())

    def candidate_input(self) -> DailySportsPrefix:
        return DailySportsPrefix(self.artificial_prefix)


def prefix_from_payload(payload: Mapping[str, object]) -> DailySportsPrefix:
    """Admit only the raw causal prefix into candidate-facing evaluation."""
    if set(payload) != {"prefix"}:
        raise ValueError("Daily Sports payload may contain only prefix; target, identity, suffix, and path fields are forbidden")
    return DailySportsPrefix(np.asarray(payload["prefix"], dtype=float))


def partition_injected_records(records: list[DailySportsInjectedRecord], split_path: str | Path) -> dict[str, list[DailySportsInjectedRecord]]:
    """Require exactly one artificial mechanics record for every frozen participant."""
    payload = json.loads(Path(split_path).read_text())
    expected_by_split = {name: [int(participant) for participant in payload.get(name, [])] for name in SPLIT_COUNTS}
    if {name: len(participants) for name, participants in expected_by_split.items()} != SPLIT_COUNTS:
        raise ValueError("Daily Sports frozen split must contain 4/2/2 participants")
    expected = {participant: split for split, participants in expected_by_split.items() for participant in participants}
    if len(expected) != sum(SPLIT_COUNTS.values()):
        raise ValueError("Daily Sports frozen split has duplicate participant IDs")
    partitioned = {name: [] for name in SPLIT_COUNTS}
    observed: set[int] = set()
    for record in records:
        if expected.get(record.participant_id) != record.split:
            raise ValueError("Daily Sports injected record crosses frozen participant split")
        if record.participant_id in observed:
            raise ValueError("Daily Sports mechanics expects exactly one injected record per participant")
        observed.add(record.participant_id)
        partitioned[record.split].append(record)
    if observed != set(expected):
        raise ValueError("Daily Sports injected records omit or add a locked participant")
    return partitioned


def predict_independent_prefixes(prefixes: list[DailySportsPrefix], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    """Use a fresh zero state for every prefix and enforce finite class probabilities."""
    probabilities = np.empty((len(prefixes), len(ACTIVITY_LABELS)), dtype=float)
    for index, prefix in enumerate(prefixes):
        values = np.asarray(predictor(0.0, prefix.prefix.copy()), dtype=float)
        if values.shape != (len(ACTIVITY_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("Daily Sports predictor emitted invalid class probabilities")
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError("Daily Sports predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

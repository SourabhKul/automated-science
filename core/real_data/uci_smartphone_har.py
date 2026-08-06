"""Isolated causal-prefix surfaces for the Phase 37 Smartphone HAR benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


CHANNEL_COUNT = 9
WINDOW_SAMPLES = 128
CAUSAL_PREFIX_SAMPLES = 64
ACTIVITY_LABELS = (1, 2, 3, 4, 5, 6)
SPLIT_COUNTS = {"train": 15, "selection": 6, "external": 9}


@dataclass(frozen=True)
class SmartphoneHARCausalPrefix:
    """Candidate-facing raw prefix without label, subject, or source-order fields."""

    prefix: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.prefix, dtype=float)
        if values.shape != (CHANNEL_COUNT, CAUSAL_PREFIX_SAMPLES) or not np.all(np.isfinite(values)):
            raise ValueError("smartphone-HAR candidate prefix must be finite shape (9, 64)")
        object.__setattr__(self, "prefix", values.copy())


@dataclass(frozen=True)
class SmartphoneHARInjectedWindow:
    """Mechanics-only wrapper; artificial outcome fields cannot enter candidates."""

    subject_id: int
    split: str
    source_row_key: str
    artificial_window: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        window = np.asarray(self.artificial_window, dtype=float)
        if self.split not in SPLIT_COUNTS or not isinstance(self.subject_id, int) or self.subject_id <= 0:
            raise ValueError("smartphone-HAR injected window has invalid subject or split")
        if not self.source_row_key or self.artificial_label not in ACTIVITY_LABELS:
            raise ValueError("smartphone-HAR injected window has invalid mechanics identity or artificial label")
        if window.shape != (CHANNEL_COUNT, WINDOW_SAMPLES) or not np.all(np.isfinite(window)):
            raise ValueError("smartphone-HAR injected window must be finite shape (9, 128)")
        object.__setattr__(self, "artificial_window", window.copy())

    def causal_input(self) -> SmartphoneHARCausalPrefix:
        return make_smartphone_har_causal_prefix(self.artificial_window)


def make_smartphone_har_causal_prefix(window: np.ndarray) -> SmartphoneHARCausalPrefix:
    """Evaluator-owned hold that exposes samples 0..63 and no suffix values."""
    values = np.asarray(window, dtype=float)
    if values.shape != (CHANNEL_COUNT, WINDOW_SAMPLES) or not np.all(np.isfinite(values)):
        raise ValueError("smartphone-HAR window must be finite shape (9, 128)")
    return SmartphoneHARCausalPrefix(values[:, :CAUSAL_PREFIX_SAMPLES])


def causal_prefix_from_payload(payload: Mapping[str, object]) -> SmartphoneHARCausalPrefix:
    """Reject target and identity fields before a candidate receives a prefix."""
    if set(payload) != {"prefix"}:
        raise ValueError("smartphone-HAR candidate payload must contain only prefix; target and subject fields are forbidden")
    return SmartphoneHARCausalPrefix(np.asarray(payload["prefix"], dtype=float))


def partition_injected_windows(
    windows: list[SmartphoneHARInjectedWindow], split_path: str | Path
) -> dict[str, list[SmartphoneHARInjectedWindow]]:
    """Verify each artificial mechanics record against the immutable subject split."""
    payload = json.loads(Path(split_path).read_text())
    expected_by_split = {name: [int(subject) for subject in payload.get(name, [])] for name in SPLIT_COUNTS}
    if {name: len(subjects) for name, subjects in expected_by_split.items()} != SPLIT_COUNTS:
        raise ValueError("smartphone-HAR frozen split must be 15/6/9 subjects")
    expected = {subject: split for split, subjects in expected_by_split.items() for subject in subjects}
    if len(expected) != sum(SPLIT_COUNTS.values()):
        raise ValueError("smartphone-HAR frozen split has duplicate subject IDs")
    partitioned = {name: [] for name in SPLIT_COUNTS}
    observed: set[int] = set()
    for window in windows:
        if expected.get(window.subject_id) != window.split:
            raise ValueError("smartphone-HAR injected record crosses frozen subject split")
        if window.subject_id in observed:
            raise ValueError("smartphone-HAR mechanics expects exactly one injected record per subject")
        observed.add(window.subject_id)
        partitioned[window.split].append(window)
    if observed != set(expected):
        raise ValueError("smartphone-HAR injected record set omits or adds a locked subject")
    if {name: len(records) for name, records in partitioned.items()} != SPLIT_COUNTS:
        raise ValueError("smartphone-HAR injected partition count mismatch")
    return partitioned


def predict_smartphone_har_prefixes(
    prefixes: list[SmartphoneHARCausalPrefix], predictor: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Evaluate independent prefixes with no retained evaluator state between records."""
    probabilities = np.empty((len(prefixes), len(ACTIVITY_LABELS)), dtype=float)
    for index, item in enumerate(prefixes):
        values = np.asarray(predictor(item.prefix.copy()), dtype=float)
        if values.shape != (len(ACTIVITY_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("smartphone-HAR predictor emitted invalid class probabilities")
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError("smartphone-HAR predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

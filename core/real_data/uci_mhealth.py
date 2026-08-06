"""Isolated raw-window surfaces for the Phase 40 MHEALTH benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


MOTION_CHANNELS = 21
WINDOW_SAMPLES = 100
ACTIVITY_LABELS = tuple(range(1, 13))
SPLIT_COUNTS = {"train": 4, "selection": 3, "external": 3}


@dataclass(frozen=True)
class MHealthMotionWindow:
    """Candidate-facing raw motion window without label, subject, or source order."""

    motion_window: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.motion_window, dtype=float)
        if values.shape != (MOTION_CHANNELS, WINDOW_SAMPLES) or not np.all(np.isfinite(values)):
            raise ValueError("MHEALTH candidate motion window must be finite shape (21, 100)")
        object.__setattr__(self, "motion_window", values.copy())


@dataclass(frozen=True)
class MHealthInjectedWindow:
    """Mechanics-only wrapper that retains artificial target and identity outside candidates."""

    subject_id: int
    split: str
    artificial_window_key: str
    artificial_motion_window: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        values = np.asarray(self.artificial_motion_window, dtype=float)
        if self.split not in SPLIT_COUNTS or not isinstance(self.subject_id, int) or self.subject_id <= 0:
            raise ValueError("MHEALTH injected window has invalid subject or split")
        if not self.artificial_window_key or self.artificial_label not in ACTIVITY_LABELS:
            raise ValueError("MHEALTH injected window has invalid artificial key or label")
        if values.shape != (MOTION_CHANNELS, WINDOW_SAMPLES) or not np.all(np.isfinite(values)):
            raise ValueError("MHEALTH injected motion window must be finite shape (21, 100)")
        object.__setattr__(self, "artificial_motion_window", values.copy())

    def candidate_input(self) -> MHealthMotionWindow:
        return MHealthMotionWindow(self.artificial_motion_window)


def motion_window_from_payload(payload: Mapping[str, object]) -> MHealthMotionWindow:
    """Reject target, identity, order, and co-intervention fields before candidate access."""
    if set(payload) != {"motion_window"}:
        raise ValueError("MHEALTH candidate payload may contain only motion_window; target and identity fields are forbidden")
    return MHealthMotionWindow(np.asarray(payload["motion_window"], dtype=float))


def partition_injected_windows(
    windows: list[MHealthInjectedWindow], split_path: str | Path
) -> dict[str, list[MHealthInjectedWindow]]:
    """Verify one artificial mechanics window for each locked source subject."""
    payload = json.loads(Path(split_path).read_text())
    expected_by_split = {name: [int(subject) for subject in payload.get(name, [])] for name in SPLIT_COUNTS}
    if {name: len(subjects) for name, subjects in expected_by_split.items()} != SPLIT_COUNTS:
        raise ValueError("MHEALTH frozen split must contain 4/3/3 subjects")
    expected = {subject: split for split, subjects in expected_by_split.items() for subject in subjects}
    if len(expected) != sum(SPLIT_COUNTS.values()):
        raise ValueError("MHEALTH frozen split has duplicate subject IDs")
    partitioned = {name: [] for name in SPLIT_COUNTS}
    observed: set[int] = set()
    for window in windows:
        if expected.get(window.subject_id) != window.split:
            raise ValueError("MHEALTH injected window crosses frozen subject split")
        if window.subject_id in observed:
            raise ValueError("MHEALTH mechanics expects exactly one injected window per subject")
        observed.add(window.subject_id)
        partitioned[window.split].append(window)
    if observed != set(expected):
        raise ValueError("MHEALTH injected windows omit or add a locked subject")
    if {name: len(items) for name, items in partitioned.items()} != SPLIT_COUNTS:
        raise ValueError("MHEALTH injected partition count mismatch")
    return partitioned


def predict_independent_windows(
    windows: list[MHealthMotionWindow], predictor: Callable[[float, np.ndarray], np.ndarray]
) -> np.ndarray:
    """Evaluate every raw motion window with a fresh zero state and bounded probabilities."""
    probabilities = np.empty((len(windows), len(ACTIVITY_LABELS)), dtype=float)
    for index, window in enumerate(windows):
        values = np.asarray(predictor(0.0, window.motion_window.copy()), dtype=float)
        if values.shape != (len(ACTIVITY_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("MHEALTH predictor emitted invalid class probabilities")
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError("MHEALTH predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

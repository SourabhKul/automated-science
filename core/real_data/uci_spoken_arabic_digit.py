"""Isolated first-frame surfaces for Phase 46 Spoken Arabic Digit."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


PREFIX_FRAMES = 4
MFCC_COEFFICIENTS = 13
DIGIT_LABELS = tuple(range(10))
SPLIT_COUNTS = {"train": 44, "selection": 22, "external": 22}


@dataclass(frozen=True)
class ArabicDigitPrefix:
    """Candidate-facing raw frames without label, speaker, ordering, or suffix."""

    frames: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.frames, dtype=float)
        if values.shape != (PREFIX_FRAMES, MFCC_COEFFICIENTS) or not np.all(np.isfinite(values)):
            raise ValueError("Arabic Digit candidate frames must be finite shape (4, 13)")
        object.__setattr__(self, "frames", values.copy())


@dataclass(frozen=True)
class ArabicDigitInjectedRecord:
    """Mechanics-only record retaining artificial target and speaker outside candidates."""

    source_partition: str
    speaker_id: int
    split: str
    artificial_key: str
    artificial_frames: np.ndarray
    artificial_suffix: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        frames, suffix = np.asarray(self.artificial_frames, dtype=float), np.asarray(self.artificial_suffix, dtype=float)
        if self.source_partition not in {"source_train", "official_test"} or self.split not in SPLIT_COUNTS or self.speaker_id <= 0:
            raise ValueError("Arabic Digit injected record has invalid source speaker or split")
        if not self.artificial_key or self.artificial_label not in DIGIT_LABELS:
            raise ValueError("Arabic Digit injected record has invalid artificial key or label")
        if frames.shape != (PREFIX_FRAMES, MFCC_COEFFICIENTS) or suffix.shape != (1, MFCC_COEFFICIENTS):
            raise ValueError("Arabic Digit injected record has invalid frame shape")
        if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(suffix)):
            raise ValueError("Arabic Digit injected record must be finite")
        object.__setattr__(self, "artificial_frames", frames.copy())
        object.__setattr__(self, "artificial_suffix", suffix.copy())

    def candidate_input(self) -> ArabicDigitPrefix:
        return ArabicDigitPrefix(self.artificial_frames)


def prefix_from_payload(payload: Mapping[str, object]) -> ArabicDigitPrefix:
    if set(payload) != {"frames"}:
        raise ValueError("Arabic Digit payload may contain only frames; target, speaker, file, order, and suffix fields are forbidden")
    return ArabicDigitPrefix(np.asarray(payload["frames"], dtype=float))


def frozen_speakers(split_path: str | Path) -> dict[str, set[tuple[str, int]]]:
    raw = json.loads(Path(split_path).read_text())
    groups = {
        "train": {("source_train", value) for key in ("train_male_speakers", "train_female_speakers") for value in raw.get(key, [])},
        "selection": {("source_train", value) for key in ("selection_male_speakers", "selection_female_speakers") for value in raw.get(key, [])},
        "external": {("official_test", value) for value in raw.get("external_official_test_speakers", [])},
    }
    if {name: len(values) for name, values in groups.items()} != SPLIT_COUNTS or len(set().union(*groups.values())) != sum(SPLIT_COUNTS.values()):
        raise ValueError("Arabic Digit frozen speaker split must contain disjoint 44/22/22 speakers")
    return groups


def partition_injected_records(records: list[ArabicDigitInjectedRecord], split_path: str | Path) -> dict[str, list[ArabicDigitInjectedRecord]]:
    expected = frozen_speakers(split_path)
    partitioned = {name: [] for name in SPLIT_COUNTS}
    observed: set[tuple[str, int]] = set()
    for record in records:
        key = (record.source_partition, record.speaker_id)
        if key not in expected[record.split] or key in observed:
            raise ValueError("Arabic Digit injected record crosses frozen speaker split")
        observed.add(key)
        partitioned[record.split].append(record)
    if observed != set().union(*expected.values()):
        raise ValueError("Arabic Digit injected records omit or add a locked speaker")
    return partitioned


def predict_independent_prefixes(prefixes: list[ArabicDigitPrefix], predictor: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    probabilities = np.empty((len(prefixes), len(DIGIT_LABELS)), dtype=float)
    for index, prefix in enumerate(prefixes):
        values = np.asarray(predictor(0.0, prefix.frames.copy()), dtype=float)
        if values.shape != (len(DIGIT_LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError("Arabic Digit predictor emitted invalid class probabilities")
        if np.any(values < 0) or np.any(values > 1) or not np.isclose(float(values.sum()), 1, atol=1e-9):
            raise ValueError("Arabic Digit predictor emitted out-of-bounds class probabilities")
        probabilities[index] = values
    return probabilities

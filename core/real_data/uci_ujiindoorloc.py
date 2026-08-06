"""Isolated raw-WLAN surfaces for Phase 49 UJIIndoorLoc."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

FEATURE_COUNT = 520
CLASS_LABELS = (0, 1, 2)
SOURCE_FILES = ("UJIndoorLoc/trainingData.csv", "UJIndoorLoc/validationData.csv")


@dataclass(frozen=True)
class UJIIndoorLocWLAN:
    waps: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.waps, dtype=float)
        if (
            values.shape != (FEATURE_COUNT,)
            or not np.all(np.isfinite(values))
            or np.any(values < -104)
            or np.any(values > 100)
            or not np.all(values == np.trunc(values))
        ):
            raise ValueError("UJIIndoorLoc candidate must be a finite integral 520-WAP source-unit vector")
        object.__setattr__(self, "waps", values.copy())


@dataclass(frozen=True)
class UJIIndoorLocInjectedRecord:
    source_file: str
    artificial_row_key: str
    artificial_waps: np.ndarray
    artificial_label: int

    def __post_init__(self) -> None:
        if self.source_file not in SOURCE_FILES or not self.artificial_row_key or self.artificial_label not in CLASS_LABELS:
            raise ValueError("invalid injected UJIIndoorLoc metadata")
        UJIIndoorLocWLAN(self.artificial_waps)

    def candidate_input(self) -> UJIIndoorLocWLAN:
        return UJIIndoorLocWLAN(self.artificial_waps)


def wlan_from_payload(payload: Mapping[str, object]) -> UJIIndoorLocWLAN:
    if set(payload) != {"waps"}:
        raise ValueError("UJIIndoorLoc candidate payload may only contain WAP values")
    return UJIIndoorLocWLAN(np.asarray(payload["waps"], dtype=float))


def verify_injected_source_files(
    records: list[UJIIndoorLocInjectedRecord], split_path: str | Path
) -> dict[str, list[UJIIndoorLocInjectedRecord]]:
    split = json.loads(Path(split_path).read_text())
    expected = {split.get("train"): "source_train", split.get("external"): "external"}
    if set(expected) != set(SOURCE_FILES):
        raise ValueError("frozen UJIIndoorLoc source-file split is invalid")
    result = {"source_train": [], "external": []}
    seen = set()
    for record in records:
        if record.source_file not in expected or record.source_file in seen:
            raise ValueError("injected records must include each frozen UJIIndoorLoc file once")
        seen.add(record.source_file)
        result[expected[record.source_file]].append(record)
    if seen != set(expected):
        raise ValueError("injected records omit a frozen UJIIndoorLoc file")
    return result


def predict_independent_wlans(
    vectors: list[UJIIndoorLocWLAN], predictor: Callable[[float, np.ndarray], np.ndarray]
) -> np.ndarray:
    probabilities = []
    for vector in vectors:
        values = np.asarray(predictor(0.0, vector.waps.copy()), dtype=float)
        if (
            values.shape != (len(CLASS_LABELS),)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
            or not np.isclose(values.sum(), 1.0, atol=1e-9)
        ):
            raise ValueError("UJIIndoorLoc predictor emitted invalid class probabilities")
        probabilities.append(values)
    return np.asarray(probabilities)

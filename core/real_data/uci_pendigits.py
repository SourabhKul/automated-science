"""Isolated raw-coordinate surfaces for the Phase 42 PenDigits benchmark."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping
import numpy as np

FEATURE_COUNT=16
CLASS_LABELS=tuple(range(10))
SOURCE_FILES=("pendigits.tra","pendigits.tes")

@dataclass(frozen=True)
class PenDigitsRawVector:
    coordinates: np.ndarray
    def __post_init__(self) -> None:
        values=np.asarray(self.coordinates,dtype=float)
        if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 100): raise ValueError("PenDigits candidate must be a finite 16-coordinate raw source-unit vector")
        object.__setattr__(self,"coordinates",values.copy())

@dataclass(frozen=True)
class PenDigitsInjectedRecord:
    source_file: str
    artificial_row_key: str
    artificial_coordinates: np.ndarray
    artificial_label: int
    def __post_init__(self) -> None:
        if self.source_file not in SOURCE_FILES or not self.artificial_row_key or self.artificial_label not in CLASS_LABELS: raise ValueError("invalid injected PenDigits metadata")
        PenDigitsRawVector(self.artificial_coordinates)
    def candidate_input(self) -> PenDigitsRawVector: return PenDigitsRawVector(self.artificial_coordinates)

def vector_from_payload(payload: Mapping[str, object]) -> PenDigitsRawVector:
    if set(payload) != {"coordinates"}: raise ValueError("PenDigits candidate payload may only contain coordinates")
    return PenDigitsRawVector(np.asarray(payload["coordinates"],dtype=float))

def verify_injected_source_files(records: list[PenDigitsInjectedRecord], split_path: str | Path) -> dict[str,list[PenDigitsInjectedRecord]]:
    split=json.loads(Path(split_path).read_text()); expected={split.get("source_train"):"source_train",split.get("external"):"external"}
    if set(expected) != set(SOURCE_FILES): raise ValueError("frozen PenDigits file split is invalid")
    result={"source_train":[],"external":[]}; seen=set()
    for record in records:
        if record.source_file not in expected or record.source_file in seen: raise ValueError("injected records must include each frozen PenDigits source file once")
        seen.add(record.source_file); result[expected[record.source_file]].append(record)
    if seen != set(expected): raise ValueError("injected records omit a frozen source file")
    return result

def predict_independent_vectors(vectors: list[PenDigitsRawVector], predictor: Callable[[float,np.ndarray],np.ndarray]) -> np.ndarray:
    probabilities=[]
    for vector in vectors:
        values=np.asarray(predictor(0.,vector.coordinates.copy()),dtype=float)
        if values.shape != (len(CLASS_LABELS),) or not np.all(np.isfinite(values)) or np.any(values<0) or np.any(values>1) or not np.isclose(values.sum(),1.,atol=1e-9): raise ValueError("PenDigits predictor emitted invalid class probabilities")
        probabilities.append(values)
    return np.asarray(probabilities)

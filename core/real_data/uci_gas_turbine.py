"""Isolated raw-schema surfaces for the Phase 38 gas-turbine NOx benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


CANDIDATE_FIELDS = ("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY")
SPLIT_NAMES = ("train", "selection", "external")
NOX_BOUNDS_MG_M3 = (0.0, 150.0)


@dataclass(frozen=True)
class UCIGasTurbineRawInput:
    """Candidate-facing raw source-unit row with no outcome or annual identity."""

    raw_source_units: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.raw_source_units, dtype=float)
        if values.shape != (len(CANDIDATE_FIELDS),) or not np.all(np.isfinite(values)):
            raise ValueError("UCI gas-turbine candidate row must be a finite raw-unit vector of nine fields")
        object.__setattr__(self, "raw_source_units", values.copy())


@dataclass(frozen=True)
class UCIGasTurbineInjectedRecord:
    """Mechanics-only wrapper that keeps artificial outcomes and member data outside candidates."""

    year: int
    split: str
    artificial_row_key: str
    raw_source_units: np.ndarray
    artificial_nox_mg_m3: float
    artificial_co_mg_m3: float

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_source_units, dtype=float)
        outcomes = np.asarray([self.artificial_nox_mg_m3, self.artificial_co_mg_m3], dtype=float)
        if self.split not in SPLIT_NAMES or not isinstance(self.year, int) or not self.artificial_row_key:
            raise ValueError("UCI gas-turbine injected record has invalid year, split, or artificial row key")
        if raw.shape != (len(CANDIDATE_FIELDS),) or not np.all(np.isfinite(raw)):
            raise ValueError("UCI gas-turbine injected record needs nine finite artificial raw-unit values")
        if not np.all(np.isfinite(outcomes)):
            raise ValueError("UCI gas-turbine injected outcomes must be finite artificial values")
        object.__setattr__(self, "raw_source_units", raw.copy())

    def candidate_input(self) -> UCIGasTurbineRawInput:
        return UCIGasTurbineRawInput(self.raw_source_units)


def candidate_input_from_payload(payload: Mapping[str, object]) -> UCIGasTurbineRawInput:
    """Reject outcome, member, ordering, and duplicate metadata before candidate access."""
    if set(payload) != {"raw_source_units"}:
        raise ValueError("UCI gas-turbine candidate payload may contain only raw_source_units; outcome and member fields are forbidden")
    return UCIGasTurbineRawInput(np.asarray(payload["raw_source_units"], dtype=float))


def _expected_year_splits(split_path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(split_path).read_text())
    train_years = [int(year) for year in payload.get("train_years", [])]
    selection_year = payload.get("selection_year")
    external_year = payload.get("external_year")
    if len(train_years) != 3 or selection_year is None or external_year is None:
        raise ValueError("UCI gas-turbine frozen split must be 2011--2013 train, 2014 selection, and 2015 external")
    expected = {year: "train" for year in train_years}
    expected[int(selection_year)] = "selection"
    expected[int(external_year)] = "external"
    if expected != {2011: "train", 2012: "train", 2013: "train", 2014: "selection", 2015: "external"}:
        raise ValueError("UCI gas-turbine frozen year split does not match the declared annual members")
    return expected


def partition_injected_records(
    records: list[UCIGasTurbineInjectedRecord], split_path: str | Path
) -> dict[str, list[UCIGasTurbineInjectedRecord]]:
    """Verify one artificial record per frozen annual member without exposing member data to candidates."""
    expected = _expected_year_splits(split_path)
    partitioned = {name: [] for name in SPLIT_NAMES}
    observed_years: set[int] = set()
    for record in records:
        if expected.get(record.year) != record.split:
            raise ValueError("UCI gas-turbine injected record crosses the frozen year split")
        if record.year in observed_years:
            raise ValueError("UCI gas-turbine mechanics smoke expects exactly one injected record per annual member")
        observed_years.add(record.year)
        partitioned[record.split].append(record)
    if observed_years != set(expected):
        raise ValueError("UCI gas-turbine injected records omit or add an annual member")
    if {name: len(items) for name, items in partitioned.items()} != {"train": 3, "selection": 1, "external": 1}:
        raise ValueError("UCI gas-turbine injected partition counts must be 3/1/1 annual members")
    return partitioned


def predict_independent_rows(
    rows: list[UCIGasTurbineRawInput],
    predictor: Callable[[float, np.ndarray], float],
    *,
    bounds_mg_m3: tuple[float, float] = NOX_BOUNDS_MG_M3,
) -> np.ndarray:
    """Evaluate each candidate row with a fresh zero state and finite source-unit bounds."""
    lower, upper = (float(bounds_mg_m3[0]), float(bounds_mg_m3[1]))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("UCI gas-turbine NOx bounds must be finite and ordered")
    predictions = np.empty(len(rows), dtype=float)
    for index, row in enumerate(rows):
        prediction = float(predictor(0.0, row.raw_source_units.copy()))
        if not np.isfinite(prediction) or prediction < lower or prediction > upper:
            raise ValueError("UCI gas-turbine predictor emitted a non-finite or out-of-bounds NOx value")
        predictions[index] = prediction
    return predictions

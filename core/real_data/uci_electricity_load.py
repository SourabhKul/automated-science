"""Isolated causal surfaces for the Phase 36 electricity-load benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np


WEEK_SAMPLES = 672
SAMPLES_PER_DAY = 96
HISTORY_LAGS = (1, 96)
SOURCE_KW_UNIT = "source kW"
SPLIT_COUNTS = {"train": 240, "selection": 65, "external": 65}


def native_week_calendar() -> np.ndarray:
    """Return the fixed weekday/quarter-hour phase for a source-labelled week."""
    quarter_hour = np.tile(np.arange(SAMPLES_PER_DAY, dtype=float), 7)
    weekday = np.repeat(np.arange(7, dtype=float), SAMPLES_PER_DAY)
    return np.column_stack([quarter_hour, weekday])


@dataclass(frozen=True)
class ElectricityLoadInputWeek:
    """Candidate-facing own-history/calendar surface without target or client identity."""

    causal_history_kw: np.ndarray
    calendar_phase: np.ndarray

    def __post_init__(self) -> None:
        history = np.asarray(self.causal_history_kw, dtype=float)
        calendar = np.asarray(self.calendar_phase, dtype=float)
        if history.shape != (WEEK_SAMPLES, len(HISTORY_LAGS)) or not np.all(np.isfinite(history)) or np.any(history < 0):
            raise ValueError("electricity-load causal history must be finite nonnegative shape (672, 2)")
        if calendar.shape != (WEEK_SAMPLES, 2) or not np.array_equal(calendar, native_week_calendar()):
            raise ValueError("electricity-load calendar must be the native 672-sample weekday/quarter-hour grid")
        object.__setattr__(self, "causal_history_kw", history.copy())
        object.__setattr__(self, "calendar_phase", calendar.copy())


@dataclass(frozen=True)
class ElectricityLoadOutcomeWeek:
    """Diagnostic wrapper that keeps outcome, client ID, and split outside candidates."""

    client_id: str
    split: str
    source_week_label: str
    target_kw: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.target_kw, dtype=float)
        if self.split not in SPLIT_COUNTS or not self.client_id.startswith("MT_") or not self.source_week_label:
            raise ValueError("electricity-load outcome week has invalid identity, split, or source week label")
        if target.shape != (WEEK_SAMPLES,) or not np.all(np.isfinite(target)) or np.any(target < 0):
            raise ValueError("electricity-load outcome week needs finite nonnegative source-kW targets")
        object.__setattr__(self, "target_kw", target.copy())

    def causal_input(self) -> ElectricityLoadInputWeek:
        return make_electricity_load_input_week(self.target_kw)


def make_electricity_load_input_week(target_kw: np.ndarray) -> ElectricityLoadInputWeek:
    """Evaluator-owned construction of strictly earlier own-client target history."""
    target = np.asarray(target_kw, dtype=float)
    if target.shape != (WEEK_SAMPLES,) or not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("electricity-load target must be finite nonnegative source kW with 672 samples")
    history = np.zeros((WEEK_SAMPLES, len(HISTORY_LAGS)), dtype=float)
    history[1:, 0] = target[:-1]
    history[HISTORY_LAGS[1] :, 1] = target[: -HISTORY_LAGS[1]]
    return ElectricityLoadInputWeek(history, native_week_calendar())


def partition_electricity_load_outcome_weeks(
    weeks: list[ElectricityLoadOutcomeWeek], split_path: str | Path
) -> dict[str, list[ElectricityLoadOutcomeWeek]]:
    """Verify the frozen client split while leaving target arrays off the input surface."""
    payload = json.loads(Path(split_path).read_text())
    partition_ids = {name: [str(client_id) for client_id in payload.get(name, [])] for name in SPLIT_COUNTS}
    if {name: len(values) for name, values in partition_ids.items()} != SPLIT_COUNTS:
        raise ValueError("electricity-load frozen split must be 240/65/65")
    expected = {client_id: split for split, client_ids in partition_ids.items() for client_id in client_ids}
    if len(expected) != sum(SPLIT_COUNTS.values()):
        raise ValueError("electricity-load frozen split has duplicate client IDs")
    partitioned = {name: [] for name in SPLIT_COUNTS}
    observed_clients: set[str] = set()
    for week in weeks:
        if expected.get(week.client_id) != week.split:
            raise ValueError("electricity-load week crosses the frozen client split")
        if week.client_id in observed_clients:
            raise ValueError("electricity-load mechanics gate expects exactly one injected week per client")
        observed_clients.add(week.client_id)
        partitioned[week.split].append(week)
    if observed_clients != set(expected):
        raise ValueError("electricity-load week set omits or adds a locked client")
    if {name: len(items) for name, items in partitioned.items()} != SPLIT_COUNTS:
        raise ValueError("electricity-load partition count mismatch")
    return partitioned


def rollout_electricity_load_causal_input(
    week: ElectricityLoadInputWeek,
    predictor: Callable[[np.ndarray, np.ndarray], float],
    *,
    upper_bound_kw: float,
) -> np.ndarray:
    """Evaluate one reset candidate surface without handing it targets or identities."""
    return rollout_electricity_load_causal_surface(
        week.causal_history_kw,
        week.calendar_phase,
        predictor,
        upper_bound_kw=upper_bound_kw,
    )


def rollout_electricity_load_causal_surface(
    causal_history_kw: np.ndarray,
    calendar_phase: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray], float],
    *,
    upper_bound_kw: float,
    require_native_calendar: bool = True,
) -> np.ndarray:
    """Evaluate an isolated causal surface; controls may supply shifted/ablated calendars."""
    upper = float(upper_bound_kw)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("electricity-load rollout upper bound must be finite and positive")
    history = np.asarray(causal_history_kw, dtype=float)
    calendar = np.asarray(calendar_phase, dtype=float)
    if history.shape != (WEEK_SAMPLES, len(HISTORY_LAGS)) or not np.all(np.isfinite(history)) or np.any(history < 0):
        raise ValueError("electricity-load causal rollout needs finite nonnegative shape (672, 2) history")
    if calendar.shape != (WEEK_SAMPLES, 2) or not np.all(np.isfinite(calendar)):
        raise ValueError("electricity-load causal rollout needs a finite shape (672, 2) calendar")
    if require_native_calendar and not np.array_equal(calendar, native_week_calendar()):
        raise ValueError("electricity-load causal rollout needs the native calendar grid")
    output = np.empty(WEEK_SAMPLES, dtype=float)
    for index, (history_row, calendar_row) in enumerate(zip(history, calendar)):
        prediction = float(predictor(np.asarray(history_row, dtype=float).copy(), np.asarray(calendar_row, dtype=float).copy()))
        if not np.isfinite(prediction) or prediction < 0 or prediction > upper:
            raise ValueError("electricity-load rollout emitted a non-finite, negative, or out-of-bounds prediction")
        output[index] = prediction
    return output

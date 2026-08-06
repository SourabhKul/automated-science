"""Isolated causal surfaces for the Phase 30 UCI gas-sensor benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np


TRACE_SAMPLES = 7500
EXPOSURE_SAMPLES = 4500
INPUT_COLUMNS = ("frame_index", "exposure", "ace_conc_vol_percent", "eth_conc_vol_percent")
SPLIT_COUNTS = {"train": 39, "selection": 11, "external": 8}
RESPONSE_BOUNDS = (-5.0, 20.0)


@dataclass(frozen=True)
class UCIFlowInputTrial:
    """Candidate-facing causal inputs with no identity or measured response."""

    input_u: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.input_u, dtype=float)
        if values.shape != (TRACE_SAMPLES, len(INPUT_COLUMNS)) or not np.all(np.isfinite(values)):
            raise ValueError("UCI flow input trial must be finite with shape (7500, 4)")
        expected_time = np.arange(TRACE_SAMPLES, dtype=float)
        expected_exposure = np.zeros(TRACE_SAMPLES, dtype=float)
        expected_exposure[:EXPOSURE_SAMPLES] = 1.0
        if not np.array_equal(values[:, 0], expected_time) or not np.array_equal(values[:, 1], expected_exposure):
            raise ValueError("UCI flow input trial must use the native frame and documented exposure schedule")
        if np.any(values[:, 2:] < 0) or not np.all(values[:, 2:] == values[0, 2:]):
            raise ValueError("UCI flow concentrations must be finite nonnegative per-trial constants")
        object.__setattr__(self, "input_u", values.copy())


@dataclass(frozen=True)
class UCIFlowOutcomeTrial:
    """Diagnostic wrapper that keeps batch/sample identity and response outside candidates."""

    sample_id: int
    batch: str
    split: str
    input_trial: UCIFlowInputTrial
    sensor_one_dr: np.ndarray

    def __post_init__(self) -> None:
        response = np.asarray(self.sensor_one_dr, dtype=float)
        if self.split not in SPLIT_COUNTS or int(self.sample_id) <= 0 or not self.batch:
            raise ValueError("UCI flow outcome trial has an invalid identity or split")
        if response.shape != (TRACE_SAMPLES,) or not np.all(np.isfinite(response)):
            raise ValueError("UCI flow outcome trial needs a finite sensor-1 trace with 7500 samples")
        object.__setattr__(self, "sensor_one_dr", response.copy())

    def causal_input(self) -> UCIFlowInputTrial:
        return self.input_trial


def make_uci_flow_input_trial(ace_conc_vol_percent: float, eth_conc_vol_percent: float) -> UCIFlowInputTrial:
    """Construct the fixed schedule and concentration-only candidate surface."""
    concentrations = np.asarray([ace_conc_vol_percent, eth_conc_vol_percent], dtype=float)
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations < 0):
        raise ValueError("UCI flow concentrations must be finite and nonnegative")
    frames = np.arange(TRACE_SAMPLES, dtype=float)
    exposure = np.zeros(TRACE_SAMPLES, dtype=float)
    exposure[:EXPOSURE_SAMPLES] = 1.0
    return UCIFlowInputTrial(np.column_stack([frames, exposure, np.full(TRACE_SAMPLES, concentrations[0]), np.full(TRACE_SAMPLES, concentrations[1])]))


def partition_uci_flow_outcome_trials(
    trials: list[UCIFlowOutcomeTrial], split_path: str | Path
) -> dict[str, list[UCIFlowOutcomeTrial]]:
    """Verify the frozen batch/sample split without exposing outcomes to candidates."""
    payload = json.loads(Path(split_path).read_text())
    roles = payload.get("roles", {})
    expected: dict[int, tuple[str, str]] = {}
    for split, entries in roles.items():
        if split not in SPLIT_COUNTS or not isinstance(entries, list):
            raise ValueError("UCI flow frozen split has an invalid role")
        for entry in entries:
            batch = str(entry["batch"])
            for sample_id in entry["samples"]:
                sample = int(sample_id)
                if sample in expected:
                    raise ValueError("UCI flow frozen split repeats a sample")
                expected[sample] = (split, batch)
    if {name: sum(len(entry["samples"]) for entry in roles.get(name, [])) for name in SPLIT_COUNTS} != SPLIT_COUNTS:
        raise ValueError("UCI flow frozen split counts must be 39/11/8")
    partitioned = {name: [] for name in SPLIT_COUNTS}
    for trial in trials:
        target = expected.get(int(trial.sample_id))
        if target != (trial.split, trial.batch):
            raise ValueError("UCI flow trial crosses the frozen batch split")
        partitioned[trial.split].append(trial)
    if {trial.sample_id for trial in trials} != set(expected):
        raise ValueError("UCI flow trial set omits or adds a locked sample")
    if {name: len(items) for name, items in partitioned.items()} != SPLIT_COUNTS:
        raise ValueError("UCI flow partition count mismatch")
    return partitioned


def rollout_uci_flow_causal_input(
    trial: UCIFlowInputTrial,
    step: Callable[[float, np.ndarray], float],
    *,
    bounds: tuple[float, float] = RESPONSE_BOUNDS,
) -> np.ndarray:
    """Run one zero-reset trial using causal inputs only and hard finite bounds."""
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("UCI flow rollout bounds must be finite and ordered")
    state = 0.0
    output = np.empty(TRACE_SAMPLES, dtype=float)
    for index, current_u in enumerate(trial.input_u):
        state = float(step(float(state), np.asarray(current_u, dtype=float).copy()))
        if not np.isfinite(state) or state < lower or state > upper:
            raise ValueError("UCI flow rollout emitted a non-finite or out-of-bounds state")
        output[index] = state
    return output

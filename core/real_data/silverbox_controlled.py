"""Opt-in controlled-system development contract for the Silverbox record.

This module exposes one fixed, bounded slice of the default development-safe
Silverbox loader. Each simulator call receives the measured input sequence and
the first 50 measured outputs for initialization, then returns only the
free-run predictions after those 50 samples. Targets stay outside the
simulator call. This is a data/simulator boundary; it does not fit or score a
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from core.real_data.silverbox import SAMPLE_TIME_SECONDS, load_silverbox_archive


BLOCK_SOURCE_START = 40_650
BLOCK_SOURCE_STOP = 48_842
TRAIN_SOURCE_START = 40_650
TRAIN_SOURCE_STOP = 46_794
UNUSED_GAP_SOURCE_START = 46_794
UNUSED_GAP_SOURCE_STOP = 47_050
VALIDATION_SOURCE_START = 47_050
VALIDATION_SOURCE_STOP = 48_842

TRAIN_WINDOW_RELATIVE_STARTS = (0, 1_536, 3_072, 4_608)
TRAIN_WINDOW_LENGTH = 256
TRAIN_WINDOW_PREDICTION_LENGTH = 206
STATE_INITIALIZATION_LENGTH = 50

SegmentRole = Literal["train", "validation"]
ControlledSimulator = Callable[..., Any]


@dataclass(frozen=True)
class SilverboxControlledSeries:
    """One fixed input/output episode with an initialization and target suffix.

    ``input_u`` covers the full source interval. ``initialization_y`` contains
    exactly its first 50 outputs. ``target_y`` contains only outputs after
    initialization and is never passed to the simulator.
    """

    role: SegmentRole
    source_start: int
    source_stop: int
    input_u: np.ndarray
    initialization_y: np.ndarray
    target_y: np.ndarray
    sampling_time: float

    def __post_init__(self) -> None:
        input_u = np.asarray(self.input_u, dtype=float)
        initialization_y = np.asarray(self.initialization_y, dtype=float)
        target_y = np.asarray(self.target_y, dtype=float)
        if self.role not in ("train", "validation"):
            raise ValueError(f"unsupported Silverbox development role: {self.role}")
        if self.source_start < 0 or self.source_stop <= self.source_start:
            raise ValueError("Silverbox controlled source range must be a positive half-open interval")
        if self.role == "train":
            fixed_starts = tuple(TRAIN_SOURCE_START + offset for offset in TRAIN_WINDOW_RELATIVE_STARTS)
            if self.source_start not in fixed_starts or self.source_stop != self.source_start + TRAIN_WINDOW_LENGTH:
                raise ValueError("Silverbox train series must use one of the four fixed 256-sample windows")
        elif self.source_start != VALIDATION_SOURCE_START or self.source_stop != VALIDATION_SOURCE_STOP:
            raise ValueError("Silverbox validation series must use the fixed validation source range")
        if self.source_stop - self.source_start != input_u.size:
            raise ValueError("Silverbox controlled source range does not match input length")
        if input_u.ndim != 1 or initialization_y.ndim != 1 or target_y.ndim != 1:
            raise ValueError("Silverbox controlled arrays must be one-dimensional")
        if input_u.size <= STATE_INITIALIZATION_LENGTH:
            raise ValueError("Silverbox controlled series must extend past its 50-sample initializer")
        if initialization_y.size != STATE_INITIALIZATION_LENGTH:
            raise ValueError("Silverbox controlled series must use exactly 50 initialization outputs")
        if target_y.size != input_u.size - STATE_INITIALIZATION_LENGTH:
            raise ValueError("Silverbox controlled target length must exclude the 50 initialization outputs")
        arrays = (input_u, initialization_y, target_y)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Silverbox controlled arrays must be finite")
        if not np.isfinite(self.sampling_time) or self.sampling_time <= 0:
            raise ValueError("Silverbox sampling time must be finite and positive")
        if self.sampling_time != SAMPLE_TIME_SECONDS:
            raise ValueError("Silverbox controlled series must use the native sample interval")
        for name, array in (
            ("input_u", input_u),
            ("initialization_y", initialization_y),
            ("target_y", target_y),
        ):
            frozen = np.array(array, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def sample_count(self) -> int:
        return int(self.input_u.size)

    @property
    def prediction_count(self) -> int:
        return int(self.target_y.size)


@dataclass(frozen=True)
class SilverboxControlledDevelopment:
    """The Phase 1 block, represented only by train and validation roles.

    The 256-sample gap is named by absolute indices and has no exposed signal
    arrays. Training contains four predeclared 256-sample windows. Validation
    is one contiguous 1,792-sample series whose first 50 outputs initialize
    the simulator and whose remaining outputs are candidate-selection targets.
    """

    train_windows: tuple[SilverboxControlledSeries, ...]
    validation: SilverboxControlledSeries
    sampling_time: float

    def __post_init__(self) -> None:
        train_windows = tuple(self.train_windows)
        object.__setattr__(self, "train_windows", train_windows)
        if len(train_windows) != len(TRAIN_WINDOW_RELATIVE_STARTS):
            raise ValueError("Silverbox controlled development requires four fixed training windows")
        if tuple(window.role for window in train_windows) != ("train",) * 4:
            raise ValueError("Silverbox controlled training windows must all have the train role")
        if self.validation.role != "validation":
            raise ValueError("Silverbox controlled validation series must have the validation role")
        expected_starts = tuple(TRAIN_SOURCE_START + start for start in TRAIN_WINDOW_RELATIVE_STARTS)
        observed_starts = tuple(window.source_start for window in train_windows)
        if observed_starts != expected_starts:
            raise ValueError("Silverbox controlled training window source indices do not match the fixed plan")
        if any(window.sample_count != TRAIN_WINDOW_LENGTH for window in train_windows):
            raise ValueError("Silverbox controlled training windows must contain exactly 256 samples")
        if (
            self.validation.source_start != VALIDATION_SOURCE_START
            or self.validation.source_stop != VALIDATION_SOURCE_STOP
        ):
            raise ValueError("Silverbox controlled validation source range does not match the fixed plan")
        if not np.isfinite(self.sampling_time) or self.sampling_time <= 0:
            raise ValueError("Silverbox sampling time must be finite and positive")
        if self.sampling_time != self.validation.sampling_time or any(
            window.sampling_time != self.sampling_time for window in train_windows
        ):
            raise ValueError("Silverbox controlled series must use one native sampling interval")

    @property
    def block_source_range(self) -> tuple[int, int]:
        return BLOCK_SOURCE_START, BLOCK_SOURCE_STOP

    @property
    def train_source_range(self) -> tuple[int, int]:
        return TRAIN_SOURCE_START, TRAIN_SOURCE_STOP

    @property
    def unused_gap_source_range(self) -> tuple[int, int]:
        return UNUSED_GAP_SOURCE_START, UNUSED_GAP_SOURCE_STOP

    @property
    def validation_source_range(self) -> tuple[int, int]:
        return VALIDATION_SOURCE_START, VALIDATION_SOURCE_STOP


@dataclass(frozen=True)
class ControlledSimulationResult:
    """A free-run result or an explicit failure with no usable trajectory."""

    status: Literal["success", "failed"]
    trajectory: np.ndarray | None
    failure_mode: str | None = None
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        if self.status == "success":
            if self.trajectory is None or self.failure_mode is not None:
                raise ValueError("successful controlled simulations require a trajectory and no failure mode")
            trajectory = np.asarray(self.trajectory, dtype=float)
            if trajectory.ndim != 1 or not np.all(np.isfinite(trajectory)):
                raise ValueError("successful controlled simulation trajectory must be finite and one-dimensional")
            trajectory = np.array(trajectory, copy=True)
            trajectory.setflags(write=False)
            object.__setattr__(self, "trajectory", trajectory)
        elif self.status == "failed":
            if self.trajectory is not None or not self.failure_mode:
                raise ValueError("failed controlled simulations must have no trajectory and a failure mode")
        else:
            raise ValueError(f"unsupported controlled simulation status: {self.status}")


def load_silverbox_controlled_development(raw_archive: str | Path) -> SilverboxControlledDevelopment:
    """Load and split only the fixed development block through the default loader."""

    dataset = load_silverbox_archive(raw_archive)
    source = dataset.train_val
    if source.record_id != "train_val_multisine" or source.source_start != BLOCK_SOURCE_START:
        raise ValueError("default Silverbox loader returned an unexpected development source")
    if source.source_stop < BLOCK_SOURCE_STOP:
        raise ValueError("Silverbox development record does not cover the fixed Phase 1 block")
    if source.sampling_time != SAMPLE_TIME_SECONDS:
        raise ValueError("Silverbox development record does not preserve its native sampling interval")

    train_windows = []
    for relative_start in TRAIN_WINDOW_RELATIVE_STARTS:
        absolute_start = BLOCK_SOURCE_START + relative_start
        source_offset = absolute_start - source.source_start
        source_stop_offset = source_offset + TRAIN_WINDOW_LENGTH
        train_windows.append(
            SilverboxControlledSeries(
                role="train",
                source_start=absolute_start,
                source_stop=absolute_start + TRAIN_WINDOW_LENGTH,
                input_u=source.input_u[source_offset:source_stop_offset],
                initialization_y=source.output_y[source_offset : source_offset + STATE_INITIALIZATION_LENGTH],
                target_y=source.output_y[source_offset + STATE_INITIALIZATION_LENGTH : source_stop_offset],
                sampling_time=source.sampling_time,
            )
        )

    validation_start = VALIDATION_SOURCE_START - source.source_start
    validation_stop = VALIDATION_SOURCE_STOP - source.source_start
    validation_u = source.input_u[validation_start:validation_stop]
    validation_y = source.output_y[validation_start:validation_stop]
    validation = SilverboxControlledSeries(
        role="validation",
        source_start=VALIDATION_SOURCE_START,
        source_stop=VALIDATION_SOURCE_STOP,
        input_u=validation_u,
        initialization_y=validation_y[:STATE_INITIALIZATION_LENGTH],
        target_y=validation_y[STATE_INITIALIZATION_LENGTH:],
        sampling_time=source.sampling_time,
    )
    return SilverboxControlledDevelopment(
        train_windows=tuple(train_windows),
        validation=validation,
        sampling_time=source.sampling_time,
    )


def simulate_controlled_series(
    series: SilverboxControlledSeries,
    simulator: ControlledSimulator,
    parameters: Any,
) -> ControlledSimulationResult:
    """Run one input-driven free-run simulation under the fixed initialization.

    ``simulator`` is called with keyword arguments ``input_u`` (the exact
    measured input sequence for the entire series), ``initialization_y`` (the
    first 50 measured outputs), ``parameters``, and ``sampling_time``. It must
    return only the predictions corresponding to ``target_y``. Targets never
    enter the simulator call. Exceptions, wrong-length returns, and nonfinite
    predictions are reported as failures; predictions are never clipped.
    """

    try:
        input_u = np.array(series.input_u, copy=True)
        initialization_y = np.array(series.initialization_y, copy=True)
        input_u.setflags(write=False)
        initialization_y.setflags(write=False)
        raw_prediction = simulator(
            input_u=input_u,
            initialization_y=initialization_y,
            parameters=parameters,
            sampling_time=series.sampling_time,
        )
    except Exception as error:
        return ControlledSimulationResult(
            status="failed",
            trajectory=None,
            failure_mode="simulator_exception",
            failure_detail=f"{type(error).__name__}: {error}",
        )

    try:
        prediction = np.asarray(raw_prediction, dtype=float)
    except Exception as error:
        return ControlledSimulationResult(
            status="failed",
            trajectory=None,
            failure_mode="invalid_output",
            failure_detail=f"{type(error).__name__}: {error}",
        )
    if prediction.shape != (series.prediction_count,):
        return ControlledSimulationResult(
            status="failed",
            trajectory=None,
            failure_mode="wrong_output_shape",
            failure_detail=f"expected {(series.prediction_count,)}, received {prediction.shape}",
        )
    if not np.all(np.isfinite(prediction)):
        return ControlledSimulationResult(
            status="failed",
            trajectory=None,
            failure_mode="nonfinite_output",
            failure_detail="simulator returned a nonfinite prediction",
        )
    return ControlledSimulationResult(status="success", trajectory=prediction)


__all__ = [
    "BLOCK_SOURCE_START",
    "BLOCK_SOURCE_STOP",
    "TRAIN_SOURCE_START",
    "TRAIN_SOURCE_STOP",
    "UNUSED_GAP_SOURCE_START",
    "UNUSED_GAP_SOURCE_STOP",
    "VALIDATION_SOURCE_START",
    "VALIDATION_SOURCE_STOP",
    "TRAIN_WINDOW_RELATIVE_STARTS",
    "TRAIN_WINDOW_LENGTH",
    "TRAIN_WINDOW_PREDICTION_LENGTH",
    "STATE_INITIALIZATION_LENGTH",
    "SilverboxControlledSeries",
    "SilverboxControlledDevelopment",
    "ControlledSimulationResult",
    "load_silverbox_controlled_development",
    "simulate_controlled_series",
]

"""Opt-in controlled-system development contract for the Silverbox record.

The development loader reads only the predeclared source rows needed for four
training windows and one validation segment. It never calls the broad dataset
loader or converts official test rows to floats. Each simulator call receives
the measured input sequence and the first 50 measured outputs for initialization,
then returns only the free-run predictions. This module does not fit or score.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal
import zipfile

import numpy as np

from core.real_data.silverbox import REQUIRED_ARCHIVE_MEMBERS, SAMPLE_TIME_SECONDS, SNLS_CSV_MEMBER


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
_SHA256_LENGTH = 64


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


@dataclass(frozen=True, repr=False)
class SilverboxControlledValidation:
    """Validation input and initializer with a deferred output target.

    The development phase keeps the measured validation inputs and exactly
    the permitted 50-output initializer. The validation suffix output is
    loaded from the source archive only when the selection API asks for it
    after verifying a complete fit receipt.
    """

    source_start: int
    source_stop: int
    input_u: np.ndarray
    initialization_y: np.ndarray
    sampling_time: float
    _raw_archive: Path = field(repr=False)
    _source_sha256: str = field(repr=False)
    _source_bytes: int = field(repr=False)
    _target_y: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _target_fit_run_id: str | None = field(default=None, init=False, repr=False, compare=False)
    _target_fit_receipt_sha256: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        input_u = np.asarray(self.input_u, dtype=float)
        initialization_y = np.asarray(self.initialization_y, dtype=float)
        if self.source_start != VALIDATION_SOURCE_START or self.source_stop != VALIDATION_SOURCE_STOP:
            raise ValueError("Silverbox validation series must use the fixed validation source range")
        if input_u.shape != (VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START,):
            raise ValueError("Silverbox validation input length does not match the fixed source range")
        if initialization_y.shape != (STATE_INITIALIZATION_LENGTH,):
            raise ValueError("Silverbox validation must use exactly 50 initialization outputs")
        if not np.all(np.isfinite(input_u)) or not np.all(np.isfinite(initialization_y)):
            raise ValueError("Silverbox validation input and initializer must be finite")
        if not np.isfinite(self.sampling_time) or self.sampling_time != SAMPLE_TIME_SECONDS:
            raise ValueError("Silverbox validation must use the native sample interval")
        if len(self._source_sha256) != _SHA256_LENGTH or any(
            char not in "0123456789abcdef" for char in self._source_sha256
        ):
            raise ValueError("Silverbox validation source hash must be a lowercase SHA-256")
        if self._source_bytes <= 0:
            raise ValueError("Silverbox validation source byte count must be positive")
        for name, array in (("input_u", input_u), ("initialization_y", initialization_y)):
            frozen = np.array(array, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "_raw_archive", Path(self._raw_archive).resolve())

    @property
    def role(self) -> Literal["validation"]:
        return "validation"

    @property
    def sample_count(self) -> int:
        return int(self.input_u.size)

    @property
    def prediction_count(self) -> int:
        return int(self.input_u.size - STATE_INITIALIZATION_LENGTH)

    @property
    def target_loaded(self) -> bool:
        return self._target_y is not None

    def load_target_y(self, fit: Any) -> np.ndarray:
        """Open targets only for a matching complete frozen fit receipt.

        The public gate is a complete content-hash-verified fit; it does not
        claim that validation forecasts have already run. The normal selector
        calls this only after every validation forecast succeeds. This method
        independently verifies fit/source/proposal/run bindings and current
        code and protocol contracts before opening the archive suffix.
        """

        fit_run_id, fit_receipt_sha256 = self._verify_complete_fit_binding(fit)
        target = self._target_y
        if target is None:
            target = _read_fixed_validation_target_suffix(
                self._raw_archive,
                expected_source_sha256=self._source_sha256,
                expected_source_bytes=self._source_bytes,
            )
            target = np.asarray(target, dtype=float)
            if target.shape != (self.prediction_count,) or not np.all(np.isfinite(target)):
                raise ValueError("Silverbox validation target suffix has an invalid shape or nonfinite value")
            target = np.array(target, copy=True)
            target.setflags(write=False)
            object.__setattr__(self, "_target_y", target)
            object.__setattr__(self, "_target_fit_run_id", fit_run_id)
            object.__setattr__(self, "_target_fit_receipt_sha256", fit_receipt_sha256)
        elif (
            fit_run_id != self._target_fit_run_id
            or fit_receipt_sha256 != self._target_fit_receipt_sha256
        ):
            raise ValueError("cached validation targets are bound to a different frozen fit")
        return target

    def _verify_complete_fit_binding(self, fit: Any) -> tuple[str, str]:
        if getattr(fit, "status", None) != "complete":
            raise ValueError("validation targets require a complete frozen fit")
        receipt_path = getattr(fit, "receipt_path", None)
        expected_digest = getattr(fit, "receipt_sha256", None)
        if receipt_path is None or not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("validation targets require a hash-bound frozen fit handle")
        try:
            receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
            embedded_digest = receipt.pop("receipt_sha256", None)
            canonical = json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            observed_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except Exception as error:
            raise ValueError("frozen fit receipt could not be verified") from error
        if embedded_digest != expected_digest or observed_digest != expected_digest:
            raise ValueError("frozen fit receipt hash mismatch")
        from core.real_data import silverbox_first_fit

        # Use the fit API's full verifier so this public deferred loader cannot
        # be bypassed with a minimally fabricated "complete" receipt. That
        # verifier binds the source split, sample interval, all frozen fit
        # constants, resource limits, fitted populations, and current code
        # hashes before this method opens the target suffix.
        silverbox_first_fit._verify_complete_fit_receipt(fit, receipt)
        if (
            receipt.get("source_sha256") != self._source_sha256
            or getattr(fit, "source_sha256", None) != self._source_sha256
        ):
            raise ValueError("frozen fit receipt does not match validation source")
        return str(fit.run_id), expected_digest

    def __repr__(self) -> str:
        return (
            "SilverboxControlledValidation("
            f"source_start={self.source_start}, source_stop={self.source_stop}, "
            f"input_count={self.input_u.size}, initializer_count={self.initialization_y.size}, "
            f"target_loaded={self.target_loaded})"
        )


@dataclass(frozen=True)
class SilverboxControlledDevelopment:
    """The Phase 1 block, represented only by train and validation roles.

    The 256-sample gap is named by absolute indices and has no exposed signal
    arrays. Training contains four predeclared 256-sample windows. Validation
    holds one contiguous 1,792-sample input sequence and its 50-output
    initializer. Candidate-selection outputs remain source-backed until a
    complete fit receipt has been verified.
    """

    train_windows: tuple[SilverboxControlledSeries, ...]
    validation: SilverboxControlledValidation | SilverboxControlledSeries
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


def _read_fixed_development_rows(
    raw_archive: str | Path,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], np.ndarray, np.ndarray]:
    """Read train windows and validation input plus the 50-value initializer.

    The source index is the CSV data-row index (the header is not counted).
    Rows before the fixed development block are skipped as raw physical lines,
    without CSV tokenization. Unused development rows are skipped as raw
    lines. Validation inputs are converted across their fixed source range,
    while validation outputs are converted only for the first 50 initializer
    rows. The reader stops before source index ``BLOCK_SOURCE_STOP``.
    """

    train_starts = tuple(TRAIN_SOURCE_START + offset for offset in TRAIN_WINDOW_RELATIVE_STARTS)
    train_input = [np.empty(TRAIN_WINDOW_LENGTH, dtype=float) for _ in train_starts]
    train_output = [np.empty(TRAIN_WINDOW_LENGTH, dtype=float) for _ in train_starts]
    validation_count = VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START
    validation_input = np.empty(validation_count, dtype=float)
    validation_initializer = np.empty(STATE_INITIALIZATION_LENGTH, dtype=float)

    train_rows: dict[int, tuple[int, int]] = {}
    for window_index, start in enumerate(train_starts):
        for offset in range(TRAIN_WINDOW_LENGTH):
            train_rows[start + offset] = (window_index, offset)

    with zipfile.ZipFile(raw_archive) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        missing = sorted(set(REQUIRED_ARCHIVE_MEMBERS) - names)
        if missing:
            raise ValueError(f"Silverbox archive is missing required members: {missing}")
        with archive.open(SNLS_CSV_MEMBER) as raw_csv:
            header_line = raw_csv.readline()
            if not header_line:
                raise ValueError("Silverbox SNLS CSV is empty")
            try:
                header = next(csv.reader([header_line.decode("utf-8")]))
            except (UnicodeDecodeError, csv.Error) as error:
                raise ValueError("Silverbox SNLS CSV header is invalid") from error
            if header[:2] != ["V1", "V2"] or any(cell.strip() for cell in header[2:]):
                raise ValueError(f"unexpected Silverbox SNLS header: {header}")

            # The official arrow tests precede this development block in the
            # CSV. Count their raw lines only; do not tokenize their fields.
            for source_index in range(BLOCK_SOURCE_START):
                if not raw_csv.readline():
                    raise ValueError(
                        f"Silverbox SNLS CSV ends before development source row {source_index}"
                    )

            for source_index in range(BLOCK_SOURCE_START, BLOCK_SOURCE_STOP):
                raw_line = raw_csv.readline()
                if not raw_line:
                    raise ValueError(
                        f"Silverbox SNLS CSV ends before required source row {source_index}"
                    )
                destination = train_rows.get(source_index)
                is_validation = VALIDATION_SOURCE_START <= source_index < VALIDATION_SOURCE_STOP
                if destination is None and not is_validation:
                    # Skip the unused gap and unselected development rows
                    # without CSV tokenization or numeric conversion.
                    continue
                if (
                    is_validation
                    and source_index >= VALIDATION_SOURCE_START + STATE_INITIALIZATION_LENGTH
                ):
                    input_value = _parse_fixed_csv_input_only(raw_line, source_index)
                    validation_input[source_index - VALIDATION_SOURCE_START] = input_value
                    continue
                row = _parse_fixed_csv_physical_line(raw_line, source_index)
                try:
                    input_value = float(row[0])
                    output_value = float(row[1])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"non-numeric Silverbox development row at source index {source_index}"
                    ) from error
                if not np.isfinite(input_value) or not np.isfinite(output_value):
                    raise ValueError(
                        f"non-finite Silverbox development row at source index {source_index}"
                    )
                if destination is not None:
                    window_index, offset = destination
                    train_input[window_index][offset] = input_value
                    train_output[window_index][offset] = output_value
                else:
                    validation_offset = source_index - VALIDATION_SOURCE_START
                    validation_input[validation_offset] = input_value
                    validation_initializer[validation_offset] = output_value

    return tuple(zip(train_input, train_output)), validation_input, validation_initializer


def _parse_fixed_csv_physical_line(raw_line: bytes, source_index: int) -> list[str]:
    try:
        text = raw_line.decode("utf-8")
        rows = list(csv.reader([text]))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"invalid Silverbox CSV row at source index {source_index}") from error
    if len(rows) != 1:
        raise ValueError(f"unexpected Silverbox CSV row at source index {source_index}")
    row = rows[0]
    if not row or not any(cell.strip() for cell in row):
        raise ValueError(f"blank Silverbox SNLS row at source index {source_index}")
    if len(row) < 2 or not row[0].strip() or not row[1].strip() or any(
        cell.strip() for cell in row[2:]
    ):
        raise ValueError(f"unexpected Silverbox SNLS row at source index {source_index}")
    return row


def _parse_fixed_csv_input_only(raw_line: bytes, source_index: int) -> float:
    """Read V1 only, without decoding or tokenizing validation target V2."""

    physical_line = raw_line.rstrip(b"\r\n")
    first_separator = physical_line.find(b",")
    second_separator = physical_line.find(b",", first_separator + 1)
    if (
        first_separator <= 0
        or second_separator <= first_separator + 1
        or second_separator != len(physical_line) - 1
    ):
        raise ValueError(f"unexpected Silverbox SNLS row at source index {source_index}")
    try:
        input_text = physical_line[:first_separator].decode("ascii").strip()
        input_value = float(input_text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"non-numeric Silverbox validation input at source index {source_index}"
        ) from error
    if not input_text or not np.isfinite(input_value):
        raise ValueError(f"non-finite Silverbox validation input at source index {source_index}")
    return input_value


def _archive_identity(raw_archive: str | Path) -> tuple[int, str]:
    path = Path(raw_archive)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_fixed_validation_target_suffix(
    raw_archive: str | Path,
    *,
    expected_source_sha256: str,
    expected_source_bytes: int,
) -> np.ndarray:
    """Load only the frozen validation output suffix after fit completion."""

    source_bytes, source_sha256 = _archive_identity(raw_archive)
    if source_bytes != expected_source_bytes or source_sha256 != expected_source_sha256:
        raise ValueError("Silverbox source archive changed before validation target loading")
    target_start = VALIDATION_SOURCE_START + STATE_INITIALIZATION_LENGTH
    target_count = VALIDATION_SOURCE_STOP - target_start
    target = np.empty(target_count, dtype=float)
    with zipfile.ZipFile(raw_archive) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        missing = sorted(set(REQUIRED_ARCHIVE_MEMBERS) - names)
        if missing:
            raise ValueError(f"Silverbox archive is missing required members: {missing}")
        with archive.open(SNLS_CSV_MEMBER) as raw_csv:
            header_line = raw_csv.readline()
            if not header_line:
                raise ValueError("Silverbox SNLS CSV is empty")
            try:
                header = next(csv.reader([header_line.decode("utf-8")]))
            except (UnicodeDecodeError, csv.Error) as error:
                raise ValueError("Silverbox SNLS CSV header is invalid") from error
            if header[:2] != ["V1", "V2"] or any(cell.strip() for cell in header[2:]):
                raise ValueError(f"unexpected Silverbox SNLS header: {header}")
            # Every preceding source row is skipped as a physical line. This
            # preserves the earlier sealed-prefix boundary and avoids
            # tokenizing any pre-target row during target loading.
            for source_index in range(target_start):
                if not raw_csv.readline():
                    raise ValueError(
                        f"Silverbox SNLS CSV ends before validation target row {source_index}"
                    )
            for offset, source_index in enumerate(range(target_start, VALIDATION_SOURCE_STOP)):
                raw_line = raw_csv.readline()
                if not raw_line:
                    raise ValueError(
                        f"Silverbox SNLS CSV ends before validation target row {source_index}"
                    )
                row = _parse_fixed_csv_physical_line(raw_line, source_index)
                try:
                    output_value = float(row[1])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"non-numeric Silverbox validation target at source index {source_index}"
                    ) from error
                if not np.isfinite(output_value):
                    raise ValueError(
                        f"non-finite Silverbox validation target at source index {source_index}"
                    )
                target[offset] = output_value
    return target


def load_silverbox_controlled_development(
    raw_archive: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected_source_bytes: int | None = None,
) -> SilverboxControlledDevelopment:
    """Load only fixed train windows and validation rows from the source CSV.

    This development-only path does not call the broad dataset loader. It
    converts no validation target-suffix output before fit completion. It stops
    reading before source index 48,842 and well before every official test
    range. Target loading rechecks the archive hash/byte count and fixed source
    indices after selection begins.
    """

    source_bytes, source_sha256 = _archive_identity(raw_archive)
    if expected_source_bytes is not None and source_bytes != expected_source_bytes:
        raise ValueError("Silverbox archive byte count does not match the verified source")
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("Silverbox archive SHA-256 does not match the verified source")
    train_arrays, validation_u, validation_initializer = _read_fixed_development_rows(raw_archive)
    train_windows = []
    for start, (input_u, output_y) in zip(
        (TRAIN_SOURCE_START + offset for offset in TRAIN_WINDOW_RELATIVE_STARTS), train_arrays
    ):
        train_windows.append(
            SilverboxControlledSeries(
                role="train",
                source_start=start,
                source_stop=start + TRAIN_WINDOW_LENGTH,
                input_u=input_u,
                initialization_y=output_y[:STATE_INITIALIZATION_LENGTH],
                target_y=output_y[STATE_INITIALIZATION_LENGTH:],
                sampling_time=SAMPLE_TIME_SECONDS,
            )
        )
    validation = SilverboxControlledValidation(
        source_start=VALIDATION_SOURCE_START,
        source_stop=VALIDATION_SOURCE_STOP,
        input_u=validation_u,
        initialization_y=validation_initializer,
        sampling_time=SAMPLE_TIME_SECONDS,
        _raw_archive=Path(raw_archive),
        _source_sha256=source_sha256,
        _source_bytes=source_bytes,
    )
    return SilverboxControlledDevelopment(
        train_windows=tuple(train_windows),
        validation=validation,
        sampling_time=SAMPLE_TIME_SECONDS,
    )


def simulate_controlled_series(
    series: SilverboxControlledSeries | SilverboxControlledValidation,
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
    "SilverboxControlledValidation",
    "SilverboxControlledDevelopment",
    "ControlledSimulationResult",
    "load_silverbox_controlled_development",
    "simulate_controlled_series",
]

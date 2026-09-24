"""Source-contract loader for the official Nonlinear Benchmark Silverbox archive.

The default loader returns a development-safe view: train/validation outputs
are available for fitting and test candidates contain only their input and the
official 50-sample initialization window. Full test outputs are available only
from the explicitly named scorer loader. This module does not fit, score, or
tune a model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ARCHIVE_ROOT = "SilverboxFiles"
SNLS_CSV_MEMBER = f"{ARCHIVE_ROOT}/SNLS80mV.csv"
SCHROEDER_CSV_MEMBER = f"{ARCHIVE_ROOT}/Schroeder80mV.csv"
SNLS_MAT_MEMBER = f"{ARCHIVE_ROOT}/SNLS80mV.mat"
SCHROEDER_MAT_MEMBER = f"{ARCHIVE_ROOT}/Schroeder80mV.mat"
README_TXT_MEMBER = f"{ARCHIVE_ROOT}/README.txt"
README_M_MEMBER = f"{ARCHIVE_ROOT}/README.m"
REQUIRED_ARCHIVE_MEMBERS = (
    SNLS_CSV_MEMBER,
    SNLS_MAT_MEMBER,
    SCHROEDER_MAT_MEMBER,
    README_TXT_MEMBER,
    README_M_MEMBER,
    SCHROEDER_CSV_MEMBER,
)

SAMPLE_COUNT = 131_072
SAMPLE_TIME_SECONDS = 1.0 / 610.35
STATE_INITIALIZATION_WINDOW_LENGTH = 50

# These are the half-open source-index ranges in the official loader.
ARROW_FULL_START = 100
ARROW_FULL_STOP = 40_575
ARROW_NO_EXTRAPOLATION_STOP = 32_100
MULTISINE_START = 40_650
MULTISINE_STOP = 127_400
MULTISINE_TRAIN_STOP = MULTISINE_START + int((MULTISINE_STOP - MULTISINE_START) * 0.75)


def _validate_record_arrays(
    *,
    record_id: str,
    source_start: int,
    source_stop: int,
    input_u: np.ndarray,
    output_y: np.ndarray,
    state_initialization_window_length: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    input_u = np.asarray(input_u, dtype=float)
    output_y = np.asarray(output_y, dtype=float)
    if input_u.ndim != 1 or output_y.ndim != 1 or input_u.shape != output_y.shape:
        raise ValueError(f"Silverbox {record_id} input and output must be matching one-dimensional arrays")
    if source_start < 0 or source_stop <= source_start:
        raise ValueError(f"Silverbox {record_id} source range must be a positive half-open interval")
    if source_stop - source_start != input_u.size:
        raise ValueError(f"Silverbox {record_id} source range does not match record length")
    if not np.all(np.isfinite(input_u)) or not np.all(np.isfinite(output_y)):
        raise ValueError(f"Silverbox {record_id} input and output must be finite")
    if state_initialization_window_length is not None:
        if state_initialization_window_length != STATE_INITIALIZATION_WINDOW_LENGTH:
            raise ValueError(
                "Silverbox test records must use the official 50-sample state initialization window"
            )
        if state_initialization_window_length > input_u.size:
            raise ValueError(f"Silverbox {record_id} state initialization window is out of range")
    return input_u.copy(), output_y.copy()


@dataclass(frozen=True)
class SilverboxCandidateView:
    """The portion of a test record that development code may receive.

    The full measured output is deliberately absent. The first 50 output
    samples are the state-initialization window permitted by the official
    submission template; the remaining test target stays with the scorer.
    """

    record_id: str
    input_u: np.ndarray
    initialization_y: np.ndarray
    sampling_time: float
    state_initialization_window_length: int

    def __post_init__(self) -> None:
        if self.state_initialization_window_length != STATE_INITIALIZATION_WINDOW_LENGTH:
            raise ValueError("Silverbox candidate views must use the official 50-sample initialization window")
        input_u = np.asarray(self.input_u, dtype=float)
        initialization_y = np.asarray(self.initialization_y, dtype=float)
        if input_u.ndim != 1 or initialization_y.ndim != 1:
            raise ValueError("Silverbox candidate arrays must be one-dimensional")
        if input_u.size < STATE_INITIALIZATION_WINDOW_LENGTH:
            raise ValueError("Silverbox candidate input is shorter than its initialization window")
        if initialization_y.size != STATE_INITIALIZATION_WINDOW_LENGTH:
            raise ValueError("Silverbox candidate initialization output has the wrong length")
        if not np.all(np.isfinite(input_u)) or not np.all(np.isfinite(initialization_y)):
            raise ValueError("Silverbox candidate arrays must be finite")
        if not np.isfinite(self.sampling_time) or self.sampling_time <= 0:
            raise ValueError("Silverbox sampling time must be finite and positive")
        object.__setattr__(self, "input_u", input_u.copy())
        object.__setattr__(self, "initialization_y", initialization_y.copy())

    @property
    def sample_count(self) -> int:
        return int(self.input_u.size)


@dataclass(frozen=True)
class SilverboxTrainValRecord:
    """The published multisine development record, including its outputs."""

    record_id: str
    source_start: int
    source_stop: int
    input_u: np.ndarray
    output_y: np.ndarray
    sampling_time: float = SAMPLE_TIME_SECONDS

    def __post_init__(self) -> None:
        input_u, output_y = _validate_record_arrays(
            record_id=self.record_id,
            source_start=self.source_start,
            source_stop=self.source_stop,
            input_u=self.input_u,
            output_y=self.output_y,
            state_initialization_window_length=None,
        )
        if not np.isfinite(self.sampling_time) or self.sampling_time <= 0:
            raise ValueError("Silverbox sampling time must be finite and positive")
        object.__setattr__(self, "input_u", input_u)
        object.__setattr__(self, "output_y", output_y)

    @property
    def sample_count(self) -> int:
        return int(self.input_u.size)


@dataclass(frozen=True)
class SilverboxScoringRecord:
    """A sealed test record for the post-lock scorer only.

    Do not pass this type to candidate or development code. The default
    ``load_silverbox_archive`` function never returns it.
    """

    record_id: str
    source_start: int
    source_stop: int
    input_u: np.ndarray
    output_y: np.ndarray
    sampling_time: float = SAMPLE_TIME_SECONDS
    state_initialization_window_length: int = STATE_INITIALIZATION_WINDOW_LENGTH

    def __post_init__(self) -> None:
        input_u, output_y = _validate_record_arrays(
            record_id=self.record_id,
            source_start=self.source_start,
            source_stop=self.source_stop,
            input_u=self.input_u,
            output_y=self.output_y,
            state_initialization_window_length=self.state_initialization_window_length,
        )
        if not np.isfinite(self.sampling_time) or self.sampling_time <= 0:
            raise ValueError("Silverbox sampling time must be finite and positive")
        object.__setattr__(self, "input_u", input_u)
        object.__setattr__(self, "output_y", output_y)

    @property
    def sample_count(self) -> int:
        return int(self.input_u.size)


@dataclass(frozen=True)
class SilverboxDevelopmentDataset:
    """Development data plus sealed test candidate views without test suffixes."""

    train_val: SilverboxTrainValRecord
    test_multisine: SilverboxCandidateView
    test_arrow_full: SilverboxCandidateView
    test_arrow_no_extrapolation: SilverboxCandidateView

    @property
    def test_candidates(self) -> tuple[SilverboxCandidateView, SilverboxCandidateView, SilverboxCandidateView]:
        return self.test_multisine, self.test_arrow_full, self.test_arrow_no_extrapolation


@dataclass(frozen=True)
class SilverboxScoringDataset:
    """Sealed test targets reserved for explicit post-lock scoring."""

    test_multisine: SilverboxScoringRecord
    test_arrow_full: SilverboxScoringRecord
    test_arrow_no_extrapolation: SilverboxScoringRecord

    @property
    def test_records(self) -> tuple[SilverboxScoringRecord, SilverboxScoringRecord, SilverboxScoringRecord]:
        return self.test_multisine, self.test_arrow_full, self.test_arrow_no_extrapolation


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inspect_silverbox_archive(raw_archive: str | Path) -> dict[str, Any]:
    """Return reproducibility metadata and reject incomplete ZIP archives."""

    archive_path = Path(raw_archive)
    payload = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = {info.filename for info in infos}
        missing = sorted(set(REQUIRED_ARCHIVE_MEMBERS) - names)
        if missing:
            raise ValueError(f"Silverbox archive is missing required members: {missing}")
        members = []
        for info in infos:
            member_payload = archive.read(info.filename)
            members.append(
                {
                    "name": info.filename,
                    "compressed_bytes": info.compress_size,
                    "expanded_bytes": info.file_size,
                    "sha256": _sha256_bytes(member_payload),
                }
            )
    return {
        "archive_bytes": len(payload),
        "archive_sha256": _sha256_bytes(payload),
        "expanded_member_bytes": sum(member["expanded_bytes"] for member in members),
        "members": members,
    }


def _read_snls_csv(archive: zipfile.ZipFile) -> tuple[np.ndarray, np.ndarray]:
    """Read the canonical V1/V2 CSV, allowing its empty trailing column/row."""

    values: list[tuple[float, float]] = []
    with archive.open(SNLS_CSV_MEMBER) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.reader(text)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("Silverbox SNLS CSV is empty") from error
            if header[:2] != ["V1", "V2"] or any(cell.strip() for cell in header[2:]):
                raise ValueError(f"unexpected Silverbox SNLS header: {header}")
            seen_blank_row = False
            for row_number, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    seen_blank_row = True
                    continue
                if seen_blank_row:
                    raise ValueError(f"non-empty Silverbox SNLS row follows trailing blank row {row_number}")
                if len(row) < 2 or any(cell.strip() for cell in row[2:]):
                    raise ValueError(f"unexpected Silverbox SNLS row {row_number}: {row}")
                try:
                    values.append((float(row[0]), float(row[1])))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"non-numeric Silverbox SNLS row {row_number}: {row}") from error
    if len(values) != SAMPLE_COUNT:
        raise ValueError(f"Silverbox SNLS CSV must contain {SAMPLE_COUNT} samples, found {len(values)}")
    signal = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(signal)):
        raise ValueError("Silverbox SNLS CSV contains non-finite values")
    return signal[:, 0], signal[:, 1]


def _load_source_arrays(raw_archive: str | Path) -> tuple[np.ndarray, np.ndarray]:
    archive_path = Path(raw_archive)
    with zipfile.ZipFile(archive_path) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        missing = sorted(set(REQUIRED_ARCHIVE_MEMBERS) - names)
        if missing:
            raise ValueError(f"Silverbox archive is missing required members: {missing}")
        return _read_snls_csv(archive)


def _record_arrays(
    u: np.ndarray, y: np.ndarray, record_id: str, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    if not (0 <= start < stop <= SAMPLE_COUNT):
        raise ValueError(f"Silverbox {record_id} range is outside the source record")
    return u[start:stop], y[start:stop]


def load_silverbox_archive(raw_archive: str | Path) -> SilverboxDevelopmentDataset:
    """Load development data and sealed candidate views without test targets."""

    u, y = _load_source_arrays(raw_archive)
    train_u, train_y = _record_arrays(u, y, "train_val_multisine", MULTISINE_START, MULTISINE_TRAIN_STOP)
    test_specs = (
        ("test_multisine", MULTISINE_TRAIN_STOP, MULTISINE_STOP),
        ("test_arrow_full", ARROW_FULL_START, ARROW_FULL_STOP),
        ("test_arrow_no_extrapolation", ARROW_FULL_START, ARROW_NO_EXTRAPOLATION_STOP),
    )
    candidates = []
    for record_id, start, stop in test_specs:
        test_u, test_y = _record_arrays(u, y, record_id, start, stop)
        candidates.append(
            SilverboxCandidateView(
                record_id=record_id,
                input_u=test_u,
                initialization_y=test_y[:STATE_INITIALIZATION_WINDOW_LENGTH],
                sampling_time=SAMPLE_TIME_SECONDS,
                state_initialization_window_length=STATE_INITIALIZATION_WINDOW_LENGTH,
            )
        )
    return SilverboxDevelopmentDataset(
        train_val=SilverboxTrainValRecord(
            record_id="train_val_multisine",
            source_start=MULTISINE_START,
            source_stop=MULTISINE_TRAIN_STOP,
            input_u=train_u,
            output_y=train_y,
        ),
        test_multisine=candidates[0],
        test_arrow_full=candidates[1],
        test_arrow_no_extrapolation=candidates[2],
    )


def load_silverbox_for_scoring(raw_archive: str | Path) -> SilverboxScoringDataset:
    """Load full sealed test outputs for a post-lock scorer only."""

    u, y = _load_source_arrays(raw_archive)
    test_specs = (
        ("test_multisine", MULTISINE_TRAIN_STOP, MULTISINE_STOP),
        ("test_arrow_full", ARROW_FULL_START, ARROW_FULL_STOP),
        ("test_arrow_no_extrapolation", ARROW_FULL_START, ARROW_NO_EXTRAPOLATION_STOP),
    )
    records = []
    for record_id, start, stop in test_specs:
        test_u, test_y = _record_arrays(u, y, record_id, start, stop)
        records.append(
            SilverboxScoringRecord(
                record_id=record_id,
                source_start=start,
                source_stop=stop,
                input_u=test_u,
                output_y=test_y,
            )
        )
    return SilverboxScoringDataset(
        test_multisine=records[0],
        test_arrow_full=records[1],
        test_arrow_no_extrapolation=records[2],
    )


def official_split_contract() -> dict[str, Any]:
    """Return the fixed source-index contract for manifests and audit reports."""

    return {
        "source_record": {"member": SNLS_CSV_MEMBER, "sample_count": SAMPLE_COUNT},
        "train_val": {
            "start": MULTISINE_START,
            "stop": MULTISINE_TRAIN_STOP,
            "count": MULTISINE_TRAIN_STOP - MULTISINE_START,
        },
        "test_multisine": {
            "start": MULTISINE_TRAIN_STOP,
            "stop": MULTISINE_STOP,
            "count": MULTISINE_STOP - MULTISINE_TRAIN_STOP,
            "state_initialization_window_length": STATE_INITIALIZATION_WINDOW_LENGTH,
        },
        "test_arrow_full": {
            "start": ARROW_FULL_START,
            "stop": ARROW_FULL_STOP,
            "count": ARROW_FULL_STOP - ARROW_FULL_START,
            "state_initialization_window_length": STATE_INITIALIZATION_WINDOW_LENGTH,
        },
        "test_arrow_no_extrapolation": {
            "start": ARROW_FULL_START,
            "stop": ARROW_NO_EXTRAPOLATION_STOP,
            "count": ARROW_NO_EXTRAPOLATION_STOP - ARROW_FULL_START,
            "state_initialization_window_length": STATE_INITIALIZATION_WINDOW_LENGTH,
            "subset_of": "test_arrow_full",
        },
        "validation": "No separate published validation slice; split train_val further before any model selection.",
    }

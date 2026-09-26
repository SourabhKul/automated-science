"""Bounded development source adapter for the Cascaded Tanks methods track.

This module verifies the pinned source archive and exposes only the estimation
input/output training prefix plus the later estimation-input suffix. It does
not provide a final/test loader, inference code, or a scorer. The CSV scanner
works field-by-field from bytes and never stores or converts the sealed
uVal/yVal fields or the development yEst suffix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import InitVar, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

TRACK_ID = "cascaded_tanks_methods_v1_20260926"
SOURCE_DOI = "10.4121/12960104.v1"
SOURCE_URL = "https://data.4tu.nl/articles/dataset/Cascaded_Tanks_Benchmark_Combining_Soft_and_Hard_Nonlinearities/12960104"
SOURCE_ARCHIVE_NAME = "CascadedTanksFiles.zip"
SOURCE_ARCHIVE_BYTES = 7_520_592
SOURCE_ARCHIVE_SHA256 = "eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51"
CSV_MEMBER = "CascadedTanksFiles/dataBenchmark.csv"
CSV_HEADER = (b"uEst", b"uVal", b"yEst", b"yVal", b"Ts")
CSV_HEADER_WIRE = b'"uEst","uVal","yEst","yVal","Ts",'
CSV_MEMBER_BYTES = 30_014
SAMPLE_INTERVAL_SECONDS = 4.0
SOURCE_ROWS = 1_024
TRAIN_STOP = 768
_MAX_HEADER_BYTES = 128
_MAX_FIELD_BYTES = 256
_OFFICIAL_LOADER_TOKEN = object()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _is_safe_member_name(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return False
    if ":" in name:
        return False
    directory = name.endswith("/")
    body = name[:-1] if directory else name
    parts = body.split("/")
    if not body or any(part in ("", ".", "..") for part in parts):
        return False
    return PurePosixPath(body).as_posix() == body


@dataclass(frozen=True, slots=True)
class ArchiveExpectation:
    """Pinned archive identity and bounded ZIP metadata limits.

    Supplying a different expectation is intended only for synthetic tests.
    Production callers should use the default fixed archive contract.
    """

    archive_bytes: int
    archive_sha256: str
    csv_member: str = CSV_MEMBER
    max_member_bytes: int = 16_000_000
    max_csv_member_bytes: int = 2_000_000
    max_total_uncompressed_bytes: int = 32_000_000
    max_members: int = 64

    def __post_init__(self) -> None:
        if self.archive_bytes <= 0:
            raise ValueError("expected archive size must be positive")
        _require_sha256(self.archive_sha256, "expected archive SHA-256")
        if not _is_safe_member_name(self.csv_member) or not self.csv_member.lower().endswith(".csv"):
            raise ValueError("expected CSV member path must be a safe .csv path")
        if min(
            self.max_member_bytes,
            self.max_csv_member_bytes,
            self.max_total_uncompressed_bytes,
            self.max_members,
        ) <= 0:
            raise ValueError("ZIP metadata limits must be positive")


OFFICIAL_ARCHIVE = ArchiveExpectation(
    archive_bytes=SOURCE_ARCHIVE_BYTES,
    archive_sha256=SOURCE_ARCHIVE_SHA256,
)


_CONTRACT_CANONICAL = {
    "track_id": TRACK_ID,
    "source_doi": SOURCE_DOI,
    "archive_name": SOURCE_ARCHIVE_NAME,
    "archive_bytes": SOURCE_ARCHIVE_BYTES,
    "archive_sha256": SOURCE_ARCHIVE_SHA256,
    "csv_member": CSV_MEMBER,
    "csv_header_fields": [part.decode("ascii") for part in CSV_HEADER],
    "csv_header_wire": (CSV_HEADER_WIRE + b"\n").decode("ascii"),
    "csv_member_bytes": CSV_MEMBER_BYTES,
    "csv_row_trailing_empty_field": True,
    "csv_line_ending": "LF",
    "csv_terminal_blank_lines": 1,
    "sample_interval_wire": {"row_0": "4.0", "rows_1_to_1023": "empty"},
    "rows": SOURCE_ROWS,
    "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
    "training_indices": [0, TRAIN_STOP],
    "development_input_indices": [TRAIN_STOP, SOURCE_ROWS],
    "development_targets_returned": False,
    "test_targets_loaded": False,
}
CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(_CONTRACT_CANONICAL, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
).hexdigest()
SYNTHETIC_FIXTURE_TRACK_ID = "synthetic-fixture-only"
SYNTHETIC_FIXTURE_CONTRACT_SHA256 = hashlib.sha256(
    b"synthetic fixture only: never use as a Cascaded Tanks production track"
).hexdigest()


class CascadedTanksSourceError(ValueError):
    """The archive or its permitted development view violates the contract."""


@dataclass(frozen=True, slots=True)
class _ArchiveInspection:
    archive_bytes: int
    archive_sha256: str
    csv_member: str
    csv_member_bytes: int
    member_count: int
    total_uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class SourcePreflight:
    """Verified archive identity and safe member metadata; contains no CSV rows."""

    source_doi: str
    source_version: str
    archive_bytes: int
    archive_sha256: str
    csv_member: str
    csv_member_bytes: int
    member_count: int
    total_uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class SyntheticFixturePreflight:
    """Synthetic ZIP metadata that is deliberately separate from source evidence."""

    fixture_marker: Literal["synthetic-fixture-only"]
    archive_bytes: int
    archive_sha256: str
    csv_member: str
    csv_member_bytes: int
    member_count: int
    total_uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class DevelopmentStageReceipts:
    """Hashes for the training view and the complete development input view."""

    training_sha256: str
    forecast_inputs_sha256: str


@dataclass(frozen=True, slots=True)
class _DevelopmentDataView:
    track_id: str
    contract_sha256: str
    source_visible_sha256: str
    sample_interval_seconds: float
    training_indices: tuple[int, ...]
    training_u_est: tuple[float, ...]
    training_y_est: tuple[float, ...]
    development_input_indices: tuple[int, ...]
    development_u_est: tuple[float, ...]
    stage_receipts: DevelopmentStageReceipts

    def __post_init__(self) -> None:
        for name in ("training_indices", "development_input_indices"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("training_u_est", "training_y_est", "development_u_est"):
            object.__setattr__(self, name, tuple(float(value) for value in getattr(self, name)))
        _require_sha256(self.source_visible_sha256, "visible source SHA-256")
        if self.sample_interval_seconds != SAMPLE_INTERVAL_SECONDS:
            raise ValueError("development data must use the fixed sample interval")
        if self.training_indices != tuple(range(TRAIN_STOP)):
            raise ValueError("training indices must be exactly [0, 768)")
        if self.development_input_indices != tuple(range(TRAIN_STOP, SOURCE_ROWS)):
            raise ValueError("development input indices must be exactly [768, 1024)")
        if len(self.training_u_est) != TRAIN_STOP or len(self.training_y_est) != TRAIN_STOP:
            raise ValueError("training inputs and outputs must contain exactly 768 samples")
        if len(self.development_u_est) != SOURCE_ROWS - TRAIN_STOP:
            raise ValueError("development inputs must contain exactly 256 samples")
        if any(not math.isfinite(value) for value in (*self.training_u_est, *self.training_y_est, *self.development_u_est)):
            raise ValueError("returned development values must be finite")
        for name in ("training_sha256", "forecast_inputs_sha256"):
            _require_sha256(getattr(self.stage_receipts, name), f"{name} receipt")


@dataclass(frozen=True, slots=True)
class CascadedTanksDevelopmentData(_DevelopmentDataView):
    """Official-source development view with production archive identity bound.

    training_y_est contains only indices [0, 768). The suffix target values
    are not fields of this object. development_u_est contains only input
    indices [768, 1024) for a later free-run development forecast.
    """

    archive_sha256: str
    _loader_token: InitVar[object]

    def __post_init__(self, _loader_token: object) -> None:
        if _loader_token is not _OFFICIAL_LOADER_TOKEN:
            raise ValueError("official development data can only be constructed by the verified source loader")
        _DevelopmentDataView.__post_init__(self)
        if self.track_id != TRACK_ID or self.contract_sha256 != CONTRACT_SHA256:
            raise ValueError("development data must use the fixed Cascaded Tanks methods contract")
        if self.archive_sha256 != SOURCE_ARCHIVE_SHA256:
            raise ValueError("production development data must bind the official archive SHA-256")
        expected_receipts = _make_stage_receipts(
            self.training_u_est,
            self.training_y_est,
            self.source_visible_sha256,
            contract_sha256=CONTRACT_SHA256,
            archive_sha256=SOURCE_ARCHIVE_SHA256,
        )
        if self.stage_receipts != expected_receipts:
            raise ValueError("production stage receipts must bind the official archive identity")


@dataclass(frozen=True, slots=True)
class SyntheticCascadedTanksDevelopmentData(_DevelopmentDataView):
    """Fixture-only development view that cannot identify as production data."""

    fixture_marker: Literal["synthetic-fixture-only"] = "synthetic-fixture-only"

    def __post_init__(self) -> None:
        _DevelopmentDataView.__post_init__(self)
        if (
            self.fixture_marker != "synthetic-fixture-only"
            or self.track_id != SYNTHETIC_FIXTURE_TRACK_ID
            or self.contract_sha256 != SYNTHETIC_FIXTURE_CONTRACT_SHA256
        ):
            raise ValueError("synthetic source fixtures cannot be labeled as production track data")
        expected_receipts = _make_stage_receipts(
            self.training_u_est,
            self.training_y_est,
            self.source_visible_sha256,
            contract_sha256=SYNTHETIC_FIXTURE_CONTRACT_SHA256,
            archive_sha256=None,
        )
        if self.stage_receipts != expected_receipts:
            raise ValueError("synthetic fixture receipts must use the synthetic-only contract")


@contextmanager
def _checked_archive(
    archive_path: str | os.PathLike[str],
    expectation: ArchiveExpectation,
) -> Iterator[tuple[zipfile.ZipFile, _ArchiveInspection, zipfile.ZipInfo]]:
    """Verify bytes and ZIP metadata while keeping one file descriptor open."""

    path = Path(archive_path)
    try:
        raw = path.open("rb")
    except OSError as error:
        raise CascadedTanksSourceError("Cascaded Tanks source archive cannot be opened") from error
    with raw:
        observed_size = os.fstat(raw.fileno()).st_size
        if observed_size != expectation.archive_bytes:
            raise CascadedTanksSourceError(
                f"archive byte count mismatch: expected {expectation.archive_bytes}, got {observed_size}"
            )
        digest = hashlib.sha256()
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
        archive_sha256 = digest.hexdigest()
        if archive_sha256 != expectation.archive_sha256:
            raise CascadedTanksSourceError("archive SHA-256 does not match the pinned source identity")
        raw.seek(0)
        try:
            archive = zipfile.ZipFile(raw, mode="r")
        except (OSError, zipfile.BadZipFile) as error:
            raise CascadedTanksSourceError("verified source bytes are not a readable ZIP archive") from error
        with archive:
            infos = archive.infolist()
            csv_info = _validate_zip_members(infos, expectation)
            inspection = _ArchiveInspection(
                archive_bytes=observed_size,
                archive_sha256=archive_sha256,
                csv_member=expectation.csv_member,
                csv_member_bytes=csv_info.file_size,
                member_count=len(infos),
                total_uncompressed_bytes=sum(info.file_size for info in infos),
            )
            yield archive, inspection, csv_info


def preflight_archive(archive_path: str | os.PathLike[str]) -> SourcePreflight:
    """Check the exact official v1 archive identity and bounded ZIP metadata."""

    with _checked_archive(archive_path, OFFICIAL_ARCHIVE) as (_, inspection, _):
        return SourcePreflight(
            source_doi=SOURCE_DOI,
            source_version="1",
            archive_bytes=inspection.archive_bytes,
            archive_sha256=inspection.archive_sha256,
            csv_member=inspection.csv_member,
            csv_member_bytes=inspection.csv_member_bytes,
            member_count=inspection.member_count,
            total_uncompressed_bytes=inspection.total_uncompressed_bytes,
        )


def load_development_data(archive_path: str | os.PathLike[str]) -> CascadedTanksDevelopmentData:
    """Load the official-source training prefix and later input sequence only."""

    preflight, parsed = _read_development_archive(archive_path, OFFICIAL_ARCHIVE)
    training_u, training_y, development_u, visible_sha256 = parsed
    receipts = _make_stage_receipts(
        training_u,
        training_y,
        visible_sha256,
        contract_sha256=CONTRACT_SHA256,
        archive_sha256=preflight.archive_sha256,
    )
    return CascadedTanksDevelopmentData(
        track_id=TRACK_ID,
        contract_sha256=CONTRACT_SHA256,
        source_visible_sha256=visible_sha256,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        training_indices=tuple(range(TRAIN_STOP)),
        training_u_est=tuple(training_u),
        training_y_est=tuple(training_y),
        development_input_indices=tuple(range(TRAIN_STOP, SOURCE_ROWS)),
        development_u_est=tuple(development_u),
        stage_receipts=receipts,
        archive_sha256=preflight.archive_sha256,
        _loader_token=_OFFICIAL_LOADER_TOKEN,
    )


def _preflight_fixture_archive(
    archive_path: str | os.PathLike[str],
    *,
    expected_archive: ArchiveExpectation,
) -> SyntheticFixturePreflight:
    """Private synthetic-test seam; never returns official source evidence."""

    with _checked_archive(archive_path, expected_archive) as (_, inspection, _):
        return SyntheticFixturePreflight(
            fixture_marker="synthetic-fixture-only",
            archive_bytes=inspection.archive_bytes,
            archive_sha256=inspection.archive_sha256,
            csv_member=inspection.csv_member,
            csv_member_bytes=inspection.csv_member_bytes,
            member_count=inspection.member_count,
            total_uncompressed_bytes=inspection.total_uncompressed_bytes,
        )


def _load_fixture_development_data(
    archive_path: str | os.PathLike[str],
    *,
    expected_archive: ArchiveExpectation,
) -> SyntheticCascadedTanksDevelopmentData:
    """Private fixture parser returning an unmistakably synthetic data type."""

    _, parsed = _read_development_archive(archive_path, expected_archive)
    training_u, training_y, development_u, visible_sha256 = parsed
    receipts = _make_stage_receipts(
        training_u,
        training_y,
        visible_sha256,
        contract_sha256=SYNTHETIC_FIXTURE_CONTRACT_SHA256,
        archive_sha256=None,
    )
    return SyntheticCascadedTanksDevelopmentData(
        track_id=SYNTHETIC_FIXTURE_TRACK_ID,
        contract_sha256=SYNTHETIC_FIXTURE_CONTRACT_SHA256,
        source_visible_sha256=visible_sha256,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        training_indices=tuple(range(TRAIN_STOP)),
        training_u_est=tuple(training_u),
        training_y_est=tuple(training_y),
        development_input_indices=tuple(range(TRAIN_STOP, SOURCE_ROWS)),
        development_u_est=tuple(development_u),
        stage_receipts=receipts,
    )


def _read_development_archive(
    archive_path: str | os.PathLike[str],
    expectation: ArchiveExpectation,
) -> tuple[_ArchiveInspection, tuple[list[float], list[float], list[float], str]]:
    with _checked_archive(archive_path, expectation) as (archive, inspection, csv_info):
        try:
            with archive.open(csv_info, mode="r") as member:
                parsed = _scan_development_csv(member, csv_info.file_size)
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise CascadedTanksSourceError("Cascaded Tanks CSV member could not be read safely") from error
    return inspection, parsed


def _scan_development_csv(
    member: BinaryIO,
    declared_member_bytes: int,
) -> tuple[list[float], list[float], list[float], str]:
    """Stream fields, retaining only allowed columns and rows.

    The byte at a time loop intentionally avoids a row tokenizer: excluded
    fields are only counted to enforce bounds and are never copied, decoded,
    or converted to numbers.
    """

    if declared_member_bytes != CSV_MEMBER_BYTES:
        raise CascadedTanksSourceError(
            f"CSV member byte count mismatch: expected {CSV_MEMBER_BYTES}, got {declared_member_bytes}"
        )
    header_line = member.readline(_MAX_HEADER_BYTES + 1)
    if len(header_line) > _MAX_HEADER_BYTES or not _valid_header_line(header_line):
        raise CascadedTanksSourceError("CSV header is not the exact five-column Cascaded Tanks schema")

    training_u: list[float] = []
    training_y: list[float] = []
    development_u: list[float] = []
    visible_digest = hashlib.sha256()
    column = 0
    row_index = 0
    field_length = 0
    field_buffer: bytearray | None = None
    rows_started = False
    terminal_blank_seen = False
    bytes_seen = len(header_line)

    def needs_buffer(column_index: int, index: int) -> bool:
        return column_index == 0 or (column_index == 2 and index < TRAIN_STOP) or (
            column_index == 4 and index == 0
        )

    def add_selected_field(column_index: int, index: int, raw_field: bytes) -> None:
        if column_index == 0:
            value = _parse_finite_number(raw_field, f"uEst row {index}")
            _update_field_hash(visible_digest, column_index, index, raw_field)
            if index < TRAIN_STOP:
                training_u.append(value)
            else:
                development_u.append(value)
        elif column_index == 2:
            value = _parse_finite_number(raw_field, f"yEst training row {index}")
            training_y.append(value)
            _update_field_hash(visible_digest, column_index, index, raw_field)
        elif column_index == 4:
            value = _parse_finite_number(raw_field, f"Ts row {index}")
            if value != SAMPLE_INTERVAL_SECONDS:
                raise CascadedTanksSourceError(
                    f"sample interval mismatch at row {index}: expected {SAMPLE_INTERVAL_SECONDS} seconds"
                )
            _update_field_hash(visible_digest, column_index, index, raw_field)

    def finish_field() -> None:
        nonlocal field_buffer
        if column == 4 and row_index > 0:
            if field_length != 0:
                raise CascadedTanksSourceError(
                    f"sample interval field must be empty after row 0; got data at row {row_index}"
                )
            _update_field_hash(visible_digest, column, row_index, b"")
            field_buffer = None
            return
        if field_length == 0:
            raise CascadedTanksSourceError(f"empty CSV field at row {row_index}, column {column}")
        if needs_buffer(column, row_index):
            if field_buffer is None:
                raise CascadedTanksSourceError("selected CSV field buffer was not initialized")
            add_selected_field(column, row_index, bytes(field_buffer))
        field_buffer = None

    def finish_row() -> None:
        nonlocal row_index, column, field_length, field_buffer, rows_started
        if not rows_started:
            raise CascadedTanksSourceError("CSV contains an unexpected blank row")
        if column != 5:
            raise CascadedTanksSourceError(
                f"CSV row {row_index} is missing the final comma and empty sixth field"
            )
        if field_length != 0:
            raise CascadedTanksSourceError(f"CSV row {row_index} sixth field must be empty")
        row_index += 1
        if row_index > SOURCE_ROWS:
            raise CascadedTanksSourceError(f"CSV contains more than {SOURCE_ROWS} data rows")
        column = 0
        field_length = 0
        field_buffer = None
        rows_started = False

    while True:
        byte = member.read(1)
        if not byte:
            break
        bytes_seen += 1
        if bytes_seen > declared_member_bytes:
            raise CascadedTanksSourceError("CSV member expanded beyond its declared bounded size")
        value = byte[0]
        if terminal_blank_seen:
            raise CascadedTanksSourceError("CSV must contain exactly one blank LF line after its 1024 rows")
        if value == 0x0A:
            if not rows_started and column == 0 and field_length == 0:
                if row_index != SOURCE_ROWS:
                    raise CascadedTanksSourceError("CSV contains a blank line before all 1024 data rows")
                terminal_blank_seen = True
                continue
            finish_row()
            continue
        rows_started = True
        if value == 0x2C:  # comma delimiter
            if column >= 5:
                raise CascadedTanksSourceError(f"CSV row {row_index} contains more than five data fields")
            finish_field()
            column += 1
            field_length = 0
            field_buffer = bytearray() if needs_buffer(column, row_index) else None
            continue
        if value == 0x22:  # quotes are not part of this fixed numeric CSV grammar
            raise CascadedTanksSourceError("quoted CSV fields are outside the fixed numeric source grammar")
        if value < 0x20 or value == 0x7F:
            raise CascadedTanksSourceError("CSV contains a control byte inside a field")
        field_length += 1
        if field_length > _MAX_FIELD_BYTES:
            raise CascadedTanksSourceError("CSV field exceeds the strict byte-length limit")
        if field_buffer is None and needs_buffer(column, row_index):
            field_buffer = bytearray()
        if field_buffer is not None:
            field_buffer.append(value)

    if rows_started or column != 0 or field_length != 0:
        raise CascadedTanksSourceError("CSV data must end after a complete LF-terminated row")
    if not terminal_blank_seen:
        raise CascadedTanksSourceError("CSV must end with exactly one blank LF line after its 1024 data rows")
    if bytes_seen != declared_member_bytes:
        raise CascadedTanksSourceError("CSV member byte count does not match its ZIP metadata")
    if row_index != SOURCE_ROWS:
        raise CascadedTanksSourceError(f"CSV must contain exactly {SOURCE_ROWS} data rows; got {row_index}")
    if len(training_u) != TRAIN_STOP or len(training_y) != TRAIN_STOP or len(development_u) != SOURCE_ROWS - TRAIN_STOP:
        raise CascadedTanksSourceError("CSV selected fields do not match the fixed development split")
    return training_u, training_y, development_u, visible_digest.hexdigest()


def _validate_zip_members(
    infos: list[zipfile.ZipInfo],
    expectation: ArchiveExpectation,
) -> zipfile.ZipInfo:
    if not infos or len(infos) > expectation.max_members:
        raise CascadedTanksSourceError("ZIP member count is outside the bounded archive contract")
    names: set[str] = set()
    csv_infos: list[zipfile.ZipInfo] = []
    total_uncompressed = 0
    for info in infos:
        if not _is_safe_member_name(info.filename):
            raise CascadedTanksSourceError(f"unsafe ZIP member path: {info.filename!r}")
        if info.filename in names:
            raise CascadedTanksSourceError(f"duplicate ZIP member path: {info.filename!r}")
        names.add(info.filename)
        if info.flag_bits & 0x1:
            raise CascadedTanksSourceError("encrypted ZIP members are not accepted")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise CascadedTanksSourceError("ZIP uses an unsupported compression method")
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise CascadedTanksSourceError("ZIP contains a non-regular or symbolic-link member")
        if kind == stat.S_IFDIR and not info.is_dir():
            raise CascadedTanksSourceError("ZIP directory member is missing its trailing slash")
        if info.is_dir() and kind not in (0, stat.S_IFDIR):
            raise CascadedTanksSourceError("ZIP directory path has a non-directory file type")
        if info.is_dir() and info.file_size != 0:
            raise CascadedTanksSourceError("ZIP directory entry declares an unexpected payload")
        if info.file_size < 0 or info.file_size > expectation.max_member_bytes:
            raise CascadedTanksSourceError("ZIP member exceeds the strict uncompressed size limit")
        if info.compress_size < 0 or info.compress_size > expectation.archive_bytes:
            raise CascadedTanksSourceError("ZIP member has an invalid compressed size")
        total_uncompressed += info.file_size
        if total_uncompressed > expectation.max_total_uncompressed_bytes:
            raise CascadedTanksSourceError("ZIP total expanded size exceeds the strict archive limit")
        if info.filename.lower().endswith(".csv"):
            csv_infos.append(info)
    if len(csv_infos) != 1 or csv_infos[0].filename != expectation.csv_member:
        raise CascadedTanksSourceError("ZIP must contain exactly one CSV at the pinned member path")
    csv_info = csv_infos[0]
    if csv_info.is_dir() or csv_info.file_size > expectation.max_csv_member_bytes:
        raise CascadedTanksSourceError("CSV member exceeds its strict uncompressed size limit")
    if csv_info.file_size != CSV_MEMBER_BYTES:
        raise CascadedTanksSourceError(
            f"CSV member byte count mismatch: expected {CSV_MEMBER_BYTES}, got {csv_info.file_size}"
        )
    return csv_info


def _valid_header_line(header_line: bytes) -> bool:
    return header_line == CSV_HEADER_WIRE + b"\n"


def _parse_finite_number(raw_field: bytes, label: str) -> float:
    try:
        value = float(raw_field)
    except (TypeError, ValueError, OverflowError) as error:
        raise CascadedTanksSourceError(f"{label} is not a valid finite number") from error
    if not math.isfinite(value):
        raise CascadedTanksSourceError(f"{label} is not finite")
    return value


def _update_field_hash(digest: object, column: int, row_index: int, raw_field: bytes) -> None:
    digest.update(column.to_bytes(1, "big"))
    digest.update(row_index.to_bytes(4, "big"))
    digest.update(len(raw_field).to_bytes(4, "big"))
    digest.update(raw_field)


def _hash_train_fields(training_u: list[float], training_y: list[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"cascaded-tanks-training-values-v1\0")
    for name, values in ((b"uEst", training_u), (b"yEst", training_y)):
        digest.update(name)
        for value in values:
            digest.update(value.hex().encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _make_stage_receipts(
    training_u: tuple[float, ...] | list[float],
    training_y: tuple[float, ...] | list[float],
    source_visible_sha256: str,
    *,
    contract_sha256: str,
    archive_sha256: str | None,
) -> DevelopmentStageReceipts:
    bound_fields: dict[str, object] = {
        "contract_sha256": contract_sha256,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
    }
    if archive_sha256 is not None:
        _require_sha256(archive_sha256, "receipt archive SHA-256")
        bound_fields["archive_sha256"] = archive_sha256
    training_receipt = _sha_fields(
        "training-stage",
        {
            **bound_fields,
            "source_visible_sha256": _hash_train_fields(list(training_u), list(training_y)),
        },
    )
    forecast_receipt = _sha_fields(
        "development-forecast-input-stage",
        {
            **bound_fields,
            "source_visible_sha256": source_visible_sha256,
        },
    )
    return DevelopmentStageReceipts(
        training_sha256=training_receipt,
        forecast_inputs_sha256=forecast_receipt,
    )


def _sha_fields(stage: str, fields: dict[str, object]) -> str:
    payload = {"stage": stage, **fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "CSV_MEMBER",
    "CSV_MEMBER_BYTES",
    "OFFICIAL_ARCHIVE",
    "SOURCE_ARCHIVE_BYTES",
    "SOURCE_ARCHIVE_SHA256",
    "SYNTHETIC_FIXTURE_CONTRACT_SHA256",
    "SYNTHETIC_FIXTURE_TRACK_ID",
    "TRACK_ID",
    "ArchiveExpectation",
    "CascadedTanksDevelopmentData",
    "CascadedTanksSourceError",
    "DevelopmentStageReceipts",
    "SourcePreflight",
    "SyntheticCascadedTanksDevelopmentData",
    "SyntheticFixturePreflight",
    "load_development_data",
    "preflight_archive",
]

from __future__ import annotations

import hashlib
import io
import stat
import warnings
import zipfile
from dataclasses import fields
from pathlib import Path

import pytest

from core.real_data.cascaded_tanks_controlled import (
    CONTRACT_SHA256,
    CSV_MEMBER,
    CSV_MEMBER_BYTES,
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_SHA256,
    SYNTHETIC_FIXTURE_CONTRACT_SHA256,
    SYNTHETIC_FIXTURE_TRACK_ID,
    TRACK_ID,
    ArchiveExpectation,
    CascadedTanksDevelopmentData,
    CascadedTanksSourceError,
    SyntheticCascadedTanksDevelopmentData,
    _load_fixture_development_data,
    _make_stage_receipts,
    _preflight_fixture_archive,
    load_development_data,
    preflight_archive,
)


def _csv_bytes(
    *,
    header: bytes = b'"uEst","uVal","yEst","yVal","Ts",',
    rows: int = 1024,
    mutate: dict[tuple[int, str], bytes] | None = None,
    line_ending: bytes = b"\n",
    trailing_comma: bool = True,
    terminal_value: bytes = b"",
    terminal_blank_lines: int = 1,
    blank_after_row: int | None = None,
) -> bytes:
    changes = mutate or {}
    result = bytearray(header + line_ending)
    for index in range(rows):
        row: dict[str, bytes] = {}
        for slot, name in enumerate(("uEst", "uVal", "yEst", "yVal")):
            width = _numeric_field_width(index, slot)
            digit = (index + slot) % 10
            row[name] = (f"{digit}.1234" if width == 6 else f"{digit}.123").encode("ascii")
        row["Ts"] = b"4.0" if index == 0 else b""
        for name in ("uEst", "uVal", "yEst", "yVal", "Ts"):
            row[name] = changes.get((index, name), row[name])
        result.extend(b",".join(row[name] for name in ("uEst", "uVal", "yEst", "yVal", "Ts")))
        if trailing_comma:
            result.extend(b",")
        result.extend(terminal_value)
        result.extend(line_ending)
        if blank_after_row == index:
            result.extend(b"\n")
    result.extend(b"\n" * terminal_blank_lines)
    return bytes(result)


def _numeric_field_width(row_index: int, slot: int) -> int:
    return 6 if row_index * 4 + slot < 3_352 else 5


def _uval_mutations(delta: int) -> dict[tuple[int, str], bytes]:
    return {
        (row_index, "uVal"): b"x" * (_numeric_field_width(row_index, 1) + delta)
        for row_index in range(1024)
    }


def _oversized_uval_mutations() -> dict[tuple[int, str], bytes]:
    target_row = 3
    overflow = 257 - _numeric_field_width(target_row, 1)
    changes = {(target_row, "uVal"): b"x" * 257}
    for row_index in range(target_row + 1, 1024):
        width = _numeric_field_width(row_index, 1)
        reduction = min(width - 1, overflow)
        if reduction:
            changes[(row_index, "uVal")] = b"x" * (width - reduction)
            overflow -= reduction
        if overflow == 0:
            break
    assert overflow == 0
    return changes


def _archive_bytes(
    csv_data: bytes,
    *,
    extras: tuple[tuple[str, bytes], ...] = (),
    duplicate_csv: bool = False,
) -> bytes:
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(CSV_MEMBER, csv_data)
            if duplicate_csv:
                archive.writestr(CSV_MEMBER, csv_data)
            for name, payload in extras:
                archive.writestr(name, payload)
    return output.getvalue()


def _archive_with_symlink(csv_data: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CSV_MEMBER, csv_data)
        symlink = zipfile.ZipInfo("CascadedTanksFiles/link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "target")
    return output.getvalue()


def _expectation(archive_data: bytes, **limits: int) -> ArchiveExpectation:
    return ArchiveExpectation(
        archive_bytes=len(archive_data),
        archive_sha256=hashlib.sha256(archive_data).hexdigest(),
        **limits,
    )


def _write(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_official_archive_contract_is_pinned_without_reading_real_archive() -> None:
    assert SOURCE_ARCHIVE_BYTES == 7_520_592
    assert SOURCE_ARCHIVE_SHA256 == "eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51"
    assert CSV_MEMBER_BYTES == 30_014
    assert TRACK_ID == "cascaded_tanks_methods_v1_20260926"
    assert len(CONTRACT_SHA256) == 64


def test_production_stage_receipts_bind_full_archive_identity() -> None:
    production_receipts = _make_stage_receipts(
        (1.0,),
        (2.0,),
        "a" * 64,
        contract_sha256=CONTRACT_SHA256,
        archive_sha256=SOURCE_ARCHIVE_SHA256,
    )
    different_archive_receipts = _make_stage_receipts(
        (1.0,),
        (2.0,),
        "a" * 64,
        contract_sha256=CONTRACT_SHA256,
        archive_sha256="0" * 64,
    )

    assert production_receipts != different_archive_receipts


def test_preflight_and_development_view_have_fixed_split_and_no_suffix_target(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes())
    archive_path = _write(tmp_path, "fixture.zip", archive_data)
    expectation = _expectation(archive_data)

    receipt = _preflight_fixture_archive(archive_path, expected_archive=expectation)
    data = _load_fixture_development_data(archive_path, expected_archive=expectation)

    assert receipt.fixture_marker == "synthetic-fixture-only"
    assert receipt.archive_bytes == len(archive_data)
    assert receipt.csv_member_bytes == CSV_MEMBER_BYTES
    assert len(_csv_bytes()) == CSV_MEMBER_BYTES
    assert receipt.archive_sha256 == hashlib.sha256(archive_data).hexdigest()
    assert receipt.csv_member == CSV_MEMBER
    assert receipt.csv_member_bytes == len(_csv_bytes())
    assert data.training_indices == tuple(range(768))
    assert data.development_input_indices == tuple(range(768, 1024))
    assert len(data.training_u_est) == len(data.training_y_est) == 768
    assert len(data.development_u_est) == 256
    assert data.sample_interval_seconds == 4.0
    assert isinstance(data, SyntheticCascadedTanksDevelopmentData)
    assert not isinstance(data, CascadedTanksDevelopmentData)
    assert data.track_id == SYNTHETIC_FIXTURE_TRACK_ID
    assert data.contract_sha256 == SYNTHETIC_FIXTURE_CONTRACT_SHA256
    assert data.contract_sha256 != CONTRACT_SHA256
    assert data.fixture_marker == "synthetic-fixture-only"
    assert all(not name.lower().endswith(("y_val", "u_val", "development_y_est")) for name in (field.name for field in fields(data)))
    assert not hasattr(data, "development_y_est")
    assert not hasattr(data, "u_val")
    assert not hasattr(data, "y_val")


def test_sealed_columns_and_development_target_suffix_do_not_change_visible_data_or_receipts(tmp_path: Path) -> None:
    baseline_csv = _csv_bytes()
    changes: dict[tuple[int, str], bytes] = {}
    for index in range(1024):
        changes[(index, "uVal")] = b"x" * _numeric_field_width(index, 1)
        changes[(index, "yVal")] = b"y" * _numeric_field_width(index, 3)
        if index >= 768:
            changes[(index, "yEst")] = b"z" * _numeric_field_width(index, 2)
    changed_csv = _csv_bytes(mutate=changes)
    baseline_archive = _archive_bytes(baseline_csv)
    changed_archive = _archive_bytes(changed_csv)
    baseline_path = _write(tmp_path, "baseline.zip", baseline_archive)
    changed_path = _write(tmp_path, "changed.zip", changed_archive)

    baseline = _load_fixture_development_data(baseline_path, expected_archive=_expectation(baseline_archive))
    changed = _load_fixture_development_data(changed_path, expected_archive=_expectation(changed_archive))

    assert baseline == changed
    assert baseline.stage_receipts == changed.stage_receipts
    assert baseline.source_visible_sha256 == changed.source_visible_sha256
    assert baseline.training_y_est == changed.training_y_est
    assert baseline.development_u_est == changed.development_u_est


def test_fixture_arrays_cannot_be_wrapped_as_official_data(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes())
    archive_path = _write(tmp_path, "fixture-wrapper-attempt.zip", archive_data)
    fixture = _load_fixture_development_data(
        archive_path,
        expected_archive=_expectation(archive_data),
    )
    official_receipts = _make_stage_receipts(
        fixture.training_u_est,
        fixture.training_y_est,
        fixture.source_visible_sha256,
        contract_sha256=CONTRACT_SHA256,
        archive_sha256=SOURCE_ARCHIVE_SHA256,
    )
    public_constructor_attempt = {
        "track_id": TRACK_ID,
        "contract_sha256": CONTRACT_SHA256,
        "source_visible_sha256": fixture.source_visible_sha256,
        "sample_interval_seconds": fixture.sample_interval_seconds,
        "training_indices": fixture.training_indices,
        "training_u_est": fixture.training_u_est,
        "training_y_est": fixture.training_y_est,
        "development_input_indices": fixture.development_input_indices,
        "development_u_est": fixture.development_u_est,
        "stage_receipts": official_receipts,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
    }

    with pytest.raises(TypeError, match="_loader_token"):
        CascadedTanksDevelopmentData(**public_constructor_attempt)
    with pytest.raises(ValueError, match="verified source loader"):
        CascadedTanksDevelopmentData(**public_constructor_attempt, _loader_token=object())


def test_selected_fields_must_be_finite_and_sample_interval_is_fixed(tmp_path: Path) -> None:
    bad_values = (
        {(7, "uEst"): b"xxxxxx"},
        {(7, "yEst"): b"xxxxxx"},
        {(0, "Ts"): b"3.0"},
        {(0, "Ts"): b"", (0, "uVal"): b"x" * 9},
        {(1, "Ts"): b"4.0", (1, "uVal"): b"xxx"},
        {(1023, "Ts"): b"4.0", (1023, "uVal"): b"xx"},
    )
    for index, changes in enumerate(bad_values):
        archive_data = _archive_bytes(_csv_bytes(mutate=changes))
        archive_path = _write(tmp_path, f"bad-value-{index}.zip", archive_data)
        with pytest.raises(CascadedTanksSourceError):
            _load_fixture_development_data(archive_path, expected_archive=_expectation(archive_data))


@pytest.mark.parametrize(
    "csv_data",
    [
        _csv_bytes(header=b'"uEst","uVal","yEst","yBad","Ts",'),
        _csv_bytes(rows=1023, mutate={(0, "uVal"): b"x" * 32}),
        _csv_bytes(mutate={(2, "uEst"): b"", (2, "uVal"): b"x" * 12}),
        _csv_bytes(mutate={(1023, "uEst"): b"", (1023, "uVal"): b"x" * 10}),
        _csv_bytes(mutate={(2, "yEst"): b"xxxxxx"}),
        _csv_bytes(mutate={(2, "uVal"): b'"1234"'}),
        _csv_bytes(mutate={(2, "uVal"): b"1.2,34"}),
        _csv_bytes(mutate={(2, "uVal"): b"x\x01xxxx"}),
        _csv_bytes(trailing_comma=False, mutate=_uval_mutations(1)),
        _csv_bytes(terminal_value=b",", mutate=_uval_mutations(-1)),
        _csv_bytes(terminal_value=b"x", mutate=_uval_mutations(-1)),
        _csv_bytes(
            line_ending=b"\r\n",
            mutate={**_uval_mutations(-1), (0, "uVal"): b"x" * (_numeric_field_width(0, 1) - 2)},
        ),
        _csv_bytes(terminal_blank_lines=0, mutate={(0, "uVal"): b"x" * 7}),
        _csv_bytes(terminal_blank_lines=2, mutate={(0, "uVal"): b"xxxxx"}),
        _csv_bytes(blank_after_row=100, terminal_blank_lines=0),
        _csv_bytes(mutate={(900, "yEst"): b"", (900, "uVal"): b"x" * 10}),
    ],
)
def test_malformed_csv_fails_closed(tmp_path: Path, csv_data: bytes) -> None:
    archive_data = _archive_bytes(csv_data)
    archive_path = _write(tmp_path, "malformed.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError):
        _load_fixture_development_data(archive_path, expected_archive=_expectation(archive_data))


def test_oversized_skipped_field_still_respects_byte_limit(tmp_path: Path) -> None:
    csv_data = _csv_bytes(mutate=_oversized_uval_mutations())
    assert len(csv_data) == CSV_MEMBER_BYTES
    archive_data = _archive_bytes(csv_data)
    archive_path = _write(tmp_path, "oversized-skipped-field.zip", archive_data)

    with pytest.raises(CascadedTanksSourceError, match="byte-length limit"):
        _load_fixture_development_data(archive_path, expected_archive=_expectation(archive_data))


def test_duplicate_member_path_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes(), duplicate_csv=True)
    archive_path = _write(tmp_path, "duplicate.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError, match="duplicate ZIP member"):
        _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))


def test_oversized_member_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes(), extras=(("large.bin", b"x" * 128),))
    archive_path = _write(tmp_path, "oversized.zip", archive_data)
    expectation = _expectation(archive_data, max_member_bytes=64)
    with pytest.raises(CascadedTanksSourceError, match="size limit"):
        _preflight_fixture_archive(archive_path, expected_archive=expectation)


def test_oversized_csv_member_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes())
    archive_path = _write(tmp_path, "oversized-csv.zip", archive_data)
    expectation = _expectation(archive_data, max_csv_member_bytes=128)
    with pytest.raises(CascadedTanksSourceError, match="CSV member exceeds"):
        _preflight_fixture_archive(archive_path, expected_archive=expectation)


def test_unsafe_member_path_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes(), extras=(("../escape.bin", b"x"),))
    archive_path = _write(tmp_path, "unsafe.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError, match="unsafe ZIP member"):
        _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))


def test_canonical_directory_entries_and_thumbs_db_are_allowed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(
        _csv_bytes(),
        extras=(
            ("CascadedTanksFiles/", b""),
            ("CascadedTanksFiles/PicsVideo/", b""),
            ("CascadedTanksFiles/Thumbs.db", b""),
        ),
    )
    archive_path = _write(tmp_path, "official-central-directory-shape.zip", archive_data)

    receipt = _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))

    assert receipt.fixture_marker == "synthetic-fixture-only"
    assert receipt.csv_member == CSV_MEMBER
    assert receipt.member_count == 4


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.bin",
        "/absolute.bin",
        "C:/drive.bin",
        "nested//empty.bin",
        r"\\server\share.bin",
        r"nested\backslash.bin",
    ],
)
def test_unsafe_member_path_variants_fail_closed(tmp_path: Path, member_name: str) -> None:
    archive_data = _archive_bytes(_csv_bytes(), extras=((member_name, b"x"),))
    archive_path = _write(tmp_path, "unsafe-path.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError, match="unsafe ZIP member"):
        _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))


def test_nonzero_directory_payload_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(
        _csv_bytes(),
        extras=(("CascadedTanksFiles/nonempty/", b"unexpected"),),
    )
    archive_path = _write(tmp_path, "nonzero-directory.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError, match="directory entry"):
        _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))


def test_symlink_member_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_with_symlink(_csv_bytes())
    archive_path = _write(tmp_path, "symlink-member.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError, match="symbolic-link"):
        _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))


def test_second_csv_member_fails_closed(tmp_path: Path) -> None:
    archive_data = _archive_bytes(
        _csv_bytes(),
        extras=(("other.csv", b'"uEst","uVal","yEst","yVal","Ts",\n'),),
    )
    archive_path = _write(tmp_path, "two-csv.zip", archive_data)
    with pytest.raises(CascadedTanksSourceError, match="exactly one CSV"):
        _preflight_fixture_archive(archive_path, expected_archive=_expectation(archive_data))


def test_size_or_hash_mismatch_fails_before_member_loading(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes())
    archive_path = _write(tmp_path, "identity-mismatch.zip", archive_data)
    wrong_size = ArchiveExpectation(
        archive_bytes=len(archive_data) + 1,
        archive_sha256=hashlib.sha256(archive_data).hexdigest(),
    )
    with pytest.raises(CascadedTanksSourceError, match="byte count mismatch"):
        _preflight_fixture_archive(archive_path, expected_archive=wrong_size)
    wrong_hash = ArchiveExpectation(
        archive_bytes=len(archive_data),
        archive_sha256="0" * 64,
    )
    with pytest.raises(CascadedTanksSourceError, match="SHA-256"):
        _preflight_fixture_archive(archive_path, expected_archive=wrong_hash)


def test_public_production_loaders_cannot_accept_fixture_archive_identity(tmp_path: Path) -> None:
    archive_data = _archive_bytes(_csv_bytes())
    archive_path = _write(tmp_path, "fixture-cannot-enter-production.zip", archive_data)
    fixture_expectation = _expectation(archive_data)

    with pytest.raises(CascadedTanksSourceError, match="byte count mismatch"):
        load_development_data(archive_path)
    with pytest.raises(CascadedTanksSourceError, match="byte count mismatch"):
        preflight_archive(archive_path)
    with pytest.raises(TypeError):
        load_development_data(archive_path, expected_archive=fixture_expectation)
    with pytest.raises(TypeError):
        preflight_archive(archive_path, expected_archive=fixture_expectation)

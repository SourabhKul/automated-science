from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from core.real_data.silverbox import (
    ARROW_FULL_START,
    ARROW_FULL_STOP,
    ARROW_NO_EXTRAPOLATION_STOP,
    MULTISINE_START,
    MULTISINE_STOP,
    MULTISINE_TRAIN_STOP,
    SAMPLE_COUNT,
    SCHROEDER_CSV_MEMBER,
    SNLS_CSV_MEMBER,
    SilverboxCandidateView,
    SilverboxScoringRecord,
    inspect_silverbox_archive,
    load_silverbox_archive,
    load_silverbox_for_scoring,
    official_split_contract,
)


RAW_ARCHIVE = Path("data/real/silverbox/raw/SilverboxFiles.zip")


def _minimal_archive(
    path: Path,
    sample_count: int = SAMPLE_COUNT,
    *,
    snls_payload: str | None = None,
    include_required_members: bool = True,
) -> None:
    rows = ["V1,V2,\n"]
    rows.extend(f"{index / 10:.8f},{index / 20:.8f},\n" for index in range(sample_count))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SNLS_CSV_MEMBER, snls_payload if snls_payload is not None else "".join(rows) + "\n")
        if include_required_members:
            archive.writestr(SCHROEDER_CSV_MEMBER, "Ovld2,Ovld1,V1,V2,\n")
            for name in (
                "SilverboxFiles/SNLS80mV.mat",
                "SilverboxFiles/Schroeder80mV.mat",
                "SilverboxFiles/README.txt",
                "SilverboxFiles/README.m",
            ):
                archive.writestr(name, "placeholder")


def test_official_split_contract_is_exact_and_has_no_published_validation_slice() -> None:
    split = official_split_contract()
    assert split["source_record"]["sample_count"] == SAMPLE_COUNT
    assert split["train_val"]["start"] == MULTISINE_START
    assert split["train_val"]["stop"] == MULTISINE_TRAIN_STOP
    assert split["train_val"]["count"] == 65062
    assert split["test_multisine"] == {
        "start": MULTISINE_TRAIN_STOP,
        "stop": MULTISINE_STOP,
        "count": 21688,
        "state_initialization_window_length": 50,
    }
    assert split["test_arrow_full"]["start"] == ARROW_FULL_START
    assert split["test_arrow_full"]["stop"] == ARROW_FULL_STOP
    assert split["test_arrow_full"]["count"] == 40475
    assert split["test_arrow_no_extrapolation"]["start"] == ARROW_FULL_START
    assert split["test_arrow_no_extrapolation"]["stop"] == ARROW_NO_EXTRAPOLATION_STOP
    assert split["test_arrow_no_extrapolation"]["subset_of"] == "test_arrow_full"
    assert "No separate published validation" in split["validation"]


def test_public_views_enforce_exact_50_and_copy_inputs() -> None:
    source_u = np.arange(60, dtype=float)
    source_y = np.arange(60, dtype=float) + 1.0
    candidate = SilverboxCandidateView(
        record_id="test",
        input_u=source_u,
        initialization_y=source_y[:50],
        sampling_time=1.0,
        state_initialization_window_length=50,
    )
    assert candidate.sample_count == 60
    assert not hasattr(candidate, "output_y")
    source_u[0] = -123.0
    source_y[0] = -456.0
    assert candidate.input_u[0] == 0.0
    assert candidate.initialization_y[0] == 1.0
    candidate.input_u[1] = -789.0
    candidate.initialization_y[1] = -987.0
    assert source_u[1] == 1.0
    assert source_y[1] == 2.0

    with pytest.raises(ValueError, match="official 50-sample"):
        SilverboxCandidateView(
            record_id="bad",
            input_u=np.arange(60, dtype=float),
            initialization_y=np.arange(60, dtype=float),
            sampling_time=1.0,
            state_initialization_window_length=60,
        )
    with pytest.raises(ValueError, match="official 50-sample"):
        SilverboxScoringRecord(
            record_id="bad",
            source_start=10,
            source_stop=70,
            input_u=np.arange(60, dtype=float),
            output_y=np.arange(60, dtype=float),
            state_initialization_window_length=60,
        )


def test_default_loader_is_development_safe_and_scoring_loader_is_explicit(tmp_path: Path) -> None:
    archive_path = tmp_path / "SilverboxFiles.zip"
    _minimal_archive(archive_path)
    development = load_silverbox_archive(archive_path)
    scoring = load_silverbox_for_scoring(archive_path)

    assert development.train_val.source_start == MULTISINE_START
    assert development.train_val.source_stop == MULTISINE_TRAIN_STOP
    assert development.train_val.sample_count == 65062
    assert [view.sample_count for view in development.test_candidates] == [21688, 40475, 32000]
    assert not hasattr(development, "test_records")
    assert all("output_y" not in vars(view) for view in development.test_candidates)
    assert [record.sample_count for record in scoring.test_records] == [21688, 40475, 32000]
    assert all("output_y" in vars(record) for record in scoring.test_records)

    for candidate, sealed in zip(development.test_candidates, scoring.test_records):
        np.testing.assert_array_equal(candidate.input_u, sealed.input_u)
        np.testing.assert_array_equal(candidate.initialization_y, sealed.output_y[:50])
        candidate.input_u[0] = -123.0
        candidate.initialization_y[0] = -456.0
        assert sealed.input_u[0] != -123.0
        assert sealed.output_y[0] != -456.0


def test_loader_reproduces_contract_on_minimal_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "SilverboxFiles.zip"
    _minimal_archive(archive_path)
    development = load_silverbox_archive(archive_path)
    scoring = load_silverbox_for_scoring(archive_path)
    assert development.train_val.sample_count == 65062
    assert scoring.test_multisine.source_start == MULTISINE_TRAIN_STOP
    assert scoring.test_multisine.source_stop == MULTISINE_STOP
    assert scoring.test_arrow_full.source_start == ARROW_FULL_START
    assert scoring.test_arrow_full.source_stop == ARROW_FULL_STOP
    assert scoring.test_arrow_no_extrapolation.source_start == ARROW_FULL_START
    assert scoring.test_arrow_no_extrapolation.source_stop == ARROW_NO_EXTRAPOLATION_STOP
    np.testing.assert_array_equal(
        scoring.test_arrow_no_extrapolation.output_y,
        scoring.test_arrow_full.output_y[:32000],
    )
    assert [view.initialization_y.size for view in development.test_candidates] == [50, 50, 50]


def test_missing_or_malformed_archives_are_rejected(tmp_path: Path) -> None:
    invalid_path = tmp_path / "not-a-zip.zip"
    invalid_path.write_bytes(b"not a zip archive")
    with pytest.raises(zipfile.BadZipFile):
        load_silverbox_archive(invalid_path)

    missing_path = tmp_path / "missing-members.zip"
    _minimal_archive(missing_path, include_required_members=False)
    with pytest.raises(ValueError, match="missing required members"):
        inspect_silverbox_archive(missing_path)
    with pytest.raises(ValueError, match="missing required members"):
        load_silverbox_archive(missing_path)

    malformed_csv_path = tmp_path / "malformed-csv.zip"
    _minimal_archive(malformed_csv_path, snls_payload="wrong,header\n")
    with pytest.raises(ValueError, match="unexpected Silverbox SNLS header"):
        load_silverbox_archive(malformed_csv_path)


def test_manifest_matches_downloaded_archive_when_present() -> None:
    if not RAW_ARCHIVE.exists():
        pytest.skip("public Silverbox archive is an ignored local input")
    manifest = json.loads(Path("data/real/silverbox/manifest.json").read_text())
    metadata = inspect_silverbox_archive(RAW_ARCHIVE)
    assert metadata["archive_bytes"] == manifest["source"]["raw_archive_bytes"] == 5793999
    assert metadata["archive_sha256"] == manifest["source"]["raw_archive_sha256"]
    assert metadata["expanded_member_bytes"] == manifest["source"]["expanded_member_bytes"] == 9738543
    expected = {member["name"]: member for member in manifest["archive_members"]}
    observed = {member["name"]: member for member in metadata["members"]}
    assert observed == expected
    fidelity = manifest["content_contract"]["source_fidelity"]
    assert fidelity["canonical_adapter_source"] == "SNLS80mV.csv"
    assert fidelity["numeric_mat_vs_csv"]["SNLS80mV.mat"]["V1"]["differing_values"] == SAMPLE_COUNT
    assert fidelity["numeric_mat_vs_csv"]["Schroeder80mV.mat"]["V2"]["differing_values"] == SAMPLE_COUNT
    development = load_silverbox_archive(RAW_ARCHIVE)
    scoring = load_silverbox_for_scoring(RAW_ARCHIVE)
    assert development.train_val.sample_count == 65062
    assert [record.sample_count for record in scoring.test_records] == [21688, 40475, 32000]


def test_downloaded_csv_fields_are_finite_when_present() -> None:
    if not RAW_ARCHIVE.exists():
        pytest.skip("public Silverbox archive is an ignored local input")
    with zipfile.ZipFile(RAW_ARCHIVE) as archive:
        for member, expected_header, value_columns in (
            (SNLS_CSV_MEMBER, ["V1", "V2"], (0, 1)),
            (SCHROEDER_CSV_MEMBER, ["Ovld2", "Ovld1", "V1", "V2"], (2, 3)),
        ):
            with io.TextIOWrapper(archive.open(member), encoding="utf-8") as text:
                rows = list(csv.reader(text))
            assert rows[0][: len(expected_header)] == expected_header
            values = np.asarray(
                [[float(row[column]) for column in value_columns] for row in rows[1:-1]], dtype=float
            )
            assert values.shape == (SAMPLE_COUNT, len(value_columns))
            assert np.all(np.isfinite(values))

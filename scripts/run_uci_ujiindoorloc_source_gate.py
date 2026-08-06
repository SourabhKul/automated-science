#!/usr/bin/env python3
"""Audit the fixed Phase 49 UCI UJIIndoorLoc source contract."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


ARCHIVE = Path("data/real/uci_ujiindoorloc/raw/uci_ujiindoorloc_310.zip")
PAGE = Path("data/real/uci_ujiindoorloc/source_provenance/dataset_page.html")
HEADERS = Path("data/real/uci_ujiindoorloc/source_provenance/archive_headers.txt")
OUT = Path("artifacts/evaluations/phase49_uci_ujiindoorloc_building_file_classification_archive_gate_20260727")
MANIFEST = Path("data/real/uci_ujiindoorloc/manifest.json")
SPLIT = Path("data/real/uci_ujiindoorloc/source_file_split.json")
DIRECTORY = "UJIndoorLoc/"
TRAIN = "UJIndoorLoc/trainingData.csv"
VALIDATION = "UJIndoorLoc/validationData.csv"
MEMBERS = (DIRECTORY, TRAIN, VALIDATION)
WAPS = tuple(f"WAP{index:03d}" for index in range(1, 521))
TRAILING = ("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP")
HEADER = WAPS + TRAILING
ROWS = {TRAIN: 19937, VALIDATION: 1111}
SOURCE_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
PAGE_URL = "https://archive.ics.uci.edu/dataset/310/ujiindoorloc%3B"


def _integer(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not integral") from exc


def audit_csv(raw: bytes, member: str) -> tuple[dict[str, object], set[str]]:
    reader = csv.reader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline=""))
    header = tuple(next(reader))
    if header != HEADER:
        raise ValueError(f"{member} header differs from fixed 529-column contract")
    rows = 0
    waps_min, waps_max, sentinel_count = None, None, 0
    buildings, floors, users, phones, timestamps = set(), set(), set(), set(), set()
    fingerprints: set[str] = set()
    coordinate_nonfinite, malformed = 0, 0
    for line_number, row in enumerate(reader, start=2):
        if len(row) != len(HEADER):
            raise ValueError(f"{member} row {line_number} has wrong field count")
        try:
            wap_values = [_integer(value, f"{member} row {line_number} WAP") for value in row[:520]]
            longitude, latitude = float(row[520]), float(row[521])
            floor = _integer(row[522], f"{member} row {line_number} floor")
            building = _integer(row[523], f"{member} row {line_number} building")
            _integer(row[524], f"{member} row {line_number} space")
            _integer(row[525], f"{member} row {line_number} relative position")
            user = _integer(row[526], f"{member} row {line_number} user")
            phone = _integer(row[527], f"{member} row {line_number} phone")
            timestamp = _integer(row[528], f"{member} row {line_number} timestamp")
        except ValueError:
            malformed += 1
            raise
        if not all(-105 <= value <= 100 for value in wap_values):
            raise ValueError(f"{member} has WAP value outside source-unit contract")
        if not all(value == 100 or value <= 0 for value in wap_values):
            raise ValueError(f"{member} has non-sentinel positive WAP value")
        if not (longitude == longitude and latitude == latitude and abs(longitude) != float("inf") and abs(latitude) != float("inf")):
            coordinate_nonfinite += 1
            raise ValueError(f"{member} has nonfinite coordinate")
        rows += 1
        waps_min = min(wap_values) if waps_min is None else min(waps_min, min(wap_values))
        waps_max = max(wap_values) if waps_max is None else max(waps_max, max(wap_values))
        sentinel_count += wap_values.count(100)
        buildings.add(building)
        floors.add(floor)
        users.add(user)
        phones.add(phone)
        timestamps.add(timestamp)
        fingerprints.add(hashlib.sha256(",".join(row).encode("utf-8")).hexdigest())
    if rows != ROWS[member] or buildings != {0, 1, 2}:
        raise ValueError(f"{member} violates fixed row count or building support")
    return {
        "rows": rows,
        "wap_minimum": waps_min,
        "wap_maximum": waps_max,
        "raw_no_detection_sentinel_count": sentinel_count,
        "building_support": sorted(buildings),
        "floor_support": sorted(floors),
        "user_ids": sorted(users),
        "phone_ids": sorted(phones),
        "unique_timestamps": len(timestamps),
        "complete_row_duplicates": rows - len(fingerprints),
        "coordinate_nonfinite": coordinate_nonfinite,
        "malformed_rows": malformed,
    }, fingerprints


def audit(archive_path: Path = ARCHIVE) -> dict[str, object]:
    if not PAGE.exists() or not HEADERS.exists():
        raise ValueError("official page or archive response provenance is missing")
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != MEMBERS:
            raise ValueError("UJIIndoorLoc archive inventory changed")
        parsed = {member: audit_csv(archive.read(member), member) for member in ROWS}
    ledgers = {member: parsed[member][0] for member in ROWS}
    cross_duplicates = len(parsed[TRAIN][1] & parsed[VALIDATION][1])
    return {
        "phase": 49,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(archive_path),
            "official_url": SOURCE_URL,
            "official_page": PAGE_URL,
            "doi": "10.24432/C5MS59",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_hash,
            "published_checksum": None,
            "archive_headers_path": str(HEADERS),
            "page_path": str(PAGE),
            "members": list(MEMBERS),
        },
        "schema": {"columns": len(HEADER), "wap_columns": len(WAPS), "trailing_metadata_columns": list(TRAILING), "headers_identical": True, "finite_coordinates": True},
        "ledgers": {"by_file": ledgers, "cross_file_complete_row_duplicates": cross_duplicates, "raw_wap_sentinel_preserved": 100},
        "frozen_file_split": {"train": TRAIN, "external": VALIDATION, "source_train_selection": "deterministic stratified 80/20 only after this gate"},
        "isolation": {"observed_feature_fitting": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed raw-WLAN adapter, train-only baseline, and output-isolated synthetic-control plan before observed fitting.",
    }


def run() -> dict[str, object]:
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    MANIFEST.write_text(json.dumps(result["source"], indent=2) + "\n")
    SPLIT.write_text(json.dumps(result["frozen_file_split"], indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

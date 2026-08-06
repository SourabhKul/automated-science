#!/usr/bin/env python3
"""Audit the fixed Phase 45 Daily and Sports Activities official source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

ARCHIVE = Path("data/real/uci_daily_sports/raw/uci_daily_sports_256.zip")
HEADERS = Path("artifacts/evaluations/phase45_uci_daily_sports_subject_prefix_classification_source_access_20260726/headers.txt")
OUT = Path("artifacts/evaluations/phase45_uci_daily_sports_subject_prefix_classification_archive_gate_20260726")
MANIFEST = Path("data/real/uci_daily_sports/manifest.json")
SPLIT = Path("data/real/uci_daily_sports/source_subject_split.json")
PATTERN = re.compile(r"data/a(?P<activity>\d{2})/p(?P<participant>[1-8])/s(?P<segment>\d{2})\.txt$")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/256/daily%2Band%2Bsports%2Bactivities.zip"
SPLIT_DATA = {"train": [1, 2, 3, 4], "selection": [5, 6], "external": [7, 8]}


def expected_members() -> set[str]:
    return {f"data/a{activity:02d}/p{participant}/s{segment:02d}.txt" for activity in range(1, 20) for participant in range(1, 9) for segment in range(1, 61)}


def parse_segment(raw: bytes, member: str) -> np.ndarray:
    lines = raw.decode("ascii").splitlines()
    if len(lines) != 125:
        raise ValueError(f"{member} has {len(lines)} rather than 125 rows")
    rows = []
    for index, line in enumerate(lines):
        values = np.fromstring(line.strip(), dtype=float, sep=",")
        if values.shape != (45,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{member} has invalid row {index}")
        rows.append(values)
    return np.asarray(rows)


def audit(archive_path: Path = ARCHIVE) -> dict[str, object]:
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as archive:
        names = tuple(archive.namelist())
        raw_members = tuple(name for name in names if not name.endswith("/"))
        expected = expected_members()
        if set(raw_members) != expected or len(raw_members) != 9120:
            raise ValueError("raw member inventory does not match the fixed grammar")
        if len(names) != 9292 or any(name.startswith("__MACOSX/") for name in names):
            raise ValueError("archive directory inventory changed")
        activity_counts = {str(activity): 0 for activity in range(1, 20)}
        participant_counts = {str(participant): 0 for participant in range(1, 9)}
        segment_fingerprints: set[str] = set()
        duplicate_segments = 0
        duplicate_rows = 0
        zero_values = 0
        minimum, maximum = float("inf"), float("-inf")
        for member in raw_members:
            match = PATTERN.fullmatch(member)
            if match is None or not 1 <= int(match["activity"]) <= 19 or not 1 <= int(match["segment"]) <= 60:
                raise ValueError(f"invalid raw member path {member}")
            values = parse_segment(archive.read(member), member)
            fingerprint = hashlib.sha256(values.tobytes()).hexdigest()
            duplicate_segments += fingerprint in segment_fingerprints
            segment_fingerprints.add(fingerprint)
            duplicate_rows += len(values) - len({hashlib.sha256(row.tobytes()).hexdigest() for row in values})
            zero_values += int(np.count_nonzero(values == 0))
            minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
            activity_counts[str(int(match["activity"]))] += 1
            participant_counts[match["participant"]] += 1
    if duplicate_segments or any(count != 480 for count in activity_counts.values()) or any(count != 1140 for count in participant_counts.values()):
        raise ValueError("locked repeated-segment coverage or duplicate contract failed")
    return {
        "phase": 45,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(archive_path),
            "official_url": SOURCE_URL,
            "doi": "10.24432/C5C59F",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "http_headers_path": str(HEADERS),
            "member_count": len(names),
            "raw_member_count": len(raw_members),
        },
        "schema": {"segment_shape": [125, 45], "finite": True, "sampling_hz_documented": 25, "units_and_calibration_limits": "UCI documents calibrated 25 Hz Xsens units but no per-axis physical-unit conversion in the raw files."},
        "coverage": {"activity_segment_counts": activity_counts, "participant_segment_counts": participant_counts, "frozen_subject_split": SPLIT_DATA},
        "ledgers": {"complete_segment_duplicates": duplicate_segments, "within_segment_duplicate_rows": duplicate_rows, "zero_values": zero_values, "minimum": minimum, "maximum": maximum},
        "isolation": {"signal_transformation": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed adapter, baseline, and output-isolated synthetic-control plan before any observed signal transformation or fitting.",
    }


def run() -> dict[str, object]:
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    MANIFEST.write_text(json.dumps(result["source"], indent=2) + "\n")
    SPLIT.write_text(json.dumps(SPLIT_DATA, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

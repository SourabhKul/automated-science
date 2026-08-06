#!/usr/bin/env python3
"""Audit the fixed Phase 51 UCI EMG Data for Gestures raw contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np


ARCHIVE = Path("data/real/uci_emg_gestures/raw/uci_emg_gestures_481.zip")
HEADERS = Path("data/real/uci_emg_gestures/source_provenance/archive_headers.txt")
API = Path("data/real/uci_emg_gestures/source_provenance/dataset_api.json")
OUT = Path("artifacts/evaluations/phase51_uci_emg_gestures_subject_holdout_archive_gate_20260728")
MANIFEST = Path("data/real/uci_emg_gestures/manifest.json")
SPLIT = Path("data/real/uci_emg_gestures/source_subject_split.json")
ROOT = "EMG_data_for_gestures-master/"
README = ROOT + "README.txt"
HEADER = ("time", "channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "class")
SUBJECTS = tuple(range(1, 37))
COMMON_CLASSES = tuple(range(1, 7))
PATTERN = re.compile(r"EMG_data_for_gestures-master/(\d{2})/([12])_raw_data_.+\.txt$")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
PAGE_URL = "https://archive.ics.uci.edu/dataset/481/emg%2Bdata%2Bfor%2Bgestures"


def _members(archive: ZipFile) -> dict[tuple[int, int], str]:
    members: dict[tuple[int, int], str] = {}
    for member in archive.namelist():
        match = PATTERN.fullmatch(member)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in members:
            raise ValueError(f"duplicate raw EMG member {member}")
        members[key] = member
    expected = {(subject, series) for subject in SUBJECTS for series in (1, 2)}
    if set(members) != expected:
        raise ValueError("EMG archive does not retain the fixed 36-subject/two-series raw member contract")
    if README not in archive.namelist():
        raise ValueError("EMG archive lacks its fixed source README")
    return members


def _audit_member(archive: ZipFile, subject: int, series: int, member: str) -> dict[str, object]:
    rows, duplicates, zero_channels, malformed = 0, 0, 0, 0
    classes: dict[int, int] = {}
    complete_segments: dict[int, int] = {}
    values_min = np.full(8, np.inf)
    values_max = np.full(8, -np.inf)
    previous_time: int | None = None
    previous_class: int | None = None
    run_length = 0
    fingerprints: set[str] = set()
    nonincreasing, class_changes = 0, 0
    with archive.open(member) as raw:
        header = raw.readline().decode("utf-8").strip().split("\t")
        if tuple(header) != HEADER:
            raise ValueError(f"{member} header changed from the fixed ten-field source schema")
        for line_number, encoded in enumerate(raw, start=2):
            line = encoded.decode("utf-8").strip()
            fields = line.split("\t")
            if len(fields) != 10:
                malformed += 1
                raise ValueError(f"{member} line {line_number} has wrong field count")
            try:
                time_ms, label = int(fields[0]), int(fields[9])
                channels = np.asarray([float(value) for value in fields[1:9]], dtype=float)
            except ValueError as exc:
                malformed += 1
                raise ValueError(f"{member} line {line_number} violates numeric schema") from exc
            if time_ms < 0 or label not in range(8) or not np.all(np.isfinite(channels)):
                raise ValueError(f"{member} line {line_number} violates time/class/finite contract")
            if previous_time is not None and time_ms <= previous_time:
                nonincreasing += 1
                raise ValueError(f"{member} line {line_number} is not strictly time-increasing")
            if previous_class == label:
                run_length += 1
            else:
                if previous_class in COMMON_CLASSES:
                    complete_segments[previous_class] = complete_segments.get(previous_class, 0) + run_length // 128
                if previous_class is not None:
                    class_changes += 1
                run_length = 1
            fingerprint = hashlib.sha256(line.encode("utf-8")).hexdigest()
            duplicates += int(fingerprint in fingerprints)
            fingerprints.add(fingerprint)
            classes[label] = classes.get(label, 0) + 1
            values_min = np.minimum(values_min, channels)
            values_max = np.maximum(values_max, channels)
            zero_channels += int(np.count_nonzero(channels == 0.0))
            rows += 1
            previous_time, previous_class = time_ms, label
    if previous_class in COMMON_CLASSES:
        complete_segments[previous_class] = complete_segments.get(previous_class, 0) + run_length // 128
    if any(complete_segments.get(label, 0) < 2 for label in COMMON_CLASSES):
        raise ValueError(f"subject {subject} series {series} lacks two complete common-class 128-sample segments")
    return {
        "rows": rows,
        "class_counts": classes,
        "complete_128_sample_segments": complete_segments,
        "channel_minimum": values_min.tolist(),
        "channel_maximum": values_max.tolist(),
        "zero_channel_values": zero_channels,
        "complete_row_duplicates": duplicates,
        "strict_time_failures": nonincreasing,
        "class_changes": class_changes,
        "malformed_rows": malformed,
    }


def audit(archive_path: Path = ARCHIVE) -> dict[str, object]:
    if not HEADERS.exists() or not API.exists():
        raise ValueError("official EMG response provenance is incomplete")
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as archive:
        members = _members(archive)
        ledgers = {
            f"{subject:02d}/{series}": _audit_member(archive, subject, series, members[(subject, series)])
            for subject in SUBJECTS
            for series in (1, 2)
        }
    subject_support = {
        subject: sorted(set().union(*(set(ledgers[f"{subject:02d}/{series}"]["class_counts"]) for series in (1, 2))))
        for subject in SUBJECTS
    }
    if any(not set(COMMON_CLASSES).issubset(subject_support[subject]) for subject in SUBJECTS):
        raise ValueError("a fixed EMG subject lacks common class support")
    split = {
        "train_subjects": list(range(1, 21)),
        "selection_subjects": list(range(21, 29)),
        "external_subjects": list(range(29, 37)),
        "source_stream": "raw_emg_eight_channels",
        "segment_samples": 128,
        "candidate_prefix_samples": 64,
        "candidate_classes": list(COMMON_CLASSES),
    }
    return {
        "phase": 51,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(archive_path), "official_url": SOURCE_URL, "official_page": PAGE_URL,
            "doi": "10.24432/C5ZP5C", "license": "CC-BY-4.0", "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256, "published_checksum": None,
            "archive_headers_path": str(HEADERS), "dataset_api_path": str(API),
            "member_inventory": archive.namelist(), "raw_member_count": len(members), "readme_member": README,
        },
        "schema": {"header": list(HEADER), "subject_count": len(SUBJECTS), "series_per_subject": 2, "common_candidate_classes": list(COMMON_CLASSES), "source_excluded_classes": [0, 7]},
        "ledgers": {"raw_member_by_subject_series": {f"{subject:02d}/{series}": members[(subject, series)] for subject in SUBJECTS for series in (1, 2)}, "by_subject_series": ledgers, "subject_class_support": subject_support},
        "frozen_subject_split": split,
        "isolation": {"observed_feature_fitting": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed raw-EMG adapter, baseline, and output-isolated controls before observed fitting.",
    }


def run() -> dict[str, object]:
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    MANIFEST.write_text(json.dumps(result["source"], indent=2, allow_nan=False) + "\n")
    SPLIT.write_text(json.dumps(result["frozen_subject_split"], indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

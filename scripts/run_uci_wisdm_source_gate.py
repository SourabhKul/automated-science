#!/usr/bin/env python3
"""Audit the fixed Phase 50 UCI WISDM raw phone-accelerometer contract."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

ARCHIVE = Path("data/real/uci_wisdm/raw/uci_wisdm_507.zip")
HEADERS = Path("data/real/uci_wisdm/source_provenance/archive_headers.txt")
API = Path("data/real/uci_wisdm/source_provenance/dataset_api.json")
OUT = Path("artifacts/evaluations/phase50_uci_wisdm_phone_accelerometer_activity_archive_gate_20260728")
MANIFEST = Path("data/real/uci_wisdm/manifest.json")
SPLIT = Path("data/real/uci_wisdm/source_subject_split.json")
OUTER_MEMBERS = ("WISDM-dataset-description.pdf", "wisdm-dataset.zip")
ROOT = "wisdm-dataset/"
RAW_ROOT = ROOT + "raw/"
PHONE_ACCEL = RAW_ROOT + "phone/accel/"
STREAMS = {
    "phone_accelerometer": PHONE_ACCEL,
    "phone_gyroscope": RAW_ROOT + "phone/gyro/",
    "watch_accelerometer": RAW_ROOT + "watch/accel/",
    "watch_gyroscope": RAW_ROOT + "watch/gyro/",
}
SUBJECTS = tuple(range(1600, 1651))
ACTIVITIES = tuple(chr(code) for code in range(ord("A"), ord("S") + 1) if chr(code) != "N")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
PAGE_URL = "https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"
FILE_PATTERN = re.compile(r"data_(\d{4})_(accel|gyro)_(phone|watch)\.txt$")


def _nested_archive(outer: ZipFile) -> ZipFile:
    if tuple(outer.namelist()) != OUTER_MEMBERS:
        raise ValueError("WISDM outer archive inventory changed")
    return ZipFile(io.BytesIO(outer.read("wisdm-dataset.zip")))


def _stream_members(archive: ZipFile, prefix: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for member in archive.namelist():
        if not member.startswith(prefix) or not member.endswith(".txt"):
            continue
        match = FILE_PATTERN.search(member)
        if match is None:
            raise ValueError(f"unexpected raw member name {member}")
        subject = int(match.group(1))
        if subject in result:
            raise ValueError(f"duplicate raw member for subject {subject} in {prefix}")
        result[subject] = member
    if tuple(sorted(result)) != SUBJECTS:
        raise ValueError(f"{prefix} does not have the fixed 51-subject member set")
    return result


def _close_segment(segments: dict[str, int], activity: str | None, length: int) -> None:
    if activity is not None:
        segments[activity] = segments.get(activity, 0) + length // 128


def _audit_phone_accel(archive: ZipFile, subject: int, member: str) -> dict[str, object]:
    rows = 0
    activities: dict[str, int] = {}
    complete_segments: dict[str, int] = {}
    values_min = np.full(3, np.inf)
    values_max = np.full(3, -np.inf)
    zero_axis_values = 0
    malformed = 0
    duplicates = 0
    nonincreasing_transitions = 0
    fingerprints: set[str] = set()
    previous_activity: str | None = None
    previous_timestamp: int | None = None
    segment_length = 0
    for line_number, line in enumerate(io.TextIOWrapper(archive.open(member), encoding="utf-8", newline=""), start=1):
        raw = line.strip()
        if not raw or not raw.endswith(";"):
            malformed += 1
            raise ValueError(f"{member} line {line_number} lacks the locked semicolon-terminated grammar")
        fields = raw[:-1].split(",")
        if len(fields) != 6:
            malformed += 1
            raise ValueError(f"{member} line {line_number} has wrong raw field count")
        try:
            parsed_subject, activity, timestamp = int(fields[0]), fields[1], int(fields[2])
            axes = np.asarray([float(value) for value in fields[3:]], dtype=float)
        except ValueError as exc:
            malformed += 1
            raise ValueError(f"{member} line {line_number} is not numeric under the fixed schema") from exc
        if parsed_subject != subject or activity not in ACTIVITIES or timestamp <= 0 or not np.all(np.isfinite(axes)):
            raise ValueError(f"{member} line {line_number} violates subject/activity/timestamp/finite contract")
        if previous_activity == activity and previous_timestamp is not None and timestamp > previous_timestamp:
            segment_length += 1
        else:
            if previous_activity == activity and previous_timestamp is not None:
                nonincreasing_transitions += 1
            _close_segment(complete_segments, previous_activity, segment_length)
            segment_length = 1
        previous_activity, previous_timestamp = activity, timestamp
        rows += 1
        activities[activity] = activities.get(activity, 0) + 1
        values_min = np.minimum(values_min, axes)
        values_max = np.maximum(values_max, axes)
        zero_axis_values += int(np.count_nonzero(axes == 0.0))
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if fingerprint in fingerprints:
            duplicates += 1
        fingerprints.add(fingerprint)
    _close_segment(complete_segments, previous_activity, segment_length)
    eligible_activities = sorted(activity for activity, count in complete_segments.items() if count >= 2)
    if len(activities) < 6 or len(eligible_activities) < 6:
        raise ValueError(f"subject {subject} does not meet the fixed activity/segment eligibility contract")
    return {
        "raw_rows": rows,
        "activity_sample_counts": activities,
        "complete_128_sample_segments": complete_segments,
        "eligible_activities": eligible_activities,
        "axis_minimum": values_min.tolist(),
        "axis_maximum": values_max.tolist(),
        "zero_axis_values": zero_axis_values,
        "complete_row_duplicates": duplicates,
        "nonincreasing_same_activity_transitions": nonincreasing_transitions,
        "malformed_rows": malformed,
    }


def audit(archive_path: Path = ARCHIVE) -> dict[str, object]:
    if not HEADERS.exists() or not API.exists():
        raise ValueError("WISDM official response provenance is missing")
    outer_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as outer:
        inner = _nested_archive(outer)
        inner_members = tuple(inner.namelist())
        required_docs = {ROOT + "README.txt", ROOT + "activity_key.txt", ROOT + "WISDM-dataset-description.pdf"}
        if not required_docs.issubset(inner_members):
            raise ValueError("WISDM nested archive lacks fixed raw documentation members")
        stream_members = {name: _stream_members(inner, prefix) for name, prefix in STREAMS.items()}
        subject_ledgers = {str(subject): _audit_phone_accel(inner, subject, stream_members["phone_accelerometer"][subject]) for subject in SUBJECTS}
    all_activities = sorted({activity for ledger in subject_ledgers.values() for activity in ledger["activity_sample_counts"]})
    if tuple(all_activities) != ACTIVITIES:
        raise ValueError("phone-accelerometer global activity support differs from the documented 18-code contract")
    split = {
        "train_subjects": list(range(1600, 1631)),
        "selection_subjects": list(range(1631, 1641)),
        "external_subjects": list(range(1641, 1651)),
        "source_stream": "phone_accelerometer",
        "segment_samples": 128,
        "candidate_prefix_samples": 64,
    }
    return {
        "phase": 50,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(archive_path),
            "official_url": SOURCE_URL,
            "official_page": PAGE_URL,
            "doi": "10.24432/C5HK59",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": outer_sha256,
            "published_checksum": None,
            "archive_headers_path": str(HEADERS),
            "dataset_api_path": str(API),
            "outer_members": list(OUTER_MEMBERS),
            "nested_archive_sha256": hashlib.sha256(ZipFile(archive_path).read("wisdm-dataset.zip")).hexdigest(),
            "nested_member_inventory": list(inner_members),
        },
        "schema": {"entry": ["subject_id", "activity_code", "timestamp", "x", "y", "z"], "phone_accelerometer_member_count": len(stream_members["phone_accelerometer"]), "all_stream_subject_sets_match": True, "activities": list(ACTIVITIES), "sampling_hz_documented": 20},
        "ledgers": {"phone_accelerometer_by_subject": subject_ledgers, "global_activity_support": all_activities, "raw_member_paths": {name: [members[subject] for subject in SUBJECTS] for name, members in stream_members.items()}},
        "frozen_subject_split": split,
        "isolation": {"observed_feature_fitting": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed phone-accelerometer adapter, baseline, and output-isolated controls before observed fitting.",
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

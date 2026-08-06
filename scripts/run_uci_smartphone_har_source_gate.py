#!/usr/bin/env python3
"""Fixed official UCI Smartphone HAR source/provenance/schema gate only."""
from __future__ import annotations

import hashlib
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARCHIVE = Path("data/real/uci_smartphone_har/raw/uci_har_dataset_240.zip")
OUT_DIR = Path("artifacts/evaluations/phase37_uci_smartphone_har_activity_tracking_archive_gate_20260724")
MANIFEST = Path("data/real/uci_smartphone_har/manifest.json")
SPLIT_PATH = Path("data/real/uci_smartphone_har/source_subject_split.json")
PREFIX = "UCI HAR Dataset/"
CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
PARTITION_ROWS = {"train": 7352, "test": 2947}
SPLIT_COUNTS = {"train": 15, "selection": 6, "external": 9}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_source_text(payload: bytes) -> str:
    """Decode the official archive documentation without changing its bytes."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _integer_vector(archive: ZipFile, member: str, expected_rows: int, field: str) -> list[int]:
    with archive.open(member) as raw:
        values = [int(line.decode("utf-8").strip()) for line in raw if line.strip()]
    if len(values) != expected_rows:
        raise ValueError(f"UCI HAR {field} vector has {len(values)}, expected {expected_rows}")
    return values


def _signal_ledger(archive: ZipFile, partition: str, subjects: list[int], labels: list[int]) -> dict[str, object]:
    rows = PARTITION_ROWS[partition]
    members = [f"{PREFIX}{partition}/Inertial Signals/{channel}_{partition}.txt" for channel in CHANNELS]
    if not all(member in archive.namelist() for member in members):
        raise ValueError(f"UCI HAR {partition} partition lacks a declared inertial matrix")
    channel_rows = {channel: 0 for channel in CHANNELS}
    nonfinite_counts = {channel: 0 for channel in CHANNELS}
    sample_counts = {channel: set() for channel in CHANNELS}
    fingerprints: set[str] = set()
    duplicate_fingerprints = 0
    raw_handles = [archive.open(member) for member in members]
    try:
        text_handles = [TextIOWrapper(handle, encoding="utf-8") for handle in raw_handles]
        for index, raw_lines in enumerate(zip(*text_handles, strict=True)):
            digest = hashlib.sha256(f"{subjects[index]}:{labels[index]}:".encode())
            for channel, line in zip(CHANNELS, raw_lines, strict=True):
                values = [float(value) for value in line.split()]
                channel_rows[channel] += 1
                sample_counts[channel].add(len(values))
                nonfinite_counts[channel] += int(not np.all(np.isfinite(values)))
                digest.update(line.strip().encode("utf-8"))
                digest.update(b"|")
            fingerprint = digest.hexdigest()
            duplicate_fingerprints += int(fingerprint in fingerprints)
            fingerprints.add(fingerprint)
        if index + 1 != rows:
            raise ValueError(f"UCI HAR {partition} signal rows have {index + 1}, expected {rows}")
        for trailing in text_handles:
            if trailing.readline():
                raise ValueError(f"UCI HAR {partition} inertial matrices are misaligned")
    finally:
        for handle in raw_handles:
            handle.close()
    return {
        "rows": rows,
        "channel_rows": channel_rows,
        "sample_counts": {channel: sorted(counts) for channel, counts in sample_counts.items()},
        "nonfinite_rows": nonfinite_counts,
        "duplicate_subject_label_window_fingerprints": duplicate_fingerprints,
        "unique_subject_label_window_fingerprints": len(fingerprints),
    }


def _subject_ledger(subjects: list[int], labels: list[int]) -> dict[str, dict[str, object]]:
    ledger: dict[str, dict[str, object]] = {}
    for subject, label in zip(subjects, labels, strict=True):
        entry = ledger.setdefault(str(subject), {"window_count": 0, "labels": set()})
        entry["window_count"] += 1
        entry["labels"].add(label)
    return {
        subject: {"window_count": int(entry["window_count"]), "labels": sorted(entry["labels"])}
        for subject, entry in sorted(ledger.items(), key=lambda item: int(item[0]))
    }


def _source_subject_split(train_subjects: list[int], test_subjects: list[int]) -> dict[str, list[int]]:
    source_train = sorted(train_subjects)
    source_test = sorted(test_subjects)
    if len(source_train) != 21 or len(source_test) != 9 or set(source_train) & set(source_test):
        raise ValueError("UCI HAR source subject partition is not 21 disjoint train plus 9 test subjects")
    return {"train": source_train[:15], "selection": source_train[15:], "external": source_test}


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT_DIR) -> dict[str, object]:
    with ZipFile(archive_path) as outer:
        outer_members = outer.namelist()
        required_outer = {"UCI HAR Dataset.names", "UCI HAR Dataset.zip"}
        if not required_outer.issubset(outer_members):
            raise ValueError("official UCI HAR wrapper lacks names or nested raw archive")
        names = outer.read("UCI HAR Dataset.names")
        inner_payload = outer.read("UCI HAR Dataset.zip")
    with ZipFile(BytesIO(inner_payload)) as archive:
        members = archive.namelist()
        required = {f"{PREFIX}README.txt", f"{PREFIX}activity_labels.txt", f"{PREFIX}features.txt"}
        for partition in PARTITION_ROWS:
            required |= {f"{PREFIX}{partition}/subject_{partition}.txt", f"{PREFIX}{partition}/y_{partition}.txt"}
            required |= {f"{PREFIX}{partition}/Inertial Signals/{channel}_{partition}.txt" for channel in CHANNELS}
        if not required.issubset(members):
            raise ValueError("nested UCI HAR archive lacks the fixed raw contract")
        readme = _decode_source_text(archive.read(f"{PREFIX}README.txt"))
        activities = [line.decode("utf-8").strip().split(maxsplit=1) for line in archive.open(f"{PREFIX}activity_labels.txt") if line.strip()]
        features = [line.decode("utf-8").strip() for line in archive.open(f"{PREFIX}features.txt") if line.strip()]
        subjects = {partition: _integer_vector(archive, f"{PREFIX}{partition}/subject_{partition}.txt", rows, f"{partition} subject") for partition, rows in PARTITION_ROWS.items()}
        labels = {partition: _integer_vector(archive, f"{PREFIX}{partition}/y_{partition}.txt", rows, f"{partition} label") for partition, rows in PARTITION_ROWS.items()}
        signal_ledgers = {partition: _signal_ledger(archive, partition, subjects[partition], labels[partition]) for partition in PARTITION_ROWS}

    activity_ids = [int(row[0]) for row in activities]
    all_subjects = set(subjects["train"]) | set(subjects["test"])
    source_split = _source_subject_split(sorted(set(subjects["train"])), sorted(set(subjects["test"])))
    subject_ledger = {partition: _subject_ledger(subjects[partition], labels[partition]) for partition in PARTITION_ROWS}
    locked_subjects = {subject for values in source_split.values() for subject in values}
    locked_eligibility = {
        str(subject): {
            "source_partition": "train" if subject in set(subjects["train"]) else "test",
            **(subject_ledger["train"].get(str(subject)) or subject_ledger["test"].get(str(subject))),
        }
        for subject in sorted(locked_subjects)
    }
    all_subjects_eligible = all(entry["window_count"] >= 50 and entry["labels"] == [1, 2, 3, 4, 5, 6] for entry in locked_eligibility.values())
    checks = {
        "outer_wrapper_contract": True,
        "nested_raw_contract": True,
        "names_bytes": len(names) == 6304,
        "readme_filtering_contract": all(term in readme.lower() for term in ("50hz", "128", "overlap", "butterworth")),
        "feature_dictionary_count": len(features) == 561,
        "activity_dictionary": activity_ids == [1, 2, 3, 4, 5, 6],
        "partition_rows": {partition: len(subjects[partition]) == rows and len(labels[partition]) == rows for partition, rows in PARTITION_ROWS.items()},
        "subject_contract": len(all_subjects) == 30 and not (set(subjects["train"]) & set(subjects["test"])),
        "label_contract": all(set(labels[partition]) == set(activity_ids) for partition in PARTITION_ROWS),
        "signal_contract": all(
            all(value == PARTITION_ROWS[partition] for value in ledger["channel_rows"].values())
            and all(value == [128] for value in ledger["sample_counts"].values())
            and all(value == 0 for value in ledger["nonfinite_rows"].values())
            for partition, ledger in signal_ledgers.items()
        ),
        "subject_activity_eligibility": all_subjects_eligible,
        "frozen_split_counts": {name: len(values) == SPLIT_COUNTS[name] for name, values in source_split.items()},
    }
    passed = all(
        checks[key] if isinstance(checks[key], bool) else all(checks[key].values())
        for key in checks
    )
    result = {
        "phase": 37,
        "status": "passed_official_source_gate_only" if passed else "blocked_official_source_gate",
        "source": {
            "dataset_page": "https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones",
            "archive_url": "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip",
            "names_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names",
            "doi": "10.24432/C54S4K",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256_file(archive_path),
            "published_checksum": None,
            "outer_members": outer_members,
            "nested_raw_sha256": _sha256_bytes(inner_payload),
            "nested_raw_members": members,
            "names_sha256": _sha256_bytes(names),
        },
        "checks": checks,
        "source_schema": {
            "train_windows": len(subjects["train"]),
            "test_windows": len(subjects["test"]),
            "total_windows": len(subjects["train"]) + len(subjects["test"]),
            "subject_ids": sorted(all_subjects),
            "activity_labels": activities,
            "channels": list(CHANNELS),
            "signal_ledgers": signal_ledgers,
            "subject_ledger": subject_ledger,
            "source_notes": {"preprocessed": True, "sampling_rate_hz": 50, "window_samples": 128, "window_overlap": "50%", "units_calibration": "not stated in source README; retained as unknown"},
        },
        "locked_subject_eligibility": locked_eligibility,
        "source_subject_split": source_split if passed else None,
        "isolation": {"signal_or_label_fitting": False, "feature_matrix_opened": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write a fixed source-normalized adapter, train-only baseline, and output-isolated synthetic/negative-control plan." if passed else "Return to source-backed application selection; do not repair or reassign this cohort.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if passed:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        SPLIT_PATH.write_text(json.dumps(source_split, indent=2) + "\n")
    return result


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "checks": result["checks"], "split": result["source_subject_split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

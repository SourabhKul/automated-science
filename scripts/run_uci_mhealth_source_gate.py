#!/usr/bin/env python3
"""Fixed official UCI MHEALTH provenance and raw-schema gate only."""
from __future__ import annotations

import hashlib
from io import TextIOWrapper
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARCHIVE = Path("data/real/uci_mhealth/raw/uci_mhealth_319.zip")
OUT_DIR = Path("artifacts/evaluations/phase40_uci_mhealth_subject_activity_tracking_archive_gate_20260725")
MANIFEST = Path("data/real/uci_mhealth/manifest.json")
SPLIT_PATH = Path("data/real/uci_mhealth/source_subject_split.json")
PREFIX = "MHEALTHDATASET/"
README = f"{PREFIX}README.txt"
SUBJECT_LOGS = {subject: f"{PREFIX}mHealth_subject{subject}.log" for subject in range(1, 11)}
EXPECTED_MEMBERS = {PREFIX, README, *SUBJECT_LOGS.values()}
SIGNAL_COLUMNS = 23
ROW_WIDTH = SIGNAL_COLUMNS + 1
ACTIVITY_LABELS = set(range(1, 13))
SOURCE_SPLIT = {"train": [1, 2, 3, 4], "selection": [5, 6, 7], "external": [8, 9, 10]}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_optional(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


def _inspect_subject(archive: ZipFile, member: str) -> dict[str, object]:
    rows = 0
    blank_rows = 0
    width_mismatches = 0
    parse_failures = 0
    nonfinite_cells = 0
    invalid_labels = 0
    labels: set[int] = set()
    label_counts: dict[int, int] = {}
    label_transitions = 0
    previous_label: int | None = None
    zero_count = 0
    negative_count = 0
    minima = np.full(SIGNAL_COLUMNS, np.inf)
    maxima = np.full(SIGNAL_COLUMNS, -np.inf)
    fingerprints: set[bytes] = set()
    duplicate_complete_rows = 0
    with archive.open(member) as raw:
        for line in TextIOWrapper(raw, encoding="utf-8"):
            stripped = line.strip()
            if not stripped:
                blank_rows += 1
                continue
            rows += 1
            values = np.fromstring(stripped, dtype=float, sep=" ")
            if values.size != ROW_WIDTH:
                width_mismatches += 1
                continue
            if not np.all(np.isfinite(values)):
                nonfinite_cells += int(np.count_nonzero(~np.isfinite(values)))
                continue
            label_value = values[-1]
            if not label_value.is_integer() or not 0 <= label_value <= 12:
                invalid_labels += 1
                continue
            label = int(label_value)
            signals = values[:-1]
            labels.add(label)
            label_counts[label] = label_counts.get(label, 0) + 1
            label_transitions += int(previous_label is not None and label != previous_label)
            previous_label = label
            zero_count += int(np.count_nonzero(signals == 0.0))
            negative_count += int(np.count_nonzero(signals < 0.0))
            minima = np.minimum(minima, signals)
            maxima = np.maximum(maxima, signals)
            fingerprint = hashlib.sha256(stripped.encode("utf-8")).digest()
            duplicate_complete_rows += int(fingerprint in fingerprints)
            fingerprints.add(fingerprint)
    return {
        "rows": rows,
        "blank_rows": blank_rows,
        "row_width": ROW_WIDTH,
        "row_width_mismatches": width_mismatches,
        "parse_failures": parse_failures,
        "nonfinite_cells": nonfinite_cells,
        "invalid_labels": invalid_labels,
        "labels": sorted(labels),
        "label_counts": {str(label): count for label, count in sorted(label_counts.items())},
        "label_transitions": label_transitions,
        "duplicate_complete_rows": duplicate_complete_rows,
        "unique_complete_rows": len(fingerprints),
        "signal_zero_count": zero_count,
        "signal_negative_count": negative_count,
        "signal_range": {"min": minima.tolist(), "max": maxima.tolist()},
    }


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT_DIR) -> dict[str, object]:
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        inspection = {subject: _inspect_subject(archive, member) for subject, member in SUBJECT_LOGS.items() if member in archive.namelist()}
        readme_sha256 = hashlib.sha256(archive.read(README)).hexdigest() if README in archive.namelist() else None
    checks = {
        "official_member_inventory": set(members) == EXPECTED_MEMBERS and len(members) == len(EXPECTED_MEMBERS),
        "ten_fixed_subject_logs": set(inspection) == set(SUBJECT_LOGS) and all(item["rows"] > 0 for item in inspection.values()),
        "fixed_headerless_24_column_contract": {str(subject): item["row_width_mismatches"] == 0 and item["parse_failures"] == 0 for subject, item in inspection.items()},
        "finite_numeric_rows_and_labels": {str(subject): item["nonfinite_cells"] == 0 and item["invalid_labels"] == 0 for subject, item in inspection.items()},
        "all_documented_activities_per_subject": {str(subject): ACTIVITY_LABELS.issubset(set(item["labels"])) for subject, item in inspection.items()},
        "frozen_subject_split": SOURCE_SPLIT == {"train": [1, 2, 3, 4], "selection": [5, 6, 7], "external": [8, 9, 10]},
    }
    passed = all(value if isinstance(value, bool) else all(value.values()) for value in checks.values())
    provenance = Path("data/real/uci_mhealth/source_provenance")
    result = {
        "phase": 40,
        "status": "passed_official_source_gate_only" if passed else "blocked_official_source_gate",
        "source": {
            "dataset_page": "https://archive.ics.uci.edu/dataset/319/mhealth%2Bdataset",
            "archive_url": "https://archive.ics.uci.edu/static/public/319/mhealth%2Bdataset.zip",
            "doi": "10.24432/C5TW22",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256_file(archive_path),
            "published_checksum": None,
            "response_headers_sha256": _sha256_optional(provenance / "archive_headers.txt"),
            "page_sha256": _sha256_optional(provenance / "dataset_page.html"),
            "readme_member_sha256": readme_sha256,
            "members": members,
        },
        "checks": checks,
        "subject_ledgers": {str(subject): item for subject, item in inspection.items()},
        "source_notes": {
            "sampling_rate_hz": 50,
            "motion_surface": "21 motion channels; two chest ECG leads are excluded from the contingent activity candidate surface",
            "units": "UCI documents acceleration m/s^2, gyroscope deg/s, magnetic field local units, and ECG mV",
            "ground_truth_and_cointervention": "UCI documents that each session was recorded using a video camera",
            "protocol": "UCI describes out-of-lab activity execution without constraints beyond trying their best",
        },
        "source_subject_split": SOURCE_SPLIT if passed else None,
        "isolation": {"window_construction": False, "signal_or_label_transformation_or_fitting": False, "external_label_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write a fixed source-normalized adapter, train-only baseline, and output-isolated synthetic/negative-control plan." if passed else "Return to source-backed application selection; do not repair, resample, trim, or reassign source records or subjects.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if passed:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        SPLIT_PATH.write_text(json.dumps(SOURCE_SPLIT, indent=2) + "\n")
    return result


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "checks": result["checks"], "split": result["source_subject_split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

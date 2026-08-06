#!/usr/bin/env python3
"""Fixed official UCI Occupancy Detection provenance and raw-schema gate only."""
from __future__ import annotations

import csv
from datetime import datetime
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


ARCHIVE = Path("data/real/uci_occupancy_detection/raw/uci_occupancy_detection_357.zip")
OUT_DIR = Path("artifacts/evaluations/phase39_uci_occupancy_detection_session_tracking_archive_gate_20260725")
MEMBERS = ("datatraining.txt", "datatest.txt", "datatest2.txt")
EXPECTED_ROWS = {"datatraining.txt": 8143, "datatest.txt": 2665, "datatest2.txt": 9752}
DECLARED_HEADER = ("date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy")
SOURCE_SPLIT = {"train": "datatraining.txt", "selection": "datatest.txt", "external": "datatest2.txt"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_optional(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


def _inspect_member(archive: ZipFile, member: str) -> dict[str, object]:
    with archive.open(member) as raw:
        reader = csv.reader(TextIOWrapper(raw, encoding="utf-8", newline=""))
        header = tuple(next(reader, ()))
        row_count = 0
        width_mismatches = 0
        malformed_rows = 0
        nonfinite_cells = 0
        invalid_labels = 0
        timestamps: list[datetime] = []
        timestamp_parse_failures = 0
        fingerprints: set[str] = set()
        duplicate_complete_rows = 0
        zeros = {"Temperature": 0, "Humidity": 0, "Light": 0, "CO2": 0, "HumidityRatio": 0}
        negatives = {name: 0 for name in zeros}
        extrema = {name: {"min": None, "max": None} for name in zeros}
        labels: set[int] = set()
        for row in reader:
            row_count += 1
            width_mismatches += int(len(row) != len(header))
            fingerprint = hashlib.sha256("\x1f".join(row).encode("utf-8")).hexdigest()
            duplicate_complete_rows += int(fingerprint in fingerprints)
            fingerprints.add(fingerprint)
            # The raw file has a physical leading identifier without a matching header.
            if len(row) != 8:
                malformed_rows += 1
                continue
            try:
                timestamps.append(datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                timestamp_parse_failures += 1
            try:
                values = {
                    "Temperature": float(row[2]),
                    "Humidity": float(row[3]),
                    "Light": float(row[4]),
                    "CO2": float(row[5]),
                    "HumidityRatio": float(row[6]),
                }
                label = int(row[7])
            except ValueError:
                malformed_rows += 1
                continue
            nonfinite_cells += sum(not np.isfinite(value) for value in values.values())
            invalid_labels += int(label not in (0, 1))
            labels.add(label)
            for name, value in values.items():
                zeros[name] += int(value == 0.0)
                negatives[name] += int(value < 0.0)
                extrema[name]["min"] = value if extrema[name]["min"] is None else min(extrema[name]["min"], value)
                extrema[name]["max"] = value if extrema[name]["max"] is None else max(extrema[name]["max"], value)
    deltas = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:], strict=False)]
    duplicate_timestamps = len(timestamps) - len(set(timestamps))
    return {
        "rows": row_count,
        "raw_header": list(header),
        "declared_header_width": len(header),
        "observed_row_width": 8,
        "row_width_mismatches": width_mismatches,
        "unlabeled_leading_identifier": width_mismatches == row_count and len(header) == 7,
        "malformed_rows": malformed_rows,
        "nonfinite_measurement_cells": int(nonfinite_cells),
        "invalid_labels": invalid_labels,
        "labels": sorted(labels),
        "timestamp_parse_failures": timestamp_parse_failures,
        "duplicate_timestamps": duplicate_timestamps,
        "strict_timestamp_order_violations": sum(delta <= 0.0 for delta in deltas),
        "timestamp_delta_seconds": {
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "nonminute_gaps": sum(delta != 60.0 for delta in deltas),
        },
        "duplicate_complete_rows": duplicate_complete_rows,
        "unique_complete_rows": len(fingerprints),
        "zero_counts": zeros,
        "negative_counts": negatives,
        "range_ledger": extrema,
    }


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT_DIR) -> dict[str, object]:
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        inspections = {member: _inspect_member(archive, member) for member in MEMBERS if member in archive.namelist()}
    checks = {
        "official_member_inventory": set(members) == set(MEMBERS) and len(members) == len(MEMBERS),
        "fixed_row_counts": {member: inspections.get(member, {}).get("rows") == expected for member, expected in EXPECTED_ROWS.items()},
        "literal_declared_headers": {member: item["raw_header"] == list(DECLARED_HEADER) for member, item in inspections.items()},
        "raw_row_width_matches_declared_schema": {member: item["row_width_mismatches"] == 0 for member, item in inspections.items()},
        "finite_measurements_and_binary_labels": {
            member: item["malformed_rows"] == 0 and item["nonfinite_measurement_cells"] == 0 and item["invalid_labels"] == 0 and item["labels"] == [0, 1]
            for member, item in inspections.items()
        },
        "valid_unique_strict_timestamps": {
            member: item["timestamp_parse_failures"] == 0 and item["duplicate_timestamps"] == 0 and item["strict_timestamp_order_violations"] == 0
            for member, item in inspections.items()
        },
        "frozen_native_session_split": SOURCE_SPLIT == {"train": "datatraining.txt", "selection": "datatest.txt", "external": "datatest2.txt"},
    }
    checks_pass = all(value if isinstance(value, bool) else all(value.values()) for value in checks.values())
    provenance = Path("data/real/uci_occupancy_detection/source_provenance")
    result = {
        "phase": 39,
        "status": "passed_official_source_gate_only" if checks_pass else "blocked_official_source_gate",
        "source": {
            "dataset_page": "https://archive.ics.uci.edu/dataset/357/occupancy%2Bdetection",
            "archive_url": "https://archive.ics.uci.edu/static/public/357/occupancy%2Bdetection.zip",
            "doi": "10.24432/C5X01N",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256_file(archive_path),
            "published_checksum": None,
            "response_headers_sha256": _sha256_optional(provenance / "archive_headers.txt"),
            "page_sha256": _sha256_optional(provenance / "dataset_page.html"),
            "members": members,
        },
        "checks": checks,
        "member_ledgers": inspections,
        "source_units": {"Temperature": "C", "Humidity": "%", "Light": "Lux", "CO2": "ppm", "HumidityRatio": "kgwater-vapor/kg-air"},
        "source_notes": "UCI describes minute-stamped photographic occupancy ground truth and reports no missing values. Its raw text members declare seven headers but emit eight physical fields, with an unlabeled leading identifier.",
        "source_session_split": SOURCE_SPLIT if checks_pass else None,
        "isolation": {"measurement_transformation_or_fitting": False, "external_label_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Blocked: preserve the raw header/row-width mismatch; do not reinterpret, relabel, drop, or repair it. Return to source-backed application selection.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

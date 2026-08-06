#!/usr/bin/env python3
"""Fixed official UCI ISOLET provenance and compressed raw-schema gate only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARCHIVE = Path("data/real/uci_isolet/raw/uci_isolet_54.zip")
OUT_DIR = Path("artifacts/evaluations/phase41_uci_isolet_speaker_group_letter_classification_archive_gate_20260725")
MANIFEST = Path("data/real/uci_isolet/manifest.json")
SPLIT_PATH = Path("data/real/uci_isolet/source_file_split.json")
TRAIN_MEMBER = "isolet1+2+3+4.data.Z"
EXTERNAL_MEMBER = "isolet5.data.Z"
REQUIRED_MEMBERS = {"Index", "isolet.info", "isolet.names", TRAIN_MEMBER, EXTERNAL_MEMBER}
EXPECTED_ROWS = {TRAIN_MEMBER: 6238, EXTERNAL_MEMBER: 1559}
ROW_WIDTH = 618
FEATURE_COUNT = 617
LABELS = set(range(1, 27))
SOURCE_SPLIT = {"source_train": TRAIN_MEMBER, "external": EXTERNAL_MEMBER, "selection_design": "stratified_80_20_seed_2041054_inside_source_train"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_optional(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


def _decompress_member(archive: ZipFile, member: str) -> bytes:
    """Use the platform's LZW reader in a temporary location; no extracted raw table persists."""
    with TemporaryDirectory() as temporary:
        compressed = Path(temporary) / member
        compressed.write_bytes(archive.read(member))
        completed = subprocess.run(["uncompress", "-c", str(compressed)], check=False, capture_output=True)
    if completed.returncode != 0:
        raise ValueError(f"ISOLET member {member} did not decompress with the recorded uncompress command")
    return completed.stdout


def _inspect_member(archive: ZipFile, member: str) -> dict[str, object]:
    payload = _decompress_member(archive, member)
    rows = 0
    blank_rows = 0
    width_mismatches = 0
    parse_failures = 0
    nonfinite_cells = 0
    invalid_labels = 0
    labels: set[int] = set()
    label_counts: dict[int, int] = {}
    zero_count = 0
    negative_count = 0
    minima = np.full(FEATURE_COUNT, np.inf)
    maxima = np.full(FEATURE_COUNT, -np.inf)
    fingerprints: set[bytes] = set()
    duplicate_complete_rows = 0
    for line in payload.decode("ascii").splitlines():
        stripped = line.strip()
        if not stripped:
            blank_rows += 1
            continue
        rows += 1
        values = np.fromstring(stripped, dtype=float, sep=",")
        if values.size != ROW_WIDTH:
            width_mismatches += 1
            continue
        if not np.all(np.isfinite(values)):
            nonfinite_cells += int(np.count_nonzero(~np.isfinite(values)))
            continue
        label_value = values[-1]
        if not label_value.is_integer() or int(label_value) not in LABELS:
            invalid_labels += 1
            continue
        label = int(label_value)
        signals = values[:-1]
        labels.add(label)
        label_counts[label] = label_counts.get(label, 0) + 1
        zero_count += int(np.count_nonzero(signals == 0.0))
        negative_count += int(np.count_nonzero(signals < 0.0))
        minima = np.minimum(minima, signals)
        maxima = np.maximum(maxima, signals)
        fingerprint = hashlib.sha256(stripped.encode("ascii")).digest()
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
        "duplicate_complete_rows": duplicate_complete_rows,
        "unique_complete_rows": len(fingerprints),
        "feature_zero_count": zero_count,
        "feature_negative_count": negative_count,
        "feature_range": {"min": minima.tolist(), "max": maxima.tolist()},
        "decompression": "uncompress -c temporary_member.Z",
    }


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT_DIR) -> dict[str, object]:
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        inspection = {member: _inspect_member(archive, member) for member in (TRAIN_MEMBER, EXTERNAL_MEMBER) if member in archive.namelist()}
        info_sha256 = hashlib.sha256(archive.read("isolet.info")).hexdigest() if "isolet.info" in archive.namelist() else None
        names_sha256 = hashlib.sha256(archive.read("isolet.names")).hexdigest() if "isolet.names" in archive.namelist() else None
    checks = {
        "official_member_inventory": set(members) == REQUIRED_MEMBERS and len(members) == len(REQUIRED_MEMBERS),
        "deterministic_decompression": set(inspection) == {TRAIN_MEMBER, EXTERNAL_MEMBER},
        "fixed_row_counts": {member: inspection.get(member, {}).get("rows") == expected for member, expected in EXPECTED_ROWS.items()},
        "finite_618_field_schema": {member: item["row_width_mismatches"] == 0 and item["parse_failures"] == 0 and item["nonfinite_cells"] == 0 for member, item in inspection.items()},
        "labels_1_to_26_per_source_member": {member: set(item["labels"]) == LABELS and item["invalid_labels"] == 0 for member, item in inspection.items()},
        "frozen_external_file_boundary": SOURCE_SPLIT["external"] == EXTERNAL_MEMBER,
    }
    passed = all(value if isinstance(value, bool) else all(value.values()) for value in checks.values())
    provenance = Path("data/real/uci_isolet/source_provenance")
    result = {
        "phase": 41,
        "status": "passed_official_source_gate_only" if passed else "blocked_official_source_gate",
        "source": {
            "dataset_page": "https://archive.ics.uci.edu/dataset/54/isolet",
            "archive_url": "https://archive.ics.uci.edu/static/public/54/isolet.zip",
            "doi": "10.24432/C51G69",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256_file(archive_path),
            "published_checksum": None,
            "response_headers_sha256": _sha256_optional(provenance / "archive_headers.txt"),
            "page_sha256": _sha256_optional(provenance / "dataset_page.html"),
            "isolet_info_sha256": info_sha256,
            "isolet_names_sha256": names_sha256,
            "members": members,
        },
        "checks": checks,
        "member_ledgers": inspection,
        "source_limitations": {"speaker_group_training_ids": "not exposed in combined isolet1+2+3+4 source file", "feature_order": "UCI documents exact feature order is not known", "missing_examples": "UCI documents three examples were dropped due to recording difficulties"},
        "source_file_split": SOURCE_SPLIT if passed else None,
        "isolation": {"feature_or_label_transformation_or_fitting": False, "selection_split_constructed": False, "external_label_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed adapter, train-only baseline, and output-isolated synthetic/negative-control plan." if passed else "Return to source-backed selection; do not repair compressed records, reconstruct speakers, or reassign rows.",
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
    print(json.dumps({"status": result["status"], "checks": result["checks"], "split": result["source_file_split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

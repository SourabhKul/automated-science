#!/usr/bin/env python3
"""Audit the fixed Phase 47 UCI Statlog Landsat source contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ARCHIVE = Path("data/real/uci_statlog_landsat/raw/uci_statlog_landsat_146.zip")
PAGE = Path("data/real/uci_statlog_landsat/source_provenance/dataset_page.html")
HEADERS = Path("data/real/uci_statlog_landsat/source_provenance/archive_headers.txt")
OUT = Path("artifacts/evaluations/phase47_uci_statlog_landsat_source_file_land_cover_archive_gate_20260727")
MANIFEST = Path("data/real/uci_statlog_landsat/manifest.json")
SPLIT = Path("data/real/uci_statlog_landsat/source_file_split.json")
MEMBERS = ("Index", "sat.doc", "sat.trn", "sat.tst")
FILES = {"sat.trn": 4435, "sat.tst": 2000}
LABELS = (1, 2, 3, 4, 5, 7)
SOURCE_URL = "https://archive.ics.uci.edu/static/public/146/statlog%2Blandsat%2Bsatellite.zip"
PAGE_URL = "https://archive.ics.uci.edu/dataset/146/statlog%2Blandsat%2Bsatellite"


def parse_rows(raw: bytes, member: str) -> tuple[np.ndarray, tuple[str, ...]]:
    values, fingerprints = [], []
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), start=1):
        stripped = line.strip()
        row = np.fromstring(stripped, dtype=float, sep=" ")
        if (
            not stripped
            or row.shape != (37,)
            or not np.all(np.isfinite(row))
            or not np.all(row == np.floor(row))
            or np.any(row[:-1] < 0)
            or np.any(row[:-1] > 255)
            or int(row[-1]) not in LABELS
        ):
            raise ValueError(f"{member} violates the locked integer schema at row {line_number}")
        values.append(row.astype(np.int16))
        fingerprints.append(hashlib.sha256(stripped.encode("ascii")).hexdigest())
    matrix = np.asarray(values)
    if matrix.shape != (FILES[member], 37) or set(matrix[:, -1]) != set(LABELS):
        raise ValueError(f"{member} violates the locked count or class-support contract")
    return matrix, tuple(fingerprints)


def audit(archive_path: Path = ARCHIVE) -> dict[str, object]:
    if not PAGE.exists() or not HEADERS.exists():
        raise ValueError("official page or archive response provenance is missing")
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != MEMBERS:
            raise ValueError("Statlog Landsat archive inventory changed")
        if not archive.read("sat.doc").startswith(b"FILE NAMES"):
            raise ValueError("Statlog Landsat documentation member changed")
        parsed = {member: parse_rows(archive.read(member), member) for member in FILES}
    matrices = {member: item[0] for member, item in parsed.items()}
    fingerprints = {member: item[1] for member, item in parsed.items()}
    all_values = np.concatenate([matrix[:, :-1] for matrix in matrices.values()])
    ledgers = {
        member: {
            "rows": len(matrix),
            "class_counts": {str(label): int(np.count_nonzero(matrix[:, -1] == label)) for label in LABELS},
            "complete_row_duplicates": len(matrix) - len(set(fingerprints[member])),
            "minimum": int(matrix[:, :-1].min()),
            "maximum": int(matrix[:, :-1].max()),
        }
        for member, matrix in matrices.items()
    }
    result = {
        "phase": 47,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(archive_path),
            "official_url": SOURCE_URL,
            "official_page": PAGE_URL,
            "doi": "10.24432/C55887",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_hash,
            "published_checksum": None,
            "archive_headers_path": str(HEADERS),
            "page_path": str(PAGE),
            "members": list(MEMBERS),
        },
        "schema": {
            "fields_per_row": 37,
            "feature_count": 36,
            "feature_range": [int(all_values.min()), int(all_values.max())],
            "labels": list(LABELS),
            "all_finite": True,
            "raw_order": "9 pixels left-to-right/top-to-bottom, each with 4 spectral bands",
        },
        "ledgers": {
            "by_file": ledgers,
            "cross_file_complete_row_duplicates": len(set(fingerprints["sat.trn"]) & set(fingerprints["sat.tst"])),
            "source_limits": "Rows are randomized and removed; scene/acquisition identity and spatial reconstruction are unavailable.",
        },
        "frozen_file_split": {"train": "sat.trn", "external": "sat.tst", "source_train_selection": "deterministic stratified 80/20 only after this gate"},
        "isolation": {"observed_feature_fitting": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed raw-neighborhood adapter, train-only baseline, and output-isolated synthetic-control plan before observed fitting.",
    }
    return result


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

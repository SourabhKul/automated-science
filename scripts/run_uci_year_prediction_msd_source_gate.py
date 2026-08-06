#!/usr/bin/env python3
"""Audit the fixed Phase 61 UCI YearPredictionMSD source contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ARCHIVE = Path("data/real/uci_year_prediction_msd/raw/yearpredictionmsd.zip")
PROVENANCE = Path("data/real/uci_year_prediction_msd/source_provenance")
OUT = Path("artifacts/evaluations/phase61_uci_year_prediction_msd_artist_holdout_archive_gate_20260801")
MANIFEST = Path("data/real/uci_year_prediction_msd/manifest.json")
SPLIT = Path("data/real/uci_year_prediction_msd/source_file_split.json")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
MEMBER = "YearPredictionMSD.txt"
TOTAL_ROWS = 515_345
SOURCE_ROWS = 463_715
EXTERNAL_ROWS = 51_630
FIELDS = 91
YEAR_BOUNDS = (1922, 2011)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_line(raw: bytes, line_number: int) -> np.ndarray:
    try:
        values = np.fromstring(raw.decode("ascii").strip(), sep=",")
    except UnicodeDecodeError as error:
        raise ValueError(f"non-ASCII source row {line_number}") from error
    if values.shape != (FIELDS,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid numeric grammar at source row {line_number}")
    if not float(values[0]).is_integer() or not YEAR_BOUNDS[0] <= int(values[0]) <= YEAR_BOUNDS[1]:
        raise ValueError(f"invalid release year at source row {line_number}")
    return values


def audit() -> dict[str, object]:
    headers = PROVENANCE / "archive.headers"
    page = PROVENANCE / "dataset_page.html"
    if not headers.exists() or not page.exists():
        raise ValueError("official UCI page or archive provenance is missing")
    with zipfile.ZipFile(ARCHIVE) as archive:
        inventory = tuple(archive.namelist())
        if inventory != (MEMBER,):
            raise ValueError("YearPredictionMSD ZIP inventory changed")
        info = archive.getinfo(MEMBER)
        line_count = 0
        years: set[int] = set()
        year_counts: dict[int, int] = {}
        feature_min = float("inf")
        feature_max = float("-inf")
        zero_features = 0
        complete_source: set[str] = set()
        complete_external: set[str] = set()
        candidate_source: set[str] = set()
        candidate_external: set[str] = set()
        source_duplicates = external_duplicates = 0
        candidate_source_duplicates = candidate_external_duplicates = 0
        with archive.open(MEMBER) as handle:
            for line_count, raw in enumerate(handle, start=1):
                values = parse_line(raw, line_count)
                year = int(values[0])
                years.add(year)
                year_counts[year] = year_counts.get(year, 0) + 1
                features = values[1:]
                feature_min = min(feature_min, float(features.min()))
                feature_max = max(feature_max, float(features.max()))
                zero_features += int(np.count_nonzero(features == 0))
                complete = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
                candidate = hashlib.sha256(features.astype("<f8", copy=False).tobytes()).hexdigest()
                if line_count <= SOURCE_ROWS:
                    source_duplicates += complete in complete_source
                    candidate_source_duplicates += candidate in candidate_source
                    complete_source.add(complete)
                    candidate_source.add(candidate)
                else:
                    external_duplicates += complete in complete_external
                    candidate_external_duplicates += candidate in candidate_external
                    complete_external.add(complete)
                    candidate_external.add(candidate)
    if line_count != TOTAL_ROWS or len(complete_source) + source_duplicates != SOURCE_ROWS or len(complete_external) + external_duplicates != EXTERNAL_ROWS:
        raise ValueError("YearPredictionMSD fixed row count changed")
    result = {
        "phase": 61,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(ARCHIVE),
            "official_url": SOURCE_URL,
            "dataset_page": "https://archive.ics.uci.edu/dataset/203/year+prediction+msd",
            "doi": "10.24432/C50K61",
            "license": "CC BY 4.0",
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": sha256(ARCHIVE),
            "archive_headers": str(headers),
            "dataset_page_evidence": str(page),
            "published_checksum": None,
            "members": list(inventory),
            "member_uncompressed_bytes": info.file_size,
        },
        "schema": {
            "header": "absent",
            "rows": line_count,
            "fields": FIELDS,
            "target": "literal first release-year field",
            "target_bounds": list(YEAR_BOUNDS),
            "target_support": sorted(years),
            "target_counts": {str(year): year_counts[year] for year in sorted(years)},
            "raw_features": 90,
            "feature_groups": {"timbre_average": 12, "timbre_covariance": 78},
            "all_finite": True,
            "feature_minimum": feature_min,
            "feature_maximum": feature_max,
            "zero_features": zero_features,
        },
        "ledgers": {
            "source_complete_row_duplicates": source_duplicates,
            "external_complete_row_duplicates": external_duplicates,
            "cross_file_complete_row_duplicates": len(complete_source & complete_external),
            "source_candidate_duplicates": candidate_source_duplicates,
            "external_candidate_duplicates": candidate_external_duplicates,
            "cross_file_candidate_duplicates": len(candidate_source & candidate_external),
        },
        "frozen_file_split": {
            "source_train_rows": SOURCE_ROWS,
            "external_rows": EXTERNAL_ROWS,
            "source_train": "first 463715 source rows",
            "external": "last 51630 source rows",
            "external_boundary": "UCI-documented artist-disjoint partition",
            "selection": "deterministic target-binned 80/20 inside source train only after this gate",
        },
        "isolation": {"observed_response_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-feature adapter, baseline, and output-isolated control plan before observed fitting.",
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

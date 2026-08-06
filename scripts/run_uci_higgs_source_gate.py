#!/usr/bin/env python3
"""Audit the fixed Phase 62 UCI HIGGS source contract."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ARCHIVE = Path("data/real/uci_higgs/raw/higgs.zip")
PROVENANCE = Path("data/real/uci_higgs/source_provenance")
OUT = Path("artifacts/evaluations/phase62_uci_higgs_source_tail_classification_archive_gate_20260801")
MANIFEST = Path("data/real/uci_higgs/manifest.json")
SPLIT = Path("data/real/uci_higgs/source_file_split.json")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/280/higgs"
MEMBER = "HIGGS.csv.gz"
TOTAL_ROWS = 11_000_000
SOURCE_ROWS = 10_500_000
EXTERNAL_ROWS = 500_000
FIELDS = 29


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
    if values[0] not in (0.0, 1.0):
        raise ValueError(f"invalid HIGGS class label at source row {line_number}")
    return values


def _digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def audit() -> dict[str, object]:
    headers = PROVENANCE / "archive.headers"
    page = PROVENANCE / "dataset_page.html"
    if not ARCHIVE.exists() or not headers.exists() or not page.exists():
        raise ValueError("official HIGGS archive or provenance is missing")
    with zipfile.ZipFile(ARCHIVE) as archive:
        inventory = tuple(archive.namelist())
        if inventory != (MEMBER,):
            raise ValueError("HIGGS ZIP inventory changed")
        info = archive.getinfo(MEMBER)
        line_count = 0
        label_counts = {0: 0, 1: 0}
        feature_min = float("inf")
        feature_max = float("-inf")
        zero_features = 0
        complete_source: set[bytes] = set()
        complete_external: set[bytes] = set()
        candidate_source: set[bytes] = set()
        candidate_external: set[bytes] = set()
        source_duplicates = external_duplicates = 0
        candidate_source_duplicates = candidate_external_duplicates = 0
        first_row_digest: str | None = None
        source_tail_boundary_digest: str | None = None
        external_first_digest: str | None = None
        last_row_digest: str | None = None
        order_digest = hashlib.sha256()
        with archive.open(MEMBER) as compressed:
            with gzip.GzipFile(fileobj=compressed) as handle:
                for line_count, raw in enumerate(handle, start=1):
                    values = parse_line(raw, line_count)
                    label = int(values[0])
                    label_counts[label] += 1
                    features = values[1:]
                    feature_min = min(feature_min, float(features.min()))
                    feature_max = max(feature_max, float(features.max()))
                    zero_features += int(np.count_nonzero(features == 0.0))
                    complete = _digest(raw.rstrip(b"\r\n"))
                    candidate = _digest(features.astype("<f8", copy=False).tobytes())
                    order_digest.update(complete)
                    if line_count == 1:
                        first_row_digest = complete.hex()
                    if line_count == SOURCE_ROWS:
                        source_tail_boundary_digest = complete.hex()
                    if line_count == SOURCE_ROWS + 1:
                        external_first_digest = complete.hex()
                    last_row_digest = complete.hex()
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
    if line_count != TOTAL_ROWS:
        raise ValueError(f"HIGGS row count changed: expected {TOTAL_ROWS}, got {line_count}")
    if len(complete_source) + source_duplicates != SOURCE_ROWS or len(complete_external) + external_duplicates != EXTERNAL_ROWS:
        raise ValueError("HIGGS fixed source-tail boundary changed")
    return {
        "phase": 62,
        "status": "passed_official_source_gate_only",
        "source": {
            "archive": str(ARCHIVE), "official_url": SOURCE_URL, "dataset_page": DATASET_PAGE,
            "doi": "10.24432/C5V312", "license": "CC BY 4.0", "archive_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": sha256(ARCHIVE), "archive_headers": str(headers),
            "dataset_page_evidence": str(page), "published_checksum": None, "members": list(inventory),
            "member_compressed_bytes": info.compress_size, "member_uncompressed_bytes": info.file_size,
        },
        "schema": {
            "header": "absent", "rows": line_count, "fields": FIELDS,
            "target": "literal first binary class field", "target_support": [0, 1],
            "target_counts": {str(label): label_counts[label] for label in sorted(label_counts)},
            "raw_features": 28, "feature_groups": {"low_level": 21, "high_level": 7},
            "all_finite": True, "feature_minimum": feature_min, "feature_maximum": feature_max,
            "zero_features": zero_features,
        },
        "source_order": {
            "first_row_sha256": first_row_digest, "source_tail_boundary_row_sha256": source_tail_boundary_digest,
            "external_first_row_sha256": external_first_digest, "last_row_sha256": last_row_digest,
            "ordered_complete_row_sha256": order_digest.hexdigest(),
        },
        "ledgers": {
            "source_complete_row_duplicates": source_duplicates, "external_complete_row_duplicates": external_duplicates,
            "cross_file_complete_row_duplicates": len(complete_source & complete_external),
            "source_candidate_duplicates": candidate_source_duplicates,
            "external_candidate_duplicates": candidate_external_duplicates,
            "cross_file_candidate_duplicates": len(candidate_source & candidate_external),
        },
        "frozen_file_split": {
            "source_train_rows": SOURCE_ROWS, "external_rows": EXTERNAL_ROWS,
            "source_train": "first 10500000 source rows", "external": "final 500000 source rows",
            "external_boundary": "UCI-documented final 500000 source rows",
            "selection": "deterministic stratified seed-2062001 80/20 inside source train only after this gate",
        },
        "isolation": {"observed_response_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-feature adapter, baseline, and output-isolated control plan before observed fitting.",
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

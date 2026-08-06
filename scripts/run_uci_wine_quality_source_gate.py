#!/usr/bin/env python3
"""Audit the fixed Phase 73 UCI Wine Quality source contract only."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


ARCHIVE = Path("data/real/uci_wine_quality/raw/wine_quality.zip")
PROVENANCE = Path("data/real/uci_wine_quality/source_provenance")
OUT = Path("artifacts/evaluations/phase73_uci_wine_quality_file_holdout_archive_gate_20260806")
MANIFEST = Path("data/real/uci_wine_quality/manifest.json")
SPLIT = Path("data/real/uci_wine_quality/source_file_split.json")
MEMBERS = ("winequality-red.csv", "winequality-white.csv", "winequality.names")
FEATURES = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
)
HEADER = FEATURES + ("quality",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_member(raw: bytes, member: str, expected_rows: int, labels: set[int]) -> dict[str, object]:
    rows = list(csv.reader(io.StringIO(raw.decode("ascii")), delimiter=";"))
    if not rows or tuple(rows[0]) != HEADER:
        raise ValueError(f"{member}: literal header contract changed")
    data = rows[1:]
    if len(data) != expected_rows:
        raise ValueError(f"{member}: row count {len(data)} != {expected_rows}")

    minimum = {name: None for name in FEATURES}
    maximum = {name: None for name in FEATURES}
    negative = {name: 0 for name in FEATURES}
    candidate, complete = set(), set()
    candidate_duplicates = complete_duplicates = missing = 0
    label_counts = {str(label): 0 for label in sorted(labels)}
    encoded_rows: list[str] = []

    for row in data:
        if len(row) != len(HEADER):
            raise ValueError(f"{member}: non-12-field data row")
        if any(value == "" for value in row):
            missing += sum(value == "" for value in row)
            raise ValueError(f"{member}: missing field")
        for index, name in enumerate(FEATURES):
            value = float(row[index])
            if not math.isfinite(value):
                raise ValueError(f"{member}: nonfinite {name}")
            minimum[name] = value if minimum[name] is None else min(minimum[name], value)
            maximum[name] = value if maximum[name] is None else max(maximum[name], value)
            negative[name] += value < 0
        try:
            quality = int(row[-1])
        except ValueError as error:
            raise ValueError(f"{member}: noninteger quality") from error
        if str(quality) != row[-1] or quality not in labels:
            raise ValueError(f"{member}: quality support contract changed")
        label_counts[str(quality)] += 1
        feature_key = "\x1f".join(row[:-1])
        complete_key = "\x1f".join(row)
        candidate_duplicates += feature_key in candidate
        complete_duplicates += complete_key in complete
        candidate.add(feature_key)
        complete.add(complete_key)
        encoded_rows.append(complete_key)

    if any(count == 0 for count in label_counts.values()):
        raise ValueError(f"{member}: quality support is incomplete")
    return {
        "rows": len(data),
        "raw_field_count": len(HEADER),
        "feature_count": len(FEATURES),
        "quality_labels": label_counts,
        "missing_fields": missing,
        "feature_minimum": minimum,
        "feature_maximum": maximum,
        "negative_measurements": negative,
        "candidate_feature_duplicates": candidate_duplicates,
        "complete_record_duplicates": complete_duplicates,
        "ordered_complete_sha256": hashlib.sha256("\n".join(encoded_rows).encode("ascii")).hexdigest(),
    }


def audit() -> dict[str, object]:
    required_provenance = ("dataset_page.html", "dataset_page.headers.txt", "archive.headers.txt")
    if any(not (PROVENANCE / name).exists() for name in required_provenance):
        raise ValueError("Wine Quality official provenance is incomplete")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if tuple(info.filename for info in archive.infolist()) != MEMBERS:
            raise ValueError("Wine Quality ZIP member inventory changed")
        if not archive.read("winequality.names"):
            raise ValueError("Wine Quality names documentation is empty")
        red = parse_member(archive.read("winequality-red.csv"), "winequality-red.csv", 1599, set(range(3, 9)))
        white = parse_member(archive.read("winequality-white.csv"), "winequality-white.csv", 4898, set(range(3, 10)))
    source = {
        "dataset_page": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "archive_url": "https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip",
        "doi": "10.24432/C56S3T",
        "license": "CC BY 4.0",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "archive_headers": str(PROVENANCE / "archive.headers.txt"),
        "dataset_page_headers": str(PROVENANCE / "dataset_page.headers.txt"),
        "members": list(MEMBERS),
    }
    return {
        "phase": 73,
        "status": "passed_official_source_gate_only",
        "source": source,
        "schema": {
            "delimiter": ";",
            "header": list(HEADER),
            "raw_field_count": len(HEADER),
            "feature_count": len(FEATURES),
            "quality_type": "integer",
            "missing_values": "none allowed",
        },
        "partitions": {"source_train": red, "external": white},
        "frozen_file_split": {
            "source_train": "winequality-red.csv",
            "external": "winequality-white.csv",
            "selection": "No selection split is constructed at the source gate; a later fixed plan must prescribe a train-only split inside winequality-red.csv.",
        },
        "isolation": {"observed_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-measurement adapter, baseline, and output-isolated controls plan before observed fitting.",
    }


def run() -> dict[str, object]:
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    MANIFEST.write_text(json.dumps(result["source"], indent=2) + "\n")
    SPLIT.write_text(json.dumps(result["frozen_file_split"], indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

#!/usr/bin/env python3
"""Audit the fixed Phase 69 UCI Adult source contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ARCHIVE = Path("data/real/uci_adult/raw/adult.zip")
PROVENANCE = Path("data/real/uci_adult/source_provenance")
OUT = Path("artifacts/evaluations/phase69_uci_adult_census_income_train_test_archive_gate_20260803")
MANIFEST = Path("data/real/uci_adult/manifest.json")
SPLIT = Path("data/real/uci_adult/source_file_split.json")
MEMBERS = ("Index", "adult.data", "adult.names", "adult.test", "old.adult.names")
NUMERIC = (0, 2, 4, 10, 11, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rows(raw: bytes, labels: set[str], comments: bool) -> dict[str, object]:
    rows, comment_count, missing = [], 0, 0
    candidate, complete = set(), set()
    candidate_duplicates = complete_duplicates = 0
    numeric_min = {str(index): None for index in NUMERIC}
    numeric_max = {str(index): None for index in NUMERIC}
    label_counts = {label: 0 for label in sorted(labels)}
    for line in raw.decode("ascii").splitlines():
        if not line:
            continue
        if line.startswith("|"):
            if not comments:
                raise ValueError("unexpected source-train comment")
            comment_count += 1
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 15 or fields[-1] not in labels:
            raise ValueError("Adult row field or label contract changed")
        for index in NUMERIC:
            value = int(fields[index])
            key = str(index)
            numeric_min[key] = value if numeric_min[key] is None else min(numeric_min[key], value)
            numeric_max[key] = value if numeric_max[key] is None else max(numeric_max[key], value)
        missing += sum(field == "?" for field in fields[:-1])
        raw_feature = "\x1f".join(fields[:-1]).encode("ascii")
        raw_complete = "\x1f".join(fields).encode("ascii")
        candidate_duplicates += raw_feature in candidate
        complete_duplicates += raw_complete in complete
        candidate.add(raw_feature)
        complete.add(raw_complete)
        label_counts[fields[-1]] += 1
        rows.append(fields)
    if any(count == 0 for count in label_counts.values()):
        raise ValueError("Adult label support changed")
    return {"rows": len(rows), "comment_lines": comment_count, "field_count": 15, "label_counts": label_counts, "missing_tokens": missing, "numeric_minimum": numeric_min, "numeric_maximum": numeric_max, "candidate_feature_duplicates": candidate_duplicates, "complete_record_duplicates": complete_duplicates, "ordered_complete_sha256": hashlib.sha256("\n".join("\x1f".join(row) for row in rows).encode("ascii")).hexdigest()}


def audit() -> dict[str, object]:
    if any(not (PROVENANCE / name).exists() for name in ("dataset_page.html", "dataset_page.headers", "archive.headers")):
        raise ValueError("UCI Adult provenance is incomplete")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if tuple(info.filename for info in archive.infolist()) != MEMBERS:
            raise ValueError("Adult ZIP inventory changed")
        if not archive.read("adult.names") or not archive.read("old.adult.names") or not archive.read("Index"):
            raise ValueError("Adult documentation member is empty")
        train = parse_rows(archive.read("adult.data"), {"<=50K", ">50K"}, False)
        test = parse_rows(archive.read("adult.test"), {"<=50K.", ">50K."}, True)
    if train["rows"] != 32561 or test["rows"] != 16281:
        raise ValueError("Adult train/test row count changed")
    source = {"dataset_page": "https://archive.ics.uci.edu/dataset/2/adult", "archive_url": "https://archive.ics.uci.edu/static/public/2/adult.zip", "doi": "10.24432/C5XW20", "license": "CC BY 4.0", "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": sha256(ARCHIVE), "archive_headers": str(PROVENANCE / "archive.headers"), "members": list(MEMBERS)}
    return {"phase": 69, "status": "passed_official_source_gate_only", "source": source, "schema": {"raw_field_count": 15, "feature_count": 14, "numeric_field_indices": list(NUMERIC), "missing_token": "?"}, "partitions": {"source_train": train, "external": test}, "frozen_file_split": {"source_train": "adult.data", "external": "adult.test", "selection": "duplicate-group-aware seed-2069001 stratified 80/20 inside adult.data only after a later plan"}, "isolation": {"observed_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0}, "next_gate": "Write the fixed raw-field adapter, baseline, and output-isolated controls plan before observed fitting."}


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

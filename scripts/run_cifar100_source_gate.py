#!/usr/bin/env python3
"""Audit the fixed Phase 60 CIFAR-100 creator source contract."""

from __future__ import annotations

import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np


ARCHIVE = Path("data/real/cifar100/raw/cifar-100-python.tar.gz")
PROVENANCE = Path("data/real/cifar100/source_provenance")
OUT = Path("artifacts/evaluations/phase60_cifar100_creator_train_test_archive_gate_20260801")
MANIFEST = Path("data/real/cifar100/manifest.json")
SPLIT = Path("data/real/cifar100/source_file_split.json")
SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
MEMBERS = ("cifar-100-python", "cifar-100-python/file.txt~", "cifar-100-python/train", "cifar-100-python/test", "cifar-100-python/meta")
FILES = {"train": 50_000, "test": 10_000}
LABELS = tuple(range(100))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_records(archive: tarfile.TarFile, name: str, expected_records: int) -> tuple[dict[str, object], tuple[str, ...]]:
    handle = archive.extractfile(f"cifar-100-python/{name}")
    if handle is None:
        raise ValueError(f"missing fixed CIFAR-100 member {name}")
    payload = pickle.load(handle, encoding="bytes")
    if set(payload) != {b"filenames", b"batch_label", b"fine_labels", b"coarse_labels", b"data"}:
        raise ValueError(f"{name} pickle schema changed")
    values = np.asarray(payload[b"data"])
    labels = np.asarray(payload[b"fine_labels"], dtype=int)
    if values.shape != (expected_records, 3072) or values.dtype != np.uint8 or labels.shape != (expected_records,):
        raise ValueError(f"{name} fixed raw fine-label surface changed")
    if set(labels) != set(LABELS) or not np.all((values >= 0) & (values <= 255)):
        raise ValueError(f"{name} fine-label support or pixel range changed")
    fingerprints = tuple(hashlib.sha256(row.tobytes()).hexdigest() for row in values)
    return {
        "records": expected_records,
        "candidate_surface": "data plus fine_labels only; coarse_labels excluded",
        "data_shape": list(values.shape),
        "data_dtype": str(values.dtype),
        "pixel_minimum": int(values.min()),
        "pixel_maximum": int(values.max()),
        "zero_pixels": int(np.count_nonzero(values == 0)),
        "all_zero_records": int(np.count_nonzero(np.all(values == 0, axis=1))),
        "fine_label_counts": {str(label): int(np.count_nonzero(labels == label)) for label in LABELS},
        "complete_record_duplicates": expected_records - len(set(fingerprints)),
    }, fingerprints


def audit() -> dict[str, object]:
    headers = PROVENANCE / "archive.headers"
    if not headers.exists():
        raise ValueError("official CIFAR-100 response headers are missing")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        inventory = tuple(member.name for member in archive.getmembers())
        if inventory != MEMBERS:
            raise ValueError("CIFAR-100 archive inventory changed")
        parsed = {name: parse_records(archive, name, expected_records) for name, expected_records in FILES.items()}
        meta_handle = archive.extractfile("cifar-100-python/meta")
        if meta_handle is None:
            raise ValueError("CIFAR-100 meta member is missing")
        meta = pickle.load(meta_handle, encoding="bytes")
    fine_names = meta.get(b"fine_label_names")
    if not isinstance(fine_names, list) or len(fine_names) != 100 or len(set(fine_names)) != 100 or not all(isinstance(name, bytes) and name for name in fine_names):
        raise ValueError("CIFAR-100 meta fine-label names changed")
    train_fingerprints = set(parsed["train"][1])
    test_fingerprints = set(parsed["test"][1])
    source = {
        "archive": str(ARCHIVE),
        "official_url": SOURCE_URL,
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "published_checksum": None,
        "archive_headers": str(headers),
        "members": list(inventory),
    }
    return {
        "phase": 60,
        "status": "passed_official_source_gate_only",
        "source": source,
        "schema": {"features": 3072, "shape": [3, 32, 32], "pixel_range": [0, 255], "fine_labels": list(LABELS), "fine_label_names": 100, "all_finite": True},
        "ledgers": {"by_file": {name: result for name, (result, _) in parsed.items()}, "cross_file_complete_record_duplicates": len(train_fingerprints & test_fingerprints)},
        "frozen_file_split": {"source_train": "train", "external": "test", "source_train_selection": "deterministic stratified 80/20 only after this gate"},
        "isolation": {"observed_pixel_fitting": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed raw-pixel adapter, train-only baseline, and output-isolated synthetic-control plan before observed fitting.",
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

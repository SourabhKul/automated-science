#!/usr/bin/env python3
"""Audit the fixed Phase 59 SVHN cropped-digit source contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path("data/real/svhn")
RAW = ROOT / "raw"
PROVENANCE = ROOT / "source_provenance"
OUT = Path("artifacts/evaluations/phase59_svhn_cropped_digit_creator_train_test_archive_gate_20260731")
MANIFEST = ROOT / "manifest.json"
SPLIT = ROOT / "source_file_split.json"
SOURCE_URL = "http://ufldl.stanford.edu/housenumbers/"
FILES = {"train_32x32.mat": 73_257, "test_32x32.mat": 26_032}
LABELS = tuple(range(1, 11))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_matrix(path: Path, expected_records: int) -> tuple[dict[str, object], tuple[str, ...]]:
    content = loadmat(path, variable_names=("X", "y"))
    if set(content) != {"__header__", "__version__", "__globals__", "X", "y"}:
        raise ValueError(f"{path.name} contains an unexpected raw-variable surface")
    images = np.asarray(content["X"])
    labels = np.asarray(content["y"])
    if images.shape != (32, 32, 3, expected_records) or images.dtype != np.uint8:
        raise ValueError(f"{path.name} violates the fixed SVHN image contract")
    if labels.shape not in {(expected_records,), (expected_records, 1)} or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{path.name} violates the fixed SVHN label shape")
    labels = labels.reshape(-1)
    if set(labels) != set(LABELS) or not np.all(np.isfinite(images)):
        raise ValueError(f"{path.name} violates label support or finite-pixel contract")
    fingerprints = tuple(hashlib.sha256(images[:, :, :, index].tobytes()).hexdigest() for index in range(expected_records))
    return {
        "records": expected_records,
        "variables": ["X", "y"],
        "x_shape": list(images.shape),
        "x_dtype": str(images.dtype),
        "y_shape": list(content["y"].shape),
        "y_dtype": str(content["y"].dtype),
        "pixel_minimum": int(images.min()),
        "pixel_maximum": int(images.max()),
        "zero_pixels": int(np.count_nonzero(images == 0)),
        "all_zero_images": int(np.count_nonzero(np.all(images == 0, axis=(0, 1, 2)))),
        "label_counts": {str(label): int(np.count_nonzero(labels == label)) for label in LABELS},
        "complete_image_duplicates": expected_records - len(set(fingerprints)),
    }, fingerprints


def audit() -> dict[str, object]:
    documentation = PROVENANCE / "documentation.html"
    headers = {name: PROVENANCE / f"{name.removesuffix('.mat').replace('_32x32', '')}.headers" for name in FILES}
    if not documentation.exists() or any(not path.exists() for path in headers.values()):
        raise ValueError("official SVHN documentation or response headers are missing")
    if b"non-commercial use only" not in documentation.read_bytes().lower():
        raise ValueError("official SVHN non-commercial terms are missing")
    matrices = {name: audit_matrix(RAW / name, expected_records) for name, expected_records in FILES.items()}
    ledgers = {name: result for name, (result, _) in matrices.items()}
    train_fingerprints = set(matrices["train_32x32.mat"][1])
    test_fingerprints = set(matrices["test_32x32.mat"][1])
    source = {
        "official_page": SOURCE_URL,
        "terms": "non-commercial use only",
        "files": [
            {
                "member": name,
                "url": f"{SOURCE_URL}{name}",
                "bytes": (RAW / name).stat().st_size,
                "sha256": sha256(RAW / name),
                "headers": str(headers[name]),
            }
            for name in FILES
        ],
        "documentation": str(documentation),
        "published_checksum": None,
    }
    return {
        "phase": 59,
        "status": "passed_official_source_gate_only",
        "source": source,
        "schema": {"shape": [32, 32, 3], "pixel_range": [0, 255], "raw_labels": list(LABELS), "all_finite": True},
        "ledgers": {"by_file": ledgers, "cross_file_complete_image_duplicates": len(train_fingerprints & test_fingerprints)},
        "frozen_file_split": {"source_train": "train_32x32.mat", "external": "test_32x32.mat", "source_train_selection": "deterministic stratified 80/20 only after this gate"},
        "isolation": {"observed_image_fitting": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
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

#!/usr/bin/env python3
"""Audit the fixed creator-published KMNIST IDX source contract."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = Path("data/real/kmnist/raw")
OUT = Path("artifacts/evaluations/phase70_kmnist_creator_raw_grid_classification_archive_gate_20260803")
MANIFEST = Path("data/real/kmnist/manifest.json")
SPLIT = Path("data/real/kmnist/source_file_split.json")
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
URL_BASE = "https://codh.rois.ac.jp/kmnist/dataset/kmnist/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_images(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        magic, count, rows, columns = struct.unpack(">IIII", header)
        payload = handle.read()
    expected = count * rows * columns
    if len(payload) != expected:
        raise ValueError(f"{path.name} payload has {len(payload)} rather than {expected} bytes")
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns), {"magic": magic, "count": count, "rows": rows, "columns": columns}


def read_labels(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with gzip.open(path, "rb") as handle:
        header = handle.read(8)
        magic, count = struct.unpack(">II", header)
        payload = handle.read()
    if len(payload) != count:
        raise ValueError(f"{path.name} payload has {len(payload)} rather than {count} bytes")
    return np.frombuffer(payload, dtype=np.uint8), {"magic": magic, "count": count}


def ledger(images: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    fingerprints = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    complete = [hashlib.sha256(image.tobytes() + bytes([label])).hexdigest() for image, label in zip(images, labels)]
    return {
        "pixel_min": int(images.min()),
        "pixel_max": int(images.max()),
        "all_zero_images": int(np.count_nonzero(np.all(images == 0, axis=(1, 2)))),
        "candidate_duplicate_rows": int(len(fingerprints) - len(set(fingerprints))),
        "complete_duplicate_rows": int(len(complete) - len(set(complete))),
        "label_counts": {str(label): int(np.count_nonzero(labels == label)) for label in range(10)},
    }


def run(raw_dir: Path = RAW, output_dir: Path = OUT, manifest_path: Path = MANIFEST, split_path: Path = SPLIT) -> dict[str, object]:
    paths = {key: raw_dir / value for key, value in FILES.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fixed KMNIST members: {missing}")
    train_images, train_image_header = read_images(paths["train_images"])
    train_labels, train_label_header = read_labels(paths["train_labels"])
    test_images, test_image_header = read_images(paths["test_images"])
    test_labels, test_label_header = read_labels(paths["test_labels"])
    headers = {"train_images": train_image_header, "train_labels": train_label_header, "test_images": test_image_header, "test_labels": test_label_header}
    contracts = {
        "literal_member_set": set(path.name for path in paths.values()) == set(FILES.values()),
        "image_magic_numbers": train_image_header["magic"] == 2051 and test_image_header["magic"] == 2051,
        "label_magic_numbers": train_label_header["magic"] == 2049 and test_label_header["magic"] == 2049,
        "counts": train_images.shape == (60000, 28, 28) and train_labels.shape == (60000,) and test_images.shape == (10000, 28, 28) and test_labels.shape == (10000,),
        "source_unit_pixels": bool(train_images.min() >= 0 and train_images.max() <= 255 and test_images.min() >= 0 and test_images.max() <= 255),
        "label_range": bool(train_labels.min() >= 0 and train_labels.max() <= 9 and test_labels.min() >= 0 and test_labels.max() <= 9),
        "full_label_support": set(train_labels.tolist()) == set(range(10)) and set(test_labels.tolist()) == set(range(10)),
    }
    ledgers = {"source_train": ledger(train_images, train_labels), "external_test": ledger(test_images, test_labels)}
    train_candidates = {hashlib.sha256(image.tobytes()).hexdigest() for image in train_images}
    test_candidates = {hashlib.sha256(image.tobytes()).hexdigest() for image in test_images}
    ledgers["cross_file_candidate_overlap"] = len(train_candidates & test_candidates)
    passed = all(contracts.values())
    manifest = {
        "phase": 70,
        "source": "CODH Kuzushiji-MNIST",
        "official_page": "https://codh.rois.ac.jp/kmnist/index.html.en",
        "doi": "10.20676/00000341",
        "license": "CC-BY-SA-4.0",
        "files": {key: {"name": path.name, "url": URL_BASE + path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for key, path in paths.items()},
        "headers": headers,
        "contracts": contracts,
        "ledgers": ledgers,
    }
    result = {"phase": 70, "status": "passed_source_gate_only" if passed else "blocked_source_contract_failure", "contracts": contracts, "headers": headers, "ledgers": ledgers, "uses_observed_values": False, "uses_observed_labels": False, "next_gate": "Write fixed raw-pixel adapter, baseline, and output-isolated control plan before observed fitting." if passed else "Return to source-backed selection.", "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    split_path.write_text(json.dumps({"source_train": FILES["train_images"], "external": FILES["test_images"], "future_selection": "duplicate-group-aware seed-2070001 stratified 80/20 inside source train only"}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

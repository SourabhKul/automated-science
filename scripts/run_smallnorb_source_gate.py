#!/usr/bin/env python3
"""Audit the fixed Phase 64 creator-published smallNORB source contract."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


RAW = Path("data/real/smallnorb/raw")
PROVENANCE = Path("data/real/smallnorb/source_provenance")
OUT = Path("artifacts/evaluations/phase64_smallnorb_creator_object_instance_holdout_archive_gate_20260802")
MANIFEST = Path("data/real/smallnorb/manifest.json")
SPLIT = Path("data/real/smallnorb/source_file_split.json")
BASE_URL = "https://cs.nyu.edu/~yann/data/norb-v1.0-small/"
ROWS = 24_300
IMAGE_SHAPE = (2, 96, 96)
MAGIC_UINT8 = 0x1E3D4C55
MAGIC_INT32 = 0x1E3D4C54
TRAIN_INSTANCES = {4, 6, 7, 8, 9}
TEST_INSTANCES = {0, 1, 2, 3, 5}


def names(partition: str) -> dict[str, str]:
    prefix = "smallnorb-5x46789x9x18x6x2x96x96-training" if partition == "train" else "smallnorb-5x01235x9x18x6x2x96x96-testing"
    return {kind: f"{prefix}-{kind}.mat.gz" for kind in ("dat", "cat", "info")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(handle: gzip.GzipFile) -> tuple[int, tuple[int, ...]]:
    raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError("smallNORB matrix header is truncated")
    magic, ndim = struct.unpack("<II", raw)
    if not 1 <= ndim <= 4:
        raise ValueError("smallNORB matrix dimensionality is invalid")
    dimensions = struct.unpack(f"<{max(ndim, 3)}I", handle.read(4 * max(ndim, 3)))
    return magic, tuple(dimensions[:ndim])


def read_int_matrix(path: Path, expected_dims: tuple[int, ...]) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, dimensions = read_header(handle)
        if magic != MAGIC_INT32 or dimensions != expected_dims:
            raise ValueError(f"smallNORB integer matrix contract changed: {path.name}")
        payload = handle.read()
    values = np.frombuffer(payload, dtype="<i4")
    if values.size != int(np.prod(expected_dims)):
        raise ValueError(f"smallNORB integer payload length changed: {path.name}")
    return values.reshape(expected_dims)


def audit_partition(partition: str) -> tuple[dict[str, object], set[bytes], set[bytes]]:
    member = names(partition)
    paths = {kind: RAW / name for kind, name in member.items()}
    if any(not path.exists() for path in paths.values()):
        raise ValueError(f"smallNORB {partition} member is missing")
    labels = read_int_matrix(paths["cat"], (ROWS,))
    info = read_int_matrix(paths["info"], (ROWS, 4))
    if not np.all((0 <= labels) & (labels <= 4)) or set(labels.tolist()) != set(range(5)):
        raise ValueError(f"smallNORB {partition} class support changed")
    expected_instances = TRAIN_INSTANCES if partition == "train" else TEST_INSTANCES
    instances = set(info[:, 0].tolist())
    if instances != expected_instances:
        raise ValueError(f"smallNORB {partition} object-instance boundary changed")
    if not np.all((0 <= info[:, 1]) & (info[:, 1] <= 8)) or not np.all((0 <= info[:, 2]) & (info[:, 2] <= 34)) or not np.all((0 <= info[:, 3]) & (info[:, 3] <= 5)):
        raise ValueError(f"smallNORB {partition} info metadata is out of documented bounds")
    pairs: set[bytes] = set()
    complete: set[bytes] = set()
    pair_duplicates = complete_duplicates = 0
    pixels_min = 255
    pixels_max = 0
    zero_pixels = 0
    order = hashlib.sha256()
    with gzip.open(paths["dat"], "rb") as handle:
        magic, dimensions = read_header(handle)
        if magic != MAGIC_UINT8 or dimensions != (ROWS, *IMAGE_SHAPE):
            raise ValueError(f"smallNORB stereo matrix contract changed: {paths['dat'].name}")
        width = int(np.prod(IMAGE_SHAPE))
        for index in range(ROWS):
            raw = handle.read(width)
            if len(raw) != width:
                raise ValueError(f"smallNORB stereo payload is truncated at record {index}")
            values = np.frombuffer(raw, dtype=np.uint8)
            pixels_min = min(pixels_min, int(values.min()))
            pixels_max = max(pixels_max, int(values.max()))
            zero_pixels += int(np.count_nonzero(values == 0))
            candidate = hashlib.sha256(raw).digest()
            complete_digest = hashlib.sha256(raw + labels[index].astype("<i4").tobytes()).digest()
            pair_duplicates += candidate in pairs
            complete_duplicates += complete_digest in complete
            pairs.add(candidate)
            complete.add(complete_digest)
            order.update(complete_digest)
        if handle.read(1):
            raise ValueError(f"smallNORB stereo payload has trailing bytes: {paths['dat'].name}")
    return ({
        "members": member,
        "rows": ROWS,
        "labels": {str(label): int(np.count_nonzero(labels == label)) for label in range(5)},
        "instances": sorted(instances),
        "info_ranges": {"elevation_index": [int(info[:, 1].min()), int(info[:, 1].max())], "azimuth_index": [int(info[:, 2].min()), int(info[:, 2].max())], "lighting": [int(info[:, 3].min()), int(info[:, 3].max())]},
        "pixel_minimum": pixels_min,
        "pixel_maximum": pixels_max,
        "zero_pixels": zero_pixels,
        "complete_record_duplicates": complete_duplicates,
        "candidate_image_duplicates": pair_duplicates,
        "ordered_complete_record_sha256": order.hexdigest(),
        "file_sha256": {kind: sha256(path) for kind, path in paths.items()},
        "file_bytes": {kind: path.stat().st_size for kind, path in paths.items()},
    }, pairs, complete)


def run() -> dict[str, object]:
    page = PROVENANCE / "dataset_page.html"
    page_headers = PROVENANCE / "dataset_page.headers"
    if not page.exists() or not page_headers.exists():
        raise ValueError("smallNORB official page provenance is missing")
    train, train_pairs, train_complete = audit_partition("train")
    test, test_pairs, test_complete = audit_partition("test")
    result = {
        "phase": 64,
        "status": "passed_official_source_gate_only",
        "source": {
            "dataset_page": "https://cs.nyu.edu/~yann/data/norb-v1.0-small/",
            "base_url": BASE_URL,
            "terms": "research purposes; cannot be sold",
            "page_headers": str(page_headers),
            "page_evidence": str(page),
            "members": [names(partition)[kind] for partition in ("train", "test") for kind in ("dat", "cat", "info")],
        },
        "schema": {"rows": {"train": ROWS, "test": ROWS}, "stereo_shape": list(IMAGE_SHAPE), "pixel_dtype": "uint8", "pixel_bounds": [0, 255], "label_bounds": [0, 4], "info_shape": [4], "all_finite": True},
        "partitions": {"train": train, "test": test},
        "ledgers": {"cross_file_candidate_image_duplicates": len(train_pairs & test_pairs), "cross_file_complete_record_duplicates": len(train_complete & test_complete)},
        "frozen_file_split": {"train": "creator training files; instances 4,6,7,8,9", "external": "creator testing files; instances 0,1,2,3,5", "selection": "duplicate-group-aware seed-2064001 stratified 80/20 inside source train only after this gate"},
        "isolation": {"observed_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-stereo adapter, baseline, and output-isolated control plan before observed fitting.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    MANIFEST.write_text(json.dumps(result["source"], indent=2) + "\n")
    SPLIT.write_text(json.dumps(result["frozen_file_split"], indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

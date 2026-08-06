#!/usr/bin/env python3
"""Audit the fixed Phase 65 NIST EMNIST Balanced source contract."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct
import zipfile


ARCHIVE = Path("data/real/emnist/raw/gzip.zip")
PROVENANCE = Path("data/real/emnist/source_provenance")
OUT = Path("artifacts/evaluations/phase65_nist_emnist_balanced_raw_grid_archive_gate_20260802")
MANIFEST = Path("data/real/emnist/manifest.json")
SPLIT = Path("data/real/emnist/source_file_split.json")
SOURCE_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
ROWS = {"train": 112_800, "test": 18_800}
LABELS = tuple(range(47))
IMAGE_BYTES = 28 * 28
IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
MEMBERS = (
    "gzip/emnist-balanced-mapping.txt",
    "gzip/emnist-balanced-test-images-idx3-ubyte.gz",
    "gzip/emnist-balanced-test-labels-idx1-ubyte.gz",
    "gzip/emnist-balanced-train-images-idx3-ubyte.gz",
    "gzip/emnist-balanced-train-labels-idx1-ubyte.gz",
    "gzip/emnist-byclass-mapping.txt",
    "gzip/emnist-byclass-test-images-idx3-ubyte.gz",
    "gzip/emnist-byclass-test-labels-idx1-ubyte.gz",
    "gzip/emnist-byclass-train-images-idx3-ubyte.gz",
    "gzip/emnist-byclass-train-labels-idx1-ubyte.gz",
    "gzip/emnist-bymerge-mapping.txt",
    "gzip/emnist-bymerge-test-images-idx3-ubyte.gz",
    "gzip/emnist-bymerge-test-labels-idx1-ubyte.gz",
    "gzip/emnist-bymerge-train-images-idx3-ubyte.gz",
    "gzip/emnist-bymerge-train-labels-idx1-ubyte.gz",
    "gzip/emnist-digits-mapping.txt",
    "gzip/emnist-digits-test-images-idx3-ubyte.gz",
    "gzip/emnist-digits-test-labels-idx1-ubyte.gz",
    "gzip/emnist-digits-train-images-idx3-ubyte.gz",
    "gzip/emnist-digits-train-labels-idx1-ubyte.gz",
    "gzip/emnist-letters-mapping.txt",
    "gzip/emnist-letters-test-images-idx3-ubyte.gz",
    "gzip/emnist-letters-test-labels-idx1-ubyte.gz",
    "gzip/emnist-letters-train-images-idx3-ubyte.gz",
    "gzip/emnist-letters-train-labels-idx1-ubyte.gz",
    "gzip/emnist-mnist-mapping.txt",
    "gzip/emnist-mnist-test-images-idx3-ubyte.gz",
    "gzip/emnist-mnist-test-labels-idx1-ubyte.gz",
    "gzip/emnist-mnist-train-images-idx3-ubyte.gz",
    "gzip/emnist-mnist-train-labels-idx1-ubyte.gz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_header(handle: gzip.GzipFile) -> tuple[int, int, int]:
    raw = handle.read(16)
    if len(raw) != 16:
        raise ValueError("EMNIST image IDX header is truncated")
    magic, count, rows, columns = struct.unpack(">IIII", raw)
    if magic != IMAGE_MAGIC:
        raise ValueError("EMNIST image IDX magic changed")
    return count, rows, columns


def read_label_header(handle: gzip.GzipFile) -> int:
    raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError("EMNIST label IDX header is truncated")
    magic, count = struct.unpack(">II", raw)
    if magic != LABEL_MAGIC:
        raise ValueError("EMNIST label IDX magic changed")
    return count


def member_names(partition: str) -> tuple[str, str]:
    return (
        f"gzip/emnist-balanced-{partition}-images-idx3-ubyte.gz",
        f"gzip/emnist-balanced-{partition}-labels-idx1-ubyte.gz",
    )


def audit_mapping(archive: zipfile.ZipFile) -> dict[str, object]:
    raw = archive.read("gzip/emnist-balanced-mapping.txt")
    rows = [line.split() for line in raw.decode("ascii").splitlines() if line]
    if len(rows) != 47 or any(len(row) != 2 for row in rows):
        raise ValueError("EMNIST Balanced mapping grammar changed")
    classes = [int(row[0]) for row in rows]
    codepoints = [int(row[1]) for row in rows]
    if classes != list(LABELS) or len(set(codepoints)) != 47:
        raise ValueError("EMNIST Balanced mapping support changed")
    return {"member": "gzip/emnist-balanced-mapping.txt", "rows": len(rows), "label_support": classes, "unique_codepoints": len(set(codepoints)), "sha256": hashlib.sha256(raw).hexdigest()}


def audit_partition(archive: zipfile.ZipFile, partition: str) -> tuple[dict[str, object], set[bytes], set[bytes]]:
    image_name, label_name = member_names(partition)
    expected_rows = ROWS[partition]
    candidates: set[bytes] = set()
    complete: set[bytes] = set()
    candidate_duplicates = complete_duplicates = zero_pixels = all_zero_images = 0
    pixel_minimum, pixel_maximum = 255, 0
    labels = [0] * len(LABELS)
    order = hashlib.sha256()
    with archive.open(image_name) as image_member, archive.open(label_name) as label_member:
        with gzip.GzipFile(fileobj=image_member) as images, gzip.GzipFile(fileobj=label_member) as label_values:
            image_count, rows, columns = read_image_header(images)
            label_count = read_label_header(label_values)
            if image_count != expected_rows or label_count != expected_rows or (rows, columns) != (28, 28):
                raise ValueError(f"EMNIST Balanced {partition} IDX dimensions changed")
            for index in range(expected_rows):
                pixels = images.read(IMAGE_BYTES)
                label_raw = label_values.read(1)
                if len(pixels) != IMAGE_BYTES or len(label_raw) != 1:
                    raise ValueError(f"EMNIST Balanced {partition} payload is truncated at row {index}")
                label = label_raw[0]
                if label not in LABELS:
                    raise ValueError(f"EMNIST Balanced {partition} label is out of range at row {index}")
                labels[label] += 1
                pixel_minimum = min(pixel_minimum, min(pixels))
                pixel_maximum = max(pixel_maximum, max(pixels))
                zero_pixels += pixels.count(0)
                all_zero_images += int(not any(pixels))
                candidate = hashlib.sha256(pixels).digest()
                full = hashlib.sha256(pixels + label_raw).digest()
                candidate_duplicates += candidate in candidates
                complete_duplicates += full in complete
                candidates.add(candidate)
                complete.add(full)
                order.update(full)
            if images.read(1) or label_values.read(1):
                raise ValueError(f"EMNIST Balanced {partition} IDX payload has trailing bytes")
    if pixel_minimum < 0 or pixel_maximum > 255 or any(count == 0 for count in labels):
        raise ValueError(f"EMNIST Balanced {partition} range or full label support changed")
    return ({
        "rows": expected_rows,
        "image_member": image_name,
        "label_member": label_name,
        "image_shape": [28, 28],
        "pixel_dtype": "uint8",
        "pixel_minimum": pixel_minimum,
        "pixel_maximum": pixel_maximum,
        "zero_pixels": zero_pixels,
        "all_zero_images": all_zero_images,
        "label_counts": {str(label): count for label, count in enumerate(labels)},
        "candidate_image_duplicates": candidate_duplicates,
        "complete_record_duplicates": complete_duplicates,
        "ordered_complete_record_sha256": order.hexdigest(),
    }, candidates, complete)


def audit() -> dict[str, object]:
    required = (PROVENANCE / "nist_page.html", PROVENANCE / "nist_page.headers", PROVENANCE / "Readme.txt", PROVENANCE / "readme.headers", PROVENANCE / "gzip_zip.head.headers")
    if any(not path.exists() for path in required):
        raise ValueError("NIST EMNIST provenance evidence is incomplete")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if tuple(archive.namelist()) != MEMBERS:
            raise ValueError("NIST EMNIST gzip.zip inventory changed")
        mapping = audit_mapping(archive)
        train, train_candidates, train_complete = audit_partition(archive, "train")
        test, test_candidates, test_complete = audit_partition(archive, "test")
    source = {
        "dataset_page": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
        "readme_url": "https://biometrics.nist.gov/cs_links/EMNIST/Readme.txt",
        "archive_url": SOURCE_URL,
        "terms": "NIST page and Readme.txt preserve citation/provenance but state no separate license.",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "archive_headers": str(PROVENANCE / "gzip_zip.head.headers"),
        "members": list(MEMBERS),
    }
    return {
        "phase": 65,
        "status": "passed_official_source_gate_only",
        "source": source,
        "mapping": mapping,
        "schema": {"rows": ROWS, "image_shape": [28, 28], "pixel_dtype": "uint8", "pixel_bounds": [0, 255], "label_bounds": [0, 46], "class_count": 47, "all_finite": True, "orientation_note": "The fixed page and Readme.txt describe 28x28 MNIST compatibility but do not specify an orientation transform."},
        "partitions": {"train": train, "test": test},
        "ledgers": {"cross_file_candidate_image_duplicates": len(train_candidates & test_candidates), "cross_file_complete_record_duplicates": len(train_complete & test_complete)},
        "frozen_file_split": {"source_train": "emnist-balanced-train IDX members", "external": "emnist-balanced-test IDX members", "selection": "duplicate-group-aware seed-2065001 stratified 80/20 inside source train only after this gate"},
        "isolation": {"observed_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-grid adapter, baseline, and output-isolated control plan before observed fitting.",
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

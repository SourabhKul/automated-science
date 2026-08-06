#!/usr/bin/env python3
"""Audit the fixed Phase 67 creator-published STL-10 source contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


ARCHIVE = Path("data/real/stl10/raw/stl10_binary.tar.gz")
PROVENANCE = Path("data/real/stl10/source_provenance")
OUT = Path("artifacts/evaluations/phase67_stl10_creator_raw_pixel_train_test_archive_gate_20260802")
MANIFEST = Path("data/real/stl10/manifest.json")
SPLIT = Path("data/real/stl10/source_file_split.json")
PAGE_URL = "https://cs.stanford.edu/~acoates/stl10/"
ARCHIVE_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
ROWS = {"train": 5_000, "test": 8_000}
PIXEL_BYTES = 3 * 96 * 96
LABELS = tuple(range(1, 11))
MEMBERS = (
    # tarfile canonicalizes the archive directory name by removing its trailing slash.
    "stl10_binary",
    "stl10_binary/test_X.bin",
    "stl10_binary/test_y.bin",
    "stl10_binary/train_X.bin",
    "stl10_binary/train_y.bin",
    "stl10_binary/unlabeled_X.bin",
    "stl10_binary/class_names.txt",
    "stl10_binary/fold_indices.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_length(headers: Path) -> int:
    for line in headers.read_text(encoding="ascii").splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("STL-10 archive response has no Content-Length")


def parse_folds(raw: bytes) -> dict[str, object]:
    lines = [line.split() for line in raw.decode("ascii").splitlines() if line]
    if len(lines) != 10 or any(len(line) != 1_000 for line in lines):
        raise ValueError("STL-10 fold-index line count or width changed")
    folds = [[int(value) for value in line] for line in lines]
    if any(any(index < 0 or index >= ROWS["train"] for index in fold) for fold in folds):
        raise ValueError("STL-10 fold index is out of train bounds")
    if any(len(set(fold)) != len(fold) for fold in folds):
        raise ValueError("STL-10 fold has repeated train indices")
    return {
        "member": "stl10_binary/fold_indices.txt",
        "line_count": len(folds),
        "indices_per_line": len(folds[0]),
        "line_sha256": [hashlib.sha256(" ".join(map(str, fold)).encode("ascii")).hexdigest() for fold in folds],
    }


def audit_partition(archive: tarfile.TarFile, partition: str) -> tuple[dict[str, object], set[bytes], set[bytes]]:
    rows = ROWS[partition]
    image_member = f"stl10_binary/{partition}_X.bin"
    label_member = f"stl10_binary/{partition}_y.bin"
    image_info = archive.getmember(image_member)
    label_info = archive.getmember(label_member)
    if image_info.size != rows * PIXEL_BYTES or label_info.size != rows:
        raise ValueError(f"STL-10 {partition} raw byte contract changed")
    candidates: set[bytes] = set()
    complete: set[bytes] = set()
    candidate_duplicates = complete_duplicates = zero_pixels = all_zero_images = 0
    minimum, maximum = 255, 0
    label_counts = {label: 0 for label in LABELS}
    order = hashlib.sha256()
    with archive.extractfile(image_info) as images, archive.extractfile(label_info) as labels:
        if images is None or labels is None:
            raise ValueError(f"STL-10 {partition} required member cannot be read")
        for row in range(rows):
            pixels = images.read(PIXEL_BYTES)
            label_raw = labels.read(1)
            if len(pixels) != PIXEL_BYTES or len(label_raw) != 1:
                raise ValueError(f"STL-10 {partition} payload is truncated at row {row}")
            label = label_raw[0]
            if label not in label_counts:
                raise ValueError(f"STL-10 {partition} label is outside 1..10 at row {row}")
            label_counts[label] += 1
            minimum = min(minimum, min(pixels))
            maximum = max(maximum, max(pixels))
            zero_pixels += pixels.count(0)
            all_zero_images += int(not any(pixels))
            candidate = hashlib.sha256(pixels).digest()
            full = hashlib.sha256(pixels + label_raw).digest()
            candidate_duplicates += candidate in candidates
            complete_duplicates += full in complete
            candidates.add(candidate)
            complete.add(full)
            order.update(full)
        if images.read(1) or labels.read(1):
            raise ValueError(f"STL-10 {partition} payload has trailing bytes")
    if minimum < 0 or maximum > 255 or any(count == 0 for count in label_counts.values()):
        raise ValueError(f"STL-10 {partition} raw range or label support changed")
    return ({
        "rows": rows,
        "image_member": image_member,
        "label_member": label_member,
        "image_shape": [3, 96, 96],
        "pixel_dtype": "uint8",
        "pixel_minimum": minimum,
        "pixel_maximum": maximum,
        "zero_pixels": zero_pixels,
        "all_zero_images": all_zero_images,
        "label_counts": {str(label): count for label, count in label_counts.items()},
        "candidate_image_duplicates": candidate_duplicates,
        "complete_record_duplicates": complete_duplicates,
        "ordered_complete_record_sha256": order.hexdigest(),
    }, candidates, complete)


def audit() -> dict[str, object]:
    required_provenance = (PROVENANCE / "dataset_page.html", PROVENANCE / "dataset_page.headers", PROVENANCE / "archive.headers")
    if any(not path.exists() for path in required_provenance):
        raise ValueError("STL-10 creator provenance evidence is incomplete")
    if ARCHIVE.stat().st_size != content_length(PROVENANCE / "archive.headers"):
        raise ValueError("STL-10 local archive size differs from creator response")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        inventory = tuple(member.name for member in archive.getmembers())
        if inventory != MEMBERS:
            raise ValueError("STL-10 TAR inventory changed")
        classes_raw = archive.extractfile("stl10_binary/class_names.txt")
        folds_raw = archive.extractfile("stl10_binary/fold_indices.txt")
        if classes_raw is None or folds_raw is None:
            raise ValueError("STL-10 metadata member cannot be read")
        classes = classes_raw.read().decode("ascii").splitlines()
        if len(classes) != 10 or any(not value for value in classes) or len(set(classes)) != 10:
            raise ValueError("STL-10 class-name contract changed")
        folds = parse_folds(folds_raw.read())
        train, train_candidates, train_complete = audit_partition(archive, "train")
        test, test_candidates, test_complete = audit_partition(archive, "test")
    source = {
        "dataset_page": PAGE_URL,
        "archive_url": ARCHIVE_URL,
        "terms": "Preserved creator-page terms and citation; no separate license statement is asserted.",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "archive_headers": str(PROVENANCE / "archive.headers"),
        "members": list(MEMBERS),
        "excluded_member": "stl10_binary/unlabeled_X.bin (inventory only; no content read)",
    }
    return {
        "phase": 67,
        "status": "passed_official_source_gate_only",
        "source": source,
        "schema": {"rows": ROWS, "image_shape": [3, 96, 96], "pixel_dtype": "uint8", "pixel_bounds": [0, 255], "label_bounds": [1, 10], "class_count": 10, "all_finite": True, "storage_order": "documented column-major, one color channel at a time"},
        "class_names": {"member": "stl10_binary/class_names.txt", "count": len(classes), "sha256": hashlib.sha256("\n".join(classes).encode("ascii")).hexdigest()},
        "fold_indices": folds,
        "partitions": {"train": train, "test": test},
        "ledgers": {"cross_file_candidate_image_duplicates": len(train_candidates & test_candidates), "cross_file_complete_record_duplicates": len(train_complete & test_complete)},
        "frozen_file_split": {"source_train": "stl10_binary/train_X.bin and train_y.bin", "external": "stl10_binary/test_X.bin and test_y.bin", "selection": "duplicate-group-aware seed-2067001 stratified 80/20 inside source train only after this gate"},
        "isolation": {"unlabeled_member_content_read": False, "observed_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-pixel adapter, baseline, and output-isolated control plan before observed fitting.",
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

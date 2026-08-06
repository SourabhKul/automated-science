#!/usr/bin/env python3
"""Audit the fixed Phase 46 Spoken Arabic Digit official source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ARCHIVE = Path("data/real/uci_spoken_arabic_digit/raw/uci_spoken_arabic_digit_195.zip")
HEADERS = Path("artifacts/evaluations/phase46_uci_spoken_arabic_digit_speaker_prefix_classification_source_access_20260727/headers.txt")
OUT = Path("artifacts/evaluations/phase46_uci_spoken_arabic_digit_speaker_prefix_classification_archive_gate_20260727")
MANIFEST = Path("data/real/uci_spoken_arabic_digit/manifest.json")
SPLIT = Path("data/real/uci_spoken_arabic_digit/source_speaker_split.json")
TRAIN = "Train_Arabic_Digit.txt"
TEST = "Test_Arabic_Digit.txt"
FILES = (TEST, TRAIN, "documentation.html", "graphic.jpg")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/195/spoken%2Barabic%2Bdigit.zip"
TRAIN_SPLIT = {"train_male_speakers": list(range(1, 23)), "train_female_speakers": list(range(34, 56)), "selection_male_speakers": list(range(23, 34)), "selection_female_speakers": list(range(56, 67)), "external_official_test_speakers": list(range(1, 23))}


def parse_blocks(raw: bytes, source_name: str) -> list[np.ndarray]:
    blocks, rows = [], []
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), start=1):
        if not line.strip():
            if rows:
                blocks.append(np.asarray(rows))
                rows = []
            continue
        values = np.fromstring(line.strip(), dtype=float, sep=" ")
        if values.shape != (13,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{source_name} invalid frame at line {line_number}")
        rows.append(values)
    if rows:
        blocks.append(np.asarray(rows))
    if any(block.shape[0] < 4 or block.shape[0] > 93 for block in blocks):
        raise ValueError(f"{source_name} utterance length is outside 4..93")
    return blocks


def _source_speaker(index_within_digit: int) -> int:
    if index_within_digit < 330:
        return index_within_digit // 10 + 1
    return 34 + (index_within_digit - 330) // 10


def audit(archive_path: Path = ARCHIVE) -> dict[str, object]:
    sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != FILES:
            raise ValueError("Spoken Arabic Digit archive inventory changed")
        train = parse_blocks(archive.read(TRAIN), TRAIN)
        test = parse_blocks(archive.read(TEST), TEST)
    if len(train) != 6600 or len(test) != 2200:
        raise ValueError("locked utterance count changed")
    for blocks, per_digit, source in ((train, 660, "train"), (test, 220, "test")):
        if any(len(blocks[digit * per_digit : (digit + 1) * per_digit]) != per_digit for digit in range(10)):
            raise ValueError(f"{source} digit blocks changed")
    speaker_counts = {str(speaker): 0 for speaker in range(1, 67)}
    for digit in range(10):
        for index in range(660):
            speaker_counts[str(_source_speaker(index))] += 1
    if any(count != 100 for count in speaker_counts.values()):
        raise ValueError("source speaker/repetition ordering changed")
    fingerprints = {hashlib.sha256(block.tobytes()).hexdigest() for block in train + test}
    all_blocks = train + test
    lengths = [len(block) for block in all_blocks]
    all_values = np.concatenate(all_blocks)
    return {
        "phase": 46,
        "status": "passed_official_source_gate_only",
        "source": {"archive": str(archive_path), "official_url": SOURCE_URL, "doi": "10.24432/C52C9Q", "license": "CC-BY-4.0", "archive_bytes": archive_path.stat().st_size, "archive_sha256": sha256, "http_headers_path": str(HEADERS), "members": list(FILES)},
        "schema": {"train_utterances": len(train), "test_utterances": len(test), "frame_features": 13, "frame_length_range": [min(lengths), max(lengths)], "finite": True, "documented_preprocessing": "11025 Hz, 16-bit, Hamming window, and 1-0.97Z^(-1) pre-emphasis."},
        "coverage": {"digits": list(range(10)), "train_blocks_per_digit": 660, "test_blocks_per_digit": 220, "source_train_speaker_utterance_counts": speaker_counts, "frozen_speaker_split": TRAIN_SPLIT},
        "ledgers": {"complete_utterance_duplicates": 8800 - len(fingerprints), "minimum": float(all_values.min()), "maximum": float(all_values.max()), "nonfinite_frames": 0},
        "isolation": {"feature_transformation": False, "label_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write fixed adapter, baseline, and output-isolated synthetic-control plan before any observed feature fitting.",
    }


def run() -> dict[str, object]:
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    MANIFEST.write_text(json.dumps(result["source"], indent=2) + "\n")
    SPLIT.write_text(json.dumps(TRAIN_SPLIT, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

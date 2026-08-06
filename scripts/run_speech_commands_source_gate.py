#!/usr/bin/env python3
"""Audit the fixed Phase 68 TensorFlow Speech Commands v0.02 source contract."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import wave

import numpy as np


ARCHIVE = Path("data/real/tensorflow_speech_commands/raw/speech_commands_v0.02.tar.gz")
PROVENANCE = Path("data/real/tensorflow_speech_commands/source_provenance")
OUT = Path("artifacts/evaluations/phase68_tensorflow_speech_commands_v0_02_raw_waveform_archive_gate_20260803")
MANIFEST = Path("data/real/tensorflow_speech_commands/manifest.json")
SPLIT = Path("data/real/tensorflow_speech_commands/source_file_split.json")
PAGE_URL = "https://www.tensorflow.org/datasets/catalog/speech_commands"
ARCHIVE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_PREFIX = "./"
REQUIRED_ROOT_MEMBERS = ("README.md", "validation_list.txt", "testing_list.txt")
BACKGROUND_DIRECTORY = "_background_noise_"
WAV_FILENAME = re.compile(r"^([0-9a-f]{8})_nohash_([0-9]+)\.wav$")


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
    raise ValueError("Speech Commands archive response has no Content-Length")


def parse_list(raw: bytes, list_name: str) -> list[str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Speech Commands {list_name} is not UTF-8") from exc
    if not lines or any(not line for line in lines):
        raise ValueError(f"Speech Commands {list_name} is empty or has blank paths")
    if len(lines) != len(set(lines)):
        raise ValueError(f"Speech Commands {list_name} has duplicate paths")
    return lines


def parse_path(path: str, word_directories: set[str]) -> tuple[str, str, int]:
    parts = PurePosixPath(path).parts
    if len(parts) != 2 or parts[0] not in word_directories or parts[0].startswith("."):
        raise ValueError(f"Speech Commands list path has invalid word-label grammar: {path}")
    if PurePosixPath(path).is_absolute() or ".." in parts or not path.endswith(".wav"):
        raise ValueError(f"Speech Commands list path is unsafe or not a WAV: {path}")
    match = WAV_FILENAME.fullmatch(parts[1])
    if match is None:
        raise ValueError(f"Speech Commands WAV filename grammar changed: {path}")
    return parts[0], match.group(1), int(match.group(2))


def member_path(relative_path: str) -> str:
    return ARCHIVE_PREFIX + relative_path


def audit_listed_wavs(
    archive: tarfile.TarFile,
    paths_by_list: dict[str, list[str]],
    word_directories: set[str],
) -> dict[str, dict[str, object]]:
    """Read all listed source WAVs once, in TAR member order."""
    list_for_path: dict[str, str] = {}
    for list_name, paths in paths_by_list.items():
        for relative_path in paths:
            parse_path(relative_path, word_directories)
            if relative_path in list_for_path:
                raise ValueError("Speech Commands creator path occurs in more than one list")
            list_for_path[relative_path] = list_name
    ledgers = {
        list_name: {
            "class_counts": {name: 0 for name in sorted(word_directories)},
            "speaker_hash_counts": {},
            "utterance_indices": {},
            "candidate_waveforms": set(),
            "complete_records": set(),
            "candidate_duplicates": 0,
            "complete_duplicates": 0,
            "total_samples": 0,
            "zero_samples": 0,
            "all_zero_wavs": 0,
            "min_sample": None,
            "max_sample": None,
            "min_frames": 16_000,
            "max_frames": 0,
            "order": hashlib.sha256(),
        }
        for list_name in paths_by_list
    }
    wanted_members = {member_path(path): path for path in list_for_path}
    ordered_members = sorted(
        (member for member in archive.getmembers() if member.name in wanted_members),
        key=lambda member: member.offset_data,
    )
    if len(ordered_members) != len(wanted_members):
        raise ValueError("Speech Commands creator list references a missing archive WAV")

    for member in ordered_members:
        relative_path = wanted_members[member.name]
        list_name = list_for_path[relative_path]
        ledger = ledgers[list_name]
        label, speaker_hash, utterance_index = parse_path(relative_path, word_directories)
        if member.isdir() or member.size <= 0:
            raise ValueError(f"Speech Commands listed member is not a readable WAV: {relative_path}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"Speech Commands listed member cannot be read: {relative_path}")
        payload = extracted.read()
        try:
            with wave.open(io.BytesIO(payload), "rb") as audio:
                channels = audio.getnchannels()
                width = audio.getsampwidth()
                rate = audio.getframerate()
                frames = audio.getnframes()
                compression = audio.getcomptype()
                samples_raw = audio.readframes(frames)
                if audio.readframes(1):
                    raise ValueError("trailing WAV frames")
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"Speech Commands invalid WAV header at {relative_path}") from exc
        if (channels, width, rate, compression) != (1, 2, 16_000, "NONE"):
            raise ValueError(f"Speech Commands WAV format changed at {relative_path}")
        if not 0 < frames <= 16_000 or len(samples_raw) != frames * 2:
            raise ValueError(f"Speech Commands WAV duration or PCM byte count changed at {relative_path}")

        samples = np.frombuffer(samples_raw, dtype="<i2")
        if samples.size != frames:
            raise ValueError(f"Speech Commands PCM sample count changed at {relative_path}")
        current_min, current_max = int(samples.min()), int(samples.max())
        ledger["min_sample"] = current_min if ledger["min_sample"] is None else min(ledger["min_sample"], current_min)
        ledger["max_sample"] = current_max if ledger["max_sample"] is None else max(ledger["max_sample"], current_max)
        ledger["class_counts"][label] += 1
        ledger["speaker_hash_counts"][speaker_hash] = ledger["speaker_hash_counts"].get(speaker_hash, 0) + 1
        ledger["utterance_indices"].setdefault(speaker_hash, []).append(utterance_index)
        ledger["total_samples"] += frames
        zero_count = int(np.count_nonzero(samples == 0))
        ledger["zero_samples"] += zero_count
        ledger["all_zero_wavs"] += int(zero_count == frames)
        ledger["min_frames"], ledger["max_frames"] = min(ledger["min_frames"], frames), max(ledger["max_frames"], frames)
        candidate = hashlib.sha256(samples_raw).digest()
        complete = hashlib.sha256(label.encode("ascii") + b"\0" + samples_raw).digest()
        ledger["candidate_duplicates"] += candidate in ledger["candidate_waveforms"]
        ledger["complete_duplicates"] += complete in ledger["complete_records"]
        ledger["candidate_waveforms"].add(candidate)
        ledger["complete_records"].add(complete)
        ledger["order"].update(relative_path.encode("utf-8") + b"\0" + complete)

    results: dict[str, dict[str, object]] = {}
    for list_name, ledger in ledgers.items():
        class_counts = ledger["class_counts"]
        if any(count == 0 for count in class_counts.values()):
            raise ValueError("Speech Commands creator lists lack one or more word classes")
        utterance_indices = ledger["utterance_indices"]
        results[list_name] = {
            "paths": len(paths_by_list[list_name]),
            "class_counts": class_counts,
            "speaker_hashes": len(ledger["speaker_hash_counts"]),
            "speaker_hash_utterance_count_minimum": min(len(values) for values in utterance_indices.values()),
            "speaker_hash_utterance_count_maximum": max(len(values) for values in utterance_indices.values()),
            "utterance_index_minimum": min(index for values in utterance_indices.values() for index in values),
            "utterance_index_maximum": max(index for values in utterance_indices.values() for index in values),
            "pcm_samples": ledger["total_samples"],
            "frames_minimum": ledger["min_frames"],
            "frames_maximum": ledger["max_frames"],
            "sample_minimum": ledger["min_sample"],
            "sample_maximum": ledger["max_sample"],
            "zero_samples": ledger["zero_samples"],
            "all_zero_wavs": ledger["all_zero_wavs"],
            "candidate_waveform_duplicates": ledger["candidate_duplicates"],
            "complete_record_duplicates": ledger["complete_duplicates"],
            "ordered_complete_record_sha256": ledger["order"].hexdigest(),
        }
    return results


def audit() -> dict[str, object]:
    required_provenance = (
        PROVENANCE / "dataset_page.html",
        PROVENANCE / "dataset_page.headers",
        PROVENANCE / "archive.headers",
        PROVENANCE / "download.log",
    )
    if any(not path.exists() for path in required_provenance):
        raise ValueError("Speech Commands TensorFlow provenance evidence is incomplete")
    if ARCHIVE.stat().st_size != content_length(PROVENANCE / "archive.headers"):
        raise ValueError("Speech Commands local archive size differs from TensorFlow response")

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        inventory = [member.name for member in archive.getmembers()]
        inventory_set = set(inventory)
        if len(inventory) != len(inventory_set):
            raise ValueError("Speech Commands TAR inventory contains duplicate member paths")
        root_members = {name.removeprefix(ARCHIVE_PREFIX) for name in inventory if name.startswith(ARCHIVE_PREFIX)}
        if any(name not in root_members for name in REQUIRED_ROOT_MEMBERS):
            raise ValueError("Speech Commands required root documentation/list member is missing")
        if member_path(BACKGROUND_DIRECTORY) not in inventory_set and member_path(BACKGROUND_DIRECTORY) + "/" not in inventory_set:
            raise ValueError("Speech Commands background-noise directory is missing")

        word_directories = {
            PurePosixPath(name).parts[0]
            for name in inventory
            if name.startswith(ARCHIVE_PREFIX)
            and len(PurePosixPath(name).parts) == 2
            and name.endswith(".wav")
            and PurePosixPath(name).parts[0] != BACKGROUND_DIRECTORY
        }
        if len(word_directories) != 35 or any(not name or name.startswith(".") for name in word_directories):
            raise ValueError("Speech Commands literal word-directory contract changed")

        validation_member = archive.extractfile(member_path("validation_list.txt"))
        testing_member = archive.extractfile(member_path("testing_list.txt"))
        readme_member = archive.extractfile(member_path("README.md"))
        if validation_member is None or testing_member is None or readme_member is None:
            raise ValueError("Speech Commands required source member cannot be read")
        validation_paths = parse_list(validation_member.read(), "validation_list.txt")
        testing_paths = parse_list(testing_member.read(), "testing_list.txt")
        if set(validation_paths) & set(testing_paths):
            raise ValueError("Speech Commands creator validation and testing paths overlap")
        if not readme_member.read():
            raise ValueError("Speech Commands README.md is empty")
        if any(member_path(path) not in inventory_set for path in validation_paths + testing_paths):
            raise ValueError("Speech Commands creator list references a missing archive WAV")

        list_audits = audit_listed_wavs(
            archive,
            {"validation": validation_paths, "testing": testing_paths},
            word_directories,
        )

    source = {
        "dataset_page": PAGE_URL,
        "archive_url": ARCHIVE_URL,
        "terms": "TensorFlow creator-published Speech Commands v0.02 archive; preserved TensorFlow documentation and response provenance.",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "archive_headers": str(PROVENANCE / "archive.headers"),
        "inventory_member_count": len(inventory),
        "required_members": [member_path(name) for name in REQUIRED_ROOT_MEMBERS] + [member_path(BACKGROUND_DIRECTORY) + "/"],
        "word_directories": sorted(word_directories),
        "background_noise_audio_content_read": False,
    }
    frozen_split = {
        "source_train": "all eligible word-label WAVs outside validation_list.txt and testing_list.txt; unavailable to models until a later fixed plan",
        "selection": "literal validation_list.txt paths only",
        "external": "literal testing_list.txt paths only",
        "path_identity": "archive member path is './' plus the creator list relative path",
        "background_noise": "inventory only; excluded from supervised candidate surface",
    }
    return {
        "phase": 68,
        "status": "passed_official_source_gate_only",
        "source": source,
        "schema": {
            "word_directory_count": len(word_directories),
            "audio_format": {"channels": 1, "sample_rate_hz": 16_000, "sample_width_bytes": 2, "encoding": "signed 16-bit little-endian PCM", "duration_samples": [1, 16_000]},
            "all_finite": True,
        },
        "creator_lists": {**list_audits, "overlap_paths": 0},
        "frozen_file_split": frozen_split,
        "isolation": {"background_noise_audio_content_read": False, "observed_fitting": False, "external_scoring": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write the fixed raw-waveform adapter, baseline, and output-isolated control plan before observed fitting.",
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

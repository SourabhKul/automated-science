"""Strict caller-supplied provenance manifest checks for Silverbox v2.

This module reads repository provenance without modifying Git state. Expected
values always come from the caller's manifest; current machine state is only
compared against those frozen expectations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "silverbox_first_fit_v2_2048_20260925"
ABC_ATTEMPTS_PER_POPULATION = 2_048
PINNED_RUNTIME = {"python": "3.12.11", "numpy": "2.4.4", "psutil": "7.2.2"}
CODE_FILES = {
    "runner": "scripts/run_silverbox_development.py",
    "fit": "core/real_data/silverbox_first_fit.py",
    "v2_manifest": "core/real_data/silverbox_v2_manifest.py",
    "controlled": "core/real_data/silverbox_controlled.py",
    "abc_reference": "core/abc_smc_reference.py",
    "proposer": "core/real_data/silverbox_proposer.py",
    "silverbox_adapter": "core/real_data/silverbox.py",
    "final_scorer": "core/real_data/silverbox_final.py",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class DuplicateManifestKeyError(ValueError):
    """Raised when a manifest JSON object repeats a key."""


def read_manifest(path: str | Path) -> tuple[bytes, dict[str, Any], str]:
    raw = Path(path).read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError("v2 preflight manifest must be one JSON object")
    return raw, value, hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_environment(repo_root: str | Path) -> dict[str, Any]:
    """Read the actual commit, working-tree state, code hashes, and runtime."""

    root = Path(repo_root).resolve()
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip()
    if not _COMMIT.fullmatch(commit):
        raise ValueError("Git did not return a valid HEAD commit")
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "psutil": importlib.metadata.version("psutil"),
    }
    hashes = {name: sha256_file(root / relative) for name, relative in CODE_FILES.items()}
    return {
        "reviewed_git_commit": commit,
        "reviewed_tree_clean": status_result.stdout == "",
        "runtime": runtime,
        "code_sha256": hashes,
    }


def verify_manifest(
    manifest: dict[str, Any],
    *,
    actual: dict[str, Any],
    archive_sha256: str,
    archive_bytes: int,
) -> None:
    """Fail closed unless expected manifest pins match all observed values."""

    if set(manifest) != {
        "protocol_id",
        "abc_attempts_per_population",
        "reviewed_git_commit",
        "reviewed_tree_clean",
        "source",
        "runtime",
        "code_sha256",
    }:
        raise ValueError("v2 preflight manifest keys do not match the frozen schema")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("preflight manifest does not name the v2 protocol")
    if manifest.get("abc_attempts_per_population") != ABC_ATTEMPTS_PER_POPULATION:
        raise ValueError("preflight manifest does not freeze the 2,048 attempt cap")
    expected_commit = manifest.get("reviewed_git_commit")
    if not isinstance(expected_commit, str) or not _COMMIT.fullmatch(expected_commit):
        raise ValueError("preflight manifest has an invalid reviewed Git commit")
    if manifest.get("reviewed_tree_clean") is not True:
        raise ValueError("preflight manifest must require a clean reviewed tree")
    if actual.get("reviewed_git_commit") != expected_commit:
        raise ValueError("current Git HEAD differs from the caller-frozen reviewed commit")
    if actual.get("reviewed_tree_clean") is not True:
        raise ValueError("current Git tree is dirty; v2 preflight requires a clean tree")

    source = manifest.get("source")
    if not isinstance(source, dict) or set(source) != {"archive_sha256", "archive_bytes"}:
        raise ValueError("preflight manifest has an invalid source binding")
    if (
        not isinstance(source.get("archive_sha256"), str)
        or not _SHA256.fullmatch(source["archive_sha256"])
        or type(source.get("archive_bytes")) is not int
        or source["archive_bytes"] <= 0
        or source["archive_sha256"] != archive_sha256
        or source["archive_bytes"] != archive_bytes
    ):
        raise ValueError("preflight manifest source archive digest or size does not match")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime != PINNED_RUNTIME or actual.get("runtime") != PINNED_RUNTIME:
        raise ValueError("preflight manifest and current runtime must match the pinned versions")

    hashes = manifest.get("code_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(CODE_FILES):
        raise ValueError("preflight manifest code hash set does not match the frozen file set")
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes.values()):
        raise ValueError("preflight manifest contains an invalid code SHA-256")
    if hashes != actual.get("code_sha256"):
        raise ValueError("current source-code hashes differ from the caller-frozen manifest")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateManifestKeyError(f"duplicate manifest key: {key}")
        result[key] = value
    return result

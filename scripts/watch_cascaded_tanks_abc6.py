"""External, one-use process-tree watchdog for the source-free ABC6 runner.

This program is a launch seam, not an approval system.  An independently
approved immutable manifest is still required before its production entry
point is used.  The watchdog supervises exactly one sequential campaign
process and never opens the deferred prospective-target gate itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

PROTOCOL_ID = "cascaded_tanks_abc6_synthetic_v1_20260928"
RUN_ID = "ct-abc6-20260928-v1"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_STATUS_RECEIPT_BYTES = 64 * 1024
MAX_CAMPAIGN_SUMMARY_BYTES = 4 * 1024 * 1024
MAX_TRAINING_EVIDENCE_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TRAINING_EVIDENCE_FILE_BYTES = 32 * 1024 * 1024
MAX_TRAINING_EVIDENCE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_TARGET_FREE_FORECAST_BYTES = 128 * 1024 * 1024
MAX_TERMINAL_RECEIPT_BYTES = 128 * 1024 * 1024
MAX_TERMINAL_READBACK_ACK_BYTES = 64 * 1024
_TERMINAL_READBACK_ACK_FILENAME = "terminal-readback.ok"
WALL_LIMIT_SECONDS = 900.0
RSS_LIMIT_BYTES = 2 * 1024**3
SAMPLE_INTERVAL_SECONDS = 0.25
TERMINATE_GRACE_SECONDS = 2.0
KILL_REAP_GRACE_SECONDS = 3.0
DIRECT_CHILD_REAP_GRACE_SECONDS = 0.05

# Keep the declared executable alias distinct from the process image observed
# through psutil.  macOS may report the Python.app image for the watchdog even
# when it was launched through the Cellar's bin/python3.14 alias.
PINNED_PYTHON_BIN = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3_1/bin/python3.14"
)
PINNED_INTERPRETER_PATH = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/"
    "Versions/3.14/bin/python3.14"
)
PINNED_INTERPRETER_SHA256 = (
    "cbf84109626aa1013bbe408fbb9590bd0f1c1548f038b2221c6b8b87de26ca43"
)
PINNED_PYTHON_APP = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/"
    "Versions/3.14/Resources/Python.app/Contents/MacOS/Python"
)
PINNED_PYTHON_APP_SHA256 = (
    "7ecc1ecbf9daa9303c4bf502ff62ffdd9010ed5c08729d470ae9380c10ce1211"
)
PINNED_PYTHON_VERSION = "3.14.3"
PINNED_NUMPY_VERSION = "2.4.2"
PINNED_NUMPY_PATH = "/opt/homebrew/lib/python3.14/site-packages/numpy/__init__.py"
PINNED_NUMPY_SHA256 = (
    "2e8da3e4385e79c4885b3f7324a8b957e6f01732b239e99e266a12c62a008b8d"
)
PINNED_PSUTIL_VERSION = "7.2.2"
PINNED_PSUTIL_PATH = (
    "/Users/sourabh/Library/Python/3.14/lib/python/site-packages/psutil/__init__.py"
)
PINNED_PSUTIL_SHA256 = (
    "d138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a"
)

_REPO_ROOT = Path(__file__).absolute().parents[1]
_SCRIPT_PATH = Path(__file__).absolute()
_RUN_DIRECTORY_RELATIVE = (
    f"artifacts/cascaded_tanks_abc6_synthetic/{RUN_ID}"
)
_MANIFEST_RELATIVE = f"{_RUN_DIRECTORY_RELATIVE}/manifest-v2.json"
_APPROVAL_RELATIVE = f"{_RUN_DIRECTORY_RELATIVE}/approval-go.json"
_RECEIPT_ROOT_RELATIVE = f"{_RUN_DIRECTORY_RELATIVE}/receipts"
_CAMPAIGN_CLAIM_PARENT_RELATIVE = (
    "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims"
)
_SCORER_CLAIM_PARENT_RELATIVE = (
    "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims"
)
_TRAINING_EVIDENCE_DIRECTORY = "training-evidence"
_TRAINING_EVIDENCE_MANIFEST = "campaign.training-evidence-manifest.json"
_TARGET_FREE_FORECAST = "campaign.target-free-forecasts.json"
_INTEGRATED_SOURCE_PATHS = (
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
)
_EXPECTED_UNUSED_LEAVES = {
    _CAMPAIGN_CLAIM_PARENT_RELATIVE: (
        f"{RUN_ID}.claim",
        f"{RUN_ID}.watchdog.claim",
    ),
    _SCORER_CLAIM_PARENT_RELATIVE: (f"{RUN_ID}.claim",),
}
_PRODUCTION_CLAIM_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "evaluations"
    / "cascaded_tanks_abc6_campaign_fit"
    / "claims"
    / f"{RUN_ID}.watchdog.claim"
)


@dataclass(frozen=True, slots=True)
class _PinnedFile:
    relative_path: str
    parent_relative: str
    leaf_name: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    raw: bytes


@dataclass(slots=True)
class _PreflightAnchors:
    """No-follow directory and control-file identities held through claim."""

    root_path: Path
    root_fd: int
    root_identity: tuple[int, int]
    directories: dict[str, int]
    directory_identities: dict[str, tuple[int, int]]
    manifest_file: _PinnedFile
    approval_file: _PinnedFile
    watchdog_file: _PinnedFile

    @property
    def receipt_root_fd(self) -> int:
        return self.directories[_RECEIPT_ROOT_RELATIVE]

    @property
    def watchdog_claim_parent_fd(self) -> int:
        return self.directories[_CAMPAIGN_CLAIM_PARENT_RELATIVE]

    @property
    def receipt_root_path(self) -> Path:
        return self.root_path / _RECEIPT_ROOT_RELATIVE

    def verify(self, *, require_unused: bool = False) -> None:
        _verify_preflight_anchors(self, require_unused=require_unused)

    def close(self) -> None:
        for descriptor in set(self.directories.values()):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.directories.clear()
        try:
            os.close(self.root_fd)
        except OSError:
            pass


class WatchdogError(RuntimeError):
    """A fail-closed launcher, monitoring, or receipt error."""


class AlreadyClaimedError(WatchdogError):
    """The watchdog's one-use run ID has already been consumed."""


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise WatchdogError("platform cannot enforce no-follow launch paths")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _relative_components(relative: str, *, label: str) -> tuple[str, ...]:
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or "\\" in relative
    ):
        raise WatchdogError(f"{label} must be a canonical relative path")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise WatchdogError(f"{label} must be a canonical relative path")
    if Path(relative).as_posix() != relative:
        raise WatchdogError(f"{label} must be a canonical relative path")
    return parts


def _open_absolute_directory_nofollow(path: Path, *, label: str) -> int:
    if not path.is_absolute() or str(path) != str(path.absolute()):
        raise WatchdogError(f"{label} must be an absolute canonical path")
    flags = _directory_flags()
    current_fd = os.open(os.sep, flags)
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise WatchdogError(f"{label} contains a noncanonical component")
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise WatchdogError(f"{label} is not a directory")
        return current_fd
    except WatchdogError:
        os.close(current_fd)
        raise
    except OSError as error:
        os.close(current_fd)
        raise WatchdogError(f"{label} is missing or contains a symlink") from error
    except BaseException:
        os.close(current_fd)
        raise


def _open_relative_directory_nofollow(
    root_fd: int,
    relative: str,
    *,
    label: str,
    create: bool = False,
) -> int:
    flags = _directory_flags()
    current_fd = os.dup(root_fd)
    try:
        for component in _relative_components(relative, label=label):
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise WatchdogError(f"{label} is not a directory")
        if create:
            os.fsync(current_fd)
        return current_fd
    except WatchdogError:
        os.close(current_fd)
        raise
    except OSError as error:
        os.close(current_fd)
        raise WatchdogError(f"{label} is missing or contains a symlink") from error
    except BaseException:
        os.close(current_fd)
        raise


def _read_regular_file_at(
    directory_fd: int,
    leaf_name: str,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    if leaf_name in {"", ".", ".."} or Path(leaf_name).name != leaf_name:
        raise WatchdogError(f"{label} name is not a leaf")
    if not hasattr(os, "O_NOFOLLOW"):
        raise WatchdogError("platform cannot enforce no-follow control-file reads")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            leaf_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise WatchdogError(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise WatchdogError(f"{label} exceeds its byte limit")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise WatchdogError(f"{label} identity changed while being read")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise WatchdogError(f"{label} changed size while being read")
        return raw, after
    except WatchdogError:
        raise
    except OSError as error:
        raise WatchdogError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _file_pin(
    root_fd: int,
    directory_fds: dict[str, int],
    relative_path: str,
    *,
    label: str,
) -> _PinnedFile:
    components = _relative_components(relative_path, label=label)
    parent_relative = "/".join(components[:-1])
    parent_fd = directory_fds.get(parent_relative)
    if parent_fd is None:
        parent_fd = _open_relative_directory_nofollow(
            root_fd, parent_relative, label=f"{label} parent"
        )
        directory_fds[parent_relative] = parent_fd
    raw, info = _read_regular_file_at(
        parent_fd,
        components[-1],
        maximum_bytes=MAX_MANIFEST_BYTES,
        label=label,
    )
    return _PinnedFile(
        relative_path=relative_path,
        parent_relative=parent_relative,
        leaf_name=components[-1],
        device=info.st_dev,
        inode=info.st_ino,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        raw=raw,
    )


def _verify_pinned_file(
    pin: _PinnedFile,
    directory_fds: dict[str, int],
    *,
    label: str,
) -> None:
    parent_fd = directory_fds[pin.parent_relative]
    raw, info = _read_regular_file_at(
        parent_fd,
        pin.leaf_name,
        maximum_bytes=MAX_MANIFEST_BYTES,
        label=label,
    )
    if (
        info.st_dev != pin.device
        or info.st_ino != pin.inode
        or info.st_size != pin.size
        or info.st_mtime_ns != pin.mtime_ns
        or info.st_ctime_ns != pin.ctime_ns
        or raw != pin.raw
    ):
        raise WatchdogError(f"{label} path or bytes changed after it was pinned")


def _stat_absent_at(directory_fd: int, leaf_name: str, *, label: str) -> None:
    try:
        os.stat(leaf_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise WatchdogError(f"cannot inspect {label}") from error
    raise WatchdogError(f"{label} already exists; this run cannot be reused")


def _verify_preflight_anchors(
    anchors: _PreflightAnchors,
    *,
    require_unused: bool,
) -> None:
    try:
        root_info = os.fstat(anchors.root_fd)
    except OSError as error:
        raise WatchdogError("pinned checkout root is closed") from error
    if not stat.S_ISDIR(root_info.st_mode) or (
        root_info.st_dev,
        root_info.st_ino,
    ) != anchors.root_identity:
        raise WatchdogError("pinned checkout root identity changed")
    reopened_root = _open_absolute_directory_nofollow(
        anchors.root_path, label="physical checkout root"
    )
    try:
        reopened_info = os.fstat(reopened_root)
        if (reopened_info.st_dev, reopened_info.st_ino) != anchors.root_identity:
            raise WatchdogError("physical checkout root path was replaced")
        for relative, descriptor in anchors.directories.items():
            opened_info = os.fstat(descriptor)
            if not stat.S_ISDIR(opened_info.st_mode) or (
                opened_info.st_dev,
                opened_info.st_ino,
            ) != anchors.directory_identities[relative]:
                raise WatchdogError(f"pinned directory identity changed: {relative}")
            current_fd = _open_relative_directory_nofollow(
                reopened_root, relative, label=f"pinned directory {relative}"
            )
            try:
                current_info = os.fstat(current_fd)
                if (current_info.st_dev, current_info.st_ino) != (
                    anchors.directory_identities[relative]
                ):
                    raise WatchdogError(f"directory path changed: {relative}")
            finally:
                os.close(current_fd)
    except WatchdogError:
        raise
    except OSError as error:
        raise WatchdogError("a pinned checkout ancestor is missing or unsafe") from error
    finally:
        os.close(reopened_root)

    _verify_pinned_file(
        anchors.manifest_file,
        anchors.directories,
        label="immutable campaign manifest",
    )
    _verify_pinned_file(
        anchors.approval_file,
        anchors.directories,
        label="separate approval record",
    )
    _verify_pinned_file(
        anchors.watchdog_file,
        anchors.directories,
        label="reviewed watchdog source",
    )
    if require_unused:
        if os.listdir(anchors.receipt_root_fd):
            raise WatchdogError("receipt root is not empty before watchdog claim")
        for relative, leaves in _EXPECTED_UNUSED_LEAVES.items():
            directory_fd = anchors.directories[relative]
            for leaf_name in leaves:
                _stat_absent_at(
                    directory_fd,
                    leaf_name,
                    label=f"one-use claim {relative}/{leaf_name}",
                )


def _open_preflight_anchors(
    *,
    manifest_path: Path,
    approval_path: Path,
    receipt_directory: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], _PreflightAnchors]:
    """Open and pin the fixed schema-v2 checkout, control files, and outputs."""

    repository_root = _REPO_ROOT
    if (
        not repository_root.is_absolute()
        or str(repository_root) != str(repository_root.absolute())
        or manifest_path != repository_root / _MANIFEST_RELATIVE
        or approval_path != repository_root / _APPROVAL_RELATIVE
    ):
        raise WatchdogError("manifest and approval paths must match the fixed run root")

    root_fd = _open_absolute_directory_nofollow(
        repository_root, label="active physical checkout root"
    )
    root_info = os.fstat(root_fd)
    root_identity = (root_info.st_dev, root_info.st_ino)
    directories: dict[str, int] = {}
    directory_identities: dict[str, tuple[int, int]] = {}
    anchors: _PreflightAnchors | None = None
    try:
        run_dir_fd = _open_relative_directory_nofollow(
            root_fd, _RUN_DIRECTORY_RELATIVE, label="fixed run directory"
        )
        directories[_RUN_DIRECTORY_RELATIVE] = run_dir_fd
        run_info = os.fstat(run_dir_fd)
        directory_identities[_RUN_DIRECTORY_RELATIVE] = (
            run_info.st_dev,
            run_info.st_ino,
        )

        manifest_file = _file_pin(
            root_fd,
            directories,
            _MANIFEST_RELATIVE,
            label="immutable campaign manifest",
        )
        manifest, manifest_sha256 = _validate_manifest_bytes(
            manifest_file.raw, expected_manifest_sha256
        )
        if manifest_sha256 != expected_manifest_sha256:
            raise WatchdogError("manifest SHA-256 changed during preflight")
        if (
            manifest.get("schema_version") != 2
            or manifest.get("repository_root_realpath") != str(repository_root)
            or manifest.get("receipt_root_relative") != _RECEIPT_ROOT_RELATIVE
            or manifest.get("reviewed_git_head") != _current_git_head()
        ):
            raise WatchdogError(
                "manifest schema-v2 root, receipt path, or reviewed HEAD differs"
            )
        if _SCRIPT_PATH != repository_root / "scripts/watch_cascaded_tanks_abc6.py":
            raise WatchdogError("active watchdog script is outside the pinned checkout")
        expected_receipt_directory = repository_root / _RECEIPT_ROOT_RELATIVE
        if receipt_directory != expected_receipt_directory:
            raise WatchdogError(
                "receipt directory must equal manifest checkout root plus fixed relative path"
            )

        approval_file = _file_pin(
            root_fd,
            directories,
            _APPROVAL_RELATIVE,
            label="separate approval record",
        )
        watchdog_file = _file_pin(
            root_fd,
            directories,
            "scripts/watch_cascaded_tanks_abc6.py",
            label="reviewed watchdog source",
        )
        receipt_fd = _open_relative_directory_nofollow(
            root_fd,
            _RECEIPT_ROOT_RELATIVE,
            label="fixed receipt root",
            create=True,
        )
        directories[_RECEIPT_ROOT_RELATIVE] = receipt_fd
        receipt_info = os.fstat(receipt_fd)
        directory_identities[_RECEIPT_ROOT_RELATIVE] = (
            receipt_info.st_dev,
            receipt_info.st_ino,
        )
        for relative in (
            _CAMPAIGN_CLAIM_PARENT_RELATIVE,
            _SCORER_CLAIM_PARENT_RELATIVE,
        ):
            descriptor = _open_relative_directory_nofollow(
                root_fd,
                relative,
                label=f"one-use claim parent {relative}",
                create=True,
            )
            directories[relative] = descriptor
            info = os.fstat(descriptor)
            directory_identities[relative] = (info.st_dev, info.st_ino)

        for relative, descriptor in directories.items():
            if relative not in directory_identities:
                info = os.fstat(descriptor)
                directory_identities[relative] = (info.st_dev, info.st_ino)

        anchors = _PreflightAnchors(
            root_path=repository_root,
            root_fd=root_fd,
            root_identity=root_identity,
            directories=directories,
            directory_identities=directory_identities,
            manifest_file=manifest_file,
            approval_file=approval_file,
            watchdog_file=watchdog_file,
        )
        if approval_file.raw == b"":
            raise WatchdogError("approval record is empty")
        _check_receipt_root_contract(manifest, receipt_directory)
        anchors.verify(require_unused=True)
        return manifest, anchors
    except BaseException:
        if anchors is not None:
            anchors.close()
        else:
            for descriptor in set(directories.values()):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            os.close(root_fd)
        raise


def _check_receipt_root_contract(
    manifest: dict[str, Any], receipt_directory: Path
) -> None:
    expected = _REPO_ROOT / _RECEIPT_ROOT_RELATIVE
    if (
        manifest.get("repository_root_realpath") != str(_REPO_ROOT)
        or manifest.get("receipt_root_relative") != _RECEIPT_ROOT_RELATIVE
        or receipt_directory != expected
    ):
        raise WatchdogError("receipt directory does not match the frozen root contract")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_exact_vector(observed: Sequence[str], expected: Sequence[str]) -> None:
    if list(observed) != list(expected):
        raise WatchdogError("live command differs from the exact declared vector")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _watchdog_intervention_snapshot(
    intervention_type: str,
    stage: str,
) -> dict[str, object]:
    """Capture a same-host monotonic ordering point matching scorer events."""

    monotonic_ns = time.monotonic_ns()
    occurred_at_utc = _utc_now()
    if (
        type(monotonic_ns) is not int
        or not 0 <= monotonic_ns <= (2**63 - 1)
        or len(occurred_at_utc) > 32
    ):
        raise WatchdogError("watchdog intervention clock snapshot is invalid")
    return {
        "type": intervention_type,
        "stage": stage,
        "monotonic_ns": monotonic_ns,
        "occurred_at_utc": occurred_at_utc,
        "clock": "host-local-monotonic-ns",
    }


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - guarded by the OS write contract.
            raise OSError("durable file write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive_durable(path: Path, value: object) -> str:
    """Publish an immutable JSON file, syncing its bytes and directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    payload = _canonical_json(value)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        # A hard-link publication is atomic and cannot overwrite an old claim
        # or terminal receipt.
        os.link(temp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temp.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive_durable_at(
    directory_fd: int, leaf_name: str, value: object
) -> str:
    """Publish a durable JSON leaf relative to an already pinned directory."""

    if leaf_name in {"", ".", ".."} or Path(leaf_name).name != leaf_name:
        raise WatchdogError("durable receipt name must be a single path component")
    payload = _canonical_json(value)
    temporary = f".{leaf_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            leaf_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        os.unlink(temporary, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except FileExistsError:
        raise
    except OSError as error:
        raise WatchdogError("could not durably publish a pinned-directory receipt") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _publish_and_verify_terminal_receipt_at(
    directory_fd: int,
    leaf_name: str,
    value: dict[str, object],
    *,
    encoded_payload: bytes | None = None,
) -> tuple[str, os.stat_result]:
    """Durably publish and read back the terminal receipt through one pinned FD."""

    if leaf_name in {"", ".", ".."} or Path(leaf_name).name != leaf_name:
        raise WatchdogError("terminal receipt name must be a single path component")
    payload = _canonical_json(value) if encoded_payload is None else encoded_payload
    if len(payload) > MAX_TERMINAL_RECEIPT_BYTES:
        raise WatchdogError("terminal receipt exceeds its byte limit")
    temporary = f".{leaf_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    temporary_exists = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        os.link(
            temporary,
            leaf_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        temporary_info = os.stat(
            temporary,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(temporary_info.st_mode):
            raise WatchdogError("published terminal receipt is not a regular file")

        readback, decoded, readback_info = _read_canonical_object_at(
            directory_fd,
            leaf_name,
            maximum_bytes=MAX_TERMINAL_RECEIPT_BYTES,
            label="watchdog terminal receipt readback",
        )
        named_info = os.stat(leaf_name, dir_fd=directory_fd, follow_symlinks=False)
        expected_identity = (
            temporary_info.st_dev,
            temporary_info.st_ino,
            temporary_info.st_size,
            temporary_info.st_mtime_ns,
            temporary_info.st_ctime_ns,
        )
        observed_identity = (
            readback_info.st_dev,
            readback_info.st_ino,
            readback_info.st_size,
            readback_info.st_mtime_ns,
            readback_info.st_ctime_ns,
        )
        named_identity = (
            named_info.st_dev,
            named_info.st_ino,
            named_info.st_size,
            named_info.st_mtime_ns,
            named_info.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(named_info.st_mode)
            or readback != payload
            or decoded != value
            or observed_identity != expected_identity
            or named_identity != expected_identity
            or hashlib.sha256(readback).hexdigest()
            != hashlib.sha256(payload).hexdigest()
        ):
            raise WatchdogError(
                "terminal receipt readback bytes, digest, or file identity differ"
            )

        os.unlink(temporary, dir_fd=directory_fd)
        temporary_exists = False
        os.fsync(directory_fd)

        # Removing the temporary hard link changes the terminal inode ctime.
        # Take the identity that the finalization sidecar will bind only after
        # that namespace change has been synced and read back once more.
        final_readback, final_decoded, final_info = _read_canonical_object_at(
            directory_fd,
            leaf_name,
            maximum_bytes=MAX_TERMINAL_RECEIPT_BYTES,
            label="watchdog terminal receipt final readback",
        )
        named_final_info = os.stat(
            leaf_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            final_readback != payload
            or final_decoded != value
            or hashlib.sha256(final_readback).hexdigest()
            != hashlib.sha256(payload).hexdigest()
            or (
                final_info.st_dev,
                final_info.st_ino,
                final_info.st_size,
                final_info.st_mtime_ns,
            )
            != (
                temporary_info.st_dev,
                temporary_info.st_ino,
                temporary_info.st_size,
                temporary_info.st_mtime_ns,
            )
            or (
                named_final_info.st_dev,
                named_final_info.st_ino,
                named_final_info.st_size,
                named_final_info.st_mtime_ns,
                named_final_info.st_ctime_ns,
            )
            != (
                final_info.st_dev,
                final_info.st_ino,
                final_info.st_size,
                final_info.st_mtime_ns,
                final_info.st_ctime_ns,
            )
        ):
            raise WatchdogError(
                "terminal receipt final readback bytes, digest, or file identity differ"
            )
        return hashlib.sha256(payload).hexdigest(), final_info
    except FileExistsError as error:
        raise WatchdogError("terminal receipt path already exists") from error
    except WatchdogError:
        raise
    except OSError as error:
        raise WatchdogError(
            "terminal receipt could not be durably published and read back"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                # The consumed claim remains authoritative if cleanup is uncertain.
                pass


def _publish_terminal_readback_ack_at(
    directory_fd: int,
    terminal_leaf_name: str,
    terminal_value: dict[str, object],
    terminal_sha256: str,
    terminal_info: os.stat_result,
) -> str:
    """Durably record that the terminal receipt passed its live readback."""

    if not _is_sha256(terminal_sha256):
        raise WatchdogError("terminal readback acknowledgment has an invalid digest")
    run_id = _validate_claim_run_id(terminal_value.get("run_id"))
    claim_sha256 = terminal_value.get("watchdog_claim_sha256")
    manifest_sha256 = terminal_value.get("manifest_sha256")
    if not _is_sha256(claim_sha256) or not _is_sha256(manifest_sha256):
        raise WatchdogError(
            "terminal readback acknowledgment lacks the bound claim or manifest digest"
        )
    if terminal_leaf_name in {"", ".", ".."} or Path(terminal_leaf_name).name != terminal_leaf_name:
        raise WatchdogError("terminal receipt name must be a single path component")

    receipt_root_info = os.fstat(directory_fd)
    expected_receipt_root = (
        terminal_value.get("receipt_root_device"),
        terminal_value.get("receipt_root_inode"),
    )
    if expected_receipt_root != (receipt_root_info.st_dev, receipt_root_info.st_ino):
        raise WatchdogError("terminal readback acknowledgment receipt-root identity differs")

    terminal_raw, terminal_decoded, bound_terminal_info = _read_canonical_object_at(
        directory_fd,
        terminal_leaf_name,
        maximum_bytes=MAX_TERMINAL_RECEIPT_BYTES,
        label="terminal receipt readback acknowledgment binding",
    )
    if (
        terminal_decoded != terminal_value
        or hashlib.sha256(terminal_raw).hexdigest() != terminal_sha256
        or (
            bound_terminal_info.st_dev,
            bound_terminal_info.st_ino,
            bound_terminal_info.st_size,
            bound_terminal_info.st_mtime_ns,
            bound_terminal_info.st_ctime_ns,
        )
        != (
            terminal_info.st_dev,
            terminal_info.st_ino,
            terminal_info.st_size,
            terminal_info.st_mtime_ns,
            terminal_info.st_ctime_ns,
        )
    ):
        raise WatchdogError(
            "terminal receipt changed before readback acknowledgment publication"
        )

    root_fields = {
        key: terminal_value.get(key)
        for key in (
            "repository_root_realpath",
            "repository_root_device",
            "repository_root_inode",
            "receipt_root_relative",
            "receipt_root_device",
            "receipt_root_inode",
        )
    }
    if (
        type(root_fields["repository_root_realpath"]) is not str
        or not Path(str(root_fields["repository_root_realpath"])).is_absolute()
        or type(root_fields["repository_root_device"]) is not int
        or type(root_fields["repository_root_inode"]) is not int
        or type(root_fields["receipt_root_relative"]) is not str
    ):
        raise WatchdogError("terminal readback acknowledgment lacks its frozen root binding")

    acknowledgement: dict[str, object] = {
        "schema_version": 1,
        "kind": "abc6-watchdog-terminal-readback-acknowledgment",
        "protocol_id": terminal_value.get("protocol_id"),
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "watchdog_claim_path": terminal_value.get("watchdog_claim_path"),
        "watchdog_claim_sha256": claim_sha256,
        "root_binding": root_fields,
        "terminal_receipt": {
            "leaf_name": terminal_leaf_name,
            "sha256": terminal_sha256,
            "file_identity": {
                "device": bound_terminal_info.st_dev,
                "inode": bound_terminal_info.st_ino,
                "size": bound_terminal_info.st_size,
                "mtime_ns": bound_terminal_info.st_mtime_ns,
                "ctime_ns": bound_terminal_info.st_ctime_ns,
            },
        },
    }
    payload = _canonical_json(acknowledgement)
    if len(payload) > MAX_TERMINAL_READBACK_ACK_BYTES:
        raise WatchdogError("terminal readback acknowledgment exceeds its byte limit")

    leaf_name = _TERMINAL_READBACK_ACK_FILENAME
    temporary = f".{leaf_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    temporary_exists = False
    acknowledgement_linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            leaf_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        acknowledgement_linked = True
        os.unlink(temporary, dir_fd=directory_fd)
        temporary_exists = False
        os.fsync(directory_fd)
        final_root_info = os.fstat(directory_fd)
        if (final_root_info.st_dev, final_root_info.st_ino) != expected_receipt_root:
            raise WatchdogError("receipt root changed while publishing readback acknowledgment")
        acknowledgement_linked = False
        return hashlib.sha256(payload).hexdigest()
    except FileExistsError as error:
        raise WatchdogError("terminal readback acknowledgment already exists") from error
    except WatchdogError:
        raise
    except OSError as error:
        raise WatchdogError(
            "terminal readback acknowledgment could not be durably published"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # If publication or its directory sync fails, remove a possibly visible
        # acknowledgment so it cannot certify a failed finalization attempt.
        # The watchdog claim remains consumed even if cleanup itself is uncertain.
        if acknowledgement_linked:
            try:
                os.unlink(leaf_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _expected_project_claim_path(project_root: Path, run_id: str) -> Path:
    _validate_claim_run_id(run_id)
    return (
        project_root
        / "artifacts"
        / "evaluations"
        / "cascaded_tanks_abc6_campaign_fit"
        / "claims"
        / f"{run_id}.watchdog.claim"
    )


def _validate_claim_run_id(run_id: object) -> str:
    """Accept only one canonical lowercase alphanumeric/hyphen basename."""

    if (
        type(run_id) is not str
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
        or run_id.startswith("-")
        or run_id.endswith("-")
        or "--" in run_id
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in run_id)
    ):
        raise WatchdogError("watchdog claim run ID must be a canonical single basename")
    return run_id


def _validate_project_claim_path(path: Path, project_root: Path, run_id: str) -> None:
    run_id = _validate_claim_run_id(run_id)
    if (
        not project_root.is_absolute()
        or project_root.resolve(strict=True) != project_root
        or not path.is_absolute()
        or path != _expected_project_claim_path(project_root, run_id)
    ):
        raise WatchdogError(
            "fixed watchdog claim must be the canonical run-ID path under the project artifacts root"
        )


def _write_project_claim_without_symlinks(
    path: Path, body: dict[str, object], project_root: Path, run_id: str
) -> str:
    """Create the fixed claim through no-follow directory descriptors.

    Walking from `/` with O_NOFOLLOW prevents an ancestor symlink from
    redirecting the O_EXCL claim outside the reviewed project artifact root.
    """

    _validate_project_claim_path(path, project_root, run_id)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise WatchdogError("platform cannot enforce no-follow claim ancestry")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd: int | None = None
    temp_name: str | None = None
    payload = _canonical_json(body)
    try:
        current_fd = os.open(os.sep, directory_flags)
        for component in project_root.parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd

        relative = path.relative_to(project_root)
        parent_components = relative.parts[:-1]
        for component in parent_components:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd

        claim_name = relative.parts[-1]
        temp_name = f".{claim_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=current_fd,
        )
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temp_name,
            claim_name,
            src_dir_fd=current_fd,
            dst_dir_fd=current_fd,
        )
        os.fsync(current_fd)
        os.unlink(temp_name, dir_fd=current_fd)
        temp_name = None
        os.fsync(current_fd)
    except FileExistsError as error:
        raise AlreadyClaimedError(
            "the fixed ABC6 watchdog claim already exists; this run cannot be retried"
        ) from error
    except OSError as error:
        raise WatchdogError(
            "cannot safely create fixed watchdog claim through no-follow project ancestry"
        ) from error
    finally:
        if current_fd is not None:
            if temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileNotFoundError:
                    pass
            os.close(current_fd)
    return hashlib.sha256(payload).hexdigest()


def _claim_once(
    path: Path,
    body: dict[str, object],
    *,
    project_root: Path | None = None,
    claim_directory_fd: int | None = None,
) -> str:
    if claim_directory_fd is not None:
        try:
            return _write_exclusive_durable_at(
                claim_directory_fd, path.name, body
            )
        except FileExistsError as error:
            raise AlreadyClaimedError(
                "the fixed ABC6 watchdog claim already exists; this run cannot be retried"
            ) from error
    if project_root is not None:
        run_id = _validate_claim_run_id(body.get("run_id"))
        return _write_project_claim_without_symlinks(path, body, project_root, run_id)
    try:
        return _write_exclusive_durable(path, body)
    except FileExistsError as error:
        raise AlreadyClaimedError(
            "the fixed ABC6 watchdog claim already exists; this run cannot be retried"
        ) from error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _load_and_check_manifest(
    path: Path,
    expected_sha256: str,
    *,
    raw_bytes: bytes | None = None,
) -> tuple[dict[str, Any], str]:
    if raw_bytes is None:
        parent_fd = _open_absolute_directory_nofollow(
            path.parent, label="manifest parent"
        )
        try:
            raw_bytes, _info = _read_regular_file_at(
                parent_fd,
                path.name,
                maximum_bytes=MAX_MANIFEST_BYTES,
                label="immutable campaign manifest",
            )
        finally:
            os.close(parent_fd)
    return _validate_manifest_bytes(raw_bytes, expected_sha256)


def _validate_manifest_bytes(
    raw: bytes, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    if not _is_sha256(expected_sha256):
        raise WatchdogError("manifest SHA-256 must be 64 lowercase hexadecimal characters")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise WatchdogError("manifest exceeds its byte limit")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise WatchdogError("manifest bytes do not match the supplied SHA-256")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise WatchdogError("manifest must be canonical ASCII JSON") from error
    if type(decoded) is not dict or _canonical_json(decoded) != raw:
        raise WatchdogError("manifest must be a canonical JSON object")
    if decoded.get("protocol_id") != PROTOCOL_ID or decoded.get("run_id") != RUN_ID:
        raise WatchdogError("manifest protocol or one-use run ID is unexpected")
    if (
        type(decoded.get("schema_version")) is not int
        or decoded.get("schema_version") != 2
        or decoded.get("repository_root_realpath") != str(_REPO_ROOT)
        or decoded.get("receipt_root_relative") != _RECEIPT_ROOT_RELATIVE
        or type(decoded.get("reviewed_git_head")) is not str
        or len(decoded["reviewed_git_head"]) != 40
        or any(char not in "0123456789abcdef" for char in decoded["reviewed_git_head"])
    ):
        raise WatchdogError("manifest schema-v2 physical-root identity is invalid")

    contract = decoded.get("execution_contract")
    if type(contract) is not dict:
        raise WatchdogError("manifest execution contract is missing")
    if (
        type(contract.get("wall_clock_limit_seconds")) is not int
        or contract["wall_clock_limit_seconds"] != int(WALL_LIMIT_SECONDS)
        or type(contract.get("runner_tree_rss_limit_bytes")) is not int
        or contract["runner_tree_rss_limit_bytes"] != RSS_LIMIT_BYTES
        or type(contract.get("worker_count")) is not int
        or contract["worker_count"] != 1
        or contract.get("watchdog_enforcement") != "external"
        or contract.get("independent_manifest_approval_required") is not True
        or contract.get("prospective_targets_before_receipts") is not False
        or type(contract.get("projected_artifact_bytes")) is not int
        or not 0 <= contract["projected_artifact_bytes"] < 1024**3
    ):
        raise WatchdogError("manifest execution contract does not match the watchdog")

    runtime = decoded.get("runtime_fingerprint")
    if type(runtime) is not dict:
        raise WatchdogError("manifest runtime fingerprint is missing")
    expected_runtime = {
        "declared_launch_interpreter_path": PINNED_INTERPRETER_PATH,
        "resolved_running_interpreter_path": PINNED_INTERPRETER_PATH,
        "python_version": PINNED_PYTHON_VERSION,
        "interpreter_sha256": PINNED_INTERPRETER_SHA256,
        "numpy_version": PINNED_NUMPY_VERSION,
        "numpy_module_path": PINNED_NUMPY_PATH,
        "numpy_module_sha256": PINNED_NUMPY_SHA256,
        "psutil_version": PINNED_PSUTIL_VERSION,
        "psutil_module_path": PINNED_PSUTIL_PATH,
        "psutil_module_sha256": PINNED_PSUTIL_SHA256,
    }
    if any(runtime.get(key) != expected for key, expected in expected_runtime.items()):
        raise WatchdogError("manifest runtime fingerprint differs from the pinned runtime")
    return decoded, expected_sha256


APPROVAL_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "status",
    "protocol_id",
    "run_id",
    "manifest_sha256",
    "reviewed_git_head",
    "reviewer_id",
    "approved_at_utc",
    "watchdog_script_sha256",
    "manifest_path",
    "receipt_directory",
    "watchdog_launch_vector_template",
    "campaign_child_launch_vector",
}
APPROVAL_SHA_PLACEHOLDER = "<APPROVAL_RECORD_SHA256>"


def _campaign_strict_preflight(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    pinned_manifest_bytes: bytes,
) -> str:
    """Call the campaign runner's exact source/runtime/roster preflight."""

    from core.real_data.cascaded_tanks_abc6_campaign_fit import (
        preflight_abc6_campaign_manifest,
    )

    return preflight_abc6_campaign_manifest(
        manifest_path,
        manifest_sha256,
        require_reviewed_runtime=True,
        pinned_manifest_bytes=pinned_manifest_bytes,
    )


def _current_git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise WatchdogError("cannot read the reviewed Git HEAD") from error
    head = result.stdout.strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise WatchdogError("Git HEAD is not a canonical commit SHA-1")
    return head


def _watchdog_launch_vector_template(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    receipt_directory: Path,
    approval_path: Path,
) -> list[str]:
    return [
        PINNED_PYTHON_BIN,
        str(_SCRIPT_PATH),
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--receipt-directory",
        str(receipt_directory),
        "--approval-record",
        str(approval_path),
        "--approval-record-sha256",
        APPROVAL_SHA_PLACEHOLDER,
    ]


def _load_and_validate_approval_record(
    *,
    approval_path: Path,
    approval_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    receipt_directory: Path,
    campaign_launch_vector: Sequence[str],
    actual_watchdog_args: Sequence[str],
    reviewed_git_head: str | None = None,
    watchdog_script_sha256: str | None = None,
    raw_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not _is_sha256(approval_sha256):
        raise WatchdogError("approval record SHA-256 must be lowercase hexadecimal")
    if raw_bytes is None:
        parent_fd = _open_absolute_directory_nofollow(
            approval_path.parent, label="approval-record parent"
        )
        try:
            raw_bytes, _info = _read_regular_file_at(
                parent_fd,
                approval_path.name,
                maximum_bytes=MAX_MANIFEST_BYTES,
                label="independent approval/launch record",
            )
        finally:
            os.close(parent_fd)
    raw = raw_bytes
    if len(raw) > MAX_MANIFEST_BYTES:
        raise WatchdogError("approval record exceeds its byte limit")
    if hashlib.sha256(raw).hexdigest() != approval_sha256:
        raise WatchdogError("approval record bytes do not match supplied SHA-256")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise WatchdogError("approval record must be canonical ASCII JSON") from error
    if (
        type(decoded) is not dict
        or set(decoded) != APPROVAL_RECORD_KEYS
        or _canonical_json(decoded) != raw
        or type(decoded.get("schema_version")) is not int
        or decoded.get("schema_version") != 1
    ):
        raise WatchdogError("approval record schema is incomplete or contains unknown fields")

    reviewer = decoded.get("reviewer_id")
    approved_at = decoded.get("approved_at_utc")
    reviewed_head = decoded.get("reviewed_git_head")
    script_sha256 = decoded.get("watchdog_script_sha256")
    expected_watchdog_script_sha256 = (
        _sha256_file(_SCRIPT_PATH)
        if watchdog_script_sha256 is None
        else watchdog_script_sha256
    )
    if (
        decoded.get("record_type") != "abc6_independent_approval_and_launch_v1"
        or decoded.get("status") != "approved"
        or decoded.get("protocol_id") != PROTOCOL_ID
        or decoded.get("run_id") != RUN_ID
        or decoded.get("manifest_sha256") != manifest_sha256
        or decoded.get("manifest_path") != str(manifest_path)
        or decoded.get("receipt_directory") != str(receipt_directory)
        or type(reviewer) is not str
        or not reviewer.strip()
        or type(approved_at) is not str
        or not approved_at.strip()
        or type(reviewed_head) is not str
        or len(reviewed_head) != 40
        or any(char not in "0123456789abcdef" for char in reviewed_head)
        or reviewed_head != _current_git_head()
        or (reviewed_git_head is not None and reviewed_head != reviewed_git_head)
        or script_sha256 != expected_watchdog_script_sha256
        or decoded.get("watchdog_launch_vector_template")
        != _watchdog_launch_vector_template(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            receipt_directory=receipt_directory,
            approval_path=approval_path,
        )
        or decoded.get("campaign_child_launch_vector") != list(campaign_launch_vector)
    ):
        raise WatchdogError(
            "approval record does not bind current manifest, reviewed Git HEAD, "
            "watchdog code, and exact launch vectors"
        )

    expected_args = [
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--receipt-directory",
        str(receipt_directory),
        "--approval-record",
        str(approval_path),
        "--approval-record-sha256",
        approval_sha256,
    ]
    _require_exact_vector(actual_watchdog_args, expected_args)
    return decoded


def _production_preflight(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    receipt_directory: Path,
    approval_path: Path,
    approval_sha256: str,
    actual_watchdog_args: Sequence[str],
) -> tuple[dict[str, Any], str, dict[str, Any], _PreflightAnchors]:
    """Validate all campaign and approval gates before any one-use claim."""

    manifest, anchors = _open_preflight_anchors(
        manifest_path=manifest_path,
        approval_path=approval_path,
        receipt_directory=receipt_directory,
        expected_manifest_sha256=manifest_sha256,
    )
    try:
        checked_manifest, loaded_sha256 = _load_and_check_manifest(
            manifest_path,
            manifest_sha256,
            raw_bytes=anchors.manifest_file.raw,
        )
        if checked_manifest != manifest:
            raise WatchdogError("pinned manifest decode changed during validation")
        anchors.verify(require_unused=True)
        verified_sha256 = _campaign_strict_preflight(
            manifest_path,
            manifest_sha256,
            pinned_manifest_bytes=anchors.manifest_file.raw,
        )
        if verified_sha256 != manifest_sha256 or loaded_sha256 != verified_sha256:
            raise WatchdogError("watchdog and campaign manifest hash checks disagree")
        anchors.verify(require_unused=True)
        launch = _build_campaign_command(
            manifest_path, loaded_sha256, receipt_directory
        )
        approval = _load_and_validate_approval_record(
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            manifest_path=manifest_path,
            manifest_sha256=loaded_sha256,
            receipt_directory=receipt_directory,
            campaign_launch_vector=launch,
            actual_watchdog_args=actual_watchdog_args,
            reviewed_git_head=str(manifest["reviewed_git_head"]),
            watchdog_script_sha256=hashlib.sha256(
                anchors.watchdog_file.raw
            ).hexdigest(),
            raw_bytes=anchors.approval_file.raw,
        )
        anchors.verify(require_unused=True)
        return manifest, loaded_sha256, approval, anchors
    except Exception as error:
        anchors.close()
        if isinstance(error, WatchdogError):
            raise
        raise WatchdogError(
            f"campaign strict manifest/source/runtime/roster preflight failed: {error}"
        ) from error


def _current_parent_attestation(actual_argv: Sequence[str]) -> dict[str, object]:
    """Verify the watchdog's exact command tokens and its observed app image."""

    try:
        launch_path = Path(PINNED_PYTHON_BIN)
        resolved_launch = launch_path.resolve(strict=True)
        image_path = Path(PINNED_PYTHON_APP).resolve(strict=True)
        resolved_interpreter = Path(sys.executable).resolve(strict=True)
        psutil_path = Path(psutil.__file__).absolute()
        process = psutil.Process(os.getpid())
        observed_argv = process.cmdline()
        observed_exe = Path(process.exe()).resolve(strict=True)
        python_version = platform.python_version()
    except (OSError, psutil.Error) as error:
        raise WatchdogError("cannot capture watchdog process/runtime identity") from error

    if (
        resolved_launch != Path(PINNED_INTERPRETER_PATH)
        or _sha256_file(launch_path) != PINNED_INTERPRETER_SHA256
        or image_path != Path(PINNED_PYTHON_APP)
        or observed_exe != image_path
        or _sha256_file(image_path) != PINNED_PYTHON_APP_SHA256
        or resolved_interpreter != Path(PINNED_INTERPRETER_PATH)
        or python_version != PINNED_PYTHON_VERSION
        or _sha256_file(resolved_interpreter) != PINNED_INTERPRETER_SHA256
        or psutil.__version__ != PINNED_PSUTIL_VERSION
        or psutil_path != Path(PINNED_PSUTIL_PATH)
        or _sha256_file(psutil_path) != PINNED_PSUTIL_SHA256
    ):
        raise WatchdogError("watchdog Python image or psutil runtime is not pinned")

    expected_tail = [str(_SCRIPT_PATH), *actual_argv]
    if (
        len(observed_argv) != len(expected_tail) + 1
        or observed_argv[0] not in {PINNED_PYTHON_BIN, str(image_path)}
        or observed_argv[1:] != expected_tail
    ):
        raise WatchdogError("watchdog live argv tail or launcher token is unexpected")
    declared_vector = [PINNED_PYTHON_BIN, *expected_tail]
    return {
        "declared_launch_vector": declared_vector,
        "declared_python_bin": PINNED_PYTHON_BIN,
        "declared_python_bin_resolved_path": str(resolved_launch),
        "declared_python_bin_sha256": _sha256_file(launch_path),
        "observed_live_argv": observed_argv,
        "observed_python_app_image_path": str(observed_exe),
        "observed_python_app_image_sha256": _sha256_file(observed_exe),
        "python_version": python_version,
        "psutil_version": psutil.__version__,
        "psutil_module_path": str(psutil_path),
        "psutil_module_sha256": _sha256_file(psutil_path),
        "original_exec_alias_attestable_from_process_apis": False,
    }


def _build_campaign_command(
    manifest_path: Path,
    manifest_sha256: str,
    receipt_directory: Path,
) -> list[str]:
    runner_path = _REPO_ROOT / "scripts/run_cascaded_tanks_abc6_synthetic.py"
    return [
        PINNED_PYTHON_BIN,
        str(runner_path),
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--receipts",
        str(receipt_directory),
    ]


def _snapshot_process_group(
    process_group_id: int,
) -> tuple[list[dict[str, object]], int, list[int], list[str]]:
    """Sample the child's private process group, even after its root exits."""

    members: list[dict[str, object]] = []
    vanished: list[int] = []
    errors: list[str] = []
    total = 0
    seen: set[tuple[int, float]] = set()
    try:
        processes = psutil.process_iter(["pid"])
        for process in processes:
            pid = process.pid
            try:
                observed_pgid = os.getpgid(pid)
            except ProcessLookupError:
                vanished.append(pid)
                continue
            except OSError as error:
                errors.append(
                    f"pid={pid} operation=getpgid error={type(error).__name__}: {error}"
                )
                continue
            if observed_pgid != process_group_id:
                continue
            try:
                create_time = process.create_time()
                identity = (pid, create_time)
                if identity in seen:
                    continue
                seen.add(identity)
                rss = process.memory_info().rss
                parent_pid = process.ppid()
                status = process.status()
                members.append(
                    {
                        "pid": pid,
                        "create_time": create_time,
                        "ppid": parent_pid,
                        "status": status,
                        "rss_bytes": rss,
                    }
                )
                total += rss
            except psutil.NoSuchProcess:
                vanished.append(pid)
            except (psutil.AccessDenied, psutil.Error, OSError) as error:
                errors.append(
                    f"pid={pid} operation=sample error={type(error).__name__}: {error}"
                )
    except (psutil.Error, OSError) as error:
        errors.append(
            f"operation=process_iter error={type(error).__name__}: {error}"
        )
    members.sort(key=lambda item: int(item["pid"]))
    return members, total, sorted(set(vanished)), errors


def _terminate_process_group(
    child: subprocess.Popen[bytes],
    process_group_id: int,
    *,
    terminate_grace_seconds: float,
    kill_reap_grace_seconds: float,
    previously_uncertain: bool = False,
) -> dict[str, object]:
    """Signal the group and report its state only after positive verification."""

    started = time.monotonic()
    term_sent_at: float | None = None
    kill_sent_at: float | None = None
    term_pids: list[int] = []
    kill_pids: list[int] = []
    term_group_signal_sent = False
    kill_group_signal_sent = False
    enumeration_errors: list[str] = []
    membership_uncertain = previously_uncertain

    def sample() -> tuple[list[dict[str, object]], bool]:
        nonlocal membership_uncertain
        members, _rss, _vanished, errors = _snapshot_process_group(process_group_id)
        enumeration_errors.extend(errors)
        if errors:
            membership_uncertain = True
        return members, bool(errors)

    members, uncertain = sample()
    if members or uncertain:
        term_sent_at = time.monotonic()
        try:
            os.killpg(process_group_id, signal.SIGTERM)
            term_group_signal_sent = True
            term_pids = [int(member["pid"]) for member in members]
        except ProcessLookupError:
            pass
        except OSError as error:
            enumeration_errors.append(
                f"operation=killpg_SIGTERM error={type(error).__name__}: {error}"
            )
            membership_uncertain = True

        deadline = time.monotonic() + terminate_grace_seconds
        while time.monotonic() < deadline:
            members, uncertain = sample()
            if not members and not uncertain:
                break
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

        members, uncertain = sample()
        if members or uncertain:
            kill_sent_at = time.monotonic()
            try:
                os.killpg(process_group_id, signal.SIGKILL)
                kill_group_signal_sent = True
                kill_pids = [int(member["pid"]) for member in members]
            except ProcessLookupError:
                pass
            except OSError as error:
                enumeration_errors.append(
                    f"operation=killpg_SIGKILL error={type(error).__name__}: {error}"
                )
                membership_uncertain = True

            deadline = time.monotonic() + kill_reap_grace_seconds
            while time.monotonic() < deadline:
                members, uncertain = sample()
                if not members and not uncertain:
                    break
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    try:
        child.wait(timeout=DIRECT_CHILD_REAP_GRACE_SECONDS)
        root_reaped = True
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            kill_group_signal_sent = True
        except OSError:
            pass
        try:
            child.wait(timeout=DIRECT_CHILD_REAP_GRACE_SECONDS)
            root_reaped = True
        except subprocess.TimeoutExpired:
            root_reaped = False

    final_members, final_uncertain = sample()
    if final_uncertain:
        membership_verification = "unknown"
        unreaped_process_pids: list[int] | None = None
    elif final_members:
        membership_verification = "members_remaining"
        unreaped_process_pids = [int(member["pid"]) for member in final_members]
    elif membership_uncertain:
        membership_verification = "unknown_after_enumeration_error"
        unreaped_process_pids = None
    else:
        membership_verification = "verified_empty"
        unreaped_process_pids = []

    verified_group_reaped = (
        root_reaped
        and membership_verification == "verified_empty"
        and not enumeration_errors
    )
    return {
        "process_group_id": process_group_id,
        "term_sent_at_elapsed_seconds": (
            None if term_sent_at is None else term_sent_at - started
        ),
        "kill_sent_at_elapsed_seconds": (
            None if kill_sent_at is None else kill_sent_at - started
        ),
        "term_pids": term_pids,
        "kill_pids": kill_pids,
        "term_process_group_signal_sent": term_group_signal_sent,
        "kill_process_group_signal_sent": kill_group_signal_sent,
        "unreaped_process_pids": unreaped_process_pids,
        "membership_verification": membership_verification,
        "membership_enumeration_errors": enumeration_errors,
        "child_reaped": root_reaped,
        "membership_scope": "isolated_process_group_only",
        "tracked_process_group_reaped": verified_group_reaped,
        "all_descendants_reaped_claimed": False,
        "escape_limitation": (
            "A descendant that creates a separate session/process group is outside this membership proof."
        ),
        "direct_child_reap_grace_seconds": DIRECT_CHILD_REAP_GRACE_SECONDS,
        "bounded_grace_seconds": (
            terminate_grace_seconds
            + kill_reap_grace_seconds
            + 2 * DIRECT_CHILD_REAP_GRACE_SECONDS
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def _observed_argv_matches(
    observed_argv: Sequence[str],
    launch_vector: Sequence[str],
    observed_executable_path: Path,
) -> bool:
    """Allow only the declared bin alias or observed app image at argv[0]."""

    return (
        len(observed_argv) == len(launch_vector)
        and observed_argv[0] in {launch_vector[0], str(observed_executable_path)}
        and list(observed_argv[1:]) == list(launch_vector[1:])
    )


def _supervise_command(
    *,
    launch_vector: Sequence[str],
    observed_executable_path: Path,
    observed_executable_sha256: str,
    claim_path: Path,
    receipt_path: Path,
    run_identity: dict[str, object],
    claim_project_root: Path | None = None,
    preflight_anchors: _PreflightAnchors | None = None,
    wall_limit_seconds: float = WALL_LIMIT_SECONDS,
    rss_limit_bytes: int = RSS_LIMIT_BYTES,
    sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
    terminate_grace_seconds: float = TERMINATE_GRACE_SECONDS,
    kill_reap_grace_seconds: float = KILL_REAP_GRACE_SECONDS,
) -> dict[str, object]:
    """Private seam used by production and fake-child tests."""

    if not launch_vector or not Path(launch_vector[0]).is_absolute():
        raise WatchdogError("child launch vector must begin with an absolute executable")
    if wall_limit_seconds <= 0 or rss_limit_bytes <= 0 or sample_interval_seconds <= 0:
        raise WatchdogError("watchdog budgets must be positive")
    if claim_project_root is not None:
        run_id = run_identity.get("run_id")
        if type(run_id) is not str:
            raise WatchdogError("project-root claim requires a string run ID")
        _validate_project_claim_path(claim_path, claim_project_root, run_id)
    executable = Path(launch_vector[0])
    try:
        resolved = executable.resolve(strict=True)
        observed = observed_executable_path.resolve(strict=True)
    except OSError as error:
        raise WatchdogError("declared child executable or image is missing") from error
    if _sha256_file(executable) == "":  # pragma: no cover - SHA-256 is never empty.
        raise WatchdogError("cannot hash declared child executable")
    if resolved != Path(PINNED_INTERPRETER_PATH) and launch_vector[0] == PINNED_PYTHON_BIN:
        raise WatchdogError("declared campaign launch alias resolves unexpectedly")
    if observed != Path(PINNED_PYTHON_APP) and launch_vector[0] == PINNED_PYTHON_BIN:
        raise WatchdogError("campaign process image differs from the pinned Python.app")
    if launch_vector[0] == PINNED_PYTHON_BIN:
        if (
            _sha256_file(executable) != PINNED_INTERPRETER_SHA256
            or observed_executable_sha256 != PINNED_PYTHON_APP_SHA256
        ):
            raise WatchdogError("campaign executable/image hash differs from pinned runtime")
    elif _sha256_file(observed) != observed_executable_sha256:
        raise WatchdogError("test child observed executable hash does not match its pin")

    # Validate the destination before consuming the one-use claim.  The
    # production path uses the descriptors pinned by _production_preflight;
    # private process-supervision tests retain their isolated Path fixture.
    receipt_directory_fd: int | None = None
    claim_directory_fd: int | None = None
    if preflight_anchors is not None:
        expected_receipt = (
            preflight_anchors.receipt_root_path
            / "watchdog-terminal-receipt.json"
        )
        expected_claim = _expected_project_claim_path(
            preflight_anchors.root_path, str(run_identity.get("run_id", ""))
        )
        if receipt_path != expected_receipt or claim_path != expected_claim:
            raise WatchdogError(
                "watchdog claim and terminal paths differ from the pinned schema-v2 root"
            )
        preflight_anchors.verify(require_unused=True)
        receipt_directory_fd = preflight_anchors.receipt_root_fd
        claim_directory_fd = preflight_anchors.watchdog_claim_parent_fd
        if os.listdir(receipt_directory_fd):
            raise WatchdogError("campaign receipt root is not empty before launch")
    else:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if receipt_path.exists() or receipt_path.is_symlink():
            raise WatchdogError("terminal watchdog receipt already exists")
        if receipt_path.parent.is_symlink() or not receipt_path.parent.is_dir():
            raise WatchdogError("campaign receipt path must be a real directory")
        try:
            entries = tuple(receipt_path.parent.iterdir())
        except OSError as error:
            raise WatchdogError("cannot inspect campaign receipt directory") from error
        if entries:
            raise WatchdogError("campaign receipt directory must be empty before launch")

    claim = {
        "schema_version": 1,
        "protocol_id": run_identity.get("protocol_id"),
        "run_id": run_identity.get("run_id"),
        "manifest_sha256": run_identity.get("manifest_sha256"),
        "approval_record_sha256": run_identity.get("approval_record_sha256"),
        "reviewed_git_head": run_identity.get("reviewed_git_head"),
        "repository_root_realpath": run_identity.get("repository_root_realpath"),
        "receipt_root_relative": run_identity.get("receipt_root_relative"),
        "repository_root_device": run_identity.get("repository_root_device"),
        "repository_root_inode": run_identity.get("repository_root_inode"),
        "receipt_root_device": run_identity.get("receipt_root_device"),
        "receipt_root_inode": run_identity.get("receipt_root_inode"),
        "watchdog_claim_parent_device": run_identity.get(
            "watchdog_claim_parent_device"
        ),
        "watchdog_claim_parent_inode": run_identity.get(
            "watchdog_claim_parent_inode"
        ),
        "scorer_claim_parent_device": run_identity.get("scorer_claim_parent_device"),
        "scorer_claim_parent_inode": run_identity.get("scorer_claim_parent_inode"),
        "claimed_at_utc": _utc_now(),
        "declared_child_launch_vector": list(launch_vector),
        "declared_child_executable_path": str(executable),
        "declared_child_executable_resolved_path": str(resolved),
        "declared_child_executable_sha256": _sha256_file(executable),
        "accepted_observed_child_argv0": [str(executable), str(observed)],
        "expected_observed_child_argv_tail": list(launch_vector[1:]),
        "expected_observed_child_image_path": str(observed),
        "expected_observed_child_image_sha256": observed_executable_sha256,
        "watchdog_attestation": run_identity.get("watchdog_attestation"),
        "claim_semantics": "consumed_once_no_resume",
    }
    claim_sha256 = _claim_once(
        claim_path,
        claim,
        project_root=claim_project_root,
        claim_directory_fd=claim_directory_fd,
    )

    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    child_env.pop("PYTHONHOME", None)
    child_env.pop("PYTHONSTARTUP", None)
    child_env["PYTHONNOUSERSITE"] = "0"

    timer_started_utc = _utc_now()
    timer_started = time.monotonic()
    child: subprocess.Popen[bytes] | None = None
    root: psutil.Process | None = None
    process_group_id: int | None = None
    samples: list[dict[str, object]] = []
    peak_rss = 0
    stop_reason = "watchdog_error"
    watchdog_intervention: dict[str, object] | None = None

    def record_watchdog_intervention(intervention_type: str, stage: str) -> None:
        nonlocal watchdog_intervention
        if watchdog_intervention is None:
            watchdog_intervention = _watchdog_intervention_snapshot(
                intervention_type, stage
            )

    process_return_code: int | None = None
    kill_reap: dict[str, object] | None = None
    observed_child_attestation: dict[str, object] | None = None
    child_image_observations: list[dict[str, object]] = []
    max_gap = 0.0
    membership_uncertain = False
    monitor_error: str | None = None
    try:
        child = subprocess.Popen(
            list(launch_vector),
            cwd=_REPO_ROOT,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # start_new_session=True makes the direct child the leader of a fresh
        # process group.  Group membership remains discoverable after the
        # leader exits, unlike a parent/child walk.
        process_group_id = child.pid
        root = psutil.Process(child.pid)
        while True:
            sampled_at = time.monotonic()
            elapsed = sampled_at - timer_started
            timestamp = _utc_now()
            try:
                # psutil may memoize exe() on a Process instance.  Reopen the
                # PID each sample so a transient Homebrew launcher image does
                # not hide the later Python.app image after exec.
                root = psutil.Process(child.pid)
                live_argv = root.cmdline()
                live_exe = Path(root.exe()).resolve(strict=True)
                live_exe_sha256 = _sha256_file(live_exe)
                if not _observed_argv_matches(live_argv, launch_vector, observed):
                    raise WatchdogError(
                        "child live argv differs from declared launch vector: "
                        f"observed={live_argv!r}; accepted_argv0="
                        f"{[launch_vector[0], str(observed)]!r}; "
                        f"expected_tail={list(launch_vector[1:])!r}"
                    )
                launch_image = resolved
                launch_image_sha256 = _sha256_file(executable)
                if live_exe == launch_image and live_exe_sha256 == launch_image_sha256:
                    image_phase = "declared_launcher_transition"
                elif (
                    live_exe == observed
                    and live_exe_sha256 == observed_executable_sha256
                ):
                    image_phase = "observed_python_app_image"
                else:
                    raise WatchdogError(
                        "child live image differs from declared launcher and pinned app: "
                        f"observed={str(live_exe)!r} ({live_exe_sha256}); "
                        f"launcher={str(launch_image)!r} ({launch_image_sha256}); "
                        f"app={str(observed)!r} ({observed_executable_sha256})"
                    )
                child_image_observations.append(
                    {
                        "timestamp_utc": timestamp,
                        "path": str(live_exe),
                        "sha256": live_exe_sha256,
                        "phase": image_phase,
                    }
                )
                if observed_child_attestation is None:
                    observed_child_attestation = {
                        "observed_live_argv": live_argv,
                        "observed_executable_path": str(live_exe),
                        "observed_executable_sha256": live_exe_sha256,
                        "verified_image_role": image_phase,
                        "observed_at_utc": timestamp,
                    }
            except psutil.NoSuchProcess:
                # The direct child completed between loop iterations.
                pass
            except (OSError, psutil.Error) as error:
                raise WatchdogError("cannot attest live child process identity") from error

            members, rss_total, vanished_pids, enumeration_errors = (
                _snapshot_process_group(process_group_id)
            )
            gap = 0.0 if not samples else sampled_at - float(samples[-1]["monotonic_seconds"])
            max_gap = max(max_gap, gap)
            peak_rss = max(peak_rss, rss_total)
            sample = {
                "timestamp_utc": timestamp,
                "monotonic_seconds": elapsed,
                "gap_seconds": gap,
                "process_group_id": process_group_id,
                "direct_child_executable_observed": (
                    child_image_observations[-1]
                    if child_image_observations
                    and child_image_observations[-1]["timestamp_utc"] == timestamp
                    else None
                ),
                "members": members,
                "vanished_during_sample_pids": vanished_pids,
                "membership_enumeration_errors": enumeration_errors,
                "membership_complete": not enumeration_errors,
                "runner_tree_rss_bytes": rss_total,
            }
            samples.append(sample)

            process_return_code = child.poll()
            if process_return_code is not None and members:
                stop_reason = "child_exited_with_live_process_group_members"
                record_watchdog_intervention("watchdog_stop", stop_reason)
                break

            if enumeration_errors:
                membership_uncertain = True
                stop_reason = "process_group_membership_unknown"
                monitor_error = "; ".join(enumeration_errors)
                record_watchdog_intervention("watchdog_stop", stop_reason)
                break

            if len(members) > 1:
                stop_reason = "unexpected_descendant_process"
                record_watchdog_intervention("watchdog_stop", stop_reason)
                break

            if process_return_code is not None:
                stop_reason = "child_exited"
                break

            if rss_total > rss_limit_bytes:
                stop_reason = "sampled_rss_limit_exceeded"
                record_watchdog_intervention("budget_cap", stop_reason)
                break
            if elapsed >= wall_limit_seconds:
                stop_reason = "wall_clock_limit_exceeded"
                record_watchdog_intervention("budget_cap", stop_reason)
                break
            time.sleep(min(sample_interval_seconds, max(0.0, wall_limit_seconds - elapsed)))

    except KeyboardInterrupt as error:
        stop_reason = "watchdog_interrupted"
        if watchdog_intervention is None:
            try:
                record_watchdog_intervention("operator_stop", stop_reason)
            except WatchdogError as clock_error:
                monitor_error = f"{type(clock_error).__name__}: {clock_error}"
        monitor_error = f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - persist terminal state after any monitor error.
        if isinstance(error, WatchdogError):
            stop_reason = "watchdog_identity_or_monitoring_error"
        else:
            stop_reason = "watchdog_monitoring_error"
        monitor_error = f"{type(error).__name__}: {error}"
        if watchdog_intervention is None:
            try:
                record_watchdog_intervention("watchdog_stop", stop_reason)
            except WatchdogError as clock_error:
                monitor_error += f"; {type(clock_error).__name__}: {clock_error}"

    if child is not None and root is not None and process_group_id is not None:
        process_return_code = child.poll()
        prior_error = membership_uncertain or stop_reason == "process_group_membership_unknown"
        kill_reap = _terminate_process_group(
            child,
            process_group_id,
            terminate_grace_seconds=terminate_grace_seconds,
            kill_reap_grace_seconds=kill_reap_grace_seconds,
            previously_uncertain=prior_error,
        )
        process_return_code = child.poll()
        if stop_reason == "child_exited" and not kill_reap["tracked_process_group_reaped"]:
            stop_reason = "child_exited_process_group_reap_unverified"
            record_watchdog_intervention("watchdog_stop", stop_reason)
        if stop_reason == "child_exited_with_live_process_group_members":
            stop_reason = "child_exited_with_live_process_group_members"

    elapsed_total = time.monotonic() - timer_started
    if (
        stop_reason == "child_exited"
        and process_return_code == 0
        and observed_child_attestation is None
    ):
        stop_reason = "child_exited_before_image_attestation"
    runner_exit_complete = stop_reason == "child_exited" and process_return_code == 0
    if stop_reason in {"sampled_rss_limit_exceeded", "wall_clock_limit_exceeded"}:
        terminal_status = "capped"
    elif runner_exit_complete:
        terminal_status = "completed"
    else:
        terminal_status = "failed"

    fit_gate_descriptor = receipt_directory_fd
    temporary_fit_gate_descriptor: int | None = None
    fit_gate_snapshot: _GateSnapshot | None = None
    try:
        if fit_gate_descriptor is None:
            temporary_fit_gate_descriptor = _open_absolute_directory_nofollow(
                receipt_path.parent,
                label="private fit-gate receipt root",
            )
            fit_gate_descriptor = temporary_fit_gate_descriptor
        fit_gate = _inspect_fit_phase_gate(
            fit_gate_descriptor,
            expected_receipt_directory=(
                preflight_anchors.receipt_root_path
                if preflight_anchors is not None
                else receipt_path.parent
            ),
            expected_manifest_sha256=str(run_identity.get("manifest_sha256", "")),
            sync_directory=runner_exit_complete,
            preflight_anchors=preflight_anchors,
            retain_snapshot=True,
        )
        private_snapshot = fit_gate.pop("_private_fit_gate_snapshot", None)
        if isinstance(private_snapshot, _GateSnapshot):
            fit_gate_snapshot = private_snapshot
    except Exception as error:
        fit_gate = {
            "status_receipt_count": 0,
            "status_receipts": [],
            "summary_path": str(receipt_path.parent / "campaign.training-summary.json"),
            "summary_sha256": None,
            "summary_status": None,
            "receipt_root_identity_valid": False,
            "file_identity_revalidation_valid": False,
            "all_48_status_receipts_present_and_linked": False,
            "all_48_fit_and_baseline_statuses_complete": False,
            "evidence_manifest_sha256": None,
            "training_evidence_chain_valid": False,
            "target_free_forecast_sha256": None,
            "target_free_forecast_chain_valid": False,
            "pre_score_artifact_chain_valid": False,
            "postfit_target_gate_opened": False,
            "problems": [
                f"cannot open descriptor-bound fit gate: {type(error).__name__}"
            ],
        }
    finally:
        if temporary_fit_gate_descriptor is not None:
            os.close(temporary_fit_gate_descriptor)
    if runner_exit_complete and not fit_gate["pre_score_artifact_chain_valid"]:
        terminal_status = "failed"
        stop_reason = "child_exited_with_invalid_pre_score_gate"
    samples_bytes = _canonical_json(samples)
    body: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": run_identity.get("protocol_id"),
        "run_id": run_identity.get("run_id"),
        "manifest_sha256": run_identity.get("manifest_sha256"),
        "approval_record_sha256": run_identity.get("approval_record_sha256"),
        "reviewed_git_head": run_identity.get("reviewed_git_head"),
        "repository_root_realpath": run_identity.get("repository_root_realpath"),
        "receipt_root_relative": run_identity.get("receipt_root_relative"),
        "repository_root_device": run_identity.get("repository_root_device"),
        "repository_root_inode": run_identity.get("repository_root_inode"),
        "receipt_root_device": run_identity.get("receipt_root_device"),
        "receipt_root_inode": run_identity.get("receipt_root_inode"),
        "watchdog_claim_parent_device": run_identity.get(
            "watchdog_claim_parent_device"
        ),
        "watchdog_claim_parent_inode": run_identity.get(
            "watchdog_claim_parent_inode"
        ),
        "scorer_claim_parent_device": run_identity.get("scorer_claim_parent_device"),
        "scorer_claim_parent_inode": run_identity.get("scorer_claim_parent_inode"),
        "watchdog_claim_path": str(claim_path),
        "watchdog_claim_sha256": claim_sha256,
        "status": terminal_status,
        "stop_reason": stop_reason,
        "monitor_error": locals().get("monitor_error"),
        "declared_budgets": {
            "wall_clock_seconds": wall_limit_seconds,
            "sampled_runner_tree_rss_bytes": rss_limit_bytes,
            "sample_interval_seconds": sample_interval_seconds,
            "terminate_grace_seconds": terminate_grace_seconds,
            "kill_reap_grace_seconds": kill_reap_grace_seconds,
            "direct_child_reap_grace_seconds": DIRECT_CHILD_REAP_GRACE_SECONDS,
        },
        "timer_start": {
            "started_at_utc": timer_started_utc,
            "started_monotonic_seconds": timer_started,
            "rule": "immediately before the single child Popen, after the durable watchdog claim",
        },
        "ended_at_utc": _utc_now(),
        "watchdog_intervention": watchdog_intervention,
        "elapsed_wall_seconds": elapsed_total,
        "process_return_code": process_return_code,
        "watchdog_process_excluded_from_runner_tree": {
            "excluded": True,
            "pid": os.getpid(),
            "observed_image_path": run_identity.get("watchdog_image_path"),
            "observed_image_sha256": run_identity.get("watchdog_image_sha256"),
        },
        "child_launch": {
            "declared_launch_vector": list(launch_vector),
            "declared_executable_path": str(executable),
            "declared_executable_sha256": _sha256_file(executable),
            "observed_process": observed_child_attestation,
            "image_observations": child_image_observations,
        },
        "rss_sampling": {
            "membership_method": "enumerate visible processes and match the private child PGID",
            "peak_sampled_runner_tree_rss_bytes": peak_rss,
            "maximum_sample_gap_seconds": max_gap,
            "raw_samples_sha256": hashlib.sha256(samples_bytes).hexdigest(),
            "raw_samples": samples,
            "between_sample_peak_caveat": (
                "RSS is sampled; brief peaks between samples may be missed. "
                "The watchdog process is excluded from runner-tree RSS."
            ),
            "process_group_escape_caveat": (
                "A child that deliberately creates a separate session/process group can escape PGID membership."
            ),
        },
        "kill_and_reap": kill_reap,
        "process_completion_scope": {
            "verified_scope": "direct child and its isolated process group only",
            "all_descendants_reaped_claimed": False,
            "detached_session_limitation": (
                "A descendant that calls setsid or otherwise leaves the process group can escape membership sampling and group signals."
            ),
        },
        "fit_phase_gate": fit_gate,
        "terminal_receipt_publication": {
            "method": (
                "pinned receipt-root FD, same-directory temporary file, file fsync, "
                "hard-link publication, directory fsync, canonical no-follow readback"
            ),
            "readback_required_for_successful_return": True,
            "readback_ack_leaf": _TERMINAL_READBACK_ACK_FILENAME,
            "readback_ack_required_for_replay": True,
            "fit_gate_revalidated_before_completed": False,
            "performed_by_this_writer": True,
        },
        "campaign_targets_generated_by_watchdog": False,
        "campaign_forecasts_run_by_watchdog": False,
    }
    terminal_receipt_fd = receipt_directory_fd
    temporary_terminal_fd: int | None = None
    try:
        if terminal_receipt_fd is None:
            temporary_terminal_fd = _open_absolute_directory_nofollow(
                receipt_path.parent,
                label="private terminal receipt root",
            )
            terminal_receipt_fd = temporary_terminal_fd
        _check_gate_receipt_root(
            terminal_receipt_fd,
            (
                preflight_anchors.receipt_root_path
                if preflight_anchors is not None
                else receipt_path.parent
            ),
            preflight_anchors=preflight_anchors,
        )

        body["terminal_receipt_publication"][
            "fit_gate_revalidated_before_completed"
        ] = terminal_status == "completed"
        terminal_payload = _canonical_json(body)
        if terminal_status == "completed":
            try:
                if fit_gate_snapshot is None:
                    raise WatchdogError("completed fit gate has no retained file snapshot")
                fit_gate_snapshot.verify()
                _check_gate_receipt_root(
                    terminal_receipt_fd,
                    (
                        preflight_anchors.receipt_root_path
                        if preflight_anchors is not None
                        else receipt_path.parent
                    ),
                    preflight_anchors=preflight_anchors,
                )
            except Exception as error:
                terminal_status = "failed"
                stop_reason = "fit_gate_changed_before_terminal_receipt"
                fit_gate["file_identity_revalidation_valid"] = False
                fit_gate["pre_score_artifact_chain_valid"] = False
                (
                    status_changed,
                    evidence_changed,
                    forecast_changed,
                ) = _gate_identity_change_categories(error)
                if status_changed:
                    fit_gate["status_chain_file_identity_valid"] = False
                    fit_gate["all_48_status_receipts_present_and_linked"] = False
                    fit_gate["all_48_fit_and_baseline_statuses_complete"] = False
                if evidence_changed:
                    fit_gate["training_evidence_chain_valid"] = False
                if forecast_changed:
                    fit_gate["target_free_forecast_chain_valid"] = False
                gate_problems = fit_gate.setdefault("problems", [])
                if isinstance(gate_problems, list):
                    gate_problems.append(
                        "terminal pre-publication revalidation failed: "
                        f"{type(error).__name__}: {str(error)[:1024]}"
                    )
                body["status"] = terminal_status
                body["stop_reason"] = stop_reason
                body["fit_phase_gate"] = fit_gate
                body["terminal_receipt_publication"][
                    "fit_gate_revalidated_before_completed"
                ] = False
                terminal_payload = _canonical_json(body)

        if fit_gate_snapshot is not None:
            fit_gate_snapshot.close()
            fit_gate_snapshot = None
        _check_gate_receipt_root(
            terminal_receipt_fd,
            (
                preflight_anchors.receipt_root_path
                if preflight_anchors is not None
                else receipt_path.parent
            ),
            preflight_anchors=preflight_anchors,
        )
        receipt_sha256, terminal_file_info = _publish_and_verify_terminal_receipt_at(
            terminal_receipt_fd,
            receipt_path.name,
            body,
            encoded_payload=terminal_payload,
        )
        _check_gate_receipt_root(
            terminal_receipt_fd,
            (
                preflight_anchors.receipt_root_path
                if preflight_anchors is not None
                else receipt_path.parent
            ),
            preflight_anchors=preflight_anchors,
        )
        terminal_readback_ack_sha256 = _publish_terminal_readback_ack_at(
            terminal_receipt_fd,
            receipt_path.name,
            body,
            receipt_sha256,
            terminal_file_info,
        )
        body["terminal_receipt_sha256"] = receipt_sha256
        # The receipt hash is returned out-of-band to avoid self-reference.
        return {
            "receipt": body,
            "receipt_sha256": receipt_sha256,
            "receipt_path": receipt_path,
            "terminal_readback_ack_path": (
                receipt_path.parent / _TERMINAL_READBACK_ACK_FILENAME
            ),
            "terminal_readback_ack_sha256": terminal_readback_ack_sha256,
            "terminal_readback_ack_published": True,
            "terminal_receipt_readback_verified": True,
        }
    finally:
        if fit_gate_snapshot is not None:
            fit_gate_snapshot.close()
        if temporary_terminal_fd is not None:
            os.close(temporary_terminal_fd)


def _payload_sha256_matches(payload: dict[str, object]) -> bool:
    digest = payload.get("payload_sha256")
    if not _is_sha256(digest):
        return False
    content = {key: value for key, value in payload.items() if key != "payload_sha256"}
    try:
        return hashlib.sha256(_canonical_json(content)).hexdigest() == digest
    except (TypeError, ValueError, RecursionError):
        return False


def _expected_case_ids() -> list[str]:
    from core.real_data.cascaded_tanks_abc6_cases import CASE_ROSTER

    if len(CASE_ROSTER) != 24:
        raise WatchdogError("reviewed campaign roster does not contain 24 cases")
    return [case.case_id for case in CASE_ROSTER]


def _status_counts(rows: Sequence[dict[str, object]], component: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("component") == component:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _summary_case_list_matches(
    observed: object, expected: Sequence[dict[str, object]]
) -> bool:
    if type(observed) is not list or len(observed) != len(expected):
        return False
    required_keys = {
        "case_index",
        "case_id",
        "fit_status",
        "baseline_status",
        "fit_receipt_sha256",
        "baseline_receipt_sha256",
    }
    for entry, expected_entry in zip(observed, expected, strict=True):
        if (
            type(entry) is not dict
            or set(entry) != required_keys
            or type(entry.get("case_index")) is not int
            or type(entry.get("case_id")) is not str
            or type(entry.get("fit_status")) is not str
            or type(entry.get("baseline_status")) is not str
            or not _is_sha256(entry.get("fit_receipt_sha256"))
            or not _is_sha256(entry.get("baseline_receipt_sha256"))
            or entry != expected_entry
        ):
            return False
    return True


def _summary_status_counts_match(observed: object, expected: dict[str, int]) -> bool:
    return (
        type(observed) is dict
        and set(observed) == set(expected)
        and all(type(value) is int for value in observed.values())
        and observed == expected
    )


@dataclass(frozen=True, slots=True)
class _GateLeafPin:
    directory_fd: int
    leaf_name: str
    maximum_bytes: int
    label: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _GateDirectoryPin:
    parent_fd: int
    leaf_name: str
    directory_fd: int
    label: str
    parent_device: int
    parent_inode: int
    device: int
    inode: int


@dataclass(slots=True)
class _GateSnapshot:
    """Retain descriptor anchors for a final on-disk fit-gate recheck."""

    leaf_pins: list[_GateLeafPin] = field(default_factory=list)
    directory_pins: list[_GateDirectoryPin] = field(default_factory=list)
    _parent_fds: dict[tuple[int, int], int] = field(default_factory=dict)
    _directory_fds: list[int] = field(default_factory=list)

    def _anchor_parent(self, directory_fd: int, *, label: str) -> tuple[int, tuple[int, int]]:
        try:
            info = os.fstat(directory_fd)
        except OSError as error:
            raise WatchdogError(f"{label} parent directory is unavailable") from error
        if not stat.S_ISDIR(info.st_mode):
            raise WatchdogError(f"{label} parent is not a directory")
        identity = (info.st_dev, info.st_ino)
        anchored = self._parent_fds.get(identity)
        if anchored is None:
            try:
                anchored = os.dup(directory_fd)
            except OSError as error:
                raise WatchdogError(f"cannot retain {label} parent descriptor") from error
            self._parent_fds[identity] = anchored
        return anchored, identity

    def pin_leaf(
        self,
        directory_fd: int,
        leaf_name: str,
        *,
        raw: bytes,
        info: os.stat_result,
        maximum_bytes: int,
        label: str,
    ) -> None:
        anchored_parent, _identity = self._anchor_parent(directory_fd, label=label)
        self.leaf_pins.append(
            _GateLeafPin(
                directory_fd=anchored_parent,
                leaf_name=leaf_name,
                maximum_bytes=maximum_bytes,
                label=label,
                device=info.st_dev,
                inode=info.st_ino,
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                ctime_ns=info.st_ctime_ns,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )

    def pin_directory(
        self,
        parent_fd: int,
        leaf_name: str,
        directory_fd: int,
        *,
        label: str,
    ) -> None:
        anchored_parent, parent_identity = self._anchor_parent(parent_fd, label=label)
        try:
            child_fd = os.dup(directory_fd)
            child_info = os.fstat(child_fd)
        except OSError as error:
            raise WatchdogError(f"cannot retain {label} descriptor") from error
        if not stat.S_ISDIR(child_info.st_mode):
            os.close(child_fd)
            raise WatchdogError(f"{label} is not a directory")
        self._directory_fds.append(child_fd)
        self.directory_pins.append(
            _GateDirectoryPin(
                parent_fd=anchored_parent,
                leaf_name=leaf_name,
                directory_fd=child_fd,
                label=label,
                parent_device=parent_identity[0],
                parent_inode=parent_identity[1],
                device=child_info.st_dev,
                inode=child_info.st_ino,
            )
        )

    def verify(self) -> None:
        problems: list[str] = []
        for pin in self.leaf_pins:
            try:
                raw, info = _read_regular_file_at(
                    pin.directory_fd,
                    pin.leaf_name,
                    maximum_bytes=pin.maximum_bytes,
                    label=pin.label,
                )
                named = os.stat(
                    pin.leaf_name,
                    dir_fd=pin.directory_fd,
                    follow_symlinks=False,
                )
                observed = (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
                named_identity = (
                    named.st_dev,
                    named.st_ino,
                    named.st_size,
                    named.st_mtime_ns,
                    named.st_ctime_ns,
                )
                expected = (
                    pin.device,
                    pin.inode,
                    pin.size,
                    pin.mtime_ns,
                    pin.ctime_ns,
                )
                if (
                    observed != expected
                    or named_identity != expected
                    or hashlib.sha256(raw).hexdigest() != pin.sha256
                ):
                    problems.append(f"{pin.label} identity or bytes changed")
            except (OSError, WatchdogError) as error:
                problems.append(
                    f"{pin.label} final no-follow read failed: {type(error).__name__}"
                )

        directory_flags = _directory_flags()
        for pin in self.directory_pins:
            try:
                parent_info = os.fstat(pin.parent_fd)
                retained_info = os.fstat(pin.directory_fd)
                reopened_fd = os.open(
                    pin.leaf_name,
                    directory_flags,
                    dir_fd=pin.parent_fd,
                )
                try:
                    reopened_info = os.fstat(reopened_fd)
                    named_info = os.stat(
                        pin.leaf_name,
                        dir_fd=pin.parent_fd,
                        follow_symlinks=False,
                    )
                finally:
                    os.close(reopened_fd)
                if (
                    (parent_info.st_dev, parent_info.st_ino)
                    != (pin.parent_device, pin.parent_inode)
                    or not stat.S_ISDIR(retained_info.st_mode)
                    or (retained_info.st_dev, retained_info.st_ino)
                    != (pin.device, pin.inode)
                    or not stat.S_ISDIR(reopened_info.st_mode)
                    or (reopened_info.st_dev, reopened_info.st_ino)
                    != (pin.device, pin.inode)
                    or not stat.S_ISDIR(named_info.st_mode)
                    or (named_info.st_dev, named_info.st_ino)
                    != (pin.device, pin.inode)
                ):
                    problems.append(f"{pin.label} directory identity changed")
            except OSError as error:
                problems.append(
                    f"{pin.label} directory re-open failed: {type(error).__name__}"
                )
        if problems:
            raise WatchdogError("; ".join(problems))

    def close(self) -> None:
        for descriptor in (*self._parent_fds.values(), *self._directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._parent_fds.clear()
        self._directory_fds.clear()


def _gate_identity_change_categories(error: BaseException) -> tuple[bool, bool, bool]:
    """Return whether a changed leaf affects status, evidence, or forecast links."""

    message = str(error).lower()
    root_changed = "receipt-root" in message or "receipt root" in message
    status_changed = root_changed or any(
        label in message
        for label in ("status receipt", "campaign claim receipt", "training summary")
    )
    evidence_changed = status_changed or any(
        label in message
        for label in ("training evidence", "training bundle evidence")
    )
    forecast_changed = evidence_changed or "target-free forecast" in message
    return status_changed, evidence_changed, forecast_changed


def _read_canonical_object_at(
    directory_fd: int,
    leaf_name: str,
    *,
    maximum_bytes: int,
    label: str,
    snapshot: _GateSnapshot | None = None,
) -> tuple[bytes, dict[str, object], os.stat_result]:
    raw, info = _read_regular_file_at(
        directory_fd,
        leaf_name,
        maximum_bytes=maximum_bytes,
        label=label,
    )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    try:
        decoded = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
        )
        if type(decoded) is not dict or _canonical_json(decoded) != raw:
            raise ValueError("JSON is not a canonical object")
        named = os.stat(leaf_name, dir_fd=directory_fd, follow_symlinks=False)
    except (
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        raise WatchdogError(f"{label} is not a stable canonical JSON object") from error
    if (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ) != (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    ):
        raise WatchdogError(f"{label} directory entry changed while being read")
    if snapshot is not None:
        snapshot.pin_leaf(
            directory_fd,
            leaf_name,
            raw=raw,
            info=info,
            maximum_bytes=maximum_bytes,
            label=label,
        )
    return raw, decoded, info


def _expected_forecast_case_identities() -> list[dict[str, object]]:
    from core.real_data.cascaded_tanks_abc6_cases import CASE_ROSTER

    if len(CASE_ROSTER) != 24:
        raise WatchdogError("reviewed campaign roster does not contain 24 cases")
    return [
        {
            "case_index": case.case_index,
            "case_id": case.case_id,
            "truth_id": case.truth_id,
            "input_window": case.input_window,
            "replicate": case.replicate,
            "fit_model": case.fit_model.value,
        }
        for case in CASE_ROSTER
    ]


def _expected_integrated_source_hashes(
    *,
    preflight_anchors: _PreflightAnchors | None,
    supplied: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if supplied is not None:
        source_hashes: object = supplied
    elif preflight_anchors is not None:
        try:
            manifest = json.loads(preflight_anchors.manifest_file.raw.decode("ascii"))
            source_hashes = (
                manifest.get("source_hashes") if type(manifest) is dict else None
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            return None
    else:
        return None
    if not isinstance(source_hashes, Mapping):
        return None
    selected = {path: source_hashes.get(path) for path in _INTEGRATED_SOURCE_PATHS}
    if any(not _is_sha256(digest) for digest in selected.values()):
        return None
    return {path: str(digest) for path, digest in selected.items()}


def _same_directory_identity(fd: int, expected_fd: int) -> bool:
    try:
        actual_info = os.fstat(fd)
        expected_info = os.fstat(expected_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(actual_info.st_mode)
        and stat.S_ISDIR(expected_info.st_mode)
        and (actual_info.st_dev, actual_info.st_ino)
        == (expected_info.st_dev, expected_info.st_ino)
    )


def _check_gate_receipt_root(
    receipt_directory_fd: int,
    expected_receipt_directory: Path,
    *,
    preflight_anchors: _PreflightAnchors | None,
) -> None:
    if (
        not expected_receipt_directory.is_absolute()
        or str(expected_receipt_directory)
        != str(expected_receipt_directory.absolute())
    ):
        raise WatchdogError("expected receipt root path is not canonical")
    if preflight_anchors is not None:
        if (
            receipt_directory_fd != preflight_anchors.receipt_root_fd
            or expected_receipt_directory != preflight_anchors.receipt_root_path
        ):
            raise WatchdogError("fit gate did not receive the pinned receipt-root FD")
        preflight_anchors.verify(require_unused=False)
        expected_identity = preflight_anchors.directory_identities[
            _RECEIPT_ROOT_RELATIVE
        ]
        info = os.fstat(receipt_directory_fd)
        if (info.st_dev, info.st_ino) != expected_identity:
            raise WatchdogError("pinned receipt-root FD identity changed")
        return

    expected_fd = _open_absolute_directory_nofollow(
        expected_receipt_directory,
        label="fit-gate receipt root",
    )
    try:
        if not _same_directory_identity(receipt_directory_fd, expected_fd):
            raise WatchdogError("fit-gate receipt root path identity changed")
    finally:
        os.close(expected_fd)


def _verify_training_evidence_chain_at(
    receipt_directory_fd: int,
    *,
    expected_manifest_sha256: str,
    claim_sha256: str,
    summary_sha256: str,
    status_receipts: Sequence[dict[str, object]],
    snapshot: _GateSnapshot,
) -> tuple[str, bool, str | None]:
    from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit

    raw, manifest, _info = _read_canonical_object_at(
        receipt_directory_fd,
        campaign_fit.EVIDENCE_MANIFEST_FILENAME,
        maximum_bytes=MAX_TRAINING_EVIDENCE_MANIFEST_BYTES,
        label="training evidence manifest",
        snapshot=snapshot,
    )
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "claim_sha256",
        "training_summary_filename",
        "training_summary_sha256",
        "roster_sha256",
        "ordered_case_identities",
        "status_receipts",
        "bundle_artifact",
        "case_artifacts",
        "payload_sha256",
    }
    expected_status_receipts = [
        {
            "filename": row["filename"],
            "sha256": row["sha256"],
            "case_index": row["case_index"],
            "case_id": row["case_id"],
            "component": row["component"],
            "status": row["status"],
        }
        for row in status_receipts
    ]
    roster_identities = campaign_fit._roster_identities()
    manifest_valid = (
        set(manifest) == expected_keys
        and type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 1
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("run_id") == RUN_ID
        and manifest.get("manifest_sha256") == expected_manifest_sha256
        and manifest.get("claim_sha256") == claim_sha256
        and manifest.get("training_summary_filename")
        == campaign_fit.SUMMARY_FILENAME
        and manifest.get("training_summary_sha256") == summary_sha256
        and manifest.get("roster_sha256") == campaign_fit._roster_sha256()
        and manifest.get("ordered_case_identities") == roster_identities
        and manifest.get("status_receipts") == expected_status_receipts
        and _payload_sha256_matches(manifest)
    )
    if not manifest_valid:
        raise WatchdogError("training evidence manifest links are invalid")

    bundle_entry = manifest.get("bundle_artifact")
    case_entries = manifest.get("case_artifacts")
    if (
        type(bundle_entry) is not dict
        or set(bundle_entry) != {"filename", "sha256"}
        or bundle_entry.get("filename") != "training-bundle.evidence.json"
        or not _is_sha256(bundle_entry.get("sha256"))
        or type(case_entries) is not list
        or len(case_entries) != 24
    ):
        raise WatchdogError("training evidence artifact roster is invalid")

    from core.real_data.cascaded_tanks_abc6_cases import CASE_ROSTER

    status_by_case = {
        int(row["case_index"]): row for row in status_receipts if row["component"] == "fit"
    }
    baseline_by_case = {
        int(row["case_index"]): row
        for row in status_receipts
        if row["component"] == "baseline"
    }
    evidence_fd = _open_relative_directory_nofollow(
        receipt_directory_fd,
        _TRAINING_EVIDENCE_DIRECTORY,
        label="training evidence directory",
    )
    total_bytes = 0
    try:
        snapshot.pin_directory(
            receipt_directory_fd,
            _TRAINING_EVIDENCE_DIRECTORY,
            evidence_fd,
            label="training evidence directory",
        )
        bundle_raw, bundle, bundle_info = _read_canonical_object_at(
            evidence_fd,
            str(bundle_entry["filename"]),
            maximum_bytes=MAX_TRAINING_EVIDENCE_FILE_BYTES,
            label="training bundle evidence",
            snapshot=snapshot,
        )
        total_bytes += bundle_info.st_size
        if (
            hashlib.sha256(bundle_raw).hexdigest() != bundle_entry["sha256"]
            or bundle.get("schema_version") != 1
            or bundle.get("protocol_id") != PROTOCOL_ID
            or bundle.get("run_id") != RUN_ID
            or bundle.get("kind") != "abc6_training_bundle"
            or bundle.get("ordered_case_identities") != roster_identities
        ):
            raise WatchdogError("training bundle evidence is invalid")

        for index, (case, entry) in enumerate(
            zip(CASE_ROSTER, case_entries, strict=True)
        ):
            expected_filename = f"case-{index:02d}.training-evidence.json"
            fit_row = status_by_case[index]
            baseline_row = baseline_by_case[index]
            case_identity = campaign_fit._case_identity(case)
            if (
                type(entry) is not dict
                or set(entry)
                != {
                    "filename",
                    "sha256",
                    "roster_index",
                    "case_identity",
                    "fit_status",
                    "baseline_status",
                }
                or entry.get("filename") != expected_filename
                or entry.get("sha256") is None
                or not _is_sha256(entry.get("sha256"))
                or type(entry.get("roster_index")) is not int
                or entry.get("roster_index") != index
                or entry.get("case_identity") != case_identity
                or entry.get("fit_status") != fit_row["status"]
                or entry.get("baseline_status") != baseline_row["status"]
            ):
                raise WatchdogError(f"training evidence link is invalid at case {index}")
            case_raw, case_payload, case_info = _read_canonical_object_at(
                evidence_fd,
                expected_filename,
                maximum_bytes=MAX_TRAINING_EVIDENCE_FILE_BYTES,
                label=f"training evidence case {index}",
                snapshot=snapshot,
            )
            total_bytes += case_info.st_size
            if total_bytes > MAX_TRAINING_EVIDENCE_TOTAL_BYTES:
                raise WatchdogError("training evidence exceeds its total byte limit")
            if (
                hashlib.sha256(case_raw).hexdigest() != entry["sha256"]
                or set(case_payload)
                != {
                    "schema_version",
                    "protocol_id",
                    "run_id",
                    "kind",
                    "roster_index",
                    "case_identity",
                    "fit_status",
                    "baseline_status",
                    "result",
                }
                or case_payload.get("schema_version") != 1
                or case_payload.get("protocol_id") != PROTOCOL_ID
                or case_payload.get("run_id") != RUN_ID
                or case_payload.get("kind") != "abc6_training_case_result"
                or case_payload.get("roster_index") != index
                or case_payload.get("case_identity") != case_identity
                or case_payload.get("fit_status") != fit_row["status"]
                or case_payload.get("baseline_status") != baseline_row["status"]
            ):
                raise WatchdogError(f"training evidence content is invalid at case {index}")
        named = os.stat(
            _TRAINING_EVIDENCE_DIRECTORY,
            dir_fd=receipt_directory_fd,
            follow_symlinks=False,
        )
        evidence_info = os.fstat(evidence_fd)
        if (
            not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino)
            != (evidence_info.st_dev, evidence_info.st_ino)
        ):
            raise WatchdogError("training evidence directory entry changed")
    finally:
        os.close(evidence_fd)
    return hashlib.sha256(raw).hexdigest(), True, None


def _verify_target_free_forecast_at(
    receipt_directory_fd: int,
    *,
    expected_manifest_sha256: str,
    claim_sha256: str,
    summary_sha256: str,
    evidence_manifest_sha256: str,
    status_receipts: Sequence[dict[str, object]],
    expected_integrated_source_hashes: Mapping[str, str] | None,
    snapshot: _GateSnapshot,
) -> tuple[str, bool, str | None]:
    raw, forecast, _info = _read_canonical_object_at(
        receipt_directory_fd,
        _TARGET_FREE_FORECAST,
        maximum_bytes=MAX_TARGET_FREE_FORECAST_BYTES,
        label="target-free forecast artifact",
        snapshot=snapshot,
    )
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "training_manifest_sha256",
        "training_claim_sha256",
        "training_summary_filename",
        "training_summary_sha256",
        "training_evidence_manifest_sha256",
        "integrated_source_hashes",
        "ordered_case_identities",
        "status_receipts",
        "forecast_case_sha256",
        "forecast_roster_sha256",
        "forecasts",
        "target_free",
        "prospective_targets_generated_by_runner",
        "retry_allowed",
        "payload_sha256",
    }
    from core.real_data.cascaded_tanks_abc6_campaign_fit import SUMMARY_FILENAME
    from core.real_data.cascaded_tanks_abc6_cases import CASE_ROSTER

    expected_identities = _expected_forecast_case_identities()
    expected_statuses = [
        {"filename": row["filename"], "sha256": row["sha256"]}
        for row in status_receipts
    ]
    expected_sources = (
        None
        if expected_integrated_source_hashes is None
        else [
            {"path": path, "sha256": expected_integrated_source_hashes[path]}
            for path in _INTEGRATED_SOURCE_PATHS
        ]
    )
    forecast_list = forecast.get("forecasts")
    case_digests = forecast.get("forecast_case_sha256")
    forecast_fields = {
        "status",
        "case_index",
        "case_id",
        "fit_window",
        "prospective_inputs",
        "common_state_index",
        "parameter_order",
        "particles",
        "weights",
        "particle_trajectories",
        "aggregate_status",
        "pointwise_weighted_mean",
        "pointwise_weighted_median",
        "pointwise_q05",
        "pointwise_q95",
        "effective_sample_size",
        "quantile_convention",
        "pointwise_summaries_are_coherent_trajectories",
        "baseline_parameter_values",
        "baseline_common_time_state",
        "baseline_trajectory",
        "baseline_failure",
    }
    forecast_valid = (
        set(forecast) == expected_keys
        and type(forecast.get("schema_version")) is int
        and forecast.get("schema_version") == 1
        and forecast.get("protocol_id") == PROTOCOL_ID
        and forecast.get("run_id") == RUN_ID
        and forecast.get("training_manifest_sha256") == expected_manifest_sha256
        and forecast.get("training_claim_sha256") == claim_sha256
        and forecast.get("training_summary_filename") == SUMMARY_FILENAME
        and forecast.get("training_summary_sha256") == summary_sha256
        and forecast.get("training_evidence_manifest_sha256")
        == evidence_manifest_sha256
        and expected_sources is not None
        and forecast.get("integrated_source_hashes") == expected_sources
        and forecast.get("ordered_case_identities") == expected_identities
        and forecast.get("status_receipts") == expected_statuses
        and type(forecast_list) is list
        and len(forecast_list) == 24
        and type(case_digests) is list
        and len(case_digests) == 24
        and all(_is_sha256(digest) for digest in case_digests)
        and forecast.get("target_free") is True
        and forecast.get("prospective_targets_generated_by_runner") is False
        and forecast.get("retry_allowed") is False
        and _payload_sha256_matches(forecast)
    )
    if not forecast_valid:
        raise WatchdogError(
            "target-free forecast identity/source/status/hash links are invalid"
        )

    computed_case_digests: list[str] = []
    for index, (case, identity, row) in enumerate(
        zip(CASE_ROSTER, expected_identities, forecast_list, strict=True)
    ):
        if (
            type(row) is not dict
            or set(row) != forecast_fields
            or type(row.get("case_index")) is not int
            or row.get("case_index") != index
            or row.get("case_id") != identity["case_id"]
            or row.get("fit_window") != case.input_window
            or row.get("status") not in {"complete", "abstained_n", "incomplete_abc_fit"}
            or (case.truth_id == "N" and row.get("status") != "abstained_n")
            or (case.truth_id != "N" and row.get("status") == "abstained_n")
        ):
            raise WatchdogError(f"target-free forecast roster is invalid at case {index}")
        case_payload = {
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "case_index": index,
            "case_id": case.case_id,
            "truth_id": case.truth_id,
            "input_window": case.input_window,
            "replicate": case.replicate,
            "fit_model": case.fit_model.value,
            "forecast": row,
        }
        computed_case_digests.append(
            hashlib.sha256(_canonical_json(case_payload)).hexdigest()
        )
    if list(case_digests) != computed_case_digests:
        raise WatchdogError("target-free forecast case digest chain is invalid")
    roster_hash = hashlib.sha256(
        _canonical_json(
            {
                "protocol_id": PROTOCOL_ID,
                "run_id": RUN_ID,
                "case_sha256": computed_case_digests,
            }
        )
    ).hexdigest()
    if forecast.get("forecast_roster_sha256") != roster_hash:
        raise WatchdogError("target-free forecast roster digest is invalid")
    return hashlib.sha256(raw).hexdigest(), True, None


def _inspect_fit_phase_gate(
    receipt_directory_fd: int,
    *,
    expected_receipt_directory: Path,
    expected_manifest_sha256: str,
    sync_directory: bool,
    preflight_anchors: _PreflightAnchors | None = None,
    expected_integrated_source_hashes: Mapping[str, str] | None = None,
    retain_snapshot: bool = False,
) -> dict[str, object]:
    snapshot = _GateSnapshot()
    snapshot_to_close: _GateSnapshot | None = snapshot
    try:
        result = _inspect_fit_phase_gate_with_snapshot(
            receipt_directory_fd,
            expected_receipt_directory=expected_receipt_directory,
            expected_manifest_sha256=expected_manifest_sha256,
            sync_directory=sync_directory,
            preflight_anchors=preflight_anchors,
            expected_integrated_source_hashes=expected_integrated_source_hashes,
            snapshot=snapshot,
        )
        if retain_snapshot:
            result["_private_fit_gate_snapshot"] = snapshot
            snapshot_to_close = None
        return result
    finally:
        if snapshot_to_close is not None:
            snapshot_to_close.close()


def _inspect_fit_phase_gate_with_snapshot(
    receipt_directory_fd: int,
    *,
    expected_receipt_directory: Path,
    expected_manifest_sha256: str,
    sync_directory: bool,
    preflight_anchors: _PreflightAnchors | None = None,
    expected_integrated_source_hashes: Mapping[str, str] | None = None,
    snapshot: _GateSnapshot,
) -> dict[str, object]:
    statuses: list[dict[str, object]] = []
    problems: list[str] = []
    receipt_root_identity_valid = True
    file_identity_revalidation_valid = True
    status_chain_file_identity_valid = True
    evidence_file_identity_valid = True
    forecast_file_identity_valid = True
    try:
        _check_gate_receipt_root(
            receipt_directory_fd,
            expected_receipt_directory,
            preflight_anchors=preflight_anchors,
        )
    except Exception as error:
        receipt_root_identity_valid = False
        problems.append(
            f"pinned receipt-root identity is invalid: {type(error).__name__}"
        )
    try:
        case_ids = _expected_case_ids()
    except (ImportError, WatchdogError) as error:
        case_ids = []
        problems.append(f"cannot load reviewed case roster: {type(error).__name__}")
    for case_index in range(24):
        for component in ("fit", "baseline"):
            filename = f"case-{case_index:02d}.{component}-status.json"
            try:
                raw, payload, _info = _read_canonical_object_at(
                    receipt_directory_fd,
                    filename,
                    maximum_bytes=MAX_STATUS_RECEIPT_BYTES,
                    label=f"status receipt {filename}",
                    snapshot=snapshot,
                )
                valid = (
                    type(payload) is dict
                    and set(payload)
                    == {
                        "schema_version",
                        "protocol_id",
                        "run_id",
                        "case_index",
                        "case_id",
                        "component",
                        "status",
                        "payload_sha256",
                    }
                    and _canonical_json(payload) == raw
                    and type(payload.get("schema_version")) is int
                    and payload.get("schema_version") == 1
                    and payload.get("protocol_id") == PROTOCOL_ID
                    and payload.get("run_id") == RUN_ID
                    and type(payload.get("case_index")) is int
                    and payload.get("case_index") == case_index
                    and len(case_ids) == 24
                    and payload.get("case_id") == case_ids[case_index]
                    and payload.get("component") == component
                    and type(payload.get("status")) is str
                    and payload.get("status") in {"complete", "incomplete", "unresolved", "failed"}
                    and _payload_sha256_matches(payload)
                )
                if not valid:
                    problems.append(f"invalid status receipt: {filename}")
                    continue
                statuses.append(
                    {
                        "case_index": case_index,
                        "case_id": payload["case_id"],
                        "component": component,
                        "status": payload["status"],
                        "filename": filename,
                        "path": str(expected_receipt_directory / filename),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            except (
                OSError,
                WatchdogError,
                UnicodeError,
                json.JSONDecodeError,
                RecursionError,
                TypeError,
                ValueError,
            ) as error:
                problems.append(f"cannot validate {filename}: {type(error).__name__}")

    summary_filename = "campaign.training-summary.json"
    summary_path = expected_receipt_directory / summary_filename
    summary: dict[str, object] | None = None
    summary_sha256: str | None = None
    campaign_claim_sha256: str | None = None
    claim_path = expected_receipt_directory / "campaign.claim"
    try:
        claim_raw, claim_payload, _claim_info = _read_canonical_object_at(
            receipt_directory_fd,
            "campaign.claim",
            maximum_bytes=MAX_STATUS_RECEIPT_BYTES,
            label="campaign claim receipt",
            snapshot=snapshot,
        )
        if (
            type(claim_payload) is not dict
            or set(claim_payload)
            != {
                "schema_version",
                "protocol_id",
                "run_id",
                "manifest_sha256",
                "receipt_directory",
                "claimed_at_utc",
                "claim_semantics",
            }
            or _canonical_json(claim_payload) != claim_raw
            or type(claim_payload.get("schema_version")) is not int
            or claim_payload.get("schema_version") != 1
            or claim_payload.get("protocol_id") != PROTOCOL_ID
            or claim_payload.get("run_id") != RUN_ID
            or claim_payload.get("manifest_sha256") != expected_manifest_sha256
            or claim_payload.get("receipt_directory")
            != str(expected_receipt_directory)
            or type(claim_payload.get("claimed_at_utc")) is not str
            or claim_payload.get("claim_semantics") != "consumed_once_no_resume"
        ):
            problems.append(
                "campaign claim is invalid or does not match the receipt directory"
            )
        else:
            campaign_claim_sha256 = hashlib.sha256(claim_raw).hexdigest()
    except (
        OSError,
        WatchdogError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        problems.append(f"cannot validate campaign claim: {type(error).__name__}")

    try:
        raw_summary, decoded, _summary_info = _read_canonical_object_at(
            receipt_directory_fd,
            summary_filename,
            maximum_bytes=MAX_CAMPAIGN_SUMMARY_BYTES,
            label="training summary",
            snapshot=snapshot,
        )
        if (
            type(decoded) is not dict
            or set(decoded)
            != {
                "schema_version",
                "protocol_id",
                "run_id",
                "manifest_sha256",
                "claim_sha256",
                "written_at_utc",
                "status",
                "case_count",
                "fit_status_counts",
                "baseline_status_counts",
                "case_statuses",
                "training_only",
                "prospective_targets_generated",
                "forecasts_run",
                "postfit_target_gate_opened",
                "external_watchdog_enforced_here",
                "independent_manifest_approval_enforced_here",
                "resume_allowed",
                "payload_sha256",
            }
            or _canonical_json(decoded) != raw_summary
            or type(decoded.get("schema_version")) is not int
            or decoded.get("schema_version") != 1
            or not _payload_sha256_matches(decoded)
        ):
            problems.append("training summary is not a valid durable campaign receipt")
        else:
            summary = decoded
            summary_sha256 = hashlib.sha256(raw_summary).hexdigest()
    except (
        OSError,
        WatchdogError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        problems.append(f"cannot validate campaign summary: {type(error).__name__}")

    summary_links_valid = False
    if summary is not None:
        summary_cases = summary.get("case_statuses")
        status_by_case_component = {
            (int(row["case_index"]), str(row["component"])): row for row in statuses
        }
        expected_summary_cases: list[dict[str, object]] = []
        if len(status_by_case_component) == 48 and len(case_ids) == 24:
            for idx, case_id in enumerate(case_ids):
                fit = status_by_case_component[(idx, "fit")]
                baseline = status_by_case_component[(idx, "baseline")]
                expected_summary_cases.append(
                    {
                        "case_index": idx,
                        "case_id": case_id,
                        "fit_status": fit["status"],
                        "baseline_status": baseline["status"],
                        "fit_receipt_sha256": fit["sha256"],
                        "baseline_receipt_sha256": baseline["sha256"],
                    }
                )
        expected_campaign_status = (
            "complete" if len(status_by_case_component) == 48
            and all(row["status"] == "complete" for row in statuses)
            else "incomplete"
        )
        summary_links_valid = (
            type(summary_cases) is list
            and len(summary_cases) == 24
            and _summary_case_list_matches(summary_cases, expected_summary_cases)
            and summary.get("protocol_id") == PROTOCOL_ID
            and summary.get("run_id") == RUN_ID
            and summary.get("manifest_sha256") == expected_manifest_sha256
            and summary.get("claim_sha256") == campaign_claim_sha256
            and type(summary.get("claim_sha256")) is str
            and _is_sha256(summary.get("claim_sha256"))
            and type(summary.get("written_at_utc")) is str
            and bool(summary.get("written_at_utc"))
            and type(summary.get("case_count")) is int
            and summary.get("case_count") == 24
            and summary.get("status") == expected_campaign_status
            and _summary_status_counts_match(
                summary.get("fit_status_counts"), _status_counts(statuses, "fit")
            )
            and _summary_status_counts_match(
                summary.get("baseline_status_counts"),
                _status_counts(statuses, "baseline"),
            )
            and summary.get("training_only") is True
            and summary.get("prospective_targets_generated") is False
            and summary.get("forecasts_run") is False
            and summary.get("postfit_target_gate_opened") is False
            and summary.get("external_watchdog_enforced_here") is False
            and summary.get("independent_manifest_approval_enforced_here") is False
            and summary.get("resume_allowed") is False
        )
        if not summary_links_valid:
            problems.append("summary identity/status links do not match durable case receipts")

    evidence_manifest_sha256: str | None = None
    evidence_chain_valid = False
    if (
        len(statuses) == 48
        and summary_links_valid
        and campaign_claim_sha256 is not None
        and summary_sha256 is not None
    ):
        try:
            (
                evidence_manifest_sha256,
                evidence_chain_valid,
                _evidence_detail,
            ) = _verify_training_evidence_chain_at(
                receipt_directory_fd,
                expected_manifest_sha256=expected_manifest_sha256,
                claim_sha256=campaign_claim_sha256,
                summary_sha256=summary_sha256,
                status_receipts=statuses,
                snapshot=snapshot,
            )
        except Exception as error:
            problems.append(
                f"cannot validate training evidence chain: {type(error).__name__}"
            )
    else:
        problems.append("training evidence chain lacks its complete status/summary links")

    integrated_sources = _expected_integrated_source_hashes(
        preflight_anchors=preflight_anchors,
        supplied=expected_integrated_source_hashes,
    )
    target_free_forecast_sha256: str | None = None
    target_free_forecast_valid = False
    if (
        evidence_chain_valid
        and evidence_manifest_sha256 is not None
        and campaign_claim_sha256 is not None
        and summary_sha256 is not None
    ):
        try:
            (
                target_free_forecast_sha256,
                target_free_forecast_valid,
                _forecast_detail,
            ) = _verify_target_free_forecast_at(
                receipt_directory_fd,
                expected_manifest_sha256=expected_manifest_sha256,
                claim_sha256=campaign_claim_sha256,
                summary_sha256=summary_sha256,
                evidence_manifest_sha256=evidence_manifest_sha256,
                status_receipts=statuses,
                expected_integrated_source_hashes=integrated_sources,
                snapshot=snapshot,
            )
        except Exception as error:
            problems.append(
                f"cannot validate target-free forecast chain: {type(error).__name__}"
            )
    else:
        problems.append("target-free forecast chain lacks verified training evidence")

    if integrated_sources is None:
        problems.append(
            "frozen manifest does not pin the integrated runner/forecast/scorer sources"
        )

    if sync_directory and len(statuses) == 48 and summary_links_valid:
        try:
            os.fsync(receipt_directory_fd)
        except OSError as error:
            problems.append(f"cannot sync campaign receipt directory: {type(error).__name__}")

    try:
        snapshot.verify()
    except WatchdogError as error:
        file_identity_revalidation_valid = False
        (
            status_changed,
            evidence_changed,
            forecast_changed,
        ) = _gate_identity_change_categories(error)
        status_chain_file_identity_valid = not status_changed
        evidence_file_identity_valid = not evidence_changed
        forecast_file_identity_valid = not forecast_changed
        problems.append(
            "fit-gate file identity changed before return: "
            f"{type(error).__name__}: {str(error)[:1024]}"
        )

    try:
        _check_gate_receipt_root(
            receipt_directory_fd,
            expected_receipt_directory,
            preflight_anchors=preflight_anchors,
        )
    except Exception as error:
        receipt_root_identity_valid = False
        problems.append(
            f"receipt-root path changed during gate verification: {type(error).__name__}"
        )

    all_statuses_complete = len(statuses) == 48 and all(
        row["status"] == "complete" for row in statuses
    )
    status_chain_valid = (
        receipt_root_identity_valid
        and status_chain_file_identity_valid
        and len(statuses) == 48
        and summary_links_valid
    )
    evidence_chain_valid = evidence_chain_valid and evidence_file_identity_valid
    target_free_forecast_valid = target_free_forecast_valid and forecast_file_identity_valid
    pre_score_artifact_chain_valid = (
        status_chain_valid and evidence_chain_valid and target_free_forecast_valid
    )
    return {
        "status_receipt_count": len(statuses),
        "status_receipts": statuses,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "summary_status": None if summary is None else summary.get("status"),
        "receipt_root_identity_valid": receipt_root_identity_valid,
        "file_identity_revalidation_valid": file_identity_revalidation_valid,
        "status_chain_file_identity_valid": status_chain_file_identity_valid,
        "all_48_status_receipts_present_and_linked": bool(
            status_chain_valid
        ),
        "all_48_fit_and_baseline_statuses_complete": bool(
            all_statuses_complete and summary_links_valid and receipt_root_identity_valid
        ),
        "evidence_manifest_sha256": evidence_manifest_sha256,
        "training_evidence_chain_valid": evidence_chain_valid,
        "target_free_forecast_sha256": target_free_forecast_sha256,
        "target_free_forecast_chain_valid": target_free_forecast_valid,
        "pre_score_artifact_chain_valid": pre_score_artifact_chain_valid,
        "postfit_target_gate_opened": (
            False if summary is None else summary.get("postfit_target_gate_opened") is True
        ),
        "problems": problems,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt-directory", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--approval-record-sha256", required=True)
    args = parser.parse_args(argv)

    actual_args = list(sys.argv[1:] if argv is None else argv)
    expected_args = [
        "--manifest",
        args.manifest,
        "--manifest-sha256",
        args.manifest_sha256,
        "--receipt-directory",
        args.receipt_directory,
        "--approval-record",
        args.approval_record,
        "--approval-record-sha256",
        args.approval_record_sha256,
    ]
    _require_exact_vector(actual_args, expected_args)

    manifest_path = Path(args.manifest)
    receipt_directory = Path(args.receipt_directory)
    approval_path = Path(args.approval_record)
    if not all(
        path.is_absolute()
        and str(path) == text
        for path, text in (
            (manifest_path, args.manifest),
            (receipt_directory, args.receipt_directory),
            (approval_path, args.approval_record),
        )
    ):
        raise WatchdogError(
            "manifest, receipt-directory, and approval-record arguments must be "
            "absolute canonical paths"
        )

    manifest, manifest_sha256, approval, anchors = _production_preflight(
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
        receipt_directory=receipt_directory,
        approval_path=approval_path,
        approval_sha256=args.approval_record_sha256,
        actual_watchdog_args=actual_args,
    )
    try:
        anchors.verify(require_unused=True)
        attestation = _current_parent_attestation(actual_args)
        launch = _build_campaign_command(
            manifest_path, manifest_sha256, receipt_directory
        )
        runtime_identity = anchors.root_identity
        receipt_info = os.fstat(anchors.receipt_root_fd)
        watchdog_claim_parent_info = os.fstat(anchors.watchdog_claim_parent_fd)
        scorer_claim_parent_info = os.fstat(
            anchors.directories[_SCORER_CLAIM_PARENT_RELATIVE]
        )
        identity: dict[str, object] = {
            "protocol_id": manifest["protocol_id"],
            "run_id": manifest["run_id"],
            "manifest_sha256": manifest_sha256,
            "approval_record_sha256": args.approval_record_sha256,
            "reviewed_git_head": approval["reviewed_git_head"],
            "repository_root_realpath": str(anchors.root_path),
            "receipt_root_relative": _RECEIPT_ROOT_RELATIVE,
            "repository_root_device": runtime_identity[0],
            "repository_root_inode": runtime_identity[1],
            "receipt_root_device": receipt_info.st_dev,
            "receipt_root_inode": receipt_info.st_ino,
            "watchdog_claim_parent_device": watchdog_claim_parent_info.st_dev,
            "watchdog_claim_parent_inode": watchdog_claim_parent_info.st_ino,
            "scorer_claim_parent_device": scorer_claim_parent_info.st_dev,
            "scorer_claim_parent_inode": scorer_claim_parent_info.st_ino,
            "watchdog_attestation": attestation,
            "watchdog_image_path": attestation["observed_python_app_image_path"],
            "watchdog_image_sha256": attestation["observed_python_app_image_sha256"],
        }
        result = _supervise_command(
            launch_vector=launch,
            observed_executable_path=Path(PINNED_PYTHON_APP),
            observed_executable_sha256=PINNED_PYTHON_APP_SHA256,
            claim_path=_expected_project_claim_path(_REPO_ROOT, RUN_ID),
            receipt_path=receipt_directory / "watchdog-terminal-receipt.json",
            run_identity=identity,
            claim_project_root=_REPO_ROOT,
            preflight_anchors=anchors,
        )
        print(
            json.dumps(
                {
                    "status": result["receipt"]["status"],
                    "receipt_path": str(result["receipt_path"]),
                    "receipt_sha256": result["receipt_sha256"],
                    "terminal_readback_ack_path": str(
                        result["terminal_readback_ack_path"]
                    ),
                    "terminal_readback_ack_sha256": result[
                        "terminal_readback_ack_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0 if result["receipt"]["status"] == "completed" else 2
    finally:
        anchors.close()


if __name__ == "__main__":  # pragma: no cover - production entry is not run in tests.
    try:
        raise SystemExit(_main())
    except (WatchdogError, OSError, psutil.Error) as error:
        print(f"ABC6 watchdog failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error

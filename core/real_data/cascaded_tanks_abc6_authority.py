"""Sealed supervised-child authority primitive for the ABC6 synthetic run.

This file is deliberately not wired into a production entrypoint. It supplies
the one-use capability types and bounded child bootstrap for later integrated
review; source pins and launch authorization remain fail-closed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import inspect
import json
import os
import re
import secrets
import selectors
import socket
import stat
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final


MAX_FRAME_BYTES: Final = 64 * 1024
MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
MAX_IMAGE_BYTES: Final = 512 * 1024 * 1024
GRANT_TTL_NS: Final = 5_000_000_000
GRANT_WAIT_NS: Final = 5_000_000_000
GRANT_RECORD: Final = "watchdog-child-grant.json"
_PRECLAIM_SCAN_ORDER: Final = (
    "root-open",
    "execution-preclaim",
    "claim-entry",
    "pre-durable-claim",
)
CASE_COUNT: Final = 24
_SHA = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_AUTHORITY_SEAL = object()
_SESSION_SEAL = object()
_CASE_SEAL = object()
_SCORE_SEAL = object()
_HANDOFF_SEAL = object()
_PENDING_SEAL = object()

# Frozen candidate source and role contract for the future ABC6 grant.  The
# campaign source appears in external source maps, but its static reviewed map
# omits it to avoid self-hash recursion.
ABC6_CAMPAIGN_SOURCE_PATH: Final = (
    "core/real_data/cascaded_tanks_abc6_campaign_fit.py"
)
ABC6_MANIFEST_SOURCE_PATHS: Final = (
    "core/abc_smc_reference.py",
    "core/real_data/__init__.py",
    "core/real_data/cascaded_tanks_abc6_authority.py",
    ABC6_CAMPAIGN_SOURCE_PATH,
    "core/real_data/cascaded_tanks_abc6_cases.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_receipt_io.py",
    "core/real_data/cascaded_tanks_abc6_replay.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
    "core/real_data/cascaded_tanks_abc6_training.py",
    "core/real_data/cascaded_tanks_models.py",
    "core/real_data/cascaded_tanks_pattern_search.py",
    "core/real_data/cascaded_tanks_synthetic_abc.py",
    "reports/cascaded-tanks-six-parameter-synthetic-abc-proposal-2026-09-28.md",
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "scripts/watch_cascaded_tanks_abc6.py",
)
ABC6_STATIC_REVIEWED_SOURCE_PATHS: Final = tuple(
    path for path in ABC6_MANIFEST_SOURCE_PATHS if path != ABC6_CAMPAIGN_SOURCE_PATH
)
ABC6_ROLE_CONTRACT: Final = (
    (
        "runner",
        "scripts/run_cascaded_tanks_abc6_synthetic.py",
        "run_cascaded_tanks_abc6_synthetic",
    ),
    (
        "campaign",
        ABC6_CAMPAIGN_SOURCE_PATH,
        "run_abc6_training_campaign_with_evidence",
    ),
    (
        "case",
        "core/real_data/cascaded_tanks_abc6_cases.py",
        "get_training_case_data",
    ),
    (
        "scorer",
        "core/real_data/cascaded_tanks_abc6_scoring.py",
        "score_deferred_abc6_synthetic",
    ),
)


class ABC6AuthorityError(RuntimeError):
    """The child grant, identity, phase, or handoff failed closed."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _is_sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _relative(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError(f"{label} is not a canonical contained path")
    return value


def _function(value: object, label: str) -> str:
    if type(value) is not str or not value.isidentifier():
        raise ValueError(f"{label} must be a function name")
    return value


def _flags(directory: bool = False) -> int:
    result = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    return result | (getattr(os, "O_DIRECTORY", 0) if directory else 0)


@dataclass(frozen=True, slots=True)
class _FileId:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> "_FileId":
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Process facts only; creating this value does not create authority."""

    pid: int
    start: str
    vector: tuple[str, ...]
    image_path: str
    image_sha256: str
    image_id: _FileId


@dataclass(frozen=True, slots=True)
class GrantRecordSnapshot:
    """Publication-time identity retained before a child grant is delivered."""

    record_sha256: str
    grant_sha256: str
    grant_fd: int
    file_identity: _FileId
    bindings: "ABC6AuthorityBindings"


@dataclass(frozen=True, slots=True)
class _ReceivedGrantRecord:
    """Child-receipt identity retained for every supervised preclaim check."""

    record_sha256: str
    file_identity: _FileId
    file_mode: int
    regular_file: bool


def _process_start(pid: int) -> str:
    if sys.platform == "darwin":
        class BsdInfo(ctypes.Structure):
            _fields_ = [
                ("flags", ctypes.c_uint32), ("status", ctypes.c_uint32), ("xstatus", ctypes.c_uint32),
                ("pid", ctypes.c_uint32), ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32),
                ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32), ("rgid", ctypes.c_uint32),
                ("svuid", ctypes.c_uint32), ("svgid", ctypes.c_uint32), ("rfu", ctypes.c_uint32),
                ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32), ("nfiles", ctypes.c_uint32),
                ("pgid", ctypes.c_uint32), ("pjobc", ctypes.c_uint32), ("tdev", ctypes.c_uint32),
                ("tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32), ("sec", ctypes.c_uint64),
                ("usec", ctypes.c_uint64),
            ]

        try:
            lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
            call = lib.proc_pidinfo
            call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
            call.restype = ctypes.c_int
            result = BsdInfo()
            size = call(pid, 3, 0, ctypes.byref(result), ctypes.sizeof(result))
        except (OSError, AttributeError) as error:
            raise ABC6AuthorityError("macOS process start identity unavailable") from error
        if size != ctypes.sizeof(result) or result.pid != pid:
            raise ABC6AuthorityError("macOS process start identity unavailable")
        return f"darwin:{result.sec}:{result.usec}"
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/stat", "rb") as stream:
                raw = stream.read(16 * 1024)
            fields = raw[raw.rfind(b")") + 2 :].split()
            return "linux:" + fields[19].decode("ascii")
        except (OSError, IndexError, UnicodeDecodeError) as error:
            raise ABC6AuthorityError("Linux process start identity unavailable") from error
    raise ABC6AuthorityError("runtime has no reviewed process-start identity provider")


def _process_vector(pid: int) -> tuple[str, ...]:
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                raw = stream.read(256 * 1024)
            if not raw or raw[-1:] != b"\0":
                raise ValueError("truncated process vector")
            values = tuple(os.fsdecode(part) for part in raw[:-1].split(b"\0"))
            if any(not part for part in values):
                raise ValueError("empty vector argument")
            return values
        except (OSError, ValueError) as error:
            raise ABC6AuthorityError("Linux process vector unavailable") from error
    if sys.platform == "darwin":
        # KERN_PROCARGS2: argc, exec path, kernel padding, then argc argv strings.
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            sysctl = libc.sysctl
            sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
            sysctl.restype = ctypes.c_int
            mib = (ctypes.c_int * 3)(1, 49, pid)
            size = ctypes.c_size_t()
            if sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or not 8 <= size.value <= 256 * 1024:
                raise OSError(ctypes.get_errno(), "KERN_PROCARGS2 size unavailable")
            buffer = ctypes.create_string_buffer(size.value)
            actual = ctypes.c_size_t(size.value)
            if sysctl(mib, 3, buffer, ctypes.byref(actual), None, 0) != 0:
                raise OSError(ctypes.get_errno(), "KERN_PROCARGS2 read failed")
            raw = buffer.raw[: actual.value]
            argc = struct.unpack_from("i", raw)[0]
            if argc <= 0 or argc > 4096:
                raise ValueError("invalid argc")
            cursor = 4
            while cursor < len(raw) and raw[cursor] == 0:
                cursor += 1
            end = raw.find(b"\0", cursor)
            if end < 0:
                raise ValueError("missing executable path")
            cursor = end + 1
            while cursor < len(raw) and raw[cursor] == 0:
                cursor += 1
            values = []
            for _ in range(argc):
                end = raw.find(b"\0", cursor)
                if end < 0:
                    raise ValueError("truncated vector")
                value = os.fsdecode(raw[cursor:end])
                if not value:
                    raise ValueError("empty vector argument")
                values.append(value)
                cursor = end + 1
            return tuple(values)
        except (OSError, ValueError, struct.error, AttributeError) as error:
            raise ABC6AuthorityError("macOS process vector unavailable") from error
    raise ABC6AuthorityError("runtime has no reviewed process-vector identity provider")


def _image_path(pid: int) -> str:
    if sys.platform == "darwin":
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
            call = lib.proc_pidpath
            call.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            call.restype = ctypes.c_int
            buffer = ctypes.create_string_buffer(4096)
            if call(pid, buffer, len(buffer)) <= 0:
                raise OSError(ctypes.get_errno(), "proc_pidpath failed")
            path = os.fsdecode(buffer.value)
        except (OSError, AttributeError) as error:
            raise ABC6AuthorityError("macOS executable image unavailable") from error
    elif sys.platform.startswith("linux"):
        try:
            path = os.readlink(f"/proc/{pid}/exe")
        except OSError as error:
            raise ABC6AuthorityError("Linux executable image unavailable") from error
        if path.endswith(" (deleted)"):
            raise ABC6AuthorityError("deleted process images are rejected")
    else:
        raise ABC6AuthorityError("runtime has no reviewed executable image provider")
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ABC6AuthorityError("process image path is not canonical")
    return path


def _vector_image_path(vector: tuple[str, ...]) -> str:
    """Resolve argv[0] as the child-observed runtime image identity.

    On Darwin a framework launcher can exec the Python.app binary after the
    supervisor has observed the launcher image. Keep that launch observation
    separate from the child's runtime image instead of requiring their paths
    or hashes to be equal.
    """

    if not vector or type(vector[0]) is not str or not os.path.isabs(vector[0]):
        raise ABC6AuthorityError("child executable vector has no absolute image argument")
    path = os.path.realpath(vector[0])
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ABC6AuthorityError("child runtime image path is not canonical")
    return path


def _read_fd(fd: int, maximum: int, label: str) -> tuple[bytes, _FileId]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > maximum:
        raise ABC6AuthorityError(f"{label} is not a bounded regular file")
    identity = _FileId.of(before)
    chunks: list[bytes] = []
    total = 0
    while total <= maximum:
        chunk = os.read(fd, min(64 * 1024, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    after = os.fstat(fd)
    if total > maximum or total != before.st_size or _FileId.of(after) != identity:
        raise ABC6AuthorityError(f"{label} changed while being read")
    return b"".join(chunks), identity


def _hash_path(path: str, maximum: int, label: str) -> tuple[str, _FileId]:
    try:
        fd = os.open(path, _flags())
    except OSError as error:
        raise ABC6AuthorityError(f"{label} cannot be opened without following links") from error
    try:
        raw, identity = _read_fd(fd, maximum, label)
        return _sha(raw), identity
    finally:
        os.close(fd)


def _open_root(path: str) -> int:
    if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ABC6AuthorityError("physical root must be canonical and absolute")
    current = os.open(os.sep, _flags(directory=True))
    try:
        for part in path.split(os.sep):
            if not part:
                continue
            next_fd = os.open(part, _flags(directory=True), dir_fd=current)
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                raise ABC6AuthorityError("physical root contains a non-directory")
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _open_beneath(root_fd: int, relative: str, *, directory: bool) -> int:
    parts = _relative(relative, "root-relative path").split("/")
    current = os.dup(root_fd)
    try:
        for part in parts[:-1] if not directory else parts:
            next_fd = os.open(part, _flags(directory=True), dir_fd=current)
            os.close(current)
            current = next_fd
        if directory:
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise ABC6AuthorityError("receipt root is not a directory")
            return current
        fd = os.open(parts[-1], _flags(), dir_fd=current)
        os.close(current)
        return fd
    except BaseException:
        os.close(current)
        raise


def _hash_beneath(root_fd: int, relative: str) -> str:
    fd = _open_beneath(root_fd, relative, directory=False)
    try:
        raw, _ = _read_fd(fd, MAX_SOURCE_BYTES, "pinned source")
        return _sha(raw)
    finally:
        os.close(fd)


@dataclass(frozen=True, slots=True)
class ABC6AuthorityBindings:
    """Constructible identity data, explicitly not an authority capability."""

    protocol_id: str
    run_id: str
    manifest_sha256: str
    approval_sha256: str
    review_sha256: str
    physical_root: str
    receipt_root_relative: str
    root_device: int
    root_inode: int
    receipt_device: int
    receipt_inode: int
    reviewed_git_head: str
    watchdog_claim_sha256: str
    watchdog_pid: int
    watchdog_start: str
    child_pid: int
    child_start: str
    child_vector: tuple[str, ...]
    child_launch_image_path: str
    child_launch_image_sha256: str
    child_image_path: str
    child_image_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    runtime_hashes: tuple[tuple[str, str], ...]
    campaign_claim_relative: str
    runner_source_path: str
    runner_function: str
    campaign_source_path: str
    campaign_function: str
    case_source_path: str
    case_function: str
    scorer_source_path: str
    scorer_function: str

    def __post_init__(self) -> None:
        for name, value in (("protocol_id", self.protocol_id), ("run_id", self.run_id)):
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise ValueError(f"{name} is not a safe identity")
        for name in ("manifest_sha256", "approval_sha256", "review_sha256", "watchdog_claim_sha256", "child_launch_image_sha256", "child_image_sha256"):
            if not _is_sha(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if type(self.reviewed_git_head) is not str or _GIT.fullmatch(self.reviewed_git_head) is None:
            raise ValueError("reviewed Git HEAD must be a lowercase commit hash")
        if not os.path.isabs(self.physical_root) or os.path.normpath(self.physical_root) != self.physical_root:
            raise ValueError("physical root must be canonical and absolute")
        _relative(self.receipt_root_relative, "receipt root")
        _relative(self.campaign_claim_relative, "campaign claim path")
        if self.receipt_root_relative != f"artifacts/cascaded_tanks_abc6_synthetic/{self.run_id}/receipts":
            raise ValueError("receipt root must follow the fixed run-relative contract")
        if self.campaign_claim_relative != f"artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/{self.run_id}.claim":
            raise ValueError("campaign claim must use its fixed root-relative path")
        for name in ("root_device", "root_inode", "receipt_device", "receipt_inode", "watchdog_pid", "child_pid"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.watchdog_start or not self.child_start:
            raise ValueError("watchdog and child start identities are required")
        if not isinstance(self.child_vector, tuple) or not self.child_vector or any(type(arg) is not str for arg in self.child_vector):
            raise ValueError("child vector must be a nonempty tuple of text arguments")
        for name in ("child_launch_image_path", "child_image_path"):
            path = getattr(self, name)
            if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path) != path:
                raise ValueError(f"{name} must be canonical and absolute")
        for role in ("runner", "campaign", "case", "scorer"):
            _relative(getattr(self, f"{role}_source_path"), f"{role} source path")
            _function(getattr(self, f"{role}_function"), f"{role} function")
        if tuple(
            (
                role,
                getattr(self, f"{role}_source_path"),
                getattr(self, f"{role}_function"),
            )
            for role in ("runner", "campaign", "case", "scorer")
        ) != ABC6_ROLE_CONTRACT:
            raise ValueError("role bindings differ from the frozen ABC6 role contract")
        for name, entries in (("source_hashes", self.source_hashes), ("runtime_hashes", self.runtime_hashes)):
            if not isinstance(entries, tuple) or not entries:
                raise ValueError(f"{name} must be a nonempty tuple")
            keys = []
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2 or type(entry[0]) is not str or not _is_sha(entry[1]):
                    raise ValueError(f"{name} entries must be (name, sha256) pairs")
                if name == "source_hashes":
                    _relative(entry[0], "source hash path")
                keys.append(entry[0])
            if keys != sorted(set(keys)):
                raise ValueError(f"{name} must be sorted and unique")
        source_paths = {path for path, _ in self.source_hashes}
        if source_paths != set(ABC6_MANIFEST_SOURCE_PATHS):
            raise ValueError("source hashes must equal the frozen sixteen-path ABC6 source roster")
        runtime = dict(self.runtime_hashes)
        if (
            runtime.get("child_launch_image_sha256") != self.child_launch_image_sha256
            or runtime.get("child_image_sha256") != self.child_image_sha256
            or runtime.get("python_version_sha256") != _sha(sys.version.encode())
        ):
            raise ValueError("runtime identities do not match both child images and the exact Python version")

    def payload(self) -> dict[str, object]:
        out = {name: getattr(self, name) for name in self.__dataclass_fields__}
        out["child_vector"] = list(self.child_vector)
        out["source_hashes"] = [list(item) for item in self.source_hashes]
        out["runtime_hashes"] = [list(item) for item in self.runtime_hashes]
        return out

    @classmethod
    def from_payload(cls, value: object) -> "ABC6AuthorityBindings":
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ABC6AuthorityError("grant bindings have an unexpected schema")
        try:
            copied = dict(value)
            copied["child_vector"] = tuple(copied["child_vector"])
            copied["source_hashes"] = tuple(tuple(x) for x in copied["source_hashes"])
            copied["runtime_hashes"] = tuple(tuple(x) for x in copied["runtime_hashes"])
            return cls(**copied)
        except (TypeError, ValueError, KeyError) as error:
            raise ABC6AuthorityError("grant bindings failed validation") from error


def _capture(pid: int) -> ProcessIdentity:
    path = _image_path(pid)
    digest, image_id = _hash_path(path, MAX_IMAGE_BYTES, "process image")
    return ProcessIdentity(pid, _process_start(pid), _process_vector(pid), path, digest, image_id)


def _require_unnamed_socket(sock: socket.socket) -> None:
    try:
        info = os.fstat(sock.fileno())
        if not stat.S_ISSOCK(info.st_mode) or sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise ABC6AuthorityError("bootstrap FD is not a stream socket")
        local, peer = sock.getsockname(), sock.getpeername()
    except OSError as error:
        raise ABC6AuthorityError("bootstrap FD is not a connected anonymous socketpair") from error
    if not isinstance(local, (str, bytes)) or not isinstance(peer, (str, bytes)) or local or peer:
        raise ABC6AuthorityError("path-bound sockets are rejected for grant delivery")


def _peer_pid(sock: socket.socket) -> int:
    try:
        if sys.platform == "darwin":
            return struct.unpack("i", sock.getsockopt(0, 2, struct.calcsize("i")))[0]
        if sys.platform.startswith("linux") and hasattr(socket, "SO_PEERCRED"):
            return struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))[0]
    except (OSError, struct.error) as error:
        raise ABC6AuthorityError("anonymous grant peer PID is unavailable") from error
    raise ABC6AuthorityError("runtime has no reviewed peer PID provider")


def _watchdog_callsite() -> None:
    frame = inspect.currentframe()
    caller = None if frame is None else frame.f_back
    if caller is not None and caller.f_code.co_name in {
        "_new_watchdog_grant_channel",
        "publish",
    }:
        caller = caller.f_back
    if caller is None or caller.f_code.co_name != "_supervise_command" or not caller.f_code.co_filename.endswith(os.path.join("scripts", "watch_cascaded_tanks_abc6.py")):
        raise ABC6AuthorityError("only the watchdog's supervised-child seam may issue grants")


class _PendingGrant:
    __slots__ = (
        "_seal", "_writer", "_child", "_child_fd", "_parent", "_used",
        "_lock", "_publication_snapshot",
    )

    def __init__(self, seal: object, writer: socket.socket, child: socket.socket, parent: ProcessIdentity):
        if seal is not _PENDING_SEAL:
            raise TypeError("grant channels are issued only by the watchdog")
        self._seal, self._writer, self._child, self._parent = seal, writer, child, parent
        self._child_fd = child.fileno()
        self._used = False
        self._lock = threading.Lock()
        self._publication_snapshot: GrantRecordSnapshot | None = None

    @property
    def child_fd(self) -> int:
        if self._seal is not _PENDING_SEAL or self._child.fileno() < 0:
            raise ABC6AuthorityError("child grant descriptor is closed")
        return self._child.fileno()

    def close_child_copy(self) -> None:
        self._child.close()

    @property
    def publication_snapshot(self) -> GrantRecordSnapshot | None:
        """Retain the durable snapshot even if delivery later fails."""

        return self._publication_snapshot

    def publish(self, bindings: ABC6AuthorityBindings) -> GrantRecordSnapshot:
        _watchdog_callsite()
        with self._lock:
            if self._used:
                raise ABC6AuthorityError("grant channel is one-use")
            self._used = True
        if self._seal is not _PENDING_SEAL or not isinstance(bindings, ABC6AuthorityBindings):
            raise ABC6AuthorityError("grant bindings are invalid")
        if (bindings.watchdog_pid, bindings.watchdog_start) != (self._parent.pid, self._parent.start):
            raise ABC6AuthorityError("watchdog identity differs from the grant")
        child = _capture(bindings.child_pid)
        if (
            child.pid == self._parent.pid
            or child.start != bindings.child_start
            or child.vector != bindings.child_vector
            or child.image_path != bindings.child_image_path
            or child.image_sha256 != bindings.child_image_sha256
        ):
            raise ABC6AuthorityError("started child differs from the frozen runtime identity")
        launch_digest, _launch_id = _hash_path(
            bindings.child_launch_image_path,
            MAX_IMAGE_BYTES,
            "frozen child launch image",
        )
        if launch_digest != bindings.child_launch_image_sha256:
            raise ABC6AuthorityError("frozen child launch image digest changed")
        runtime_path = _vector_image_path(child.vector)
        if runtime_path != bindings.child_image_path:
            raise ABC6AuthorityError("child vector resolves to a different runtime image")
        runtime_digest, _ = _hash_path(runtime_path, MAX_IMAGE_BYTES, "child runtime image")
        if runtime_digest != bindings.child_image_sha256:
            raise ABC6AuthorityError("child runtime image digest differs from its frozen identity")
        secret = secrets.token_bytes(32)
        digest = _sha(secret)
        grant_fd = self._child_fd
        snapshot = _write_grant_record(bindings, digest, grant_fd)
        # This assignment deliberately precedes any frame construction or
        # socket operation. A failed delivery must not erase durable evidence.
        self._publication_snapshot = snapshot
        now = time.monotonic_ns()
        frame = _json({
            "schema_version": 1,
            "issued_monotonic_ns": now,
            "expires_monotonic_ns": now + GRANT_TTL_NS,
            "secret_hex": secret.hex(),
            "grant_fd": grant_fd,
            "bindings": bindings.payload(),
        })
        if not frame or len(frame) > MAX_FRAME_BYTES:
            raise ABC6AuthorityError("grant frame exceeds its fixed size bound")
        wire = struct.pack("!I", len(frame)) + frame
        self._writer.setblocking(False)
        sent = 0
        deadline = time.monotonic_ns() + 1_000_000_000
        while sent < len(wire):
            try:
                sent += self._writer.send(wire[sent:])
            except BlockingIOError:
                remain = max(0.0, (deadline - time.monotonic_ns()) / 1e9)
                with selectors.DefaultSelector() as selector:
                    selector.register(self._writer, selectors.EVENT_WRITE)
                    if remain == 0 or not selector.select(remain):
                        raise ABC6AuthorityError("grant write exceeded its deadline")
            except OSError as error:
                raise ABC6AuthorityError("anonymous grant channel failed") from error
        try:
            self._writer.shutdown(socket.SHUT_WR)
        except OSError as error:
            raise ABC6AuthorityError("grant channel could not be sealed") from error
        return snapshot

    def close(self) -> None:
        self._writer.close()
        self._child.close()


def _new_watchdog_grant_channel() -> _PendingGrant:
    _watchdog_callsite()
    parent = _capture(os.getpid())
    try:
        child, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        _require_unnamed_socket(child)
        _require_unnamed_socket(writer)
        child.set_inheritable(True)
        writer.set_inheritable(False)
        return _PendingGrant(_PENDING_SEAL, writer, child, parent)
    except OSError as error:
        raise ABC6AuthorityError("could not create anonymous grant socketpair") from error


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write")
        view = view[count:]


def _write_grant_record(
    bindings: ABC6AuthorityBindings, digest: str, grant_fd: int
) -> GrantRecordSnapshot:
    root_fd = _open_root(bindings.physical_root)
    receipt_fd = -1
    try:
        root = os.fstat(root_fd)
        if (root.st_dev, root.st_ino) != (bindings.root_device, bindings.root_inode):
            raise ABC6AuthorityError("grant root identity changed")
        receipt_fd = _open_beneath(root_fd, bindings.receipt_root_relative, directory=True)
        receipt = os.fstat(receipt_fd)
        if (receipt.st_dev, receipt.st_ino) != (bindings.receipt_device, bindings.receipt_inode) or os.listdir(receipt_fd):
            raise ABC6AuthorityError("grant receipt root changed or is already nonempty")
        metadata = _json({
            "schema_version": 1,
            "grant_sha256": digest,
            "grant_fd": grant_fd,
            "bindings_sha256": _sha(_json(bindings.payload())),
            "bindings": bindings.payload(),
        })
        fd = os.open(
            GRANT_RECORD,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=receipt_fd,
        )
        try:
            initial_identity = _FileId.of(os.fstat(fd))
            _write_all(fd, metadata)
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            raw, file_identity = _read_fd(fd, MAX_FRAME_BYTES, "published grant receipt")
            named_identity = _FileId.of(
                os.stat(GRANT_RECORD, dir_fd=receipt_fd, follow_symlinks=False)
            )
            if (
                raw != metadata
                or _sha(raw) != _sha(metadata)
                or _FileId.of(os.fstat(fd)) != file_identity
                or named_identity != file_identity
                or file_identity.device != initial_identity.device
                or file_identity.inode != initial_identity.inode
            ):
                raise ABC6AuthorityError("published grant receipt could not be identity-bound")
            os.fsync(receipt_fd)
        finally:
            os.close(fd)
        snapshot = GrantRecordSnapshot(
            record_sha256=_sha(metadata),
            grant_sha256=digest,
            grant_fd=grant_fd,
            file_identity=file_identity,
            bindings=bindings,
        )
    except OSError as error:
        raise ABC6AuthorityError("one-use grant digest receipt could not be published") from error
    finally:
        if receipt_fd >= 0:
            os.close(receipt_fd)
        os.close(root_fd)
    return snapshot


def verify_grant_record_snapshot(snapshot: GrantRecordSnapshot) -> None:
    """Re-read a published grant under pinned no-follow root and compare it."""

    if not isinstance(snapshot, GrantRecordSnapshot):
        raise ABC6AuthorityError("grant publication snapshot is unavailable")
    bindings = snapshot.bindings
    root_fd = _open_root(bindings.physical_root)
    receipt_fd = -1
    try:
        root = os.fstat(root_fd)
        if (root.st_dev, root.st_ino) != (bindings.root_device, bindings.root_inode):
            raise ABC6AuthorityError("grant root identity changed after publication")
        receipt_fd = _open_beneath(root_fd, bindings.receipt_root_relative, directory=True)
        receipt = os.fstat(receipt_fd)
        if (receipt.st_dev, receipt.st_ino) != (
            bindings.receipt_device,
            bindings.receipt_inode,
        ):
            raise ABC6AuthorityError("grant receipt root changed after publication")
        fd = os.open(GRANT_RECORD, _flags(), dir_fd=receipt_fd)
        try:
            raw, file_identity = _read_fd(fd, MAX_FRAME_BYTES, "published grant receipt")
            named_identity = _FileId.of(
                os.stat(GRANT_RECORD, dir_fd=receipt_fd, follow_symlinks=False)
            )
        finally:
            os.close(fd)
        if (
            _sha(raw) != snapshot.record_sha256
            or file_identity != snapshot.file_identity
            or named_identity != snapshot.file_identity
        ):
            raise ABC6AuthorityError("published grant receipt changed after delivery")
    except OSError as error:
        raise ABC6AuthorityError("published grant receipt is unavailable after delivery") from error
    finally:
        if receipt_fd >= 0:
            os.close(receipt_fd)
        os.close(root_fd)


def _read_exact(sock: socket.socket, size: int, deadline: int) -> bytes:
    result = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(sock, selectors.EVENT_READ)
        while len(result) < size:
            remaining = (deadline - time.monotonic_ns()) / 1e9
            if remaining <= 0 or not selector.select(remaining):
                raise ABC6AuthorityError("grant was absent or delayed past the acquisition deadline")
            try:
                chunk = sock.recv(min(64 * 1024, size - len(result)))
            except BlockingIOError:
                continue
            if not chunk:
                raise ABC6AuthorityError("grant frame is truncated")
            result.extend(chunk)
    return bytes(result)


def _read_frame(fd: int) -> tuple[dict[str, object], socket.socket, int]:
    if type(fd) is not int or fd < 0:
        raise ABC6AuthorityError("authority requires an inherited anonymous socket FD")
    try:
        sock = socket.socket(fileno=os.dup(fd))
    except OSError as error:
        raise ABC6AuthorityError("authority FD is not a socket") from error
    try:
        _require_unnamed_socket(sock)
        peer = _peer_pid(sock)
        sock.setblocking(False)
        deadline = time.monotonic_ns() + GRANT_WAIT_NS
        (size,) = struct.unpack("!I", _read_exact(sock, 4, deadline))
        if size <= 0 or size > MAX_FRAME_BYTES:
            raise ABC6AuthorityError("grant frame length exceeds its fixed bound")
        raw = _read_exact(sock, size, deadline)
        with selectors.DefaultSelector() as selector:
            selector.register(sock, selectors.EVENT_READ)
            remain = (deadline - time.monotonic_ns()) / 1e9
            if remain <= 0 or not selector.select(remain):
                raise ABC6AuthorityError("grant writer did not seal the single-frame stream")
        if sock.recv(1) != b"":
            raise ABC6AuthorityError("grant stream contains trailing data")
        try:
            frame = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ABC6AuthorityError("grant frame is malformed") from error
        if type(frame) is not dict or _json(frame) != raw or set(frame) != {"schema_version", "issued_monotonic_ns", "expires_monotonic_ns", "secret_hex", "grant_fd", "bindings"}:
            raise ABC6AuthorityError("grant frame schema or canonical encoding is invalid")
        if type(frame["schema_version"]) is not int or frame["schema_version"] != 1:
            raise ABC6AuthorityError("unsupported grant frame version")
        issued, expires, now = frame["issued_monotonic_ns"], frame["expires_monotonic_ns"], time.monotonic_ns()
        if type(issued) is not int or type(expires) is not int or issued > now or expires < now or expires - issued != GRANT_TTL_NS:
            raise ABC6AuthorityError("grant has expired or has an invalid issue time")
        if type(frame["secret_hex"]) is not str or re.fullmatch(r"[0-9a-f]{64}", frame["secret_hex"]) is None:
            raise ABC6AuthorityError("grant secret is malformed")
        bindings = ABC6AuthorityBindings.from_payload(frame["bindings"])
        if type(frame["grant_fd"]) is not int or frame["grant_fd"] != fd:
            raise ABC6AuthorityError("grant frame names a different inherited FD")
        if bindings.watchdog_pid != peer:
            raise ABC6AuthorityError("anonymous grant peer does not match watchdog PID")
        frame["bindings"] = bindings
        return frame, sock, peer
    except BaseException:
        sock.close()
        raise


class _State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.runner_entry_consumed = False
        self.training_started = False
        self.issued: set[int] = set()
        self.consumed: set[int] = set()
        self.activation_attempted = False
        self.activated = False
        self.score_consumed = False
        self.handoff: ABC6RunnerHandoff | None = None


class ABC6LaunchAuthority:
    """Opaque in-memory capability tied to one live child process."""

    __slots__ = (
        "_seal",
        "_bindings",
        "_grant_digest",
        "_grant_fd",
        "_grant_record",
        "_root_fd",
        "_receipt_fd",
        "_identity",
        "_state",
    )

    def __new__(cls, seal: object = None, *_args: object, **_kwargs: object):
        if cls is not ABC6LaunchAuthority or seal is not _AUTHORITY_SEAL:
            raise TypeError("launch authority is minted only by a verified inherited grant")
        return super().__new__(cls)

    def __init__(
        self,
        seal: object,
        bindings: ABC6AuthorityBindings,
        grant_digest: str,
        grant_fd: int,
        grant_record: _ReceivedGrantRecord,
        root_fd: int,
        receipt_fd: int,
        identity: ProcessIdentity,
    ):
        self._seal, self._bindings, self._grant_digest = seal, bindings, grant_digest
        self._grant_fd = grant_fd
        self._grant_record = grant_record
        self._root_fd, self._receipt_fd, self._identity = root_fd, receipt_fd, identity
        self._state = _State()

    def __reduce__(self):
        raise TypeError("process-bound authority cannot be serialized")

    def _assert_live(self) -> None:
        if self._seal is not _AUTHORITY_SEAL or self._root_fd < 0 or self._receipt_fd < 0:
            raise ABC6AuthorityError("authority capability is closed or invalid")
        if os.getpid() != self._bindings.child_pid:
            raise ABC6AuthorityError("authority cannot cross fork boundaries")
        if _process_start(os.getpid()) != self._bindings.child_start or _process_vector(os.getpid()) != self._bindings.child_vector:
            raise ABC6AuthorityError("child PID/start/vector changed after grant receipt")
        path = _image_path(os.getpid())
        try:
            current = _FileId.of(os.stat(path, follow_symlinks=False))
        except OSError as error:
            raise ABC6AuthorityError("child image identity is unavailable") from error
        if (
            path != self._bindings.child_image_path
            or self._identity.image_path != self._bindings.child_image_path
            or self._identity.image_sha256 != self._bindings.child_image_sha256
            or current != self._identity.image_id
        ):
            raise ABC6AuthorityError("child image changed after grant receipt")
        if _sha(sys.version.encode()) != dict(self._bindings.runtime_hashes)["python_version_sha256"]:
            raise ABC6AuthorityError("Python runtime identity changed")
        if (os.fstat(self._root_fd).st_dev, os.fstat(self._root_fd).st_ino) != (self._bindings.root_device, self._bindings.root_inode):
            raise ABC6AuthorityError("anchored physical root identity changed")
        if (os.fstat(self._receipt_fd).st_dev, os.fstat(self._receipt_fd).st_ino) != (self._bindings.receipt_device, self._bindings.receipt_inode):
            raise ABC6AuthorityError("anchored receipt root identity changed")

    @property
    def grant_digest(self) -> str:
        self._assert_live()
        return self._grant_digest

    def begin_training(self) -> "ABC6TrainingSession":
        self._assert_live()
        _require_role(self, "campaign")
        with self._state.lock:
            if self._state.training_started or self._state.activation_attempted:
                raise ABC6AuthorityError("training phase is one-use")
            self._state.training_started = True
        return ABC6TrainingSession(_SESSION_SEAL, self)

    def _verify_training_preclaim_root(
        self,
        repository_root_fd: int,
        receipt_root_fd: int,
        *,
        repository_root_realpath: str,
        receipt_root_relative: str,
        protocol_id: str,
        run_id: str,
        manifest_sha256: str,
    ) -> None:
        """Verify the campaign's opened root and the exact received grant leaf."""

        self._assert_live()
        bindings = self._bindings
        if (
            repository_root_realpath != bindings.physical_root
            or receipt_root_relative != bindings.receipt_root_relative
            or protocol_id != bindings.protocol_id
            or run_id != bindings.run_id
            or manifest_sha256 != bindings.manifest_sha256
        ):
            raise ABC6AuthorityError(
                "campaign preclaim identity differs from the received grant bindings"
            )
        try:
            root_info = os.fstat(repository_root_fd)
            receipt_info = os.fstat(receipt_root_fd)
            authority_root_info = os.fstat(self._root_fd)
            authority_receipt_info = os.fstat(self._receipt_fd)
        except (OSError, TypeError, ValueError) as error:
            raise ABC6AuthorityError("campaign preclaim root descriptor is unavailable") from error
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or not stat.S_ISDIR(receipt_info.st_mode)
            or (root_info.st_dev, root_info.st_ino)
            != (bindings.root_device, bindings.root_inode)
            or (receipt_info.st_dev, receipt_info.st_ino)
            != (bindings.receipt_device, bindings.receipt_inode)
            or (authority_root_info.st_dev, authority_root_info.st_ino)
            != (root_info.st_dev, root_info.st_ino)
            or (authority_receipt_info.st_dev, authority_receipt_info.st_ino)
            != (receipt_info.st_dev, receipt_info.st_ino)
        ):
            raise ABC6AuthorityError(
                "campaign preclaim descriptors differ from the authority-pinned root"
            )

        try:
            if set(os.listdir(receipt_root_fd)) != {GRANT_RECORD}:
                raise ABC6AuthorityError(
                    "supervised receipt root must contain only the pinned grant leaf"
                )
            fd = os.open(GRANT_RECORD, _flags(), dir_fd=receipt_root_fd)
        except OSError as error:
            raise ABC6AuthorityError(
                "pinned watchdog grant leaf is missing or unsafe"
            ) from error
        try:
            before = os.fstat(fd)
            snapshot = self._grant_record
            if (
                not snapshot.regular_file
                or not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != snapshot.file_mode
            ):
                raise ABC6AuthorityError(
                    "pinned watchdog grant leaf type or mode changed"
                )
            raw, file_identity = _read_fd(
                fd, MAX_FRAME_BYTES, "pinned watchdog grant leaf"
            )
            after = os.fstat(fd)
            named = os.stat(
                GRANT_RECORD,
                dir_fd=receipt_root_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ABC6AuthorityError(
                "pinned watchdog grant leaf could not be read safely"
            ) from error
        finally:
            os.close(fd)

        expected_identity = snapshot.file_identity
        if (
            _FileId.of(after) != expected_identity
            or file_identity != expected_identity
            or _FileId.of(named) != expected_identity
            or not stat.S_ISREG(named.st_mode)
            or stat.S_IMODE(named.st_mode) != snapshot.file_mode
            or _sha(raw) != snapshot.record_sha256
        ):
            raise ABC6AuthorityError(
                "pinned watchdog grant leaf identity or digest changed"
            )

        expected = _json(
            {
                "schema_version": 1,
                "grant_sha256": self._grant_digest,
                "grant_fd": self._grant_fd,
                "bindings_sha256": _sha(_json(bindings.payload())),
                "bindings": bindings.payload(),
            }
        )
        if raw != expected:
            raise ABC6AuthorityError(
                "pinned watchdog grant leaf payload differs from live authority"
            )

    def consume_runner_startup(self) -> None:
        """Consume the one-use authenticated runner entry before preflight."""

        self._assert_live()
        _require_role(self, "runner")
        with self._state.lock:
            if self._state.runner_entry_consumed:
                raise ABC6AuthorityError("runner startup grant is one-use")
            self._state.runner_entry_consumed = True

    def create_runner_handoff(
        self,
        status_receipts: tuple[tuple[int, str, str], ...],
        summary_sha256: str,
        forecast_sha256: str,
    ) -> "ABC6RunnerHandoff":
        """Seal the runner's already-reverified 48 status + summary + forecast digests."""

        self._assert_live()
        _require_role(self, "runner")
        if not isinstance(status_receipts, tuple) or len(status_receipts) != 48:
            raise ABC6AuthorityError("runner handoff requires exactly 48 ordered status digests")
        expected = tuple((i, component) for i in range(CASE_COUNT) for component in ("fit", "baseline"))
        actual: list[tuple[int, str]] = []
        for entry in status_receipts:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ABC6AuthorityError("runner status handoff entry has an invalid shape")
            index, component, digest = entry
            if type(index) is not int or type(component) is not str or not _is_sha(digest):
                raise ABC6AuthorityError("runner status handoff identity or digest is invalid")
            actual.append((index, component))
        if tuple(actual) != expected or not _is_sha(summary_sha256) or not _is_sha(forecast_sha256):
            raise ABC6AuthorityError("runner handoff is incomplete or not in frozen roster order")
        handoff = ABC6RunnerHandoff(_HANDOFF_SEAL, self, status_receipts, summary_sha256, forecast_sha256)
        return handoff

    def activate_scoring(self, handoff: "ABC6RunnerHandoff") -> "ABC6ScoringPermit":
        self._assert_live()
        _require_role(self, "runner")
        with self._state.lock:
            if self._state.activation_attempted:
                raise ABC6AuthorityError("scoring activation is one-use")
            self._state.activation_attempted = True
        if not self._state.training_started or self._state.issued != set(range(CASE_COUNT)) or self._state.consumed != set(range(CASE_COUNT)):
            raise ABC6AuthorityError("scoring requires the one complete 24-case training campaign")
        if not isinstance(handoff, ABC6RunnerHandoff) or handoff._seal is not _HANDOFF_SEAL or handoff._authority is not self:
            raise ABC6AuthorityError("scoring activation requires this child's sealed runner handoff")
        expected = tuple((i, component) for i in range(CASE_COUNT) for component in ("fit", "baseline"))
        if (
            len(handoff._status_receipts) != 48
            or tuple((item[0], item[1]) for item in handoff._status_receipts) != expected
            or any(not _is_sha(item[2]) for item in handoff._status_receipts)
            or not _is_sha(handoff._summary_sha256)
            or not _is_sha(handoff._forecast_sha256)
        ):
            raise ABC6AuthorityError("sealed runner handoff no longer matches its verified digest shape")
        self._state.handoff = handoff
        self._state.activated = True
        return ABC6ScoringPermit(_SCORE_SEAL, self)

    def receipt_metadata(self) -> dict[str, object]:
        self._assert_live()
        handoff = self._state.handoff
        return {
            "schema_version": 1,
            "protocol_id": self._bindings.protocol_id,
            "run_id": self._bindings.run_id,
            "grant_sha256": self._grant_digest,
            "grant_fd": self._grant_fd,
            "manifest_sha256": self._bindings.manifest_sha256,
            "approval_sha256": self._bindings.approval_sha256,
            "review_sha256": self._bindings.review_sha256,
            "physical_root": self._bindings.physical_root,
            "receipt_root_relative": self._bindings.receipt_root_relative,
            "reviewed_git_head": self._bindings.reviewed_git_head,
            "watchdog_claim_sha256": self._bindings.watchdog_claim_sha256,
            "watchdog_pid": self._bindings.watchdog_pid,
            "watchdog_start": self._bindings.watchdog_start,
            "child_pid": self._bindings.child_pid,
            "child_start": self._bindings.child_start,
            "child_vector": list(self._bindings.child_vector),
            "child_launch_image_path": self._bindings.child_launch_image_path,
            "child_launch_image_sha256": self._bindings.child_launch_image_sha256,
            "child_image_path": self._bindings.child_image_path,
            "child_image_sha256": self._bindings.child_image_sha256,
            "source_hashes": [list(pair) for pair in self._bindings.source_hashes],
            "runtime_hashes": [list(pair) for pair in self._bindings.runtime_hashes],
            "training_started": self._state.training_started,
            "issued_case_indices": sorted(self._state.issued),
            "consumed_case_indices": sorted(self._state.consumed),
            "scoring_activation_attempted": self._state.activation_attempted,
            "scoring_activated": self._state.activated,
            "scoring_consumed": self._state.score_consumed,
            "status_receipts": [] if handoff is None else [list(item) for item in handoff._status_receipts],
            "summary_sha256": None if handoff is None else handoff._summary_sha256,
            "forecast_sha256": None if handoff is None else handoff._forecast_sha256,
        }

    def close(self) -> None:
        if self._seal is not _AUTHORITY_SEAL:
            return
        self._seal = None
        for fd in (self._receipt_fd, self._root_fd):
            try:
                os.close(fd)
            except OSError:
                pass
        self._receipt_fd = self._root_fd = -1


def _read_grant_record(
    receipt_fd: int, bindings: ABC6AuthorityBindings, digest: str, grant_fd: int
) -> _ReceivedGrantRecord:
    try:
        fd = os.open(GRANT_RECORD, _flags(), dir_fd=receipt_fd)
    except OSError as error:
        raise ABC6AuthorityError("watchdog grant digest receipt is missing or unsafe") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600:
            raise ABC6AuthorityError("watchdog grant receipt has an unsafe file type or mode")
        raw, file_identity = _read_fd(fd, MAX_FRAME_BYTES, "watchdog grant receipt")
        after = os.fstat(fd)
        named = os.stat(GRANT_RECORD, dir_fd=receipt_fd, follow_symlinks=False)
        if (
            _FileId.of(after) != file_identity
            or _FileId.of(named) != file_identity
            or not stat.S_ISREG(named.st_mode)
            or stat.S_IMODE(named.st_mode) != stat.S_IMODE(before.st_mode)
        ):
            raise ABC6AuthorityError("watchdog grant receipt name changed during read")
    finally:
        os.close(fd)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ABC6AuthorityError("watchdog grant receipt is malformed") from error
    expected = {
        "schema_version": 1,
        "grant_sha256": digest,
        "grant_fd": grant_fd,
        "bindings_sha256": _sha(_json(bindings.payload())),
        "bindings": bindings.payload(),
    }
    if type(value) is not dict or _json(value) != raw or value != expected:
        raise ABC6AuthorityError("watchdog grant receipt does not match the received secret and bindings")
    return _ReceivedGrantRecord(
        record_sha256=_sha(raw),
        file_identity=file_identity,
        file_mode=stat.S_IMODE(before.st_mode),
        regular_file=True,
    )


def receive_child_grant(fd: int) -> ABC6LaunchAuthority:
    """Consume one bounded grant from an inherited anonymous socket descriptor."""

    frame, sock, peer_pid = _read_frame(fd)
    bindings = frame["bindings"]
    assert isinstance(bindings, ABC6AuthorityBindings)
    secret = bytearray.fromhex(str(frame["secret_hex"]))
    root_fd = receipt_fd = -1
    try:
        if bindings.child_pid != os.getpid():
            raise ABC6AuthorityError("grant is bound to another child PID")
        identity = _capture(os.getpid())
        if (
            identity.start != bindings.child_start
            or identity.vector != bindings.child_vector
            or identity.image_path != bindings.child_image_path
            or identity.image_sha256 != bindings.child_image_sha256
            or _vector_image_path(identity.vector) != bindings.child_image_path
        ):
            raise ABC6AuthorityError("live child process differs from its runtime image binding")
        launch_digest, _ = _hash_path(bindings.child_launch_image_path, MAX_IMAGE_BYTES, "parent-observed launch image")
        if launch_digest != bindings.child_launch_image_sha256:
            raise ABC6AuthorityError("parent-observed launch image digest differs from its frozen identity")
        if _process_start(peer_pid) != bindings.watchdog_start:
            raise ABC6AuthorityError("watchdog process start identity changed")
        root_fd = _open_root(bindings.physical_root)
        root = os.fstat(root_fd)
        if (root.st_dev, root.st_ino) != (bindings.root_device, bindings.root_inode):
            raise ABC6AuthorityError("physical root differs from the grant binding")
        receipt_fd = _open_beneath(root_fd, bindings.receipt_root_relative, directory=True)
        receipt = os.fstat(receipt_fd)
        if (receipt.st_dev, receipt.st_ino) != (bindings.receipt_device, bindings.receipt_inode):
            raise ABC6AuthorityError("receipt root differs from the grant binding")
        for source_path, digest in bindings.source_hashes:
            if _hash_beneath(root_fd, source_path) != digest:
                raise ABC6AuthorityError("pinned source identity differs from the grant")
        if _sha(sys.version.encode()) != dict(bindings.runtime_hashes)["python_version_sha256"]:
            raise ABC6AuthorityError("Python runtime differs from the grant")
        grant_record = _read_grant_record(
            receipt_fd, bindings, _sha(bytes(secret)), fd
        )
        authority = ABC6LaunchAuthority(
            _AUTHORITY_SEAL,
            bindings,
            _sha(bytes(secret)),
            fd,
            grant_record,
            root_fd,
            receipt_fd,
            identity,
        )
        root_fd = receipt_fd = -1
        return authority
    except BaseException:
        if root_fd >= 0:
            os.close(root_fd)
        if receipt_fd >= 0:
            os.close(receipt_fd)
        raise
    finally:
        for i in range(len(secret)):
            secret[i] = 0
        sock.close()


class ABC6TrainingSession:
    __slots__ = ("_seal", "_authority", "_receipt_scan_index", "_receipt_scan_failed")

    def __new__(cls, seal: object = None, *_args: object, **_kwargs: object):
        if cls is not ABC6TrainingSession or seal is not _SESSION_SEAL:
            raise TypeError("training sessions require a sealed launch authority")
        return super().__new__(cls)

    def __init__(self, seal: object, authority: ABC6LaunchAuthority):
        if seal is not _SESSION_SEAL or authority._seal is not _AUTHORITY_SEAL:
            raise TypeError("invalid training session")
        self._seal, self._authority = seal, authority
        self._receipt_scan_index = 0
        self._receipt_scan_failed = False

    def verify_preclaim_root(
        self,
        repository_root_fd: int,
        receipt_root_fd: int,
        *,
        scan: str,
        repository_root_realpath: str,
        receipt_root_relative: str,
        protocol_id: str,
        run_id: str,
        manifest_sha256: str,
    ) -> None:
        """Authorize one ordered scan of the sole pinned grant leaf."""

        authority = self._authority
        if (
            self._seal is not _SESSION_SEAL
            or authority._seal is not _AUTHORITY_SEAL
            or self._receipt_scan_failed
            or not authority._state.training_started
            or self._receipt_scan_index >= len(_PRECLAIM_SCAN_ORDER)
            or scan != _PRECLAIM_SCAN_ORDER[self._receipt_scan_index]
        ):
            self._receipt_scan_failed = True
            raise ABC6AuthorityError(
                "training session is stale, replayed, or out of preclaim order"
            )
        self._receipt_scan_index += 1
        try:
            authority._verify_training_preclaim_root(
                repository_root_fd,
                receipt_root_fd,
                repository_root_realpath=repository_root_realpath,
                receipt_root_relative=receipt_root_relative,
                protocol_id=protocol_id,
                run_id=run_id,
                manifest_sha256=manifest_sha256,
            )
        except BaseException:
            self._receipt_scan_failed = True
            raise

    def issue_case_permit(self, index: int) -> "ABC6TrainingPermit":
        authority = self._authority
        authority._assert_live()
        _require_role(authority, "campaign")
        if type(index) is not int or not 0 <= index < CASE_COUNT:
            raise ABC6AuthorityError("training permit index must be in 0..23")
        with authority._state.lock:
            if not authority._state.training_started or index in authority._state.issued:
                raise ABC6AuthorityError("training index permit is unavailable or already issued")
            authority._state.issued.add(index)
        return ABC6TrainingPermit(_CASE_SEAL, authority, index)


class ABC6TrainingPermit:
    __slots__ = ("_seal", "_authority", "_index", "_used")

    def __new__(cls, seal: object = None, *_args: object, **_kwargs: object):
        if cls is not ABC6TrainingPermit or seal is not _CASE_SEAL:
            raise TypeError("case permits require a sealed training session")
        return super().__new__(cls)

    def __init__(self, seal: object, authority: ABC6LaunchAuthority, index: int):
        if seal is not _CASE_SEAL or authority._seal is not _AUTHORITY_SEAL:
            raise TypeError("invalid training permit")
        self._seal, self._authority, self._index, self._used = seal, authority, index, False

    @property
    def case_index(self) -> int:
        return self._index

    def consume(self, index: int) -> None:
        authority = self._authority
        authority._assert_live()
        _require_role(authority, "case")
        if self._seal is not _CASE_SEAL or type(index) is not int or index != self._index:
            raise ABC6AuthorityError("permit does not match this exact case index")
        with authority._state.lock:
            if self._used or index in authority._state.consumed:
                raise ABC6AuthorityError("case permit is one-use")
            self._used = True
            authority._state.consumed.add(index)


class ABC6RunnerHandoff:
    __slots__ = ("_seal", "_authority", "_status_receipts", "_summary_sha256", "_forecast_sha256")

    def __new__(cls, seal: object = None, *_args: object, **_kwargs: object):
        if cls is not ABC6RunnerHandoff or seal is not _HANDOFF_SEAL:
            raise TypeError("runner handoff is sealed to the attested runner")
        return super().__new__(cls)

    def __init__(self, seal: object, authority: ABC6LaunchAuthority, statuses: tuple[tuple[int, str, str], ...], summary: str, forecast: str):
        self._seal, self._authority = seal, authority
        self._status_receipts, self._summary_sha256, self._forecast_sha256 = statuses, summary, forecast


class ABC6ScoringPermit:
    __slots__ = ("_seal", "_authority", "_used")

    def __new__(cls, seal: object = None, *_args: object, **_kwargs: object):
        if cls is not ABC6ScoringPermit or seal is not _SCORE_SEAL:
            raise TypeError("scoring permit requires one activated runner handoff")
        return super().__new__(cls)

    def __init__(self, seal: object, authority: ABC6LaunchAuthority):
        if seal is not _SCORE_SEAL or authority._seal is not _AUTHORITY_SEAL:
            raise TypeError("invalid scoring permit")
        self._seal, self._authority, self._used = seal, authority, False

    def consume(self) -> dict[str, object]:
        authority = self._authority
        authority._assert_live()
        _require_role(authority, "scorer")
        with authority._state.lock:
            if self._seal is not _SCORE_SEAL or not authority._state.activated or self._used or authority._state.score_consumed:
                raise ABC6AuthorityError("scoring permit is dormant or already consumed")
            self._used = authority._state.score_consumed = True
        return authority.receipt_metadata()


def _require_role(authority: ABC6LaunchAuthority, role: str) -> None:
    expected_path = getattr(authority._bindings, f"{role}_source_path")
    expected_function = getattr(authority._bindings, f"{role}_function")
    frame = inspect.currentframe()
    frame = None if frame is None else frame.f_back
    frame = None if frame is None else frame.f_back
    if frame is None or frame.f_code.co_name != expected_function:
        raise ABC6AuthorityError(f"only the attested {role} entrypoint may use this authority")
    actual = os.path.abspath(frame.f_code.co_filename)
    expected = os.path.join(authority._bindings.physical_root, *expected_path.split("/"))
    if actual != expected or _hash_beneath(authority._root_fd, expected_path) != dict(authority._bindings.source_hashes)[expected_path]:
        raise ABC6AuthorityError(f"attested {role} source identity changed")


__all__ = [
    "ABC6AuthorityBindings",
    "ABC6AuthorityError",
    "ABC6LaunchAuthority",
    "ABC6RunnerHandoff",
    "ABC6ScoringPermit",
    "ABC6TrainingPermit",
    "ABC6TrainingSession",
    "ProcessIdentity",
    "receive_child_grant",
]

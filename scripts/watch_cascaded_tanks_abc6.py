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
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

PROTOCOL_ID = "cascaded_tanks_abc6_synthetic_v1_20260928"
RUN_ID = "ct-abc6-20260928-v1"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_CLAIM_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "evaluations"
    / "cascaded_tanks_abc6_campaign_fit"
    / "claims"
    / f"{RUN_ID}.watchdog.claim"
)
_SCRIPT_PATH = Path(__file__).resolve()


class WatchdogError(RuntimeError):
    """A fail-closed launcher, monitoring, or receipt error."""


class AlreadyClaimedError(WatchdogError):
    """The watchdog's one-use run ID has already been consumed."""


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
) -> str:
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


def _load_and_check_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not _is_sha256(expected_sha256):
        raise WatchdogError("manifest SHA-256 must be 64 lowercase hexadecimal characters")
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_MANIFEST_BYTES:
            raise WatchdogError("manifest must be a small regular file, not a symlink")
        raw = path.read_bytes()
    except OSError as error:
        raise WatchdogError("manifest is missing or unreadable") from error
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


def _campaign_strict_preflight(manifest_path: Path, manifest_sha256: str) -> str:
    """Call the campaign runner's exact source/runtime/roster preflight."""

    from core.real_data.cascaded_tanks_abc6_campaign_fit import (
        preflight_abc6_campaign_manifest,
    )

    return preflight_abc6_campaign_manifest(
        manifest_path,
        manifest_sha256,
        require_reviewed_runtime=True,
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
) -> dict[str, Any]:
    if not _is_sha256(approval_sha256):
        raise WatchdogError("approval record SHA-256 must be lowercase hexadecimal")
    try:
        metadata = approval_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_MANIFEST_BYTES:
            raise WatchdogError("approval record must be a small regular file")
        raw = approval_path.read_bytes()
    except OSError as error:
        raise WatchdogError("independent approval/launch record is missing") from error
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
        or script_sha256 != _sha256_file(_SCRIPT_PATH)
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
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate all campaign and approval gates before any one-use claim."""

    try:
        verified_sha256 = _campaign_strict_preflight(manifest_path, manifest_sha256)
    except Exception as error:
        raise WatchdogError(
            f"campaign strict manifest/source/runtime/roster preflight failed: {error}"
        ) from error
    if verified_sha256 != manifest_sha256:
        raise WatchdogError("campaign strict preflight returned a different manifest hash")
    manifest, loaded_sha256 = _load_and_check_manifest(manifest_path, manifest_sha256)
    if loaded_sha256 != verified_sha256:
        raise WatchdogError("watchdog and campaign manifest hash checks disagree")
    receipt_resolved = receipt_directory.resolve(strict=True)
    if receipt_resolved != receipt_directory or receipt_directory.is_symlink():
        raise WatchdogError("receipt directory must be canonical and not a symlink")
    launch = _build_campaign_command(manifest_path, loaded_sha256, receipt_directory)
    approval = _load_and_validate_approval_record(
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        manifest_path=manifest_path,
        manifest_sha256=loaded_sha256,
        receipt_directory=receipt_directory,
        campaign_launch_vector=launch,
        actual_watchdog_args=actual_watchdog_args,
    )
    return manifest, loaded_sha256, approval


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
    code = (
        "from core.real_data.cascaded_tanks_abc6_campaign_fit import "
        "run_abc6_training_campaign; "
        f"run_abc6_training_campaign({str(manifest_path)!r}, "
        f"{manifest_sha256!r}, {str(receipt_directory)!r})"
    )
    return [PINNED_PYTHON_BIN, "-c", code]


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

    # Validate the destination before consuming the one-use claim.  Once the
    # claim exists, every failure is terminal and the run ID stays consumed.
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
                break

            if enumeration_errors:
                membership_uncertain = True
                stop_reason = "process_group_membership_unknown"
                monitor_error = "; ".join(enumeration_errors)
                break

            if len(members) > 1:
                stop_reason = "unexpected_descendant_process"
                break

            if rss_total > rss_limit_bytes:
                stop_reason = "sampled_rss_limit_exceeded"
                break
            if elapsed >= wall_limit_seconds:
                stop_reason = "wall_clock_limit_exceeded"
                break
            if process_return_code is not None:
                stop_reason = (
                    "child_exited"
                    if not members
                    else "child_exited_with_live_process_group_members"
                )
                break
            time.sleep(min(sample_interval_seconds, max(0.0, wall_limit_seconds - elapsed)))

    except KeyboardInterrupt as error:
        stop_reason = "watchdog_interrupted"
        monitor_error = f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - persist terminal state after any monitor error.
        if isinstance(error, WatchdogError):
            stop_reason = "watchdog_identity_or_monitoring_error"
        else:
            stop_reason = "watchdog_monitoring_error"
        monitor_error = f"{type(error).__name__}: {error}"

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

    fit_gate = _inspect_fit_phase_gate(
        receipt_path.parent,
        expected_manifest_sha256=str(run_identity.get("manifest_sha256", "")),
        sync_directory=runner_exit_complete,
    )
    samples_bytes = _canonical_json(samples)
    body: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": run_identity.get("protocol_id"),
        "run_id": run_identity.get("run_id"),
        "manifest_sha256": run_identity.get("manifest_sha256"),
        "approval_record_sha256": run_identity.get("approval_record_sha256"),
        "reviewed_git_head": run_identity.get("reviewed_git_head"),
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
            "method": "same-directory temporary file, file fsync, hard-link publication, directory fsync",
            "performed_by_this_writer": True,
        },
        "campaign_targets_generated_by_watchdog": False,
        "campaign_forecasts_run_by_watchdog": False,
    }
    receipt_sha256 = _write_exclusive_durable(receipt_path, body)
    body["terminal_receipt_sha256"] = receipt_sha256
    # The receipt hash is returned out-of-band: adding it to the already
    # published receipt would make the recorded digest self-referential.
    return {"receipt": body, "receipt_sha256": receipt_sha256, "receipt_path": receipt_path}


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


def _inspect_fit_phase_gate(
    receipt_directory: Path,
    *,
    expected_manifest_sha256: str,
    sync_directory: bool,
) -> dict[str, object]:
    statuses: list[dict[str, object]] = []
    problems: list[str] = []
    try:
        case_ids = _expected_case_ids()
    except (ImportError, WatchdogError) as error:
        case_ids = []
        problems.append(f"cannot load reviewed case roster: {type(error).__name__}")
    for case_index in range(24):
        for component in ("fit", "baseline"):
            path = receipt_directory / f"case-{case_index:02d}.{component}-status.json"
            try:
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise OSError("status path is not a regular file")
                raw = path.read_bytes()
                payload = json.loads(raw.decode("ascii"))
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
                    problems.append(f"invalid status receipt: {path.name}")
                    continue
                statuses.append(
                    {
                        "case_index": case_index,
                        "case_id": payload["case_id"],
                        "component": component,
                        "status": payload["status"],
                        "path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            except FileNotFoundError:
                continue
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                RecursionError,
                TypeError,
                ValueError,
            ) as error:
                problems.append(f"cannot validate {path.name}: {type(error).__name__}")

    summary_path = receipt_directory / "campaign.training-summary.json"
    summary: dict[str, object] | None = None
    summary_sha256: str | None = None
    campaign_claim_sha256: str | None = None
    claim_path = receipt_directory / "campaign.claim"
    try:
        claim_metadata = claim_path.lstat()
        if not stat.S_ISREG(claim_metadata.st_mode):
            raise OSError("campaign claim is not a regular file")
        claim_raw = claim_path.read_bytes()
        claim_payload = json.loads(claim_raw.decode("ascii"))
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
            or claim_payload.get("receipt_directory") != str(receipt_directory.resolve())
            or type(claim_payload.get("claimed_at_utc")) is not str
            or claim_payload.get("claim_semantics") != "consumed_once_no_resume"
        ):
            problems.append("campaign claim is invalid or does not match the receipt directory")
        else:
            campaign_claim_sha256 = hashlib.sha256(claim_raw).hexdigest()
    except FileNotFoundError:
        problems.append("campaign claim receipt is missing")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        problems.append(f"cannot validate campaign claim: {type(error).__name__}")

    try:
        metadata = summary_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("summary path is not a regular file")
        raw_summary = summary_path.read_bytes()
        decoded = json.loads(raw_summary.decode("ascii"))
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
    except FileNotFoundError:
        pass
    except (
        OSError,
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

    if sync_directory and len(statuses) == 48 and summary_links_valid:
        try:
            _fsync_directory(receipt_directory)
        except OSError as error:
            problems.append(f"cannot sync campaign receipt directory: {type(error).__name__}")

    all_statuses_complete = len(statuses) == 48 and all(
        row["status"] == "complete" for row in statuses
    )
    return {
        "status_receipt_count": len(statuses),
        "status_receipts": statuses,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "summary_status": None if summary is None else summary.get("status"),
        "all_48_status_receipts_present_and_linked": bool(
            len(statuses) == 48 and summary_links_valid
        ),
        "all_48_fit_and_baseline_statuses_complete": bool(
            all_statuses_complete and summary_links_valid
        ),
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
    if not all(path.is_absolute() for path in (manifest_path, receipt_directory, approval_path)):
        raise WatchdogError(
            "manifest, receipt-directory, and approval-record arguments must be absolute"
        )
    if any(
        path.resolve() != path or path.is_symlink()
        for path in (manifest_path, receipt_directory, approval_path)
    ):
        raise WatchdogError("manifest, receipt-directory, and approval paths must be canonical")

    manifest, manifest_sha256, approval = _production_preflight(
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
        receipt_directory=receipt_directory,
        approval_path=approval_path,
        approval_sha256=args.approval_record_sha256,
        actual_watchdog_args=actual_args,
    )
    attestation = _current_parent_attestation(actual_args)
    if not receipt_directory.is_dir():
        raise WatchdogError("receipt directory must be created before launch")
    launch = _build_campaign_command(manifest_path, manifest_sha256, receipt_directory)
    identity: dict[str, object] = {
        "protocol_id": manifest["protocol_id"],
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256,
        "approval_record_sha256": args.approval_record_sha256,
        "reviewed_git_head": approval["reviewed_git_head"],
        "watchdog_attestation": attestation,
        "watchdog_image_path": attestation["observed_python_app_image_path"],
        "watchdog_image_sha256": attestation["observed_python_app_image_sha256"],
    }
    result = _supervise_command(
        launch_vector=launch,
        observed_executable_path=Path(PINNED_PYTHON_APP),
        observed_executable_sha256=PINNED_PYTHON_APP_SHA256,
        claim_path=_PRODUCTION_CLAIM_PATH,
        receipt_path=receipt_directory / "watchdog-terminal-receipt.json",
        run_identity=identity,
        claim_project_root=_REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "status": result["receipt"]["status"],
                "receipt_path": str(result["receipt_path"]),
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["receipt"]["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover - production entry is not run in tests.
    try:
        raise SystemExit(_main())
    except (WatchdogError, OSError, psutil.Error) as error:
        print(f"ABC6 watchdog failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error

"""One-use, training-only orchestration for the frozen ABC6 case roster.

This module connects the reviewed 24-case roster to the reviewed one-case fit
adapter.  It creates only synthetic training data and writes immutable
fit/baseline status receipts.  It never opens the prospective-target gate,
materializes future outputs, forecasts, or scores targets.

The manifest hash/source/runtime checks below are preflight checks, not an
independent approval system.  The 900-second/2-GiB sampled watchdog and final
manifest approval remain external launch gates; this module does not enforce
either.  The caller must not invoke the production entry point until those
separate gates have passed.  A fixed project-root O_EXCL registry claim, not
the caller-selected receipt directory, consumes the run ID permanently:
partial runs cannot be resumed or retried under it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from core.real_data import cascaded_tanks_abc6_training as training
from core.real_data.cascaded_tanks_abc6_cases import (
    CASE_COUNT,
    CASE_ROSTER,
    PROTOCOL_ID,
    RUN_ID,
    ABC6Case,
    ABC6TrainingCaseData,
    build_synthetic_training_bundle,
    case_by_index,
)

MANIFEST_SCHEMA_VERSION: Final = 1
CLAIM_FILENAME: Final = "campaign.claim"
SUMMARY_FILENAME: Final = "campaign.training-summary.json"
FAILURE_FILENAME: Final = "campaign.failure.json"
MAX_MANIFEST_BYTES: Final = 2 * 1024 * 1024
GLOBAL_WATCHDOG_SECONDS: Final = 900
RUNNER_TREE_RSS_LIMIT_BYTES: Final = 2 * 1024**3
PROJECTED_ARTIFACT_LIMIT_BYTES: Final = 1024**3
MAX_WORKERS: Final = 1

REVIEWED_PROPOSAL_PATH: Final = (
    "reports/cascaded-tanks-six-parameter-synthetic-abc-proposal-2026-09-28.md"
)
REVIEWED_PROPOSAL_SHA256: Final = (
    "8f406a1e3ceb12cc7d835b9ed686e5dd2ce181725c97f9c40047ba84a1e92b0f"
)

# These are checks against the already-reviewed stable local runtime named by
# the proposal.  The final manifest must independently pin and hash them too.
PINNED_INTERPRETER_PATH: Final = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/"
    "Versions/3.14/bin/python3.14"
)
PINNED_INTERPRETER_SHA256: Final = (
    "cbf84109626aa1013bbe408fbb9590bd0f1c1548f038b2221c6b8b87de26ca43"
)
PINNED_NUMPY_PATH: Final = (
    "/opt/homebrew/lib/python3.14/site-packages/numpy/__init__.py"
)
PINNED_NUMPY_SHA256: Final = (
    "2e8da3e4385e79c4885b3f7324a8b957e6f01732b239e99e266a12c62a008b8d"
)
PINNED_PSUTIL_PATH: Final = (
    "/Users/sourabh/Library/Python/3.14/lib/python/site-packages/psutil/__init__.py"
)
PINNED_PSUTIL_SHA256: Final = (
    "d138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a"
)

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_PROJECT_RUN_CLAIM_PATH: Final = (
    _REPO_ROOT
    / "artifacts"
    / "evaluations"
    / "cascaded_tanks_abc6_campaign_fit"
    / "claims"
    / f"{RUN_ID}.claim"
)
_REQUIRED_SOURCE_PATHS: Final = (
    "core/real_data/cascaded_tanks_abc6_campaign_fit.py",
    "core/real_data/cascaded_tanks_abc6_cases.py",
    "core/real_data/cascaded_tanks_abc6_training.py",
    "core/real_data/cascaded_tanks_pattern_search.py",
    "core/real_data/cascaded_tanks_synthetic_abc.py",
    "core/real_data/cascaded_tanks_models.py",
    REVIEWED_PROPOSAL_PATH,
)
_REVIEWED_SOURCE_SHA256: Final = {
    "core/real_data/cascaded_tanks_abc6_cases.py": (
        "3e20935ab4324dd4ffc8b352281ce02fae05f8f57516f5ac8c9cd8c606cd119b"
    ),
    "core/real_data/cascaded_tanks_abc6_training.py": (
        "a12e0f1e2c4fb40e632b0785fdd47f8467ecffbdc36e1396e62d147b2981c59e"
    ),
    "core/real_data/cascaded_tanks_pattern_search.py": (
        "c7800c76c73b5d6063a6e15ea45cd756e7a542b7ec584e99a18108e3ea914458"
    ),
    REVIEWED_PROPOSAL_PATH: REVIEWED_PROPOSAL_SHA256,
}
_FINAL_STATUSES: Final = frozenset({"complete", "incomplete", "unresolved", "failed"})
_COMPONENTS: Final = ("fit", "baseline")


class ABC6CampaignError(RuntimeError):
    """Base error for fail-closed campaign preflight or orchestration."""


class ABC6CampaignPreflightError(ABC6CampaignError):
    """The immutable run manifest or output inventory failed preflight."""


class ABC6CampaignAlreadyClaimedError(ABC6CampaignError):
    """The one-use run ID has already been claimed and cannot be resumed."""


class ABC6CampaignExecutionError(ABC6CampaignError):
    """Training stopped after the run claim; the run ID remains consumed."""


@dataclass(frozen=True, slots=True)
class ABC6CaseStatus:
    case_index: int
    case_id: str
    fit_status: str
    baseline_status: str
    fit_receipt_sha256: str
    baseline_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ABC6CampaignResult:
    """Compact terminal status only; training evidence stays caller-owned."""

    protocol_id: str
    run_id: str
    manifest_sha256: str
    claim_sha256: str
    status: Literal["complete", "incomplete"]
    case_statuses: tuple[ABC6CaseStatus, ...]
    summary_path: Path
    summary_sha256: str
    prospective_targets_generated: Literal[False] = False
    forecasts_run: Literal[False] = False


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    try:
        file_stat = path.stat()
    except OSError as error:
        raise ABC6CampaignPreflightError(f"cannot stat pinned file: {path}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ABC6CampaignPreflightError(f"pinned path is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ABC6CampaignPreflightError(f"cannot read pinned file: {path}") from error
    return digest.hexdigest()


def _case_identity(case: ABC6Case) -> dict[str, object]:
    return {
        "case_index": case.case_index,
        "case_id": case.case_id,
        "truth_id": case.truth_id,
        "input_window": case.input_window,
        "replicate": case.replicate,
        "fit_model": case.fit_model.value,
        "calibration_seed": case.calibration_seed,
        "abc_seed": case.abc_seed,
        "input_length": case.input_length,
        "prior_bounds": [list(bounds) for bounds in case.prior_bounds],
    }


def _manifest_identity() -> dict[str, object]:
    """Stable identity fields; source/runtime digests remain externally pinned."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "reviewed_proposal": {
            "path": REVIEWED_PROPOSAL_PATH,
            "sha256": REVIEWED_PROPOSAL_SHA256,
        },
        "training_control_manifest_sha256": training.TRAINING_CONTROL_MANIFEST_SHA256,
        "ordered_cases": [_case_identity(case) for case in CASE_ROSTER],
    }


def _current_source_hashes() -> dict[str, str]:
    return {
        relative_path: _sha256_file(_REPO_ROOT / relative_path)
        for relative_path in _REQUIRED_SOURCE_PATHS
    }


def _current_runtime_fingerprint() -> dict[str, object]:
    """Return actual in-process runtime evidence for manifest comparison."""

    import numpy
    import psutil

    try:
        resolved_interpreter = Path(sys.executable).resolve(strict=True)
        numpy_path = Path(numpy.__file__).absolute()
        psutil_path = Path(psutil.__file__).absolute()
    except OSError as error:
        raise ABC6CampaignPreflightError("runtime path resolution failed") from error
    return {
        "declared_launch_interpreter_path": PINNED_INTERPRETER_PATH,
        "resolved_running_interpreter_path": str(resolved_interpreter),
        "python_version": platform.python_version(),
        "interpreter_sha256": _sha256_file(Path(PINNED_INTERPRETER_PATH)),
        "numpy_version": numpy.__version__,
        "numpy_module_path": str(numpy_path),
        "numpy_module_sha256": _sha256_file(numpy_path),
        "psutil_version": psutil.__version__,
        "psutil_module_path": str(psutil_path),
        "psutil_module_sha256": _sha256_file(psutil_path),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _require_reviewed_runtime(fingerprint: Mapping[str, object]) -> None:
    expected = {
        "declared_launch_interpreter_path": PINNED_INTERPRETER_PATH,
        "resolved_running_interpreter_path": str(
            Path(PINNED_INTERPRETER_PATH).resolve(strict=True)
        ),
        "python_version": "3.14.3",
        "interpreter_sha256": PINNED_INTERPRETER_SHA256,
        "numpy_version": "2.4.2",
        "numpy_module_path": PINNED_NUMPY_PATH,
        "numpy_module_sha256": PINNED_NUMPY_SHA256,
        "psutil_version": "7.2.2",
        "psutil_module_path": PINNED_PSUTIL_PATH,
        "psutil_module_sha256": PINNED_PSUTIL_SHA256,
    }
    if any(fingerprint.get(key) != value for key, value in expected.items()):
        raise ABC6CampaignPreflightError(
            "process runtime does not match the reviewed Python/NumPy/psutil pins"
        )


def _execution_contract_is_valid(value: object) -> bool:
    expected_keys = {
        "wall_clock_limit_seconds",
        "runner_tree_rss_limit_bytes",
        "projected_artifact_bytes",
        "worker_count",
        "watchdog_enforcement",
        "independent_manifest_approval_required",
        "prospective_targets_before_receipts",
    }
    if type(value) is not dict or set(value) != expected_keys:
        return False
    projected = value.get("projected_artifact_bytes")
    return (
        type(value.get("wall_clock_limit_seconds")) is int
        and value["wall_clock_limit_seconds"] == GLOBAL_WATCHDOG_SECONDS
        and type(value.get("runner_tree_rss_limit_bytes")) is int
        and value["runner_tree_rss_limit_bytes"] == RUNNER_TREE_RSS_LIMIT_BYTES
        and type(projected) is int
        and 0 <= projected < PROJECTED_ARTIFACT_LIMIT_BYTES
        and type(value.get("worker_count")) is int
        and value["worker_count"] == MAX_WORKERS
        and value.get("watchdog_enforcement") == "external"
        and value.get("independent_manifest_approval_required") is True
        and value.get("prospective_targets_before_receipts") is False
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_case_entry(entry: object) -> bool:
    expected_keys = {
        "case_index",
        "case_id",
        "truth_id",
        "input_window",
        "replicate",
        "fit_model",
        "calibration_seed",
        "abc_seed",
        "input_length",
        "prior_bounds",
    }
    if type(entry) is not dict or set(entry) != expected_keys:
        return False
    for key in ("case_index", "calibration_seed", "abc_seed", "input_length"):
        if type(entry[key]) is not int:
            return False
    if entry["replicate"] is not None and type(entry["replicate"]) is not int:
        return False
    for key in ("case_id", "truth_id", "input_window", "fit_model"):
        if type(entry[key]) is not str:
            return False
    bounds = entry["prior_bounds"]
    if type(bounds) is not list or len(bounds) != 6:
        return False
    for pair in bounds:
        if type(pair) is not list or len(pair) != 2:
            return False
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in pair
        ):
            return False
    return True


def _strict_manifest_schema(decoded: object) -> bool:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "reviewed_proposal",
        "training_control_manifest_sha256",
        "ordered_cases",
        "source_hashes",
        "runtime_fingerprint",
        "execution_contract",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys:
        return False
    if type(decoded["schema_version"]) is not int:
        return False
    if type(decoded["protocol_id"]) is not str or type(decoded["run_id"]) is not str:
        return False
    if not _is_sha256(decoded["training_control_manifest_sha256"]):
        return False

    proposal = decoded["reviewed_proposal"]
    if (
        type(proposal) is not dict
        or set(proposal) != {"path", "sha256"}
        or type(proposal["path"]) is not str
        or not _is_sha256(proposal["sha256"])
    ):
        return False

    roster = decoded["ordered_cases"]
    if (
        type(roster) is not list
        or len(roster) != CASE_COUNT
        or any(not _strict_case_entry(entry) for entry in roster)
    ):
        return False

    source_hashes = decoded["source_hashes"]
    if (
        type(source_hashes) is not dict
        or set(source_hashes) != set(_REQUIRED_SOURCE_PATHS)
        or any(
            type(path) is not str or not _is_sha256(digest)
            for path, digest in source_hashes.items()
        )
    ):
        return False

    runtime = decoded["runtime_fingerprint"]
    runtime_keys = {
        "declared_launch_interpreter_path",
        "resolved_running_interpreter_path",
        "python_version",
        "interpreter_sha256",
        "numpy_version",
        "numpy_module_path",
        "numpy_module_sha256",
        "psutil_version",
        "psutil_module_path",
        "psutil_module_sha256",
        "platform",
        "machine",
    }
    if (
        type(runtime) is not dict
        or set(runtime) != runtime_keys
        or any(type(value) is not str for value in runtime.values())
        or any(
            not _is_sha256(runtime[key])
            for key in (
                "interpreter_sha256",
                "numpy_module_sha256",
                "psutil_module_sha256",
            )
        )
    ):
        return False
    return _execution_contract_is_valid(decoded["execution_contract"])


def preflight_abc6_campaign_manifest(
    manifest_path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    require_reviewed_runtime: bool = True,
) -> str:
    """Validate an external immutable manifest and return its exact file hash.

    The expected hash is supplied by the caller after the manifest has been
    frozen.  This function deliberately does not approve the manifest, attest
    an independent reviewer, or verify/enforce an external watchdog receipt.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ABC6CampaignPreflightError(
            "expected_sha256 must be 64 lowercase hex chars"
        )
    path = Path(manifest_path)
    try:
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_size > MAX_MANIFEST_BYTES
        ):
            raise ABC6CampaignPreflightError(
                "manifest must be a small regular file, not a symlink"
            )
        raw = path.read_bytes()
    except ABC6CampaignPreflightError:
        raise
    except OSError as error:
        raise ABC6CampaignPreflightError("manifest is missing or unreadable") from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ABC6CampaignPreflightError("manifest SHA-256 does not match caller pin")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ABC6CampaignPreflightError(
            "manifest must be canonical ASCII JSON"
        ) from error
    try:
        canonical_decoded = _canonical_json(decoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise ABC6CampaignPreflightError(
            "manifest JSON contains values outside the strict schema"
        ) from error
    if type(decoded) is not dict or canonical_decoded != raw:
        raise ABC6CampaignPreflightError("manifest JSON is not a canonical object")
    if not _strict_manifest_schema(decoded):
        raise ABC6CampaignPreflightError("manifest schema/type validation failed")

    identity = _manifest_identity()
    if any(decoded.get(key) != value for key, value in identity.items()):
        raise ABC6CampaignPreflightError(
            "manifest protocol, report, training controls, or ordered roster mismatch"
        )
    source_hashes = decoded.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ABC6CampaignPreflightError("manifest lacks source_hashes mapping")
    current_sources = _current_source_hashes()
    if any(
        current_sources.get(name) != digest
        for name, digest in _REVIEWED_SOURCE_SHA256.items()
    ):
        raise ABC6CampaignPreflightError(
            "reviewed proposal, case roster, training seam, or baseline source changed"
        )
    if any(
        source_hashes.get(name) != digest for name, digest in current_sources.items()
    ):
        raise ABC6CampaignPreflightError("manifest source hash mismatch")
    runtime = decoded.get("runtime_fingerprint")
    if not isinstance(runtime, dict):
        raise ABC6CampaignPreflightError("manifest lacks runtime_fingerprint object")
    current_runtime = _current_runtime_fingerprint()
    if runtime != current_runtime:
        raise ABC6CampaignPreflightError("manifest runtime fingerprint mismatch")
    if require_reviewed_runtime:
        _require_reviewed_runtime(current_runtime)
    if not _execution_contract_is_valid(decoded.get("execution_contract")):
        raise ABC6CampaignPreflightError(
            "manifest execution contract must declare the external 900s/2-GiB "
            "single-worker limits and a projected artifact size below 1 GiB"
        )
    return actual_sha256


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:  # pragma: no cover - guarded by the OS write contract.
            raise OSError("durable file write made no progress")
        remaining = remaining[count:]


def _write_exclusive_durable(path: Path, payload: bytes) -> None:
    """Atomically publish immutable bytes with no replacement on collision."""

    directory = path.parent
    temporary_path = directory / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # Same-directory hard-link publication is atomic and fails if the final
        # immutable receipt already exists; os.replace would permit rewriting.
        os.link(temporary_path, path)
        _fsync_directory(directory)
    finally:
        try:
            temporary_path.unlink()
            _fsync_directory(directory)
        except FileNotFoundError:
            pass


def _status_receipt_payload(case_index: int, component: str, status: str) -> bytes:
    case = case_by_index(case_index)
    body: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "case_index": case_index,
        "case_id": case.case_id,
        "component": component,
        "status": status,
    }
    body["payload_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return _canonical_json(body)


def _write_case_status(
    receipt_directory: Path,
    case_index: int,
    component: str,
    status: str,
) -> tuple[Path, str]:
    if component not in _COMPONENTS:
        raise ValueError("component must be 'fit' or 'baseline'")
    if status not in _FINAL_STATUSES:
        raise ValueError("status must be a final receipt status")
    path = receipt_directory / f"case-{case_index:02d}.{component}-status.json"
    payload = _status_receipt_payload(case_index, component, status)
    _write_exclusive_durable(path, payload)
    return path, hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _claim_run(
    receipt_directory: Path,
    manifest_sha256: str,
    claim_registry_path: Path,
) -> str:
    local_claim_path = receipt_directory / CLAIM_FILENAME
    if local_claim_path.exists() or local_claim_path.is_symlink():
        raise ABC6CampaignAlreadyClaimedError(
            "the one-use run ID is already claimed; resume and retry are forbidden"
        )
    try:
        entries = tuple(receipt_directory.iterdir())
    except OSError as error:
        raise ABC6CampaignPreflightError(
            "cannot inventory run output directory"
        ) from error
    if entries:
        raise ABC6CampaignPreflightError(
            "run output directory must be empty before the one-use claim"
        )
    if not receipt_directory.is_dir() or receipt_directory.is_symlink():
        raise ABC6CampaignPreflightError(
            "run output directory must be a real directory"
        )
    try:
        claim_registry_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ABC6CampaignPreflightError(
            "cannot create the fixed project-root run-ID registry"
        ) from error
    if (
        claim_registry_path.parent.is_symlink()
        or not claim_registry_path.parent.is_dir()
    ):
        raise ABC6CampaignPreflightError(
            "run-ID registry parent must be a real directory"
        )
    claim = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "receipt_directory": str(receipt_directory.resolve()),
        "claimed_at_utc": _utc_now(),
        "claim_semantics": "consumed_once_no_resume",
    }
    payload = _canonical_json(claim)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            claim_registry_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _fsync_directory(claim_registry_path.parent)
    except FileExistsError as error:
        raise ABC6CampaignAlreadyClaimedError(
            "the fixed project-root registry already consumed this one-use run ID"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    # Retain a local linkable receipt in the per-run output folder.  The
    # registry claim above is authoritative and remains consumed if this
    # secondary publication fails.
    _write_exclusive_durable(local_claim_path, payload)
    return hashlib.sha256(payload).hexdigest()


def _write_failure_receipt(
    receipt_directory: Path,
    *,
    manifest_sha256: str,
    claim_sha256: str,
    completed_cases: int,
    case_index: int | None,
    stop_reason: str,
    exception_type: str,
) -> None:
    failure: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "claim_sha256": claim_sha256,
        "failed_at_utc": _utc_now(),
        "completed_case_count": completed_cases,
        "case_index": case_index,
        "case_id": None if case_index is None else case_by_index(case_index).case_id,
        "stop_reason": stop_reason,
        "exception_type": exception_type,
        "resume_allowed": False,
    }
    failure["payload_sha256"] = hashlib.sha256(_canonical_json(failure)).hexdigest()
    _write_exclusive_durable(
        receipt_directory / FAILURE_FILENAME,
        _canonical_json(failure),
    )


def _result_statuses(
    result: object,
    case: ABC6Case,
) -> tuple[str, str]:
    # Attribute validation keeps the private injected seam lightweight for
    # control-flow tests.  The production callable is the reviewed typed
    # one-case adapter above.
    required = ("case_index", "case_id", "model", "abc_status", "baseline_status")
    if any(not hasattr(result, name) for name in required):
        raise TypeError("fit callable returned an object without case status fields")
    if (
        type(result.case_index) is not int
        or result.case_index != case.case_index
        or result.case_id != case.case_id
        or result.model != case.fit_model.value
    ):
        raise ValueError("one-case training result identity does not match roster")
    fit_status = result.abc_status
    baseline_status = result.baseline_status
    if fit_status not in {"complete", "incomplete", "unresolved"}:
        raise ValueError("one-case training result has an invalid ABC status")
    if baseline_status not in {"complete", "incomplete"}:
        raise ValueError("one-case training result has an invalid baseline status")
    return fit_status, baseline_status


def _record_failed_case(
    receipt_directory: Path,
    *,
    case: ABC6Case,
    manifest_sha256: str,
    claim_sha256: str,
    completed_cases: int,
    stop_reason: str,
    error: BaseException,
) -> None:
    # The case-level schema intentionally accepts "failed" for both components.
    # Existing immutable receipts are never rewritten if storage failed midway.
    for component in _COMPONENTS:
        path = receipt_directory / f"case-{case.case_index:02d}.{component}-status.json"
        if not path.exists():
            _write_case_status(receipt_directory, case.case_index, component, "failed")
    _write_failure_receipt(
        receipt_directory,
        manifest_sha256=manifest_sha256,
        claim_sha256=claim_sha256,
        completed_cases=completed_cases,
        case_index=case.case_index,
        stop_reason=stop_reason,
        exception_type=type(error).__name__,
    )


def _summary_payload(
    manifest_sha256: str,
    claim_sha256: str,
    case_statuses: tuple[ABC6CaseStatus, ...],
) -> bytes:
    overall_status = (
        "complete"
        if len(case_statuses) == CASE_COUNT
        and all(
            item.fit_status == "complete" and item.baseline_status == "complete"
            for item in case_statuses
        )
        else "incomplete"
    )
    fit_counts = Counter(item.fit_status for item in case_statuses)
    baseline_counts = Counter(item.baseline_status for item in case_statuses)
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "claim_sha256": claim_sha256,
        "written_at_utc": _utc_now(),
        "status": overall_status,
        "case_count": len(case_statuses),
        "fit_status_counts": dict(sorted(fit_counts.items())),
        "baseline_status_counts": dict(sorted(baseline_counts.items())),
        "case_statuses": [
            {
                "case_index": item.case_index,
                "case_id": item.case_id,
                "fit_status": item.fit_status,
                "baseline_status": item.baseline_status,
                "fit_receipt_sha256": item.fit_receipt_sha256,
                "baseline_receipt_sha256": item.baseline_receipt_sha256,
            }
            for item in case_statuses
        ],
        "training_only": True,
        "prospective_targets_generated": False,
        "forecasts_run": False,
        "postfit_target_gate_opened": False,
        "external_watchdog_enforced_here": False,
        "independent_manifest_approval_enforced_here": False,
        "resume_allowed": False,
    }
    summary["payload_sha256"] = hashlib.sha256(_canonical_json(summary)).hexdigest()
    return _canonical_json(summary)


def _execute_campaign(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    require_reviewed_runtime: bool,
    claim_registry_path: Path,
) -> ABC6CampaignResult:
    # Hash and source/runtime checks happen before the irreversible O_EXCL claim.
    verified_manifest_sha256 = preflight_abc6_campaign_manifest(
        manifest_path,
        manifest_sha256,
        require_reviewed_runtime=require_reviewed_runtime,
    )
    directory = Path(receipt_directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ABC6CampaignPreflightError(
            "run output directory must already exist as a real directory"
        )
    claim_sha256 = _claim_run(
        directory,
        verified_manifest_sha256,
        claim_registry_path,
    )

    try:
        bundle = build_synthetic_training_bundle()
    except BaseException as error:
        _write_failure_receipt(
            directory,
            manifest_sha256=verified_manifest_sha256,
            claim_sha256=claim_sha256,
            completed_cases=0,
            case_index=None,
            stop_reason="synthetic_training_bundle_construction_failed",
            exception_type=type(error).__name__,
        )
        raise ABC6CampaignExecutionError(
            "training-data construction failed after the one-use claim; run is terminal"
        ) from error

    case_statuses: list[ABC6CaseStatus] = []
    for expected_index, expected_case in enumerate(CASE_ROSTER):
        # Never trust a consumer-supplied order: verify the frozen view and then
        # retrieve it again by its exact roster index.
        if expected_case != case_by_index(expected_index):
            raise ABC6CampaignPreflightError(
                "runtime case roster changed after preflight"
            )
        data = bundle.data_for_case(expected_index)
        if data.case != expected_case:
            error = ValueError("training bundle view identity differs from roster")
            _record_failed_case(
                directory,
                case=expected_case,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(case_statuses),
                stop_reason="training_case_view_identity_mismatch",
                error=error,
            )
            raise ABC6CampaignExecutionError(
                f"case {expected_index} identity mismatch; run is terminal"
            ) from error

        try:
            result = fit_callable(data)
        except BaseException as error:
            try:
                _record_failed_case(
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="one_case_training_callable_raised",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "one-case training stopped and failure receipt could not be sealed; "
                    "run remains claimed and non-resumable"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} training failed; run is terminal and non-resumable"
            ) from error
        try:
            fit_status, baseline_status = _result_statuses(result, expected_case)
        except BaseException as error:
            try:
                _record_failed_case(
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="one_case_result_validation_failed",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "invalid one-case result and failure receipt could not be sealed; "
                    "run remains claimed and non-resumable"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} result failed validation; run is terminal"
            ) from error

        try:
            fit_path, fit_receipt_sha256 = _write_case_status(
                directory, expected_index, "fit", fit_status
            )
            baseline_path, baseline_receipt_sha256 = _write_case_status(
                directory, expected_index, "baseline", baseline_status
            )
        except BaseException as error:
            try:
                _write_failure_receipt(
                    directory,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    case_index=expected_index,
                    stop_reason="case_status_receipt_publication_failed",
                    exception_type=type(error).__name__,
                )
            except OSError as receipt_error:
                raise ABC6CampaignExecutionError(
                    "case status publication and failure receipt both failed; "
                    "run is terminal"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} status publication failed; run is terminal"
            ) from error
        del fit_path, baseline_path
        case_statuses.append(
            ABC6CaseStatus(
                case_index=expected_index,
                case_id=expected_case.case_id,
                fit_status=fit_status,
                baseline_status=baseline_status,
                fit_receipt_sha256=fit_receipt_sha256,
                baseline_receipt_sha256=baseline_receipt_sha256,
            )
        )

    statuses = tuple(case_statuses)
    summary_bytes = _summary_payload(
        verified_manifest_sha256,
        claim_sha256,
        statuses,
    )
    summary_path = directory / SUMMARY_FILENAME
    try:
        _write_exclusive_durable(summary_path, summary_bytes)
    except OSError as error:
        try:
            _write_failure_receipt(
                directory,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(statuses),
                case_index=None,
                stop_reason="campaign_summary_publication_failed",
                exception_type=type(error).__name__,
            )
        except OSError as receipt_error:
            raise ABC6CampaignExecutionError(
                "campaign summary and failure receipt both failed; run is terminal"
            ) from receipt_error
        raise ABC6CampaignExecutionError(
            "all case receipts exist but summary publication failed; run is terminal"
        ) from error
    overall_status: Literal["complete", "incomplete"] = (
        "complete"
        if all(
            item.fit_status == "complete" and item.baseline_status == "complete"
            for item in statuses
        )
        else "incomplete"
    )
    return ABC6CampaignResult(
        protocol_id=PROTOCOL_ID,
        run_id=RUN_ID,
        manifest_sha256=verified_manifest_sha256,
        claim_sha256=claim_sha256,
        status=overall_status,
        case_statuses=statuses,
        summary_path=summary_path,
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
    )


def run_abc6_training_campaign(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
) -> ABC6CampaignResult:
    """Run the 24 production training fits once, after separate external gates.

    This is not a launch approval.  Before calling it, an independent reviewer
    must approve the immutable manifest and a separate watchdog must enforce
    and record the full runner-tree budget.  The function itself verifies the
    manifest hash, exact roster/source/runtime identity, and declared limits;
    it cannot prove reviewer independence or monitor global time/RSS.
    """

    return _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=training.run_abc6_training_case,
        require_reviewed_runtime=True,
        claim_registry_path=_PROJECT_RUN_CLAIM_PATH,
    )


def _run_campaign_with_fit_callable_for_test(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    claim_registry_path: str | os.PathLike[str],
) -> ABC6CampaignResult:
    """Test-only bounded seam; permits fake fits and a non-production runtime."""

    if not callable(fit_callable):
        raise TypeError("fit_callable must be callable")
    return _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=fit_callable,
        require_reviewed_runtime=False,
        claim_registry_path=Path(claim_registry_path),
    )


__all__ = [
    "ABC6CampaignAlreadyClaimedError",
    "ABC6CampaignError",
    "ABC6CampaignExecutionError",
    "ABC6CampaignPreflightError",
    "ABC6CampaignResult",
    "ABC6CaseStatus",
    "preflight_abc6_campaign_manifest",
    "run_abc6_training_campaign",
]

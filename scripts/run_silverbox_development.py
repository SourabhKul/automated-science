"""Run the bounded, development-only Silverbox first-fit protocol.

Reproducible explicit runtime invocation (from the repository root):

    uv run --python 3.12 --with numpy==2.4.4 --with psutil==7.2.2 \
      python scripts/run_silverbox_development.py

The bridge verifies the archive against the tracked manifest before making
one fixed-model proposal request. It does not expose any sealed scoring API.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Any

from core.real_data.silverbox_controlled import (
    SilverboxControlledDevelopment,
    load_silverbox_controlled_development,
)
from core.real_data.silverbox_first_fit import (
    PILOT_WALL_SECONDS,
    FrozenSilverboxFit,
    SilverboxSelection,
    fit_silverbox_development,
    select_silverbox_development,
)
from core.real_data.silverbox_proposer import (
    ALLOWED_TERM_IDS,
    ENDPOINT,
    FROZEN_REQUEST_PAYLOAD_SHA256,
    MODEL_ID,
    MODELS_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    SilverboxProposalResult,
    request_silverbox_proposal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "data/real/silverbox/manifest.json"
DEFAULT_RECEIPT_ROOT = REPOSITORY_ROOT / "artifacts/silverbox_first_fit"
PROPOSAL_MAX_WALL_SECONDS = REQUEST_TIMEOUT_SECONDS
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@dataclass(frozen=True)
class SilverboxDevelopmentRun:
    """Compact outcome handle for one locally receipted pilot attempt."""

    run_id: str
    status: str
    reason: str
    stage_directory: Path
    source_sha256: str | None = None
    proposal_receipt_sha256: str | None = None
    fit: FrozenSilverboxFit | Any | None = None
    selection: SilverboxSelection | Any | None = None


def run_silverbox_development(
    *,
    archive_path: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    receipt_root: str | Path = DEFAULT_RECEIPT_ROOT,
    loader: Callable[..., SilverboxControlledDevelopment] | None = None,
    proposal_request: Callable[..., SilverboxProposalResult] | None = None,
    fit: Callable[..., FrozenSilverboxFit] | None = None,
    select: Callable[..., SilverboxSelection] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    run_id_factory: Callable[[], str] | None = None,
) -> SilverboxDevelopmentRun:
    """Execute one bounded proposal → train-only fit → development selection.

    Dependencies are injectable for offline orchestration tests. The default
    path always uses the fixed Silverbox loader, fixed Qwen proposer, checked
    first-fit API, and separate selection API.
    """

    started_monotonic = monotonic()
    if not math.isfinite(started_monotonic):
        raise ValueError("monotonic clock must return a finite timestamp")
    started_unix = time.time()
    run_id = (run_id_factory or _new_run_id)()
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a filesystem-safe 1–128 character identifier")

    root = Path(receipt_root)
    stage_directory = root / "_orchestration" / run_id
    root.mkdir(parents=True, exist_ok=True)
    if (root / run_id).exists():
        raise FileExistsError(f"fit receipt directory already exists for run ID {run_id}")
    stage_directory.mkdir(parents=True, exist_ok=False)
    _write_stage_receipt(
        stage_directory / "run_started.json",
        {
            "protocol_id": "silverbox-first-fit-2026-09-25",
            "run_id": run_id,
            "status": "started",
            "started_unix": started_unix,
            "started_monotonic": started_monotonic,
            "wall_limit_seconds": PILOT_WALL_SECONDS,
        },
    )

    try:
        verified_archive, source_sha256, expected_bytes = _verify_archive(
            manifest_path=Path(manifest_path), archive_path=None if archive_path is None else Path(archive_path)
        )
    except Exception as error:
        return _finish_unresolved(
            run_id,
            stage_directory,
            "source_manifest_or_archive_verification_failed",
            "source_failed.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "validation_accessed": False,
            },
        )
    _write_stage_receipt(
        stage_directory / "source_verified.json",
        {
            "run_id": run_id,
            "status": "verified",
            "manifest_path": str(Path(manifest_path).resolve()),
            "archive_path": str(verified_archive),
            "archive_bytes": expected_bytes,
            "archive_sha256": source_sha256,
        },
    )

    try:
        if loader is None:
            development = load_silverbox_controlled_development(
                verified_archive,
                expected_source_sha256=source_sha256,
                expected_source_bytes=expected_bytes,
            )
        else:
            development = loader(verified_archive)
        train_windows = development.train_windows
        if not isinstance(train_windows, tuple) or len(train_windows) != 4:
            raise ValueError("development loader did not return the four fixed training windows")
    except Exception as error:
        return _finish_unresolved(
            run_id,
            stage_directory,
            "development_data_load_failed",
            "data_load_failed.json",
            {"error_type": type(error).__name__, "error": str(error), "validation_accessed": False},
            source_sha256=source_sha256,
        )
    _write_stage_receipt(
        stage_directory / "data_loaded.json",
        {
            "run_id": run_id,
            "status": "loaded",
            "train_window_count": len(train_windows),
            "train_window_source_ranges": [
                [window.source_start, window.source_stop] for window in train_windows
            ],
            "validation_accessed": False,
        },
    )

    if _pilot_expired(started_monotonic, monotonic()):
        return _finish_unresolved(
            run_id,
            stage_directory,
            "wall_time_limit_before_proposal",
            "proposal_skipped.json",
            {"validation_accessed": False},
            source_sha256=source_sha256,
        )

    request_fn = proposal_request or request_silverbox_proposal
    proposal_receipt_directory = stage_directory / "proposal_receipts"
    # This directory is an empty, run-local capability: the one proposal call
    # is allowed to place exactly one receipt here, and the verifier rejects
    # any receipt that resolves elsewhere.
    proposal_receipt_directory.mkdir(parents=False, exist_ok=False)
    _write_stage_receipt(
        stage_directory / "proposal_dispatching.json",
        {"run_id": run_id, "status": "dispatching", "model_id": MODEL_ID},
    )
    request_started = monotonic()
    proposal_result: Any = None
    try:
        proposal_result = request_fn(receipt_dir=proposal_receipt_directory)
        request_elapsed = max(0.0, monotonic() - request_started)
        proposal_receipt = _verify_proposal_receipt(
            proposal_result, expected_receipt_dir=proposal_receipt_directory
        )
    except Exception as error:
        receipt_path = getattr(error, "receipt_path", None)
        if receipt_path is None and proposal_result is not None:
            receipt_path = getattr(proposal_result, "receipt_path", None)
        _write_stage_receipt(
            stage_directory / "proposal_failed.json",
            {
                "run_id": run_id,
                "status": "failed",
                "failure_code": getattr(error, "failure_code", "proposal_or_receipt_failure"),
                "error_type": type(error).__name__,
                "error": str(error),
                "proposal_receipt_path": str(receipt_path) if receipt_path is not None else None,
                "proposal_receipt_file_sha256": _optional_file_sha256(receipt_path),
                "validation_accessed": False,
            },
        )
        return _finish_unresolved(
            run_id,
            stage_directory,
            "terminal_qwen_proposal_failure",
            None,
            {},
            source_sha256=source_sha256,
        )

    proposal_sha256 = proposal_receipt["receipt_sha256"]
    if request_elapsed >= PROPOSAL_MAX_WALL_SECONDS:
        return _finish_unresolved(
            run_id,
            stage_directory,
            "proposal_request_wall_time_limit",
            "proposal_failed.json",
            {
                "status": "failed",
                "failure_code": "proposal_request_wall_time_limit",
                "request_elapsed_seconds": request_elapsed,
                "proposal_receipt_sha256": proposal_sha256,
                "validation_accessed": False,
            },
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
        )
    _write_stage_receipt(
        stage_directory / "proposal_succeeded.json",
        {
            "run_id": run_id,
            "status": "success",
            "model_id": MODEL_ID,
            "term_id": proposal_receipt["term_id"],
            "proposal_receipt_path": str(proposal_receipt["receipt_path"]),
            "proposal_receipt_sha256": proposal_sha256,
            "request_payload_sha256": proposal_receipt["request_payload_sha256"],
            "response_sha256": proposal_receipt.get("response_sha256"),
            "falsifying_prediction_status": "unscored_selection_independent",
            "request_elapsed_seconds": request_elapsed,
            "validation_accessed": False,
        },
    )

    if _pilot_expired(started_monotonic, monotonic()):
        return _finish_unresolved(
            run_id,
            stage_directory,
            "wall_time_limit_before_fit",
            "fit_skipped.json",
            {"validation_accessed": False, "proposal_receipt_sha256": proposal_sha256},
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
        )

    fit_fn = fit or fit_silverbox_development
    try:
        fit_result = fit_fn(
            train_windows,
            proposal_receipt["term_id"],
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
            run_id=run_id,
            receipt_dir=root,
            pilot_started_monotonic=started_monotonic,
        )
    except Exception as error:
        _write_stage_receipt(
            stage_directory / "fit_failed.json",
            {
                "run_id": run_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "proposal_receipt_sha256": proposal_sha256,
                "validation_accessed": False,
            },
        )
        return _finish_unresolved(
            run_id,
            stage_directory,
            "fit_failed",
            None,
            {},
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
        )

    fit_status = getattr(fit_result, "status", None)
    fit_receipt_path = getattr(fit_result, "receipt_path", None)
    fit_receipt_sha256 = getattr(fit_result, "receipt_sha256", None)
    fit_deadline_exceeded = _pilot_expired(started_monotonic, monotonic())
    _write_stage_receipt(
        stage_directory / "fit_completed.json",
        {
            "run_id": run_id,
            "status": "unresolved" if fit_deadline_exceeded else fit_status,
            "reason": "wall_time_limit" if fit_deadline_exceeded else None,
            "fit_receipt_path": str(fit_receipt_path) if fit_receipt_path is not None else None,
            "fit_receipt_sha256": fit_receipt_sha256,
            "proposal_receipt_sha256": proposal_sha256,
            "validation_accessed": False,
        },
    )

    if fit_deadline_exceeded:
        return _finish_unresolved(
            run_id,
            stage_directory,
            "wall_time_limit_after_fit",
            "selection_skipped.json",
            {
                "status": "unresolved",
                "fit_status": fit_status,
                "fit_receipt_sha256": fit_receipt_sha256,
                "validation_accessed": False,
            },
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
            fit=fit_result,
        )

    if fit_status != "complete":
        reason = "fit_incomplete" if fit_status == "incomplete" else "fit_returned_invalid_status"
        return _finish_unresolved(
            run_id,
            stage_directory,
            reason,
            "selection_skipped.json",
            {
                "status": "unresolved",
                "fit_status": fit_status,
                "fit_receipt_sha256": fit_receipt_sha256,
                "validation_accessed": False,
            },
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
            fit=fit_result,
        )

    if _pilot_expired(started_monotonic, monotonic()):
        return _finish_unresolved(
            run_id,
            stage_directory,
            "wall_time_limit_before_selection",
            "selection_skipped.json",
            {"validation_accessed": False, "fit_receipt_sha256": fit_receipt_sha256},
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
            fit=fit_result,
        )

    select_fn = select or select_silverbox_development
    try:
        selection_result = select_fn(fit_result, development.validation)
    except Exception as error:
        _write_stage_receipt(
            stage_directory / "selection_failed.json",
            {
                "run_id": run_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "fit_receipt_sha256": fit_receipt_sha256,
                "validation_accessed": True,
            },
        )
        return _finish_unresolved(
            run_id,
            stage_directory,
            "development_selection_failed",
            None,
            {},
            source_sha256=source_sha256,
            proposal_receipt_sha256=proposal_sha256,
            fit=fit_result,
        )

    selection_status = getattr(selection_result, "status", None)
    selection_api_status = selection_status
    selection_reason = getattr(selection_result, "reason", "selection_returned_invalid_status")
    selection_selected_hypothesis = getattr(selection_result, "selected_hypothesis", None)
    selection_deadline_exceeded = _pilot_expired(started_monotonic, monotonic())
    if selection_deadline_exceeded:
        selection_status = "unresolved"
        selection_reason = "wall_time_limit"
        selection_selected_hypothesis = None
    selection_stage_payload = {
        "run_id": run_id,
        "status": selection_status,
        "reason": selection_reason,
        "selected_hypothesis": selection_selected_hypothesis,
        "selection_receipt_path": (
            str(selection_result.receipt_path)
            if getattr(selection_result, "receipt_path", None) is not None
            else None
        ),
        "selection_receipt_sha256": getattr(selection_result, "receipt_sha256", None),
        "fit_receipt_sha256": fit_receipt_sha256,
        "validation_accessed": True,
    }
    selection_stage_path = stage_directory / "selection_completed.json"
    _write_stage_receipt(selection_stage_path, selection_stage_payload)
    status = "selected" if selection_status == "selected" else "unresolved"
    run_finished_deadline_exceeded = _pilot_expired(started_monotonic, monotonic())
    if run_finished_deadline_exceeded:
        status = "unresolved"
        selection_reason = "wall_time_limit"
    selection_blocked_by_deadline = selection_deadline_exceeded or run_finished_deadline_exceeded
    if selection_blocked_by_deadline:
        selection_stage_payload.update(
            {"status": "unresolved", "reason": "wall_time_limit", "selected_hypothesis": None}
        )
        _write_stage_receipt(selection_stage_path, selection_stage_payload)

    run_finished_path = stage_directory / "run_finished.json"

    def write_run_finished() -> None:
        _write_stage_receipt(
            run_finished_path,
            {
                "run_id": run_id,
                "status": status,
                "reason": selection_reason,
                "source_sha256": source_sha256,
                "proposal_receipt_sha256": proposal_sha256,
                "fit_receipt_sha256": fit_receipt_sha256,
                "selection_receipt_sha256": getattr(selection_result, "receipt_sha256", None),
                "finished_unix": time.time(),
            },
        )

    write_run_finished()
    # Final receipt serialization is in the same wall-time budget. If it
    # crossed the limit while writing, atomically downgrade both stage and
    # terminal receipts so no selected run can survive reconstruction.
    if status == "selected" and _pilot_expired(started_monotonic, monotonic()):
        status = "unresolved"
        selection_reason = "wall_time_limit"
        selection_blocked_by_deadline = True
        selection_stage_payload.update(
            {"status": "unresolved", "reason": "wall_time_limit", "selected_hypothesis": None}
        )
        _write_stage_receipt(selection_stage_path, selection_stage_payload)
        write_run_finished()
    return SilverboxDevelopmentRun(
        run_id,
        status,
        str(selection_reason),
        stage_directory,
        source_sha256,
        proposal_sha256,
        fit_result,
        (
            None
            if selection_api_status == "selected"
            and selection_blocked_by_deadline
            else selection_result
        ),
    )


def _verify_archive(
    *, manifest_path: Path, archive_path: Path | None
) -> tuple[Path, str, int]:
    manifest = _read_json_no_duplicate_keys(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("source"), dict):
        raise ValueError("tracked Silverbox manifest has no source object")
    source = manifest["source"]
    if source.get("archive_name") != "SilverboxFiles.zip":
        raise ValueError("tracked manifest does not identify the official Silverbox archive")
    expected_sha256 = source.get("raw_archive_sha256")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("tracked manifest has no valid lowercase archive SHA-256")
    expected_bytes = source.get("raw_archive_bytes")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("tracked manifest has no valid archive byte count")
    if archive_path is None:
        manifest_archive_path = source.get("raw_archive_path")
        if not isinstance(manifest_archive_path, str) or not manifest_archive_path:
            raise ValueError("tracked manifest has no raw archive path")
        resolved_archive = Path(manifest_archive_path)
        if not resolved_archive.is_absolute():
            resolved_archive = REPOSITORY_ROOT / resolved_archive
    else:
        resolved_archive = archive_path
    resolved_archive = resolved_archive.resolve()
    observed_sha256, observed_bytes = _sha256_file(resolved_archive)
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"official archive byte count mismatch: expected {expected_bytes}, observed {observed_bytes}"
        )
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"official archive SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}"
        )
    return resolved_archive, observed_sha256, observed_bytes


def _verify_proposal_receipt(
    result: Any, *, expected_receipt_dir: str | Path
) -> dict[str, Any]:
    receipt_path = getattr(result, "receipt_path", None)
    returned_digest = getattr(result, "receipt_sha256", None)
    if not isinstance(receipt_path, (str, Path)):
        raise ValueError("proposal result has no receipt path")
    if not isinstance(returned_digest, str) or not _SHA256_RE.fullmatch(returned_digest):
        raise ValueError("proposal result has no valid lowercase receipt SHA-256")
    expected_directory = Path(expected_receipt_dir).resolve(strict=True)
    supplied_path = Path(receipt_path)
    if supplied_path.is_symlink():
        raise ValueError("proposal receipt must be a regular file in the fresh run receipt directory")
    resolved_path = supplied_path.resolve(strict=True)
    if resolved_path.parent != expected_directory or not resolved_path.is_file():
        raise ValueError("proposal receipt is outside this run's fresh proposal_receipts directory")
    directory_entries = tuple(expected_directory.iterdir())
    if len(directory_entries) != 1 or directory_entries[0].resolve(strict=True) != resolved_path:
        raise ValueError("fresh proposal_receipts directory must contain exactly the returned receipt")
    raw_bytes = resolved_path.read_bytes()
    observed_digest = hashlib.sha256(raw_bytes).hexdigest()
    if observed_digest != returned_digest:
        raise ValueError("proposal receipt bytes do not match the returned receipt SHA-256")
    receipt = _json_no_duplicate_keys(raw_bytes)
    if not isinstance(receipt, dict):
        raise ValueError("proposal receipt must be one JSON object")

    request_payload = receipt.get("request_payload")
    if not isinstance(request_payload, dict):
        raise ValueError("proposal receipt has no frozen request payload")
    canonical_payload = json.dumps(
        request_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    observed_payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    if (
        request_payload.get("model") != MODEL_ID
        or observed_payload_sha256 != FROZEN_REQUEST_PAYLOAD_SHA256
        or receipt.get("request_payload_sha256") != observed_payload_sha256
    ):
        raise ValueError("proposal receipt request payload does not match the frozen request")

    if (
        receipt.get("endpoint") != ENDPOINT
        or receipt.get("model_preflight_endpoint") != MODELS_ENDPOINT
        or receipt.get("model_preflight_status") != "success"
        or receipt.get("model_preflight_http_status") != 200
        or not _fixed_timeout(receipt.get("timeout_seconds"))
        or not _fixed_timeout(receipt.get("model_preflight_timeout_seconds"))
    ):
        raise ValueError("proposal receipt does not record the fixed successful 120-second request/preflight")
    preflight_body = _decode_receipt_base64(receipt.get("model_preflight_raw_response_base64"))
    preflight_body_sha256 = hashlib.sha256(preflight_body).hexdigest()
    if receipt.get("model_preflight_response_sha256") != preflight_body_sha256:
        raise ValueError("proposal preflight response bytes do not match their recorded SHA-256")
    model_listing = _json_no_duplicate_keys(preflight_body)
    if (
        not isinstance(model_listing, dict)
        or "error" in model_listing
        or not isinstance(model_listing.get("data"), list)
    ):
        raise ValueError("proposal receipt preflight response has an invalid model-list shape")
    observed_model_ids = [
        item["id"]
        for item in model_listing["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if (
        receipt.get("model_preflight_model_ids") != observed_model_ids
        or MODEL_ID not in observed_model_ids
    ):
        raise ValueError("proposal receipt preflight model list does not prove the exact model ID")

    response_body = _decode_receipt_base64(receipt.get("raw_response_base64"))
    response_sha256 = hashlib.sha256(response_body).hexdigest()
    if receipt.get("response_sha256") != response_sha256:
        raise ValueError("proposal response bytes do not match their recorded SHA-256")
    response_envelope = _json_no_duplicate_keys(response_body)
    if (
        receipt.get("http_status") != 200
        or receipt.get("finish_reason") != "stop"
        or not isinstance(response_envelope, dict)
        or "error" in response_envelope
        or response_envelope.get("model") != MODEL_ID
        or not isinstance(response_envelope.get("choices"), list)
        or len(response_envelope["choices"]) != 1
    ):
        raise ValueError("proposal receipt does not record a successful fixed-model chat response")
    choice = response_envelope["choices"][0]
    if (
        not isinstance(choice, dict)
        or choice.get("finish_reason") != "stop"
        or not isinstance(choice.get("message"), dict)
        or choice["message"].get("role") != "assistant"
        or not isinstance(choice["message"].get("content"), str)
    ):
        raise ValueError("proposal raw response does not contain one complete assistant answer")
    assistant_proposal = _json_no_duplicate_keys(choice["message"]["content"].encode("utf-8"))
    if (
        not isinstance(assistant_proposal, dict)
        or set(assistant_proposal) != {"term_id", "falsifying_prediction"}
        or assistant_proposal.get("term_id") not in ALLOWED_TERM_IDS
        or not isinstance(assistant_proposal.get("falsifying_prediction"), str)
        or not assistant_proposal["falsifying_prediction"].strip()
        or len(assistant_proposal["falsifying_prediction"]) > 240
    ):
        raise ValueError("proposal raw assistant answer violates the frozen proposal schema")

    term_id = getattr(result, "term_id", None)
    if (
        receipt.get("status") != "success"
        or receipt.get("parser_outcome") != "success"
        or receipt.get("returned_model") != MODEL_ID
        or receipt.get("chat_dispatched") is not True
        or receipt.get("term_id") not in ALLOWED_TERM_IDS
        or receipt.get("term_id") != term_id
        or receipt.get("term_id") != assistant_proposal["term_id"]
        or receipt.get("falsifying_prediction")
        != getattr(result, "falsifying_prediction", None)
        or receipt.get("falsifying_prediction")
        != assistant_proposal["falsifying_prediction"]
        or receipt.get("falsifying_prediction_status") != "unscored_selection_independent"
    ):
        raise ValueError("proposal receipt status, model, term, or answer does not match the result")
    if getattr(result, "receipt_sha256", None) != observed_digest:
        raise ValueError("proposal handle digest changed while verifying receipt")
    return {
        "receipt_path": resolved_path,
        "receipt_sha256": observed_digest,
        "request_payload_sha256": receipt["request_payload_sha256"],
        "response_sha256": response_sha256,
        "term_id": receipt["term_id"],
    }


def _fixed_timeout(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and float(value) == REQUEST_TIMEOUT_SECONDS


def _finish_unresolved(
    run_id: str,
    stage_directory: Path,
    reason: str,
    stage_filename: str | None,
    stage_payload: Mapping[str, Any],
    *,
    source_sha256: str | None = None,
    proposal_receipt_sha256: str | None = None,
    fit: Any = None,
) -> SilverboxDevelopmentRun:
    payload = {"run_id": run_id, "reason": reason, **dict(stage_payload)}
    if stage_filename is not None:
        _write_stage_receipt(stage_directory / stage_filename, payload)
    _write_stage_receipt(
        stage_directory / "run_finished.json",
        {
            "run_id": run_id,
            "status": "unresolved",
            "reason": reason,
            "source_sha256": source_sha256,
            "proposal_receipt_sha256": proposal_receipt_sha256,
            "fit_receipt_sha256": getattr(fit, "receipt_sha256", None),
            "validation_accessed": bool(payload.get("validation_accessed", False)),
            "finished_unix": time.time(),
        },
    )
    return SilverboxDevelopmentRun(
        run_id,
        "unresolved",
        reason,
        stage_directory,
        source_sha256,
        proposal_receipt_sha256,
        fit,
        None,
    )


def _write_stage_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(encoded)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def _read_json_no_duplicate_keys(path: Path) -> Any:
    return _json_no_duplicate_keys(path.read_bytes())


def _json_no_duplicate_keys(raw_bytes: bytes) -> Any:
    return json.loads(
        raw_bytes.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_json_constant,
    )


def _decode_receipt_base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ValueError("proposal receipt has no raw response bytes")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("proposal receipt raw response is not valid base64") from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _optional_file_sha256(value: Any) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        return _sha256_file(Path(value))[0]
    except OSError:
        return None


def _pilot_expired(started_monotonic: float, now_monotonic: float) -> bool:
    return not math.isfinite(now_monotonic) or now_monotonic - started_monotonic >= PILOT_WALL_SECONDS


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"silverbox-{stamp}-{uuid.uuid4().hex[:12]}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="path to the official archive verified against the manifest")
    parser.add_argument(
        "--receipt-root",
        type=Path,
        default=DEFAULT_RECEIPT_ROOT,
        help="ignored local directory for fit and orchestration receipts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outcome = run_silverbox_development(archive_path=args.archive, receipt_root=args.receipt_root)
    print(
        json.dumps(
            {
                "run_id": outcome.run_id,
                "status": outcome.status,
                "reason": outcome.reason,
                "stage_directory": str(outcome.stage_directory),
                "source_sha256": outcome.source_sha256,
                "proposal_receipt_sha256": outcome.proposal_receipt_sha256,
                "fit_receipt_sha256": getattr(outcome.fit, "receipt_sha256", None),
                "selection_receipt_sha256": getattr(outcome.selection, "receipt_sha256", None),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if outcome.status == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())

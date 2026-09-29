"""Descriptor-relative readers for ABC6 campaign evidence and status receipts.

This module deliberately depends only on the Python standard library. It keeps
the campaign-fit evidence verifier independent from the deferred-target gate,
while providing the same bounded, no-follow reads used for durable receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Final

STATUS_RECEIPT_SCHEMA_VERSION: Final = 1
REQUIRED_CASE_COUNT: Final = 24
MAX_STATUS_RECEIPT_BYTES: Final = 4096
_STATUS_COMPONENTS: Final = ("fit", "baseline")
_FINAL_STATUSES: Final = frozenset({"complete", "incomplete", "unresolved", "failed"})
_STATUS_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_id",
        "case_index",
        "case_id",
        "component",
        "status",
        "payload_sha256",
    }
)


class ABC6ReceiptIOError(ValueError):
    """A bounded descriptor-relative receipt read or validation failed."""


def read_regular_file_at(
    directory_fd: int,
    filename: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one bounded regular-file leaf without following a symlink."""

    if (
        not isinstance(filename, str)
        or filename in {"", ".", ".."}
        or Path(filename).name != filename
    ):
        raise ABC6ReceiptIOError(f"{label} filename is not a leaf")
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise ABC6ReceiptIOError("maximum_bytes must be a nonnegative integer")

    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise ABC6ReceiptIOError(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise ABC6ReceiptIOError(f"{label} exceeds its size limit")
        return b"".join(chunks)
    except ABC6ReceiptIOError:
        raise
    except FileNotFoundError as error:
        if label == "status receipt":
            raise ABC6ReceiptIOError(
                f"missing durable receipt: {filename}"
            ) from error
        raise ABC6ReceiptIOError(f"{label} is missing") from error
    except OSError as error:
        raise ABC6ReceiptIOError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_status_receipt_bytes(
    raw: bytes,
    filename: str,
    *,
    case_index: int,
    case_id: str,
    component: str,
    protocol_id: str,
    run_id: str,
) -> tuple[str, str]:
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ABC6ReceiptIOError(f"receipt is unreadable: {filename}") from error
    if not isinstance(decoded, dict):
        raise ABC6ReceiptIOError(f"receipt must contain a JSON object: {filename}")
    if len(raw) > MAX_STATUS_RECEIPT_BYTES:
        raise ABC6ReceiptIOError(f"receipt is too large: {filename}")
    if set(decoded) != _STATUS_RECEIPT_FIELDS:
        raise ABC6ReceiptIOError(f"receipt fields do not match schema: {filename}")
    if (
        type(decoded["schema_version"]) is not int
        or type(decoded["case_index"]) is not int
    ):
        raise ABC6ReceiptIOError(
            f"receipt integer fields have wrong types: {filename}"
        )
    if any(
        not isinstance(decoded[key], str)
        for key in (
            "protocol_id",
            "run_id",
            "case_id",
            "component",
            "status",
            "payload_sha256",
        )
    ):
        raise ABC6ReceiptIOError(
            f"receipt text fields have wrong types: {filename}"
        )
    expected_identity = {
        "schema_version": STATUS_RECEIPT_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "run_id": run_id,
        "case_index": case_index,
        "case_id": case_id,
        "component": component,
    }
    if any(decoded.get(key) != value for key, value in expected_identity.items()):
        raise ABC6ReceiptIOError(f"receipt identity mismatch: {filename}")
    status = decoded["status"]
    if status not in _FINAL_STATUSES:
        raise ABC6ReceiptIOError(f"receipt status is not final: {filename}")
    body = {key: value for key, value in decoded.items() if key != "payload_sha256"}
    expected_payload_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    if (
        decoded["payload_sha256"] != expected_payload_sha256
        or raw != _canonical_json(decoded)
    ):
        raise ABC6ReceiptIOError(f"receipt hash or encoding mismatch: {filename}")
    return status, hashlib.sha256(raw).hexdigest()


def verify_status_receipts_at(
    receipt_directory_fd: int,
    *,
    protocol_id: str,
    run_id: str,
    case_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Verify the ordered fit/baseline receipt set against a frozen roster."""

    if not isinstance(protocol_id, str) or not isinstance(run_id, str):
        raise ABC6ReceiptIOError("status receipt identity must use text values")
    if (
        not isinstance(case_ids, tuple)
        or len(case_ids) != REQUIRED_CASE_COUNT
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ABC6ReceiptIOError("ordered status receipt case IDs are invalid")

    verified: list[tuple[str, str]] = []
    expected_names: set[str] = set()
    for case_index, case_id in enumerate(case_ids):
        for component in _STATUS_COMPONENTS:
            filename = f"case-{case_index:02d}.{component}-status.json"
            expected_names.add(filename)
            raw = read_regular_file_at(
                receipt_directory_fd,
                filename,
                maximum_bytes=MAX_STATUS_RECEIPT_BYTES,
                label="status receipt",
            )
            _status, digest = _validate_status_receipt_bytes(
                raw,
                filename,
                case_index=case_index,
                case_id=case_id,
                component=component,
                protocol_id=protocol_id,
                run_id=run_id,
            )
            verified.append((filename, digest))
    try:
        entries = set(os.listdir(receipt_directory_fd))
    except OSError as error:
        raise ABC6ReceiptIOError(
            "cannot enumerate the anchored status receipt directory"
        ) from error
    unexpected = {
        name
        for name in entries
        if name.startswith("case-") and "-status.json" in name
    } - expected_names
    if unexpected:
        raise ABC6ReceiptIOError(
            "unexpected fit/baseline status receipt(s): "
            + ", ".join(sorted(unexpected))
        )
    return tuple(verified)


__all__ = (
    "ABC6ReceiptIOError",
    "MAX_STATUS_RECEIPT_BYTES",
    "REQUIRED_CASE_COUNT",
    "STATUS_RECEIPT_SCHEMA_VERSION",
    "read_regular_file_at",
    "verify_status_receipts_at",
)

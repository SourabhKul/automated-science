"""One-shot, fixed-prompt Qwen proposal for the Silverbox first-fit pilot.

This module deliberately has no fallback, retry, repair, or model selection
logic. A successful call returns only a validated ``term_id``. The accompanying
``falsifying_prediction`` is archived as unscored receipt metadata.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODELS_ENDPOINT = "http://127.0.0.1:8000/v1/models"
REQUEST_TIMEOUT_SECONDS = 120.0
FROZEN_REQUEST_PAYLOAD_SHA256 = "59823e7cbcae3a9c92340c7582462872dcb218e7377230432caa3b44bcf5f038"
RECEIPT_DIRECTORY = Path("artifacts/silverbox_qwen_proposal_receipts")
ALLOWED_TERM_IDS = frozenset({"y_cubed", "u_cubed", "u_y_product"})

SYSTEM_MESSAGE = (
    "You propose one compact nonlinear extension to a forced observed-signal recurrence. "
    "Use only the training facts provided. Return exactly one JSON object with keys term_id "
    "and falsifying_prediction, no markdown or other text. term_id must be one of y_cubed, "
    "u_cubed, u_y_product. falsifying_prediction must be one short sentence about an input "
    "or amplitude condition under which the proposed term should matter."
)

USER_MESSAGE = (
    "A measured signal y responds to a recorded input u at 610.35 samples/second. On 824 "
    "training prediction steps in four separated windows, a linear recurrence "
    "y[k]=a1*y[k-1]+a2*y[k-2]+b*u[k-1]+c fitted a1=1.4631070, a2=-0.9361293, "
    "b=0.4190676, c=-0.00213506. Training one-step residual RMS was 0.00112682 and "
    "free-run RMS was 0.00719364. Correlation of one-step residuals with observed-lag "
    "features was -0.290589 for y[k-1]^3, +0.0503233 for u[k-1]^3, and +0.0669048 for "
    "u[k-1]*y[k-1]. The training input lay roughly in [-0.08,0.10] and the output in "
    "[-0.21,0.21]. Choose exactly one feature as the next explicit hypothesis. The allowed "
    "term_id mapping is y_cubed -> y[k-1]^3, u_cubed -> u[k-1]^3, u_y_product -> "
    "u[k-1]*y[k-1]. Do not infer a named physical mechanism from these observations."
)

_REQUIRED_PROPOSAL_KEYS = frozenset({"term_id", "falsifying_prediction"})
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class DuplicateJSONKeyError(ValueError):
    """Raised when any JSON object repeats a key."""


class SilverboxProposalError(RuntimeError):
    """Terminal proposal failure; the associated receipt records the outcome."""

    def __init__(self, failure_code: str, receipt_path: Path):
        super().__init__(f"Silverbox Qwen proposal failed: {failure_code}")
        self.failure_code = failure_code
        self.receipt_path = receipt_path


@dataclass(frozen=True)
class ChatHTTPResponse:
    """HTTP status and exact response bytes returned by an injected transport."""

    status_code: int
    body: bytes

    def __post_init__(self) -> None:
        if type(self.status_code) is not int:
            raise TypeError("ChatHTTPResponse.status_code must be an integer")
        if not isinstance(self.body, bytes):
            raise TypeError("ChatHTTPResponse.body must be bytes")


@dataclass(frozen=True)
class SilverboxProposalResult:
    """Validated proposal plus the exact successful receipt identity."""

    term_id: str
    falsifying_prediction: str
    receipt_path: Path
    receipt_sha256: str


class _ResponseRejected(Exception):
    def __init__(self, failure_code: str, parser_outcome: str, metadata: Mapping[str, Any]):
        super().__init__(failure_code)
        self.failure_code = failure_code
        self.parser_outcome = parser_outcome
        self.metadata = dict(metadata)


def _build_request_payload() -> dict[str, Any]:
    """Build a fresh copy of the protocol's frozen chat-completions payload."""

    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the fixed endpoint fixed and ensure no redirect dispatch occurs."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _http_post_once(url: str, request_body: bytes, timeout_seconds: float) -> ChatHTTPResponse:
    """Send one POST without redirects or retries."""

    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return ChatHTTPResponse(status_code=int(response.status), body=response.read())
    except urllib.error.HTTPError as error:
        return ChatHTTPResponse(status_code=int(error.code), body=error.read())


def _http_get_once(url: str, timeout_seconds: float) -> ChatHTTPResponse:
    """Read the model-list endpoint once without redirects or retries."""

    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return ChatHTTPResponse(status_code=int(response.status), body=response.read())
    except urllib.error.HTTPError as error:
        return ChatHTTPResponse(status_code=int(error.code), body=error.read())


Transport = Callable[[str, bytes, float], ChatHTTPResponse]
ReadTransport = Callable[[str, float], ChatHTTPResponse]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one receipt in the same ignored directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
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
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _receipt_path(receipt_dir: str | Path | None) -> Path:
    directory = Path(receipt_dir) if receipt_dir is not None else RECEIPT_DIRECTORY
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return directory / f"silverbox_qwen_proposal_{stamp}_{uuid.uuid4().hex}.json"


def _base_receipt(payload: Mapping[str, Any], payload_bytes: bytes) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "completed_at_utc": None,
        "status": "preflight_pending",
        "endpoint": ENDPOINT,
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "chat_dispatched": False,
        "model_preflight_endpoint": MODELS_ENDPOINT,
        "model_preflight_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "model_preflight_status": "pending",
        "model_preflight_http_status": None,
        "model_preflight_response_sha256": None,
        "model_preflight_raw_response_base64": None,
        "model_preflight_model_ids": None,
        "model_preflight_failure_type": None,
        "request_payload": payload,
        "request_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "http_status": None,
        "returned_model": None,
        "finish_reason": None,
        "usage": None,
        "response_sha256": None,
        "raw_response_base64": None,
        "parser_outcome": "not_run",
        "term_id": None,
        "falsifying_prediction": None,
        "falsifying_prediction_status": "unscored_selection_independent",
        "failure_code": None,
        "transport_failure_type": None,
    }


def _response_metadata(envelope: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"returned_model": None, "finish_reason": None, "usage": None}
    if not isinstance(envelope, dict):
        return metadata
    model = envelope.get("model")
    if isinstance(model, str):
        metadata["returned_model"] = model
    if "usage" in envelope:
        metadata["usage"] = envelope["usage"]
    choices = envelope.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
        if isinstance(finish_reason, str):
            metadata["finish_reason"] = finish_reason
    return metadata


def _parse_model_preflight(response: ChatHTTPResponse) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_preflight_http_status": response.status_code,
        "model_preflight_response_sha256": hashlib.sha256(response.body).hexdigest(),
        "model_preflight_raw_response_base64": base64.b64encode(response.body).decode("ascii"),
    }
    if response.status_code != 200:
        raise _ResponseRejected(
            "preflight_http_status_not_200",
            "preflight_http_status_rejected",
            metadata,
        )
    try:
        model_listing = json.loads(
            response.body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except DuplicateJSONKeyError as error:
        raise _ResponseRejected(
            "preflight_duplicate_json_key",
            "preflight_duplicate_key_rejected",
            metadata,
        ) from error
    except UnicodeDecodeError as error:
        raise _ResponseRejected(
            "preflight_response_not_utf8",
            "preflight_invalid_json",
            metadata,
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise _ResponseRejected(
            "preflight_invalid_json",
            "preflight_invalid_json",
            metadata,
        ) from error

    if not isinstance(model_listing, dict):
        raise _ResponseRejected("preflight_response_not_object", "preflight_invalid_schema", metadata)
    if "error" in model_listing:
        raise _ResponseRejected("preflight_api_error", "preflight_api_error", metadata)
    if "data" not in model_listing or not isinstance(model_listing.get("data"), list):
        raise _ResponseRejected("preflight_models_missing", "preflight_missing_required_field", metadata)
    model_ids = [
        item["id"]
        for item in model_listing["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    metadata["model_preflight_model_ids"] = model_ids
    if MODEL_ID not in model_ids:
        raise _ResponseRejected("preflight_model_id_missing", "preflight_model_id_rejected", metadata)
    return metadata


def _parse_assistant_content(content: str) -> dict[str, str]:
    try:
        proposal = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except DuplicateJSONKeyError as error:
        raise _ResponseRejected("duplicate_assistant_json_key", "duplicate_key_rejected", {}) from error
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise _ResponseRejected("invalid_assistant_json", "invalid_assistant_json", {}) from error

    if not isinstance(proposal, dict):
        raise _ResponseRejected("assistant_json_not_object", "schema_rejected", {})
    if set(proposal) != _REQUIRED_PROPOSAL_KEYS:
        raise _ResponseRejected("assistant_json_keys_mismatch", "schema_rejected", {})
    term_id = proposal.get("term_id")
    falsifying_prediction = proposal.get("falsifying_prediction")
    if not isinstance(term_id, str) or not isinstance(falsifying_prediction, str):
        raise _ResponseRejected("assistant_json_values_must_be_strings", "schema_rejected", {})
    if term_id not in ALLOWED_TERM_IDS:
        raise _ResponseRejected("term_id_not_allowed", "schema_rejected", {})
    if not falsifying_prediction.strip() or not 1 <= len(falsifying_prediction) <= 240:
        raise _ResponseRejected("falsifying_prediction_invalid_length", "schema_rejected", {})
    return {"term_id": term_id, "falsifying_prediction": falsifying_prediction}


def _parse_chat_response(body: bytes, http_status: int | None) -> tuple[dict[str, str], dict[str, Any]]:
    metadata: dict[str, Any] = {"returned_model": None, "finish_reason": None, "usage": None}
    try:
        response_text = body.decode("utf-8")
        envelope = json.loads(
            response_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except DuplicateJSONKeyError as error:
        raise _ResponseRejected("duplicate_response_json_key", "duplicate_key_rejected", metadata) from error
    except UnicodeDecodeError as error:
        raise _ResponseRejected("response_not_utf8", "invalid_envelope_json", metadata) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise _ResponseRejected("invalid_response_json", "invalid_envelope_json", metadata) from error

    metadata = _response_metadata(envelope)
    if http_status != 200:
        raise _ResponseRejected("http_status_not_200", "http_status_rejected", metadata)
    if not isinstance(envelope, dict):
        raise _ResponseRejected("response_json_not_object", "invalid_envelope_schema", metadata)
    if "error" in envelope:
        raise _ResponseRejected("api_error", "api_error", metadata)
    if "model" not in envelope or not isinstance(envelope.get("model"), str):
        raise _ResponseRejected("missing_returned_model", "missing_required_field", metadata)
    if envelope["model"] != MODEL_ID:
        raise _ResponseRejected("returned_model_mismatch", "model_rejected", metadata)
    if "choices" not in envelope or not isinstance(envelope.get("choices"), list) or not envelope["choices"]:
        raise _ResponseRejected("missing_choices", "missing_required_field", metadata)
    if len(envelope["choices"]) != 1:
        raise _ResponseRejected("unexpected_choice_count", "invalid_envelope_schema", metadata)

    choice = envelope["choices"][0]
    if not isinstance(choice, dict):
        raise _ResponseRejected("choice_not_object", "invalid_envelope_schema", metadata)
    if "finish_reason" not in choice:
        raise _ResponseRejected("missing_finish_reason", "missing_required_field", metadata)
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise _ResponseRejected("truncated_response", "truncated_finish", metadata)
    if finish_reason != "stop":
        raise _ResponseRejected("finish_reason_not_stop", "finish_reason_rejected", metadata)
    if "message" not in choice or not isinstance(choice.get("message"), dict):
        raise _ResponseRejected("missing_message", "missing_required_field", metadata)
    message = choice["message"]
    if message.get("role") != "assistant":
        raise _ResponseRejected("message_role_not_assistant", "assistant_role_rejected", metadata)
    if "content" not in message or not isinstance(message.get("content"), str):
        raise _ResponseRejected("missing_assistant_content", "missing_required_field", metadata)

    try:
        proposal = _parse_assistant_content(message["content"])
    except _ResponseRejected as error:
        error.metadata.update(metadata)
        raise
    return proposal, {**metadata, "parser_outcome": "success"}


def request_silverbox_proposal(
    *,
    transport: Transport | None = None,
    read_transport: ReadTransport | None = None,
    receipt_dir: str | Path | None = None,
) -> SilverboxProposalResult:
    """Make one fixed Qwen request and return its validated proposal receipt.

    The complete chat payload and hash are atomically persisted before model
    preflight. The separate read transport contract is ``(url, timeout_seconds)
    -> ChatHTTPResponse``; the chat transport contract is ``(url, body_bytes,
    timeout_seconds) -> ChatHTTPResponse``. A fresh exact-ID model-list check
    must pass before one chat request is dispatched. The prediction sentence is
    retained only in the ignored receipt and is explicitly marked unscored.
    The returned digest hashes the exact UTF-8 receipt file bytes.
    """

    request_payload = _build_request_payload()
    request_body = _canonical_json_bytes(request_payload)
    receipt_file = _receipt_path(receipt_dir)
    receipt = _base_receipt(request_payload, request_body)
    _atomic_write_json(receipt_file, receipt)
    if receipt["request_payload_sha256"] != FROZEN_REQUEST_PAYLOAD_SHA256:
        receipt.update(
            {
                "completed_at_utc": _utc_now(),
                "status": "failed",
                "parser_outcome": "frozen_payload_hash_mismatch",
                "failure_code": "frozen_payload_hash_mismatch",
            }
        )
        _atomic_write_json(receipt_file, receipt)
        raise SilverboxProposalError("frozen_payload_hash_mismatch", receipt_file)

    one_shot_read_transport = read_transport if read_transport is not None else _http_get_once
    try:
        model_response = one_shot_read_transport(MODELS_ENDPOINT, REQUEST_TIMEOUT_SECONDS)
        if not isinstance(model_response, ChatHTTPResponse):
            raise TypeError("read_transport must return ChatHTTPResponse")
    except Exception as error:
        receipt.update(
            {
                "completed_at_utc": _utc_now(),
                "status": "failed",
                "model_preflight_status": "failed",
                "model_preflight_response_sha256": _EMPTY_SHA256,
                "model_preflight_raw_response_base64": "",
                "model_preflight_failure_type": type(error).__name__,
                "parser_outcome": "preflight_transport_failed",
                "failure_code": "preflight_transport_failure",
            }
        )
        _atomic_write_json(receipt_file, receipt)
        raise SilverboxProposalError("preflight_transport_failure", receipt_file) from error

    try:
        preflight_metadata = _parse_model_preflight(model_response)
    except _ResponseRejected as error:
        receipt.update(error.metadata)
        receipt.update(
            {
                "completed_at_utc": _utc_now(),
                "status": "failed",
                "model_preflight_status": "failed",
                "parser_outcome": error.parser_outcome,
                "failure_code": error.failure_code,
            }
        )
        _atomic_write_json(receipt_file, receipt)
        raise SilverboxProposalError(error.failure_code, receipt_file) from error

    receipt.update(preflight_metadata)
    receipt.update(
        {
            "model_preflight_status": "success",
            "model_preflight_failure_type": None,
            "status": "dispatching",
            "chat_dispatched": True,
            "chat_dispatched_at_utc": _utc_now(),
        }
    )
    _atomic_write_json(receipt_file, receipt)

    one_shot_transport = transport if transport is not None else _http_post_once
    try:
        response = one_shot_transport(ENDPOINT, request_body, REQUEST_TIMEOUT_SECONDS)
        if not isinstance(response, ChatHTTPResponse):
            raise TypeError("transport must return ChatHTTPResponse")
    except Exception as error:
        receipt.update(
            {
                "completed_at_utc": _utc_now(),
                "status": "failed",
                "http_status": None,
                "response_sha256": _EMPTY_SHA256,
                "raw_response_base64": "",
                "parser_outcome": "transport_failed",
                "failure_code": "transport_failure",
                "transport_failure_type": type(error).__name__,
            }
        )
        _atomic_write_json(receipt_file, receipt)
        raise SilverboxProposalError("transport_failure", receipt_file) from error

    response_hash = hashlib.sha256(response.body).hexdigest()
    receipt.update(
        {
            "http_status": response.status_code,
            "response_sha256": response_hash,
            "raw_response_base64": base64.b64encode(response.body).decode("ascii"),
        }
    )
    try:
        proposal, metadata = _parse_chat_response(response.body, response.status_code)
    except _ResponseRejected as error:
        receipt.update(error.metadata)
        receipt.update(
            {
                "completed_at_utc": _utc_now(),
                "status": "failed",
                "parser_outcome": error.parser_outcome,
                "failure_code": error.failure_code,
            }
        )
        _atomic_write_json(receipt_file, receipt)
        raise SilverboxProposalError(error.failure_code, receipt_file) from error

    receipt.update(metadata)
    receipt.update(
        {
            "completed_at_utc": _utc_now(),
            "status": "success",
            "parser_outcome": "success",
            "term_id": proposal["term_id"],
            "falsifying_prediction": proposal["falsifying_prediction"],
            "falsifying_prediction_status": "unscored_selection_independent",
        }
    )
    _atomic_write_json(receipt_file, receipt)
    receipt_sha256 = hashlib.sha256(receipt_file.read_bytes()).hexdigest()
    return SilverboxProposalResult(
        term_id=proposal["term_id"],
        falsifying_prediction=proposal["falsifying_prediction"],
        receipt_path=receipt_file,
        receipt_sha256=receipt_sha256,
    )


def request_silverbox_term_id(
    *,
    transport: Transport | None = None,
    read_transport: ReadTransport | None = None,
    receipt_dir: str | Path | None = None,
) -> str:
    """Return only the validated term ID; use proposal API for receipt identity."""

    return request_silverbox_proposal(
        transport=transport,
        read_transport=read_transport,
        receipt_dir=receipt_dir,
    ).term_id


__all__ = [
    "ALLOWED_TERM_IDS",
    "ENDPOINT",
    "FROZEN_REQUEST_PAYLOAD_SHA256",
    "MODELS_ENDPOINT",
    "MODEL_ID",
    "REQUEST_TIMEOUT_SECONDS",
    "ChatHTTPResponse",
    "ReadTransport",
    "SilverboxProposalError",
    "SilverboxProposalResult",
    "Transport",
    "request_silverbox_proposal",
    "request_silverbox_term_id",
]

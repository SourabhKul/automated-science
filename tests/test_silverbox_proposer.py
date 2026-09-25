import base64
import hashlib
import json
from pathlib import Path

import pytest

from core.real_data import silverbox_proposer
from core.real_data.silverbox_proposer import (
    ENDPOINT,
    FROZEN_REQUEST_PAYLOAD_SHA256,
    MODEL_ID,
    MODELS_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    ChatHTTPResponse,
    SilverboxProposalError,
    SilverboxProposalResult,
    request_silverbox_proposal,
    request_silverbox_term_id,
)


def _response_body(
    content: str,
    *,
    model: str = MODEL_ID,
    finish_reason: str = "stop",
    role: str | None = "assistant",
) -> bytes:
    message = {"content": content}
    if role is not None:
        message["role"] = role
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "model": model,
            "choices": [
                {"message": message, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 203, "completion_tokens": 24, "total_tokens": 227},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_content(term_id: str = "y_cubed") -> str:
    return json.dumps(
        {
            "term_id": term_id,
            "falsifying_prediction": "The cubic output term should matter at larger output amplitudes.",
        },
        separators=(",", ":"),
    )


def _models_response(*model_ids: str, status_code: int = 200) -> ChatHTTPResponse:
    body = json.dumps({"object": "list", "data": [{"id": model_id} for model_id in model_ids]}).encode("utf-8")
    return ChatHTTPResponse(status_code=status_code, body=body)


def _read_single_receipt(receipt_dir: Path) -> dict:
    receipts = list(receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    return json.loads(receipts[0].read_text(encoding="utf-8"))


class _RecordingTransport:
    def __init__(self, response=None, error=None, *, receipt_dir: Path | None = None):
        self.response = response
        self.error = error
        self.receipt_dir = receipt_dir
        self.calls = []

    def __call__(self, url: str, body: bytes, timeout_seconds: float) -> ChatHTTPResponse:
        self.calls.append((url, body, timeout_seconds))
        if self.receipt_dir is not None:
            dispatch_receipts = list(self.receipt_dir.glob("*.json"))
            assert len(dispatch_receipts) == 1
            dispatch_receipt = json.loads(dispatch_receipts[0].read_text(encoding="utf-8"))
            assert dispatch_receipt["status"] == "dispatching"
            assert dispatch_receipt["model_preflight_status"] == "success"
            assert dispatch_receipt["chat_dispatched"] is True
            assert dispatch_receipt["request_payload_sha256"] == hashlib.sha256(body).hexdigest()
        if self.error is not None:
            raise self.error
        return self.response


class _RecordingReadTransport:
    def __init__(self, response=None, error=None, *, receipt_dir: Path | None = None):
        self.response = response if response is not None else _models_response(MODEL_ID)
        self.error = error
        self.receipt_dir = receipt_dir
        self.calls = []

    def __call__(self, url: str, timeout_seconds: float) -> ChatHTTPResponse:
        self.calls.append((url, timeout_seconds))
        assert url == MODELS_ENDPOINT
        assert timeout_seconds == REQUEST_TIMEOUT_SECONDS == 120.0
        if self.receipt_dir is not None:
            dispatch_receipts = list(self.receipt_dir.glob("*.json"))
            assert len(dispatch_receipts) == 1
            dispatch_receipt = json.loads(dispatch_receipts[0].read_text(encoding="utf-8"))
            assert dispatch_receipt["status"] == "preflight_pending"
            assert dispatch_receipt["model_preflight_status"] == "pending"
            assert dispatch_receipt["request_payload_sha256"]
        if self.error is not None:
            raise self.error
        return self.response


def test_valid_proposal_sends_exact_frozen_request_and_writes_atomic_receipt(tmp_path):
    raw_body = _response_body(_valid_content())
    transport = _RecordingTransport(
        ChatHTTPResponse(status_code=200, body=raw_body),
        receipt_dir=tmp_path,
    )
    read_transport = _RecordingReadTransport(receipt_dir=tmp_path)

    result = request_silverbox_proposal(
        transport=transport,
        read_transport=read_transport,
        receipt_dir=tmp_path,
    )

    assert isinstance(result, SilverboxProposalResult)
    assert result.term_id == "y_cubed"
    assert result.falsifying_prediction == "The cubic output term should matter at larger output amplitudes."
    assert len(transport.calls) == 1
    assert len(read_transport.calls) == 1
    assert read_transport.calls[0] == (MODELS_ENDPOINT, REQUEST_TIMEOUT_SECONDS)
    url, request_body, timeout_seconds = transport.calls[0]
    assert url == ENDPOINT
    assert timeout_seconds == REQUEST_TIMEOUT_SECONDS == 120.0
    payload = json.loads(request_body)
    assert set(payload) == {"model", "messages", "temperature", "max_tokens", "chat_template_kwargs"}
    assert payload["model"] == MODEL_ID
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert "Silverbox" not in payload["messages"][1]["content"]
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 256
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    receipt = _read_single_receipt(tmp_path)
    assert result.receipt_path.parent == tmp_path
    assert result.receipt_path.read_bytes()
    assert result.receipt_sha256 == hashlib.sha256(result.receipt_path.read_bytes()).hexdigest()
    assert result.receipt_path.name.endswith(".json")
    assert receipt["status"] == "success"
    assert receipt["http_status"] == 200
    assert receipt["returned_model"] == MODEL_ID
    assert receipt["finish_reason"] == "stop"
    assert receipt["usage"] == {"prompt_tokens": 203, "completion_tokens": 24, "total_tokens": 227}
    assert receipt["request_payload"] == payload
    assert receipt["request_payload_sha256"] == hashlib.sha256(request_body).hexdigest()
    assert receipt["request_payload_sha256"] == FROZEN_REQUEST_PAYLOAD_SHA256
    assert receipt["response_sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert base64.b64decode(receipt["raw_response_base64"]) == raw_body
    assert receipt["falsifying_prediction"] == "The cubic output term should matter at larger output amplitudes."
    assert receipt["falsifying_prediction_status"] == "unscored_selection_independent"
    assert receipt["parser_outcome"] == "success"
    assert receipt["created_at_utc"].endswith("Z")
    assert receipt["completed_at_utc"].endswith("Z")
    assert receipt["model_preflight_status"] == "success"
    assert receipt["model_preflight_http_status"] == 200
    assert receipt["model_preflight_model_ids"] == [MODEL_ID]
    assert receipt["model_preflight_response_sha256"]
    assert receipt["chat_dispatched"] is True
    assert "authorization" not in receipt


def test_term_id_helper_returns_plain_string_with_one_chat_request(tmp_path):
    transport = _RecordingTransport(
        ChatHTTPResponse(status_code=200, body=_response_body(_valid_content()))
    )
    read_transport = _RecordingReadTransport()

    result = request_silverbox_term_id(
        transport=transport,
        read_transport=read_transport,
        receipt_dir=tmp_path,
    )

    assert result == "y_cubed"
    assert type(result) is str
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("content", "failure_code"),
    [
        ("not json", "invalid_assistant_json"),
        (_valid_content() + " trailing", "invalid_assistant_json"),
        (
            (
                '{"term_id":"y_cubed","term_id":"u_cubed",'
                '"falsifying_prediction":"The term should matter at high amplitude."}'
            ),
            "duplicate_assistant_json_key",
        ),
        ('{"term_id":"y_cubed"}', "assistant_json_keys_mismatch"),
        (
            '{"term_id":"y_cubed","falsifying_prediction":"x","extra":1}',
            "assistant_json_keys_mismatch",
        ),
    ],
)
def test_malformed_content_is_terminal_and_never_retried(tmp_path, content, failure_code):
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=_response_body(content)))
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == failure_code
    assert receipt["failure_code"] == failure_code
    assert receipt["status"] == "failed"
    assert receipt["term_id"] is None
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_duplicate_key_in_outer_response_is_rejected(tmp_path):
    body = (
        b'{"model":"Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",'
        b'"model":"Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",'
        b'"choices":[],"usage":{"total_tokens":1}}'
    )
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=body))
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError, match="duplicate_response_json_key"):
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert receipt["parser_outcome"] == "duplicate_key_rejected"
    assert receipt["response_sha256"] == hashlib.sha256(body).hexdigest()
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_wrong_returned_model_is_terminal_without_fallback(tmp_path):
    transport = _RecordingTransport(
        ChatHTTPResponse(status_code=200, body=_response_body(_valid_content(), model="other-model"))
    )
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "returned_model_mismatch"
    assert receipt["returned_model"] == "other-model"
    assert receipt["term_id"] is None
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_non_stop_finish_reason_is_terminal_and_length_is_marked_truncated(tmp_path):
    transport = _RecordingTransport(
        ChatHTTPResponse(status_code=200, body=_response_body(_valid_content(), finish_reason="length"))
    )
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "truncated_response"
    assert receipt["finish_reason"] == "length"
    assert receipt["parser_outcome"] == "truncated_finish"
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_non_200_http_response_is_terminal_and_raw_response_is_retained(tmp_path):
    raw_body = b'{"error":{"message":"rate limited"}}'
    transport = _RecordingTransport(ChatHTTPResponse(status_code=429, body=raw_body))
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "http_status_not_200"
    assert receipt["http_status"] == 429
    assert receipt["response_sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert base64.b64decode(receipt["raw_response_base64"]) == raw_body
    assert receipt["parser_outcome"] == "http_status_rejected"
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_timeout_is_terminal_and_does_not_retry_or_choose_a_fallback(tmp_path):
    transport = _RecordingTransport(error=TimeoutError("private timeout detail"))
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "transport_failure"
    assert receipt["transport_failure_type"] == "TimeoutError"
    assert receipt["parser_outcome"] == "transport_failed"
    assert receipt["http_status"] is None
    assert receipt["response_sha256"] == hashlib.sha256(b"").hexdigest()
    assert receipt["raw_response_base64"] == ""
    assert "private timeout detail" not in json.dumps(receipt)
    assert receipt["term_id"] is None
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_frozen_payload_hash_mismatch_is_receipted_without_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(silverbox_proposer, "SYSTEM_MESSAGE", "changed prompt")
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=_response_body(_valid_content())))
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "frozen_payload_hash_mismatch"
    assert receipt["status"] == "failed"
    assert receipt["request_payload_sha256"] != FROZEN_REQUEST_PAYLOAD_SHA256
    assert receipt["parser_outcome"] == "frozen_payload_hash_mismatch"
    assert len(read_transport.calls) == 0
    assert len(transport.calls) == 0


@pytest.mark.parametrize(
    ("response_body", "failure_code"),
    [
        (json.dumps({"model": MODEL_ID}).encode("utf-8"), "missing_choices"),
        (
            json.dumps(
                {
                    "model": MODEL_ID,
                    "choices": [{"message": {"content": _valid_content()}}],
                }
            ).encode("utf-8"),
            "missing_finish_reason",
        ),
    ],
)
def test_missing_required_response_fields_are_terminal(tmp_path, response_body, failure_code):
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=response_body))
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    assert raised.value.failure_code == failure_code
    assert _read_single_receipt(tmp_path)["failure_code"] == failure_code
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


@pytest.mark.parametrize("role", ["user", None])
def test_non_assistant_message_role_is_terminal(tmp_path, role):
    transport = _RecordingTransport(
        ChatHTTPResponse(status_code=200, body=_response_body(_valid_content(), role=role))
    )
    read_transport = _RecordingReadTransport()

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "message_role_not_assistant"
    assert receipt["parser_outcome"] == "assistant_role_rejected"
    assert receipt["status"] == "failed"
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 1


def test_preflight_without_exact_model_id_blocks_chat_post(tmp_path):
    read_transport = _RecordingReadTransport(response=_models_response(MODEL_ID + "-other"))
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=_response_body(_valid_content())))

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "preflight_model_id_missing"
    assert receipt["model_preflight_status"] == "failed"
    assert receipt["model_preflight_model_ids"] == [MODEL_ID + "-other"]
    assert receipt["chat_dispatched"] is False
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 0


def test_preflight_http_failure_is_terminal_without_chat_post(tmp_path):
    read_transport = _RecordingReadTransport(response=_models_response(MODEL_ID, status_code=503))
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=_response_body(_valid_content())))

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "preflight_http_status_not_200"
    assert receipt["model_preflight_http_status"] == 503
    assert receipt["model_preflight_status"] == "failed"
    assert receipt["chat_dispatched"] is False
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 0


def test_preflight_timeout_is_terminal_without_chat_post(tmp_path):
    read_transport = _RecordingReadTransport(error=TimeoutError("private preflight detail"))
    transport = _RecordingTransport(ChatHTTPResponse(status_code=200, body=_response_body(_valid_content())))

    with pytest.raises(SilverboxProposalError) as raised:
        request_silverbox_term_id(transport=transport, read_transport=read_transport, receipt_dir=tmp_path)

    receipt = _read_single_receipt(tmp_path)
    assert raised.value.failure_code == "preflight_transport_failure"
    assert receipt["model_preflight_status"] == "failed"
    assert receipt["model_preflight_failure_type"] == "TimeoutError"
    assert receipt["chat_dispatched"] is False
    assert "private preflight detail" not in json.dumps(receipt)
    assert len(read_transport.calls) == 1
    assert len(transport.calls) == 0

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests

from core.evaluation_boundary import CANONICAL_PROTOCOL_VERSION, split_trajectory, write_protocol_manifests
from scripts import run_domain


class _FakeResponse:
    def __init__(self, status_code, text, payload):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} response", response=self)


def test_call_llm_metadata_captures_success_and_preserves_legacy_content():
    payload = {
        "id": "chatcmpl-test",
        "model": "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
        "created": 1789368411,
        "choices": [{
            "message": {"content": "```python\npass\n```"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    raw = json.dumps(payload, separators=(",", ":"))
    metadata = {}
    with patch.object(run_domain.requests, "post", return_value=_FakeResponse(200, raw, payload)) as post:
        result = run_domain.call_llm(
            "prompt",
            payload["model"],
            "http://127.0.0.1:8000/v1/chat/completions",
            max_tokens=32,
            timeout_seconds=9,
            chat_template_kwargs={"enable_thinking": False},
            response_metadata=metadata,
        )

    assert result == "```python\npass\n```"
    assert post.call_args.kwargs["timeout"] == 9
    request_payload = post.call_args.kwargs["json"]
    assert request_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert metadata["status"] == "success"
    assert metadata["status_code"] == 200
    assert metadata["finish_reason"] == "stop"
    assert metadata["usage"] == payload["usage"]
    assert metadata["elapsed_seconds"] >= 0
    assert metadata["raw_response"] == raw
    assert metadata["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert metadata["error"] is None


def test_call_llm_metadata_captures_http_failure_and_raw_error_body():
    payload = {"error": {"message": "rate limited"}}
    raw = json.dumps(payload)
    metadata = {}
    with patch.object(
        run_domain.requests,
        "post",
        return_value=_FakeResponse(429, raw, payload),
    ):
        result = run_domain.call_llm("prompt", "fixed-model", "http://endpoint", response_metadata=metadata)

    assert result is None
    assert metadata["status"] == "failed"
    assert metadata["status_code"] == 429
    assert metadata["finish_reason"] is None
    assert metadata["usage"] is None
    assert metadata["elapsed_seconds"] >= 0
    assert metadata["raw_response"] == raw
    assert metadata["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert "429" in metadata["error"]


def test_call_llm_canonical_rejects_non_string_content_but_legacy_return_stays_unchanged():
    payload = {
        "choices": [{"message": {"content": [{"text": "chunk"}]}, "finish_reason": "stop"}],
    }
    raw = json.dumps(payload)
    with patch.object(
        run_domain.requests,
        "post",
        return_value=_FakeResponse(200, raw, payload),
    ):
        legacy_result = run_domain.call_llm("prompt", "fixed-model", "http://endpoint")
        canonical_metadata = {}
        canonical_result = run_domain.call_llm(
            "prompt",
            "fixed-model",
            "http://endpoint",
            response_metadata=canonical_metadata,
        )

    assert legacy_result == [{"text": "chunk"}]
    assert canonical_result is None
    assert canonical_metadata["status"] == "failed"
    assert canonical_metadata["status_code"] == 200
    assert "must be a string" in canonical_metadata["error"]
    assert canonical_metadata["raw_response"] == raw


def test_call_llm_canonical_records_empty_content_as_unusable_http_success():
    payload = {
        "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
    }
    raw = json.dumps(payload)
    metadata = {}
    with patch.object(
        run_domain.requests,
        "post",
        return_value=_FakeResponse(200, raw, payload),
    ):
        result = run_domain.call_llm("prompt", "fixed-model", "http://endpoint", response_metadata=metadata)

    assert result is None
    assert metadata["status"] == "failed"
    assert metadata["status_code"] == 200
    assert metadata["finish_reason"] == "stop"
    assert "empty" in metadata["error"]
    assert metadata["raw_response"] == raw


def test_call_llm_metadata_captures_transport_timeout_without_http_status():
    metadata = {}

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("request deadline")

    with patch.object(run_domain.requests, "post", side_effect=raise_timeout):
        result = run_domain.call_llm("prompt", "fixed-model", "http://endpoint", response_metadata=metadata)

    assert result is None
    assert metadata["status"] == "failed"
    assert metadata["status_code"] is None
    assert metadata["raw_response"] == ""
    assert metadata["raw_response_sha256"] == hashlib.sha256(b"").hexdigest()
    assert "Timeout" in metadata["error"]
    assert metadata["elapsed_seconds"] >= 0


def test_canonical_receipts_persist_proposal_and_repair_metadata_after_each_call():
    original_eval = run_domain.run_evaluation
    original_llm = run_domain.call_llm
    calls = []

    def fake_eval(code, domain, held_out, target_samples, generations, initial_particles, seed, models_dir, **kwargs):
        manifest = run_domain.load_partition_manifest(
            kwargs["protocol_manifest"],
            required_roles=("train", "validation"),
            forbidden_roles=("final",),
        )
        return (
            1.0,
            {
                "status": "success",
                "median_distance": 1.0,
                "min_distance": 1.0,
                "train_metrics": {"mse": 1.0},
                "validation_metrics": {"mse": 1.0},
                "test_metrics": None,
                "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
                "development_data_receipt": manifest["receipt"],
                "diagnostics": {
                    "schema_version": 1,
                    "fit": {"median_distance": 1.0, "min_distance": 1.0},
                    "best_parameters": {"r": 0.2, "a": 0.1, "b": 0.1, "m": 0.2},
                    "residual_summary": {},
                    "warnings": [],
                    "failure_modes": [],
                },
            },
            None,
        )

    invalid_reply = "```python\ndef dynamics(t, y, args):\n    return [\n\nmetadata = []\n```"

    def fake_llm(prompt, model_id, endpoint, response_metadata=None, **kwargs):
        index = len(calls)
        calls.append(prompt)
        raw = json.dumps({"choices": [{"finish_reason": "length"}], "call": index})
        response_metadata.update({
            "status": "success",
            "status_code": 200,
            "finish_reason": "length",
            "usage": {"total_tokens": 32},
            "elapsed_seconds": 0.01,
            "error": None,
            "raw_response": raw,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        })
        return invalid_reply

    run_domain.run_evaluation = fake_eval
    run_domain.call_llm = fake_llm
    try:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = split_trajectory(
                np.arange(15, dtype=float),
                np.column_stack([np.arange(15, dtype=float), np.arange(15, dtype=float)]),
            )
            development_manifest, _ = write_protocol_manifests(protocol, root / "inputs")
            old_cwd = Path.cwd()
            import os
            os.chdir(root)
            try:
                code = run_domain.run_domain_main(
                    "ecology",
                    model="Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
                    endpoint="http://127.0.0.1:8000/v1/chat/completions",
                    tag_prefix="response_receipt_test",
                    epochs=1,
                    seed=7,
                    evaluation_protocol=CANONICAL_PROTOCOL_VERSION,
                    protocol_manifest=str(development_manifest),
                    target_samples_override=2,
                    generations_override=1,
                    initial_particles_override=4,
                )
                assert code == 0
                receipts_path = root / "models/response_receipt_test_ecology/proposal_prompt_receipts.json"
                receipts = json.loads(receipts_path.read_text())
                assert [item["stage"] for item in receipts] == ["proposal", "repair_1"]
                assert len(calls) == 2
                for item in receipts:
                    response = item["response_metadata"]
                    assert response["status_code"] == 200
                    assert response["finish_reason"] == "length"
                    assert response["usage"]["total_tokens"] == 32
                    assert response["elapsed_seconds"] == 0.01
                    assert response["raw_response"].startswith("{\"choices\"")
                    assert response["raw_response_sha256"] == hashlib.sha256(
                        response["raw_response"].encode()
                    ).hexdigest()
            finally:
                os.chdir(old_cwd)
    finally:
        run_domain.run_evaluation = original_eval
        run_domain.call_llm = original_llm

from pathlib import Path

from scripts.run_model_matrix import (
    build_gauntlet_command,
    classify_completed_model_record,
    preflight_error_result,
    preflight_request_payload,
    preflight_repair_request_payload,
    preflight_result_from_completion,
)


VALID_COMPLETION = {
    "id": "chatcmpl-valid",
    "model": "demo-model",
    "choices": [
        {
            "finish_reason": "stop",
            "message": {
                "content": """```python
def dynamics(t, y, args):
    k = args[0]
    return jnp.array([-k * y[0]])

metadata = [
    {"name": "k", "range": (0.01, 1.0)}
]
```""",
            },
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40},
}


def main():
    payload = preflight_request_payload("demo-model")
    assert payload["model"] == "demo-model"
    assert payload["max_tokens"] >= 128
    user_prompt = payload["messages"][1]["content"]
    assert "{'name': 'k', 'range': (0.01, 1.0)}" in user_prompt
    assert "parameter range tuple" not in user_prompt

    repair_payload = preflight_repair_request_payload(
        "demo-model",
        invalid_content="metadata = [(0.0, 1.0)]",
        detail="metadata entries must be dicts",
    )
    assert repair_payload["model"] == "demo-model"
    repair_prompt = repair_payload["messages"][1]["content"]
    assert "metadata entries must be dicts" in repair_prompt
    assert "metadata` is a non-empty list of dicts" in repair_prompt

    valid = preflight_result_from_completion(
        VALID_COMPLETION,
        status_code=200,
        raw_text="raw-valid",
    )
    assert valid["ok"] is True
    assert valid["category"] is None
    assert valid["finish_reason"] == "stop"
    assert valid["output_length"] > 0
    assert valid["normalization_applied"] is False
    assert valid["response_metadata"]["status_code"] == 200
    assert valid["response_metadata"]["usage"]["total_tokens"] == 40

    import_valid = preflight_result_from_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": """```python
import jax.numpy as jnp

def dynamics(t, y, args):
    return jnp.array([y[0]])

metadata = [{"name": "k", "range": (0.0, 1.0)}]
```""",
                    },
                }
            ]
        },
        status_code=200,
        raw_text="raw-import-valid",
    )
    assert import_valid["ok"] is True
    assert import_valid["category"] is None
    assert import_valid["normalization_applied"] is True

    invalid_metadata = preflight_result_from_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": """```python
def dynamics(t, y, args):
    return jnp.array([y[0]])

metadata = [(0.0, 1.0)]
```""",
                    },
                }
            ]
        },
        status_code=200,
        raw_text="raw-invalid-metadata",
    )
    assert invalid_metadata["ok"] is False
    assert invalid_metadata["category"] == "invalid_code"
    assert "metadata" in invalid_metadata["detail"]

    empty = preflight_result_from_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": ""},
                }
            ]
        },
        status_code=200,
        raw_text="",
    )
    assert empty["ok"] is False
    assert empty["category"] == "chat_empty"
    assert empty["output_length"] == 0

    invalid = preflight_result_from_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": """```python
def dynamics(t, y, args):
    return jnp.array([0.0])
```""",
                    },
                }
            ]
        },
        status_code=200,
        raw_text="raw-invalid",
    )
    assert invalid["ok"] is False
    assert invalid["category"] == "invalid_code"
    assert "metadata" in invalid["detail"]

    unavailable = preflight_error_result(
        detail="HTTP 503 model unavailable",
        status_code=503,
        raw_text='{"error":"model unavailable"}',
    )
    assert unavailable["ok"] is False
    assert unavailable["category"] == "model_unavailable"

    timed_out = preflight_error_result(detail="ReadTimeout: request timed out")
    assert timed_out["ok"] is False
    assert timed_out["category"] == "timeout"

    cmd = build_gauntlet_command(
        python_executable=".venv/bin/python",
        family="qwen3_coder_next",
        model_id="demo-model",
        tag_prefix="demo-tag",
        epochs=1,
        gauntlet_summary=Path("artifacts/model_matrix/demo/summary.json"),
        endpoint="http://localhost:1234/v1/chat/completions",
        max_tokens=512,
        target_samples=500,
        generations=15,
        initial_particles=50000,
        domains=["ecology"],
        held_out=True,
        seed=123,
        inference_strategy="bdss",
        include_quarantined_domains=False,
    )
    assert "--domains" in cmd
    assert cmd[cmd.index("--domains") + 1] == "ecology"
    assert cmd[cmd.index("--inference-strategy") + 1] == "bdss"
    assert cmd[cmd.index("--target-samples") + 1] == "500"
    assert cmd[cmd.index("--generations") + 1] == "15"
    assert cmd[cmd.index("--initial-particles") + 1] == "50000"
    assert "--seed" in cmd

    completed_status = classify_completed_model_record(
        {
            "status": "running",
            "timed_out": False,
            "metrics": {
                "valid_domains": 1,
                "rows": [{"domain": "ecology", "metrics_path": "models/demo/held_out_metrics.json"}],
            },
        },
        requested_domain_count=1,
    )
    assert completed_status == "completed"

    print("SUCCESS: model-matrix preflight captures code readiness and actionable failures")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.lmstudio_models import (  # noqa: E402
    DEFAULT_BASE_URL,
    load_lmstudio_model,
    loaded_model_instances,
    unload_loaded_models,
)
from scripts.run_model_matrix import (  # noqa: E402
    chat_endpoint,
    preflight_error_result,
    preflight_result_from_completion,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_minimal_code_payload(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "Return exactly one fenced Python code block and nothing else.",
            },
            {
                "role": "user",
                "content": (
                    "Return this tiny module. No imports. No comments. No examples.\n"
                    "```python\n"
                    "def dynamics(t, y, args):\n"
                    "    return jnp.array([-args[0] * y[0]])\n\n"
                    "metadata = [{'name': 'k', 'range': (0.01, 1.0)}]\n"
                    "```"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 96,
    }


def raw_python_payload(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "Return raw Python only. No markdown. No prose.",
            },
            {
                "role": "user",
                "content": (
                    "Write exactly these two top-level objects and nothing else:\n"
                    "def dynamics(t, y, args):\n"
                    "    return jnp.array([-args[0] * y[0]])\n\n"
                    "metadata = [{'name': 'k', 'range': (0.01, 1.0)}]"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 96,
    }


def repair_only_payload(model_id: str, *, invalid_content: str, detail: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Repair code. Return only Python. Do not add imports, helper functions, "
                    "classes, validation code, examples, comments, or prose."
                ),
            },
            {
                "role": "user",
                "content": f"""The previous tiny scientific module failed validation.

Failure: {detail}

Return exactly two top-level objects:
1. def dynamics(t, y, args):
2. metadata = [{{'name': 'k', 'range': (0.01, 1.0)}}]

The dynamics function must use args[0] and return jnp.array([...]).

Previous output:
```python
{invalid_content}
```""",
            },
        ],
        "temperature": 0,
        "max_tokens": 160,
    }


def ready_payload_with_system(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Return exactly the literal text READY."},
            {"role": "user", "content": "READY"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }


def ready_payload_user_only(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Return exactly the literal text READY."},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }


def choice_content(payload: dict[str, Any] | None) -> str:
    payload = payload or {}
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        fragments = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    fragments.append(str(text))
        return "".join(fragments).strip()
    return str(content or "").strip()


def finish_reason(payload: dict[str, Any] | None) -> str | None:
    payload = payload or {}
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    return choice.get("finish_reason")


def run_chat_request(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
    validate_code: bool,
) -> dict[str, Any]:
    try:
        response = requests.post(chat_endpoint(base_url), json=payload, timeout=timeout)
        raw_text = response.text[:2000]
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = None
        if not response.ok:
            detail = raw_text.strip() or f"HTTP {response.status_code}"
            return preflight_error_result(
                detail=detail,
                status_code=response.status_code,
                raw_text=raw_text,
                payload=response_payload,
            )
        if validate_code:
            return preflight_result_from_completion(
                response_payload,
                status_code=response.status_code,
                raw_text=raw_text,
            )
        content = choice_content(response_payload)
        if not content:
            return {
                "ok": False,
                "category": "chat_template_or_runtime_empty",
                "detail": "readiness probe returned empty content",
                "status_code": response.status_code,
                "finish_reason": finish_reason(response_payload),
                "output_length": 0,
                "content_preview": "",
                "raw_response_preview": raw_text[:500],
            }
        return {
            "ok": True,
            "category": None,
            "detail": "readiness probe returned non-empty content",
            "status_code": response.status_code,
            "finish_reason": finish_reason(response_payload),
            "output_length": len(content),
            "content_preview": content[:200],
            "raw_response_preview": raw_text[:500],
        }
    except requests.Timeout as exc:
        return preflight_error_result(detail=f"{type(exc).__name__}: {exc}")
    except requests.RequestException as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raw_text = exc.response.text[:2000] if exc.response is not None else ""
        payload_json = None
        if exc.response is not None:
            try:
                payload_json = exc.response.json()
            except ValueError:
                payload_json = None
        return preflight_error_result(
            detail=f"{type(exc).__name__}: {exc}",
            status_code=status_code,
            raw_text=raw_text,
            payload=payload_json,
        )
    except Exception as exc:
        return preflight_error_result(detail=f"{type(exc).__name__}: {exc}")


def classify_final(variants: list[dict[str, Any]]) -> str:
    if any(v["result"].get("ok") and v.get("validates_code") for v in variants):
        return "recovered"
    if any(v["result"].get("ok") and not v.get("validates_code") for v in variants):
        return "chat_nonempty_code_invalid"
    categories = {v["result"].get("category") for v in variants if v["result"].get("category")}
    if categories == {"chat_template_or_runtime_empty"} or categories == {"chat_empty"}:
        return "chat_template_or_runtime_empty"
    if "invalid_code" in categories:
        return "invalid_code"
    if "timeout" in categories:
        return "timeout"
    if "request_error" in categories:
        return "request_error"
    return "unrecovered"


def recover_model(
    *,
    model_id: str,
    base_url: str,
    context_length: int,
    load_timeout: int,
    unload_timeout: int,
    request_timeout: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": model_id,
        "started_at": utc_now(),
        "pre_unload": {},
        "load": {},
        "variants": [],
    }
    before = loaded_model_instances(base_url, unload_timeout)
    pre_results = unload_loaded_models(base_url, unload_timeout) if before else []
    after_pre = loaded_model_instances(base_url, unload_timeout)
    record["pre_unload"] = {
        "before": before,
        "results": pre_results,
        "after": after_pre,
        "ok": not after_pre,
    }
    if after_pre:
        record["status"] = "blocked_loaded_instances"
        record["finished_at"] = utc_now()
        return record

    record["load"] = load_lmstudio_model(
        model_id,
        base_url=base_url,
        context_length=context_length,
        timeout=load_timeout,
    )
    if not record["load"].get("ok"):
        record["status"] = "load_failed"
        post_before = loaded_model_instances(base_url, unload_timeout)
        post_results = unload_loaded_models(base_url, unload_timeout) if post_before else []
        post_after = loaded_model_instances(base_url, unload_timeout)
        record["post_unload"] = {
            "before": post_before,
            "results": post_results,
            "after": post_after,
            "ok": not post_after,
        }
        record["finished_at"] = utc_now()
        return record

    last_invalid_content = ""
    last_invalid_detail = ""
    variant_builders = [
        ("strict_minimal_code", lambda: strict_minimal_code_payload(model_id), True),
        ("raw_python_no_fence", lambda: raw_python_payload(model_id), True),
    ]
    for variant_id, payload_builder, validates_code in variant_builders:
        result = run_chat_request(
            base_url=base_url,
            payload=payload_builder(),
            timeout=request_timeout,
            validate_code=validates_code,
        )
        record["variants"].append({
            "id": variant_id,
            "validates_code": validates_code,
            "result": result,
        })
        if result.get("ok"):
            break
        if result.get("category") == "invalid_code":
            last_invalid_content = result.get("content_preview", "")
            last_invalid_detail = result.get("detail", "")

    if not any(v["result"].get("ok") and v.get("validates_code") for v in record["variants"]):
        if last_invalid_content or last_invalid_detail:
            result = run_chat_request(
                base_url=base_url,
                payload=repair_only_payload(
                    model_id,
                    invalid_content=last_invalid_content,
                    detail=last_invalid_detail,
                ),
                timeout=request_timeout,
                validate_code=True,
            )
            record["variants"].append({
                "id": "repair_only_retry",
                "validates_code": True,
                "result": result,
            })

    if not any(v["result"].get("ok") and v.get("validates_code") for v in record["variants"]):
        for variant_id, payload_builder in [
            ("ready_with_system", lambda: ready_payload_with_system(model_id)),
            ("ready_user_only", lambda: ready_payload_user_only(model_id)),
        ]:
            result = run_chat_request(
                base_url=base_url,
                payload=payload_builder(),
                timeout=request_timeout,
                validate_code=False,
            )
            record["variants"].append({
                "id": variant_id,
                "validates_code": False,
                "result": result,
            })
            if result.get("ok"):
                break

    record["final_category"] = classify_final(record["variants"])
    record["recovered"] = record["final_category"] == "recovered"
    recovered_variant = next(
        (
            variant["id"]
            for variant in record["variants"]
            if variant["result"].get("ok") and variant.get("validates_code")
        ),
        None,
    )
    record["recovery_variant"] = recovered_variant
    post_before = loaded_model_instances(base_url, unload_timeout)
    post_results = unload_loaded_models(base_url, unload_timeout) if post_before else []
    post_after = loaded_model_instances(base_url, unload_timeout)
    record["post_unload"] = {
        "before": post_before,
        "results": post_results,
        "after": post_after,
        "ok": not post_after,
    }
    record["status"] = "completed"
    record["finished_at"] = utc_now()
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted LM Studio preflight recovery probes.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/evaluations/phase12_preflight_recovery_20260707/recovery_summary.json")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--load-timeout", type=int, default=600)
    parser.add_argument("--unload-timeout", type=int, default=30)
    parser.add_argument("--request-timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("r") as f:
            summary = json.load(f)
        previous_models = [
            model
            for model in summary.get("models", [])
            if model.get("model_id") not in set(args.models)
        ]
    else:
        summary = {
            "created_at": utc_now(),
            "models": [],
        }
        previous_models = []
    summary["base_url"] = args.base_url
    summary["context_length"] = args.context_length
    summary["models_requested"] = args.models
    summary["models"] = previous_models
    failures = 0
    for model_id in args.models:
        record = recover_model(
            model_id=model_id,
            base_url=args.base_url,
            context_length=args.context_length,
            load_timeout=args.load_timeout,
            unload_timeout=args.unload_timeout,
            request_timeout=args.request_timeout,
        )
        summary["models"].append(record)
        summary["updated_at"] = utc_now()
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    failures = sum(1 for record in summary["models"] if not record.get("recovered"))
    summary["finished_at"] = utc_now()
    summary["failures"] = failures
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote preflight recovery summary to {output_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

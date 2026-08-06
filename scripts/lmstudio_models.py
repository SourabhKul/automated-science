from __future__ import annotations

import argparse
import json
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:1234/v1"


def native_base_url(base_url: str = DEFAULT_BASE_URL) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return f"{stripped[:-3]}/api/v1"
    return f"{stripped}/api/v1"


def list_lmstudio_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> list[str]:
    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return [item["id"] for item in payload.get("data", []) if item.get("id")]


def native_lmstudio_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> list[dict[str, Any]]:
    response = requests.get(f"{native_base_url(base_url)}/models", timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return list(payload.get("models", []))


def loaded_model_instances(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> list[dict[str, str]]:
    instances: list[dict[str, str]] = []
    for model in native_lmstudio_models(base_url, timeout):
        for instance in model.get("loaded_instances") or []:
            instance_id = instance.get("id")
            if instance_id:
                instances.append({
                    "model_key": model.get("key", ""),
                    "instance_id": instance_id,
                })
    return instances


def unload_loaded_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 30) -> list[dict[str, Any]]:
    results = []
    for instance in loaded_model_instances(base_url, timeout):
        instance_id = instance["instance_id"]
        try:
            response = requests.post(
                f"{native_base_url(base_url)}/models/unload",
                json={"instance_id": instance_id},
                timeout=timeout,
            )
            results.append({
                **instance,
                "ok": response.ok,
                "status_code": response.status_code,
                "detail": response.text[:500],
            })
        except Exception as exc:
            results.append({
                **instance,
                "ok": False,
                "status_code": None,
                "detail": f"{type(exc).__name__}: {exc}",
            })
    return results


def load_lmstudio_model(
    model_id: str,
    base_url: str = DEFAULT_BASE_URL,
    context_length: int | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "echo_load_config": True,
    }
    if context_length:
        body["context_length"] = context_length
    response = requests.post(f"{native_base_url(base_url)}/models/load", json=body, timeout=timeout)
    detail = response.text[:2000]
    try:
        payload = response.json()
    except Exception:
        payload = None
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "detail": detail,
        "payload": payload,
    }


def is_generation_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return "embed" not in lowered and "embedding" not in lowered


def generation_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> list[str]:
    return [model_id for model_id in list_lmstudio_models(base_url, timeout) if is_generation_model(model_id)]


def main() -> int:
    parser = argparse.ArgumentParser(description="List LM Studio models from the OpenAI-compatible REST API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print a JSON array instead of one model per line.")
    args = parser.parse_args()

    models = list_lmstudio_models(args.base_url)
    if not args.include_embeddings:
        models = [model_id for model_id in models if is_generation_model(model_id)]

    if args.json:
        print(json.dumps(models, indent=2))
    else:
        for model_id in models:
            print(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

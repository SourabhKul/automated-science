from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.domain_quarantine import partition_requested_domains
from core.domain_configs import DOMAIN_CONFIGS
from core.generated_code import (
    GeneratedCodeError,
    extract_code,
    normalize_generated_math_code,
    validate_generated_math_code,
)
from core.model_matrix_status import (
    TERMINAL_MODEL_STATUSES,
    classify_model_status,
    failed_domains,
    preflight_failure_category,
)
from scripts.lmstudio_models import (
    DEFAULT_BASE_URL,
    generation_models,
    load_lmstudio_model,
    loaded_model_instances,
    unload_loaded_models,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_model(model_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
    return re.sub(r"_+", "_", slug)[:80] or "model"


def chat_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def preflight_request_payload(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deterministic code generator. "
                    "Return only Python for a tiny valid module."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write the smallest valid Python module for this scientific runner. "
                    "Define exactly one function `dynamics(t, y, args)` that returns a "
                    "single-state `jnp.array`, and define `metadata` as a list with one "
                    "dict: `{'name': 'k', 'range': (0.01, 1.0)}`. No prose."
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 192,
    }


def preflight_repair_request_payload(
    model_id: str,
    *,
    invalid_content: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You repair tiny scientific Python modules. "
                    "Return only a Python code block with `dynamics(t, y, args)` and `metadata`."
                ),
            },
            {
                "role": "user",
                "content": f"""The previous preflight module failed validation.

Failure detail: `{detail}`

Fix it so it follows this exact contract:
- no imports
- exactly one `dynamics(t, y, args)` function
- `metadata` is a non-empty list of dicts
- each metadata dict has a string `name` and a finite ascending `(low, high)` `range`
- every `args[i]` used by dynamics has a matching metadata entry

Previous output:
```python
{invalid_content}
```""",
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
    }


def _choice_content_text(choice: dict[str, Any]) -> str:
    message = choice.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        fragments = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    fragments.append(str(text))
        return "".join(fragments)
    return str(content or "")


def _response_metadata(payload: dict[str, Any] | None, status_code: int | None) -> dict[str, Any]:
    if not payload:
        return {"status_code": status_code}
    usage = payload.get("usage")
    metadata = {
        "status_code": status_code,
        "id": payload.get("id"),
        "model": payload.get("model"),
        "created": payload.get("created"),
        "system_fingerprint": payload.get("system_fingerprint"),
    }
    if usage is not None:
        metadata["usage"] = usage
    return metadata


def preflight_result_from_completion(
    payload: dict[str, Any] | None,
    *,
    status_code: int | None,
    raw_text: str,
) -> dict[str, Any]:
    payload = payload or {}
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    content = _choice_content_text(choice).strip()
    finish_reason = choice.get("finish_reason")
    result = {
        "ok": False,
        "category": None,
        "detail": "",
        "status_code": status_code,
        "finish_reason": finish_reason,
        "output_length": len(content),
        "content_preview": content[:200],
        "raw_response_preview": raw_text[:500],
        "normalization_applied": False,
        "response_metadata": _response_metadata(payload, status_code),
    }
    if not content:
        result["category"] = "chat_empty"
        result["detail"] = "chat completion returned empty content"
        return result

    extracted = extract_code(content)
    if extracted is None:
        result["category"] = "invalid_code"
        result["detail"] = "unable to extract a Python module from preflight output"
        return result

    normalized_code, normalization_applied = normalize_generated_math_code(extracted.code)
    result["normalization_applied"] = normalization_applied
    try:
        validate_generated_math_code(normalized_code)
    except GeneratedCodeError as exc:
        result["category"] = "invalid_code"
        result["detail"] = str(exc)
        return result

    result["ok"] = True
    result["detail"] = "validated dynamics(t, y, args) and metadata preflight module"
    return result


def preflight_error_result(
    *,
    detail: str,
    status_code: int | None = None,
    raw_text: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "category": preflight_failure_category({
            "ok": False,
            "detail": detail,
            "status_code": status_code,
            "raw_response_preview": raw_text,
        }),
        "detail": detail,
        "status_code": status_code,
        "finish_reason": None,
        "output_length": 0,
        "content_preview": "",
        "raw_response_preview": raw_text[:500],
        "response_metadata": _response_metadata(payload, status_code),
    }


def preflight_model(model_id: str, base_url: str, timeout: int) -> dict[str, Any]:
    payload = {
        **preflight_request_payload(model_id),
    }
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
        first_result = preflight_result_from_completion(
            response_payload,
            status_code=response.status_code,
            raw_text=raw_text,
        )
        if first_result["ok"] or first_result.get("category") != "invalid_code" or not first_result.get("output_length"):
            return first_result

        repair_payload = preflight_repair_request_payload(
            model_id,
            invalid_content=first_result.get("content_preview", ""),
            detail=first_result.get("detail", ""),
        )
        repair_response = requests.post(chat_endpoint(base_url), json=repair_payload, timeout=timeout)
        repair_raw_text = repair_response.text[:2000]
        try:
            repair_response_payload = repair_response.json()
        except ValueError:
            repair_response_payload = None
        if not repair_response.ok:
            first_result["repair_attempted"] = True
            first_result["repair_error"] = preflight_error_result(
                detail=repair_raw_text.strip() or f"HTTP {repair_response.status_code}",
                status_code=repair_response.status_code,
                raw_text=repair_raw_text,
                payload=repair_response_payload,
            )
            return first_result

        repair_result = preflight_result_from_completion(
            repair_response_payload,
            status_code=repair_response.status_code,
            raw_text=repair_raw_text,
        )
        repair_result["repair_attempted"] = True
        repair_result["initial_result"] = first_result
        if repair_result["ok"]:
            repair_result["detail"] = "validated dynamics(t, y, args) and metadata preflight module after repair"
        return repair_result
    except requests.Timeout as exc:
        return preflight_error_result(detail=f"{type(exc).__name__}: {exc}")
    except requests.RequestException as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raw_text = exc.response.text[:2000] if exc.response is not None else ""
        payload = None
        if exc.response is not None:
            try:
                payload = exc.response.json()
            except ValueError:
                payload = None
        return preflight_error_result(
            detail=f"{type(exc).__name__}: {exc}",
            status_code=status_code,
            raw_text=raw_text,
            payload=payload,
        )
    except Exception as exc:
        return preflight_error_result(detail=f"{type(exc).__name__}: {exc}")


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def collect_model_metrics(
    tag_prefix: str,
    domains: list[str],
    quarantined_domains: list[dict[str, str]] | None = None,
) -> dict:
    rows = []
    valid = 0
    accepted_domains = 0
    failed = []
    final_scores = []
    quarantined_by_domain = {
        item["domain"]: item["reason"]
        for item in quarantined_domains or []
    }
    for domain in domains:
        if domain in quarantined_by_domain:
            rows.append({
                "domain": domain,
                "domain_status": "quarantined",
                "quarantine_reason": quarantined_by_domain[domain],
                "metrics_path": None,
                "run_config_path": None,
                "accepted_updates": 0,
                "seed_test_mse": None,
                "final_train_mse": None,
                "final_test_mse": None,
                "relative_improvement_vs_seed": None,
            })
            continue
        metrics_path = Path(f"models/{tag_prefix}_{domain}/held_out_metrics.json")
        run_config_path = Path(f"models/{tag_prefix}_{domain}/run_config.json")
        history = []
        if metrics_path.exists():
            history = read_json(metrics_path, default=[])
        seed_test = None
        final_test = None
        final_train = None
        accepted = 0
        if history:
            seed_test = float(history[0]["test"]["mse"])
            final_test = float(history[-1]["test"]["mse"])
            final_train = float(history[-1]["train"]["mse"])
            accepted = max(0, len(history) - 1)
            valid += 1
            if accepted:
                accepted_domains += 1
            if math.isfinite(final_test):
                final_scores.append(final_test)
        else:
            failed.append(domain)
        rows.append({
            "domain": domain,
            "domain_status": "completed" if history else "failed",
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "run_config_path": str(run_config_path) if run_config_path.exists() else None,
            "accepted_updates": accepted,
            "seed_test_mse": seed_test,
            "final_train_mse": final_train,
            "final_test_mse": final_test,
            "relative_improvement_vs_seed": ((seed_test - final_test) / seed_test if seed_test and final_test is not None else None),
        })
    return {
        "valid_domains": valid,
        "accepted_domains": accepted_domains,
        "failed_domain_count": len(failed),
        "failed_domains": failed,
        "quarantined_domain_count": len(quarantined_by_domain),
        "quarantined_domains": quarantined_domains or [],
        "mean_final_test_mse": (sum(final_scores) / len(final_scores) if final_scores else None),
        "median_final_test_mse": (sorted(final_scores)[len(final_scores) // 2] if final_scores else None),
        "rows": rows,
    }


def load_matrix(
    path: Path,
    args: argparse.Namespace,
    models: list[str],
    requested_domains: list[str],
    stable_domains: list[str],
    quarantined_domains: list[dict[str, str]],
) -> dict:
    if args.resume and path.exists():
        matrix = read_json(path)
        matrix.setdefault("domains", requested_domains)
        matrix["stable_domains"] = stable_domains
        matrix["stable_domain_count"] = len(stable_domains)
        matrix["quarantined_domains"] = quarantined_domains
        matrix["target_samples"] = args.target_samples
        matrix["generations"] = args.generations
        matrix["initial_particles"] = args.initial_particles
        matrix["inference_strategy"] = args.inference_strategy
        return matrix
    return {
        "created_at": utc_now(),
        "base_url": args.base_url,
        "chat_endpoint": chat_endpoint(args.base_url),
        "domains": requested_domains,
        "stable_domains": stable_domains,
        "stable_domain_count": len(stable_domains),
        "quarantined_domains": quarantined_domains,
        "epochs": args.epochs,
        "held_out": args.held_out,
        "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "target_samples": args.target_samples,
        "generations": args.generations,
        "initial_particles": args.initial_particles,
        "inference_strategy": args.inference_strategy,
        "base_seed": args.seed,
        "run_id": args.run_id,
        "requested_models": models,
        "models": [],
    }


def completed_models(matrix: dict) -> set[str]:
    terminal_statuses = TERMINAL_MODEL_STATUSES | {"failed", "timeout"}
    return {
        item["model_id"]
        for item in matrix.get("models", [])
        if item.get("status") in terminal_statuses
    }


def upsert_model_record(matrix: dict, model_record: dict) -> None:
    records = matrix.setdefault("models", [])
    for index, item in enumerate(records):
        if item.get("model_id") == model_record.get("model_id"):
            records[index] = model_record
            return
    records.append(model_record)


def run_streaming(cmd: list[str], cwd: Path, timeout_seconds: int | None) -> tuple[int, float, bool]:
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    try:
        return_code = proc.wait(timeout=timeout_seconds)
        return return_code, time.monotonic() - started, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        return proc.returncode if proc.returncode is not None else -9, time.monotonic() - started, True


def unload_all_loaded_models(base_url: str, timeout: int) -> dict:
    before = loaded_model_instances(base_url, timeout)
    results = unload_loaded_models(base_url, timeout)
    time.sleep(2)
    after = loaded_model_instances(base_url, timeout)
    return {
        "before": before,
        "results": results,
        "after": after,
        "ok": not after and all(item.get("ok") for item in results),
    }


def build_gauntlet_command(
    *,
    python_executable: str,
    family: str,
    model_id: str,
    tag_prefix: str,
    epochs: int,
    gauntlet_summary: Path,
    endpoint: str,
    max_tokens: int,
    target_samples: int | None,
    generations: int | None,
    initial_particles: int | None,
    domains: list[str],
    held_out: bool,
    seed: int | None,
    inference_strategy: str,
    include_quarantined_domains: bool,
) -> list[str]:
    cmd = [
        python_executable,
        "scripts/run_gauntlet.py",
        "--family",
        family,
        "--model",
        model_id,
        "--tag-prefix",
        tag_prefix,
        "--epochs",
        str(epochs),
        "--summary",
        str(gauntlet_summary),
        "--endpoint",
        endpoint,
        "--max-tokens",
        str(max_tokens),
        "--inference-strategy",
        inference_strategy,
        "--domains",
        *domains,
    ]
    if target_samples is not None:
        cmd.extend(["--target-samples", str(target_samples)])
    if generations is not None:
        cmd.extend(["--generations", str(generations)])
    if initial_particles is not None:
        cmd.extend(["--initial-particles", str(initial_particles)])
    if held_out:
        cmd.append("--held-out")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if include_quarantined_domains:
        cmd.append("--include-quarantined-domains")
    return cmd


def classify_completed_model_record(model_record: dict[str, Any], requested_domain_count: int) -> str:
    finalized_record = dict(model_record)
    finalized_record["status"] = "timeout" if finalized_record.get("timed_out") else "completed"
    return classify_model_status(
        finalized_record,
        requested_domain_count=requested_domain_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a held-out gauntlet for every available LM Studio generation model.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--models", nargs="+", default=None, help="Explicit model IDs. Defaults to LM Studio /v1/models generation models.")
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_CONFIGS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--held-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-timeout", type=int, default=120)
    parser.add_argument("--per-model-timeout", type=int, default=None, help="Wall-clock seconds allowed for each model gauntlet.")
    parser.add_argument("--context-length", type=int, default=32768, help="LM Studio context length to request when loading each model.")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum output tokens for each LLM call.")
    parser.add_argument("--target-samples", type=int, default=None, help="Override domain ABC-SMC target samples.")
    parser.add_argument("--generations", type=int, default=None, help="Override domain ABC-SMC generations.")
    parser.add_argument("--initial-particles", type=int, default=None, help="Override domain ABC-SMC initial particles.")
    parser.add_argument(
        "--inference-strategy",
        default="gaussian_weighted",
        help="ABC-SMC transition strategy passed through gauntlet/domain evaluation.",
    )
    parser.add_argument("--load-timeout", type=int, default=600, help="Timeout seconds for LM Studio native load calls.")
    parser.add_argument("--unload-timeout", type=int, default=30, help="Timeout seconds for LM Studio native unload calls.")
    parser.add_argument("--unload-between-models", action=argparse.BooleanOptionalAction, default=True, help="Unload all LM Studio loaded instances before and after each model.")
    parser.add_argument(
        "--include-quarantined-domains",
        action="store_true",
        help="Run known unstable domains instead of recording them as quarantined skips.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid = [domain for domain in args.domains if domain not in DOMAIN_CONFIGS]
    if invalid:
        print(f"Unknown domains: {', '.join(invalid)}", file=sys.stderr)
        return 2
    requested_domains = list(args.domains)
    stable_domains, quarantined_domains = partition_requested_domains(
        requested_domains,
        include_quarantined=args.include_quarantined_domains,
    )

    models = args.models or generation_models(args.base_url)
    excluded = set(args.exclude_models)
    models = [model_id for model_id in models if model_id not in excluded]
    if not models:
        print("No LM Studio generation models found.", file=sys.stderr)
        return 1

    summary_path = Path(args.summary or f"artifacts/model_matrix/{args.run_id}/summary.json")
    matrix = load_matrix(
        summary_path,
        args,
        models,
        requested_domains,
        stable_domains,
        quarantined_domains,
    )
    done = completed_models(matrix) if args.resume else set()
    project_root = Path(__file__).resolve().parents[1]
    failures = 0

    for model_index, model_id in enumerate(models):
        if model_id in done:
            print(f"\n=== Skipping {model_id}: already recorded in {summary_path} ===")
            continue

        slug = slugify_model(model_id)
        tag_prefix = f"lmstudio_{args.run_id}_{slug}"
        gauntlet_summary = Path(f"artifacts/model_matrix/{args.run_id}/{slug}_gauntlet.json")
        model_record = {
            "model_id": model_id,
            "slug": slug,
            "tag_prefix": tag_prefix,
            "gauntlet_summary": str(gauntlet_summary),
            "inference_strategy": args.inference_strategy,
            "target_samples": args.target_samples,
            "generations": args.generations,
            "initial_particles": args.initial_particles,
            "started_at": utc_now(),
            "status": "starting",
        }

        print(f"\n=== Model {model_index + 1}/{len(models)}: {model_id} ===")
        if args.unload_between_models:
            model_record["pre_unload"] = unload_all_loaded_models(args.base_url, args.unload_timeout)
            matrix["updated_at"] = utc_now()
            upsert_model_record(matrix, model_record)
            write_json(summary_path, matrix)
            if not model_record["pre_unload"]["ok"]:
                print(f"Warning: loaded LM Studio instances remain before {model_id}: {model_record['pre_unload']['after']}")

        model_record["status"] = "loading"
        matrix["updated_at"] = utc_now()
        upsert_model_record(matrix, model_record)
        write_json(summary_path, matrix)
        model_record["load"] = load_lmstudio_model(
            model_id,
            base_url=args.base_url,
            context_length=args.context_length,
            timeout=args.load_timeout,
        )
        if not model_record["load"]["ok"]:
            model_record["status"] = "load_failed"
            model_record["finished_at"] = utc_now()
            if args.unload_between_models:
                model_record["post_unload"] = unload_all_loaded_models(args.base_url, args.unload_timeout)
            matrix["updated_at"] = utc_now()
            upsert_model_record(matrix, model_record)
            write_json(summary_path, matrix)
            failures += 1
            if args.stop_on_failure:
                break
            continue

        if not args.skip_preflight:
            model_record["status"] = "preflighting"
            matrix["updated_at"] = utc_now()
            upsert_model_record(matrix, model_record)
            write_json(summary_path, matrix)
            model_record["preflight"] = preflight_model(model_id, args.base_url, args.preflight_timeout)
            if not model_record["preflight"]["ok"]:
                model_record["status"] = "preflight_failed"
                model_record["finished_at"] = utc_now()
                if args.unload_between_models:
                    model_record["post_unload"] = unload_all_loaded_models(args.base_url, args.unload_timeout)
                matrix["updated_at"] = utc_now()
                upsert_model_record(matrix, model_record)
                write_json(summary_path, matrix)
                failures += 1
                if args.stop_on_failure:
                    break
                continue

        cmd = build_gauntlet_command(
            python_executable=sys.executable,
            family="qwen3_coder_next",
            model_id=model_id,
            tag_prefix=tag_prefix,
            epochs=args.epochs,
            gauntlet_summary=gauntlet_summary,
            endpoint=chat_endpoint(args.base_url),
            max_tokens=args.max_tokens,
            target_samples=args.target_samples,
            generations=args.generations,
            initial_particles=args.initial_particles,
            domains=requested_domains,
            held_out=args.held_out,
            seed=None if args.seed is None else args.seed + model_index * 1000,
            inference_strategy=args.inference_strategy,
            include_quarantined_domains=args.include_quarantined_domains,
        )

        model_record["cmd"] = cmd
        model_record["status"] = "running"
        model_record["per_model_timeout_seconds"] = args.per_model_timeout
        matrix["updated_at"] = utc_now()
        upsert_model_record(matrix, model_record)
        write_json(summary_path, matrix)

        return_code, duration, timed_out = run_streaming(cmd, project_root, args.per_model_timeout)
        model_record["duration_seconds"] = duration
        model_record["return_code"] = return_code
        model_record["finished_at"] = utc_now()
        model_record["metrics"] = collect_model_metrics(
            tag_prefix,
            requested_domains,
            quarantined_domains,
        )
        model_record["timed_out"] = timed_out
        model_record["status"] = classify_completed_model_record(
            model_record,
            requested_domain_count=len(stable_domains),
        )
        if model_record["status"] == "completed_with_domain_failures":
            model_record["domain_failures"] = failed_domains(model_record)
        if args.unload_between_models:
            model_record["post_unload"] = unload_all_loaded_models(args.base_url, args.unload_timeout)
        matrix["updated_at"] = utc_now()
        upsert_model_record(matrix, model_record)
        write_json(summary_path, matrix)

        if return_code != 0:
            failures += 1
            if args.stop_on_failure:
                break

    matrix["finished_at"] = utc_now()
    matrix["failures"] = failures
    write_json(summary_path, matrix)
    print(f"\nModel matrix summary written to {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

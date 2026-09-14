#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

# Ensure path includes workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.domain_configs import DOMAIN_CONFIGS
from core.evaluation_boundary import (
    CANONICAL_PROTOCOL_VERSION,
    frozen_parameters_hash,
    load_partition_manifest,
    parameter_semantics_from_code,
    sha256_json,
    sha256_text,
)
from core.generated_code import GeneratedCodeError, extract_and_validate, extract_code, proposal_fingerprint

DEFAULT_ENDPOINT = "http://localhost:1234/v1/chat/completions"
FIXED_RESEARCH_MODEL_ID = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
FIXED_RESEARCH_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MAX_LOOP_FEEDBACK_ITEMS = 6
DUPLICATE_DIVERSIFY_STREAK = 2
DUPLICATE_EARLY_STOP_STREAK = 5
DISTINCT_CANDIDATE_STALL_WINDOW = 12
MIN_ACCEPTED_UPDATE_YIELD = 0.05
EXTREME_TEST_MSE_THRESHOLD = 1e8
CANONICAL_CANDIDATE_TIMEOUT_SECONDS = 120
CANONICAL_MAX_REPAIR_ATTEMPTS = 1
CANONICAL_MAX_PROPOSALS = 2
CANONICAL_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
FAMILY_DEFAULTS = {
    "qwen36": {
        "model_id": "qwen3.6-27b-nvfp4",
        "tag_prefix": "qwen36_27b",
    },
    "qwen3_coder_next": {
        "model_id": "qwen/qwen3-coder-next",
        "tag_prefix": "qwen3_coder_next",
    },
    "gemma4": {
        "model_id": "gemma-4-26b-it",
        "tag_prefix": "gemma4_26b",
    },
}


class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def persist_diagnostic_packet(models_dir: str, artifact_name: str, eval_data: dict | None) -> str | None:
    diagnostics = eval_data.get("diagnostics") if eval_data else None
    if not diagnostics:
        return None
    out_dir = os.path.join(models_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{artifact_name}.json")
    payload = {
        "artifact_name": artifact_name,
        "status": eval_data.get("status"),
        "median_distance": eval_data.get("median_distance"),
        "min_distance": eval_data.get("min_distance"),
        "train_metrics": eval_data.get("train_metrics"),
        "validation_metrics": eval_data.get("validation_metrics"),
        "test_metrics": eval_data.get("test_metrics"),
        "evaluation_protocol": eval_data.get("evaluation_protocol", "legacy_held_out"),
        "development_data_receipt": eval_data.get("development_data_receipt"),
        "diagnostics": diagnostics,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def diagnostic_prompt_context(eval_data: dict | None) -> str:
    diagnostics = eval_data.get("diagnostics") if eval_data else None
    if not diagnostics:
        return "No incumbent diagnostics available."
    compact = {
        "fit": diagnostics.get("fit"),
        "best_parameters": diagnostics.get("best_parameters"),
        "residual_summary": diagnostics.get("residual_summary"),
        "warnings": diagnostics.get("warnings"),
        "failure_modes": diagnostics.get("failure_modes"),
    }
    return json.dumps(compact, indent=2, sort_keys=True)


def call_llm(
    prompt,
    model_id,
    endpoint,
    max_tokens=8192,
    system_prompt="You are a mathematical AI. Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`. No prose.",
    timeout_seconds=900,
    chat_template_kwargs=None,
    response_metadata=None,
):
    """Call the configured chat endpoint and preserve the legacy return API.

    Callers still receive the extracted message content or ``None``.  Canonical
    callers may pass a mutable ``response_metadata`` dict to receive the full
    transport/result receipt, including failures and the untruncated response
    body.  Legacy callers that omit the sink are unaffected.
    """
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
    request_started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    status_code = None
    raw_text = ""
    response_payload = None
    error = None
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout_seconds)
        status_code = getattr(r, "status_code", None)
        raw_text = str(getattr(r, "text", "") or "")
        try:
            response_payload = r.json()
        except (TypeError, ValueError):
            response_payload = None
        r.raise_for_status()
        content = response_payload["choices"][0]["message"]["content"]
        if isinstance(response_metadata, dict) and not isinstance(content, str):
            raise TypeError("chat completion message.content must be a string")
        if isinstance(response_metadata, dict) and not content.strip():
            raise ValueError("chat completion message.content is empty")
        return content
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"LLM Error: {e}")
        return None
    finally:
        if isinstance(response_metadata, dict):
            choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else {}
            usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
            response_metadata.update({
                "status": "success" if error is None else "failed",
                "status_code": status_code,
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                "usage": usage,
                "response_id": response_payload.get("id") if isinstance(response_payload, dict) else None,
                "response_model": response_payload.get("model") if isinstance(response_payload, dict) else None,
                "response_created": response_payload.get("created") if isinstance(response_payload, dict) else None,
                "system_fingerprint": response_payload.get("system_fingerprint") if isinstance(response_payload, dict) else None,
                "elapsed_seconds": time.monotonic() - started,
                "request_started_at": request_started_at,
                "request_finished_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
                "raw_response": raw_text,
                "raw_response_sha256": sha256_text(raw_text),
            })


def invoke_llm_with_metadata(*args, **kwargs):
    """Invoke ``call_llm`` while guaranteeing a receipt for canonical calls."""

    response_metadata = {}
    started = time.monotonic()
    kwargs["response_metadata"] = response_metadata
    try:
        reply = call_llm(*args, **kwargs)
    except Exception as exc:  # Defensive for injected/mocked clients too.
        reply = None
        response_metadata.update({
            "status": "failed",
            "status_code": None,
            "finish_reason": None,
            "usage": None,
            "elapsed_seconds": time.monotonic() - started,
            "request_started_at": datetime.now(timezone.utc).isoformat(),
            "request_finished_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "raw_response": "",
            "raw_response_sha256": sha256_text(""),
        })
    if reply is not None and (not isinstance(reply, str) or not reply.strip()):
        response_metadata.update({
            "status": "failed",
            "error": (
                "chat completion message.content must be a non-empty string"
                if not isinstance(reply, str)
                else "chat completion message.content is empty"
            ),
        })
        reply = None
    return reply, response_metadata


def append_prompt_receipt(
    receipts: list[dict],
    *,
    iteration: int,
    stage: str,
    prompt: str,
    response: str | None,
    protocol: str,
    response_metadata: dict | None = None,
) -> None:
    """Record canonical prompt/response hashes and payloads without final data."""

    metadata = dict(response_metadata or {})
    raw_response = str(metadata.get("raw_response") or "")
    metadata.setdefault("status", "success" if response is not None else "failed")
    metadata.setdefault("status_code", None)
    metadata.setdefault("finish_reason", None)
    metadata.setdefault("usage", None)
    metadata.setdefault("elapsed_seconds", None)
    metadata.setdefault("error", None if response is not None else "LLM returned no content")
    metadata["raw_response"] = raw_response
    metadata["raw_response_sha256"] = sha256_text(raw_response)
    receipts.append({
        "iteration": iteration,
        "stage": stage,
        "protocol": protocol,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "response_sha256": sha256_text(response or ""),
        "final_outcomes_in_context": False,
        "chat_template_kwargs": dict(CANONICAL_CHAT_TEMPLATE_KWARGS),
        "response_metadata": metadata,
    })


def write_prompt_receipts(path: str, receipts: list[dict]) -> None:
    """Durably publish canonical LLM receipts after each request."""

    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as f:
        json.dump(receipts, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary_path, path)


def build_repair_prompt(domain: str, previous_code: str | None, error_msg: str | None) -> str:
    detail = (error_msg or "").strip()
    lowered = detail.lower()
    previous_code_block = previous_code or "# No previous code was extracted."

    contract = """Output ONLY a Python code block.
- Define exactly one `dynamics(t, y, args)` function.
- Define a top-level `metadata = [...]` list of dicts with parameter names and `(low, high)` ranges.
- Do not include any import statements; `jnp` is already available in the execution sandbox.
- Preserve the mathematical intent when possible while fixing the failure."""

    if "missing dynamics" in lowered:
        repair_focus = (
            "The validator could not find `dynamics(t, y, args)`. "
            "Add that exact function signature and return clipped derivatives with `jnp.clip(jnp.array([...]), -1e6, 1e6)`."
        )
    elif "missing metadata" in lowered:
        repair_focus = (
            "The validator could not find a valid `metadata` assignment. "
            "Add `metadata = [...]` so each `args[i]` used in `dynamics` has a matching parameter entry in order."
        )
    elif "imports are not allowed" in lowered or "forbidden" in lowered:
        repair_focus = (
            "Remove forbidden imports or unsafe file/network/system access. "
            "Use only sandbox-provided symbols such as `jnp`, `t`, `y`, and `args`."
        )
    else:
        repair_focus = "Fix the bug reported below and return only the corrected mathematical code."

    return f"""Your previous code for {domain} failed.

Failure detail: `{detail or "unknown error"}`

Repair focus:
{repair_focus}

Required output contract:
{contract}

Previous code:
```python
{previous_code_block}
```"""


def register_proposal_fingerprint(
    ledger: list[dict],
    known_fingerprints: dict[str, str],
    *,
    iteration: int,
    stage: str,
    code: str,
) -> tuple[str, str | None]:
    fingerprint = proposal_fingerprint(code)
    duplicate_of = known_fingerprints.get(fingerprint)
    record_id = f"{iteration}:{stage}"
    ledger.append({
        "id": record_id,
        "iteration": iteration,
        "stage": stage,
        "fingerprint": fingerprint,
        "duplicate_of": duplicate_of,
        "is_duplicate": duplicate_of is not None,
    })
    if duplicate_of is None:
        known_fingerprints[fingerprint] = record_id
    return fingerprint, duplicate_of


def write_proposal_fingerprint_ledger(path: str, ledger: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump({"proposals": ledger}, f, indent=2, sort_keys=True)


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def metric_mse(eval_data: dict | None, split: str) -> float | None:
    if not eval_data:
        return None
    metrics = eval_data.get(f"{split}_metrics") or {}
    return finite_float(metrics.get("mse"))


def metric_failure_feedback(
    eval_data: dict | None,
    *,
    threshold: float = EXTREME_TEST_MSE_THRESHOLD,
    metrics_key: str = "test_metrics",
    metric_label: str = "held-out",
) -> str | None:
    if not eval_data:
        return None
    raw_metric = (eval_data.get(metrics_key) or {}).get("mse")
    metric_value = finite_float(raw_metric)
    if metric_value is None:
        return f"Previous candidate produced non-finite {metric_label} metrics; constrain dynamics, state clipping, and parameter ranges before adding mechanisms."
    if abs(metric_value) > threshold:
        return (
            f"Previous candidate produced extreme {metric_label} MSE {metric_value:.6g}; "
            "avoid explosive dynamics and tighten parameter ranges around plausible scales."
        )
    return None


def held_out_regression_feedback(
    incumbent_eval: dict | None,
    candidate_eval: dict | None,
    *,
    incumbent_score: float,
    candidate_score: float,
) -> str | None:
    incumbent_train = metric_mse(incumbent_eval, "train")
    candidate_train = metric_mse(candidate_eval, "train")
    if incumbent_train is None or candidate_train is None:
        return None
    if candidate_train < incumbent_train and candidate_score >= incumbent_score:
        return (
            f"Previous candidate improved train MSE {incumbent_train:.6g} -> {candidate_train:.6g} "
            f"but worsened or failed held-out score {incumbent_score:.6g} -> {candidate_score:.6g}; "
            "target generalization and residual structure, not train-only fit."
        )
    return None


def loop_feedback_context(events: list[str], duplicate_streak: int) -> str:
    lines = []
    if duplicate_streak >= DUPLICATE_DIVERSIFY_STREAK:
        lines.append(
            f"Duplicate proposal streak is {duplicate_streak}. Propose a materially different mechanism family, "
            "different state coupling, or different parameterization; do not repeat the incumbent or previous failed candidate."
        )
    lines.extend(events[-MAX_LOOP_FEEDBACK_ITEMS:])
    if not lines:
        return "No loop waste feedback yet."
    return "\n".join(f"- {line}" for line in lines)


def adaptive_yield_stop_reason(candidate_records: list[dict]) -> str | None:
    """Return a stop reason when recent distinct evaluated candidates yield no updates."""
    if len(candidate_records) < DISTINCT_CANDIDATE_STALL_WINDOW:
        return None
    recent = candidate_records[-DISTINCT_CANDIDATE_STALL_WINDOW:]
    accepted = sum(bool(record.get("accepted")) for record in recent)
    yield_rate = accepted / DISTINCT_CANDIDATE_STALL_WINDOW
    if accepted == 0:
        return (
            f"Stopped after {DISTINCT_CANDIDATE_STALL_WINDOW} distinct evaluated candidates without an accepted update; "
            "further inference is likely wasted without a prompt or architecture change."
        )
    if yield_rate < MIN_ACCEPTED_UPDATE_YIELD:
        return (
            f"Stopped because the rolling {DISTINCT_CANDIDATE_STALL_WINDOW}-candidate accepted-update yield "
            f"was {yield_rate:.1%}, below {MIN_ACCEPTED_UPDATE_YIELD:.1%}."
        )
    return None


def summarize_proposal_waste(
    *,
    proposal_fingerprint_ledger: list[dict],
    duplicate_early_stop: bool = False,
    adaptive_yield_early_stop: bool = False,
    early_stop_reason: str | None = None,
    held_out_history: list[dict] | None = None,
    models_dir: str | None = None,
    candidate_evaluation_records: list[dict] | None = None,
) -> dict:
    non_seed = [p for p in proposal_fingerprint_ledger if p.get("stage") != "seed"]
    duplicate_proposals = [p for p in non_seed if p.get("is_duplicate")]
    repair_proposals = [p for p in non_seed if str(p.get("stage", "")).startswith("repair")]
    candidate_evaluation_records = candidate_evaluation_records or []
    recent = candidate_evaluation_records[-DISTINCT_CANDIDATE_STALL_WINDOW:]
    recent_accepted = sum(bool(record.get("accepted")) for record in recent)
    summary = {
        "schema_version": 1,
        "proposal_records": len(non_seed),
        "duplicate_proposals": len(duplicate_proposals),
        "duplicate_rate": (len(duplicate_proposals) / len(non_seed)) if non_seed else 0.0,
        "repair_proposals": len(repair_proposals),
        "repair_rate": (len(repair_proposals) / len(non_seed)) if non_seed else 0.0,
        "unique_proposals": len(non_seed) - len(duplicate_proposals),
        "duplicate_early_stop": duplicate_early_stop,
        "early_stop_reason": early_stop_reason,
        "accepted_updates": max(len(held_out_history or []) - 1, 0),
        "distinct_evaluated_candidates": len(candidate_evaluation_records),
        "accepted_update_yield_over_last_12": (recent_accepted / len(recent)) if recent else 0.0,
        "adaptive_yield_early_stop": adaptive_yield_early_stop,
    }
    if models_dir:
        diagnostics_dir = os.path.join(models_dir, "diagnostics")
        candidate_files = []
        repair_files = []
        if os.path.isdir(diagnostics_dir):
            candidate_files = [
                name for name in os.listdir(diagnostics_dir)
                if name.startswith("iteration_") and name.endswith("_candidate.json")
            ]
            repair_files = [
                name for name in os.listdir(diagnostics_dir)
                if "_repair_" in name and name.endswith(".json")
            ]
        nonfinite = 0
        extreme = 0
        for name in candidate_files + repair_files:
            path = os.path.join(diagnostics_dir, name)
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except Exception:
                continue
            raw_test = (payload.get("test_metrics") or {}).get("mse")
            test_mse = finite_float(raw_test)
            if test_mse is None:
                nonfinite += 1
            elif abs(test_mse) > EXTREME_TEST_MSE_THRESHOLD:
                extreme += 1
        summary.update({
            "candidate_evaluations": len(candidate_files),
            "repair_evaluations": len(repair_files),
            "nonfinite_candidate_metrics": nonfinite,
            "extreme_candidate_metrics": extreme,
        })
    return summary


def write_proposal_waste_summary(path: str, summary: dict) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def write_canonical_freeze(
    *,
    models_dir: str,
    domain: str,
    domain_config: dict,
    candidate_path: str,
    selection_path: str,
    search_receipt_path: str,
    prompt_receipts_path: str,
    protocol_manifest: str,
    current_logic: str,
    best_loss: float,
    best_score: float,
    data: dict | None,
    proposal_fingerprint_ledger: list[dict],
    candidate_evaluation_records: list[dict],
    prompt_receipts: list[dict],
) -> dict:
    """Persist a frozen development result without touching sealed inputs."""

    candidate_code_path = os.path.abspath(candidate_path)
    with open(candidate_code_path, "w") as f:
        f.write(current_logic)
    best_parameters = ((data or {}).get("diagnostics") or {}).get("best_parameters") or {}
    development_data_receipt = (data or {}).get("development_data_receipt") or {}
    parameter_semantics = parameter_semantics_from_code(current_logic)
    development_manifest_data = load_partition_manifest(
        protocol_manifest,
        required_roles=("train", "validation"),
        forbidden_roles=("final",),
    )
    development_train_times = development_manifest_data["roles"]["train"].time_points
    if len(development_train_times) == 0:
        raise ValueError("canonical development train partition has no time origin")
    time_origin = float(development_train_times[0])
    selection = {
        "schema_version": 1,
        "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
        "domain": domain,
        "protocol_manifest": os.path.abspath(protocol_manifest),
        "development_receipt_sha256": development_data_receipt.get("development_receipt_sha256"),
        "development_split_metadata": development_data_receipt.get("metadata", {}),
        "candidate_code_path": candidate_code_path,
        "candidate_code_sha256": sha256_text(current_logic),
        "parameter_semantics": parameter_semantics,
        "parameter_semantics_sha256": sha256_json(parameter_semantics),
        "model_contract": {
            "initial_conditions": list(domain_config["y0"]),
            "solver_dt0": float(domain_config["dt0"]),
            "solver_max_steps": int(domain_config["max_steps"]),
            "time_origin": time_origin,
            "time_origin_source": "development_train_manifest",
            "time_origin_policy": "declared_manifest_time_points",
        },
        "llm_request_settings": {
            "chat_template_kwargs": dict(CANONICAL_CHAT_TEMPLATE_KWARGS),
        },
        "parameters": best_parameters,
        "parameters_sha256": frozen_parameters_hash(best_parameters),
        "selected_median_distance": finite_float(best_loss),
        "selected_validation_mse": finite_float(best_score),
        "selected_metric_role": "development_validation_mse",
        "final_evaluation_required": True,
        "final_outcomes_available_to_search": False,
    }
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2, sort_keys=True)
    write_prompt_receipts(prompt_receipts_path, prompt_receipts)
    search_receipt = {
        "schema_version": 1,
        "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
        "domain": domain,
        "development_data_receipt": (data or {}).get("development_data_receipt"),
        "protocol_manifest": os.path.abspath(protocol_manifest),
        "candidate_code_sha256": sha256_text(current_logic),
        "parameters_sha256": frozen_parameters_hash(best_parameters),
        "parameter_semantics": parameter_semantics,
        "parameter_semantics_sha256": sha256_json(parameter_semantics),
        "model_contract": selection["model_contract"],
        "llm_request_settings": selection["llm_request_settings"],
        "selected_metric_role": "development_validation_mse",
        "selected_validation_mse": finite_float(best_score),
        "selected_median_distance": finite_float(best_loss),
        "proposal_fingerprints": proposal_fingerprint_ledger,
        "candidate_evaluations": candidate_evaluation_records,
        "prompt_receipt_sha256": sha256_text(json.dumps(prompt_receipts, sort_keys=True)),
        "final_outcomes_available_to_search": False,
        "final_data_hash": None,
    }
    with open(search_receipt_path, "w") as f:
        json.dump(search_receipt, f, indent=2, sort_keys=True)
    return selection


def run_evaluation(
    code,
    domain,
    held_out,
    target_samples,
    generations,
    initial_particles,
    seed,
    models_dir,
    inference_strategy="gaussian_weighted",
    evaluation_protocol="legacy",
    protocol_manifest=None,
    timeout_seconds=1200,
):
    # Write code to temporary file
    temp_code_path = os.path.join(models_dir, "temp_proposed.py")
    temp_out_path = os.path.join(models_dir, "temp_eval.json")
    
    with open(temp_code_path, "w") as f:
        f.write(code)
        
    cmd = [
        sys.executable,
        "-m", "core.sandbox_eval",
        "--code-file", temp_code_path,
        "--domain", domain,
        "--output", temp_out_path,
        "--target-samples", str(target_samples),
        "--generations", str(generations),
        "--initial-particles", str(initial_particles),
        "--inference-strategy", inference_strategy,
    ]
    if evaluation_protocol != "legacy":
        cmd.extend(["--evaluation-protocol", evaluation_protocol])
        if protocol_manifest:
            cmd.extend(["--protocol-manifest", str(protocol_manifest)])
    if held_out:
        cmd.append("--held-out")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
        
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        if res.returncode != 0:
            err = res.stderr or res.stdout or "Subprocess returned non-zero exit code"
            return float("inf"), None, f"SubprocessError: {err}"
            
        if not os.path.exists(temp_out_path):
            return float("inf"), None, "Evaluation output file missing"
            
        with open(temp_out_path, "r") as f:
            data = json.load(f)
            
        if data["status"] == "error":
            return float("inf"), None, data["error"] or "Unknown evaluation error"
            
        return data["median_distance"], data, None
    except subprocess.TimeoutExpired:
        return float("inf"), None, "TimeoutExpired: Subprocess evaluation timed out"
    except Exception as e:
        return float("inf"), None, f"Exception: {str(e)}"
    finally:
        # Cleanup temp files
        for p in [temp_code_path, temp_out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def resolve_family(
    family: str,
    domain: str,
    model: str | None = None,
    endpoint: str | None = None,
    tag_prefix: str | None = None,
):
    if family not in FAMILY_DEFAULTS:
        raise ValueError(f"Unknown family '{family}'")
    defaults = FAMILY_DEFAULTS[family]
    model_id = model or defaults["model_id"]
    llm_endpoint = endpoint or DEFAULT_ENDPOINT
    resolved_tag_prefix = tag_prefix or defaults["tag_prefix"]
    experiment_tag = f"{resolved_tag_prefix}_{domain}"
    return model_id, llm_endpoint, experiment_tag


def selection_score(
    median_distance: float,
    eval_data: dict | None,
    held_out: bool,
    evaluation_protocol: str = "legacy",
) -> float:
    if evaluation_protocol == CANONICAL_PROTOCOL_VERSION and eval_data:
        score = finite_float((eval_data.get("validation_metrics") or {}).get("mse"))
        return score if score is not None else float("inf")
    if held_out and eval_data and eval_data.get("test_metrics"):
        score = finite_float(eval_data["test_metrics"].get("mse"))
        return score if score is not None else float("inf")
    score = finite_float(median_distance)
    return score if score is not None else float("inf")


def score_label(held_out: bool, evaluation_protocol: str = "legacy") -> str:
    if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
        return "Development validation MSE"
    return "Held-out test MSE" if held_out else "ABC-SMC median distance"


def run_domain_main(
    domain: str,
    family: str = "qwen36",
    model: str | None = None,
    epochs: int = 100,
    held_out: bool = False,
    seed: int | None = None,
    endpoint: str | None = None,
    tag_prefix: str | None = None,
    max_tokens: int = 8192,
    inference_strategy: str = "gaussian_weighted",
    target_samples_override: int | None = None,
    generations_override: int | None = None,
    initial_particles_override: int | None = None,
    evaluation_protocol: str = "legacy",
    protocol_manifest: str | None = None,
    dry_run: bool = False,
) -> int:
    if domain not in DOMAIN_CONFIGS:
        print(f"Error: Unknown domain '{domain}'")
        return 1
        
    config = DOMAIN_CONFIGS[domain]

    if evaluation_protocol not in {"legacy", CANONICAL_PROTOCOL_VERSION}:
        print(f"Error: Unknown evaluation protocol '{evaluation_protocol}'")
        return 1
    if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
        if held_out:
            print("Error: canonical trajectory protocol cannot be combined with legacy --held-out")
            return 1
        if not protocol_manifest:
            print("Error: canonical trajectory protocol requires --protocol-manifest")
            return 1
        try:
            # Validate the development input before any LLM request. The
            # manifest loader rejects a final role, preventing accidental
            # sealed-outcome exposure through the canonical runner.
            load_partition_manifest(
                protocol_manifest,
                required_roles=("train", "validation"),
                forbidden_roles=("final",),
            )
        except Exception as exc:
            print(f"Error: invalid canonical development manifest: {exc}")
            return 1
        epochs = min(int(epochs), CANONICAL_MAX_PROPOSALS)
    
    try:
        model_id, llm_endpoint, experiment_tag = resolve_family(
            family,
            domain,
            model=model,
            endpoint=endpoint,
            tag_prefix=tag_prefix,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
        if model_id != FIXED_RESEARCH_MODEL_ID or llm_endpoint != FIXED_RESEARCH_ENDPOINT:
            print(
                "Error: canonical research protocol requires the fixed Qwen model "
                f"{FIXED_RESEARCH_MODEL_ID} at {FIXED_RESEARCH_ENDPOINT}"
            )
            return 1

    run_seed = seed if seed is not None else (int(os.environ["E3_SEED"]) if os.environ.get("E3_SEED") else None)

    MODELS_DIR = f"models/{experiment_tag}"
    LOG_FILE = f"{experiment_tag}_run_history.csv"
    BEST_LOGIC_PATH = f"{MODELS_DIR}/best_logic.py"
    BEST_DIAGNOSTICS_PATH = f"{MODELS_DIR}/best_diagnostics.json"
    PROPOSAL_FINGERPRINTS_PATH = f"{MODELS_DIR}/proposal_fingerprints.json"
    PROPOSAL_WASTE_PATH = f"{MODELS_DIR}/proposal_waste_summary.json"
    PROMPT_RECEIPTS_PATH = f"{MODELS_DIR}/proposal_prompt_receipts.json"
    SEARCH_RECEIPT_PATH = f"{MODELS_DIR}/search_receipt.json"
    FROZEN_SELECTION_PATH = f"{MODELS_DIR}/frozen_selection.json"
    FROZEN_CANDIDATE_PATH = f"{MODELS_DIR}/frozen_candidate.py"
    RUN_LOG_PATH = f"{experiment_tag}_run.log"

    # Dynamic target samples, generations, and initial particles from config,
    # with explicit run-level overrides for strategy and scaling experiments.
    target_samples = target_samples_override if target_samples_override is not None else config.get("target_samples", 500)
    generations = generations_override if generations_override is not None else config.get("generations", 15)
    initial_particles = (
        initial_particles_override
        if initial_particles_override is not None
        else config.get("initial_particles", 150000)
    )

    if dry_run:
        print(json.dumps({
            "domain": domain,
            "family": family,
            "model_id": model_id,
            "llm_endpoint": llm_endpoint,
            "experiment_tag": experiment_tag,
            "models_dir": MODELS_DIR,
            "log_file": LOG_FILE,
            "held_out": held_out,
            "evaluation_protocol": evaluation_protocol,
            "protocol_manifest": protocol_manifest,
            "seed": run_seed,
            "max_tokens": max_tokens,
            "inference_strategy": inference_strategy,
            "target_samples": target_samples,
            "generations": generations,
            "initial_particles": initial_particles,
            "candidate_timeout_seconds": CANONICAL_CANDIDATE_TIMEOUT_SECONDS if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else 1200,
            "llm_request_settings": {
                "chat_template_kwargs": dict(CANONICAL_CHAT_TEMPLATE_KWARGS),
            } if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else None,
        }, indent=2, sort_keys=True))
        return 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    proposal_fingerprint_ledger: list[dict] = []
    known_proposal_fingerprints: dict[str, str] = {}
    loop_feedback_events: list[str] = []
    duplicate_streak = 0
    duplicate_early_stop = False
    adaptive_yield_early_stop = False
    early_stop_reason = None
    candidate_evaluation_records: list[dict] = []
    prompt_receipts: list[dict] = []
    validation_history: list[dict] = []
    canonical_repairs_used = 0

    # Setup Tee Logging
    _log_fh = open(RUN_LOG_PATH, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

    print("=" * 70)
    print(f"  SBI-Factory Config Runner · {config['name']}")
    print(f"  Domain      : {domain} | Family: {family} | Model: {model_id}")
    print(f"  Held-out Eval: {held_out}")
    print(f"  Evaluation Protocol: {evaluation_protocol}")
    print(f"  Inference Strategy: {inference_strategy}")
    print(f"  ABC-SMC     : target_samples={target_samples} generations={generations} initial_particles={initial_particles}")
    print("=" * 70)

    from core.run_config import RunConfig, write_run_config
    write_run_config(
        RunConfig(
            experiment_tag=experiment_tag,
            domain=domain,
            model_id=model_id,
            llm_endpoint=llm_endpoint,
            seed=run_seed,
            target_samples=target_samples,
            generations=generations,
            initial_particles=initial_particles,
            extra={
                "inference_strategy": inference_strategy,
                "evaluation_protocol": evaluation_protocol,
                "protocol_manifest": protocol_manifest,
                "llm_request_settings": {
                    "chat_template_kwargs": dict(CANONICAL_CHAT_TEMPLATE_KWARGS),
                } if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else None,
            },
        ),
        f"{MODELS_DIR}/run_config.json",
    )

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Iteration", "Action", "Proposed_Loss", "Baseline_Loss"])

    def log_action(iteration, action, prop_loss, base_loss):
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iteration, action, prop_loss, base_loss])

    # Evaluate Seed
    current_logic = config["seed_logic"]
    register_proposal_fingerprint(
        proposal_fingerprint_ledger,
        known_proposal_fingerprints,
        iteration=0,
        stage="seed",
        code=current_logic,
    )
    write_proposal_fingerprint_ledger(PROPOSAL_FINGERPRINTS_PATH, proposal_fingerprint_ledger)
    write_proposal_waste_summary(
        PROPOSAL_WASTE_PATH,
        summarize_proposal_waste(
            proposal_fingerprint_ledger=proposal_fingerprint_ledger,
            held_out_history=[],
            models_dir=MODELS_DIR,
        ),
    )
    print("Evaluating seed logic...")
    best_loss, data, err_msg = run_evaluation(
        current_logic,
        domain,
        held_out,
        target_samples,
        generations,
        initial_particles,
        run_seed,
        MODELS_DIR,
        inference_strategy=inference_strategy,
        evaluation_protocol=evaluation_protocol,
        protocol_manifest=protocol_manifest,
        timeout_seconds=(CANONICAL_CANDIDATE_TIMEOUT_SECONDS if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else 1200),
    )
    if best_loss == float("inf"):
        print(f"Error evaluating seed: {err_msg}")
        return 1
    seed_diag_path = persist_diagnostic_packet(MODELS_DIR, "seed", data)
    if seed_diag_path:
        with open(seed_diag_path, "r") as src, open(BEST_DIAGNOSTICS_PATH, "w") as dst:
            dst.write(src.read())

    best_score = selection_score(best_loss, data, held_out, evaluation_protocol)
    label = score_label(held_out, evaluation_protocol)
    print(f"Seed ABC-SMC median distance: {best_loss:.6f}")
    if held_out or evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
        print(f"Seed {label}: {best_score:.6f}")
    log_action(0, "SEED_EVALUATED", best_score, best_score)
    
    # Save seed logic
    best_scaff_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{current_logic}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0=float(time_points[0]), t1=float(time_points[-1]), dt0={config['dt0']}, y0=y0, args=params, saveat=saveat, max_steps={config['max_steps']})
        return sol.ys
    def get_parameter_metadata(self):
        return metadata
    def get_initial_conditions(self):
        return jnp.array({config['y0']})
    def get_latex(self):
        return "{family.upper()} Generated {domain}"
"""
    with open(BEST_LOGIC_PATH, "w") as f:
        f.write(best_scaff_module)

    # Meta hints from prior runs
    dynamic_hints = ""
    meta_hints_path = f"{MODELS_DIR}/meta_hints.txt"
    if os.path.exists(meta_hints_path):
        with open(meta_hints_path, "r") as f:
            dynamic_hints = f.read()

    # Track held-out metrics separately
    held_out_metrics_path = f"{MODELS_DIR}/held_out_metrics.json"
    held_out_history = []
    if held_out and data.get("train_metrics"):
        held_out_history.append({
            "iteration": 0,
            "train": data["train_metrics"],
            "test": data["test_metrics"],
        })
        with open(held_out_metrics_path, "w") as f:
            json.dump(held_out_history, f, indent=2)
    if evaluation_protocol == CANONICAL_PROTOCOL_VERSION and data.get("train_metrics"):
        validation_history.append({
            "iteration": 0,
            "train": data.get("train_metrics"),
            "validation": data.get("validation_metrics"),
            "validation_mse": best_score,
        })
        with open(f"{MODELS_DIR}/validation_metrics.json", "w") as f:
            json.dump(validation_history, f, indent=2)

    # Main Evolutionary Loop
    for iteration in range(1, epochs + 1):
        if yield_stop_reason := adaptive_yield_stop_reason(candidate_evaluation_records):
            adaptive_yield_early_stop = True
            early_stop_reason = yield_stop_reason
            print(f"Distinct-candidate yield early stop: {early_stop_reason}")
            log_action(iteration, "DISTINCT_CANDIDATE_YIELD_EARLY_STOP", float("inf"), best_score)
            break
        if duplicate_streak >= DUPLICATE_EARLY_STOP_STREAK:
            duplicate_early_stop = True
            early_stop_reason = (
                f"Stopped after {duplicate_streak} consecutive duplicate proposals; "
                "further epochs are likely wasted without prompt or architecture changes."
            )
            print(f"Duplicate churn early stop: {early_stop_reason}")
            log_action(iteration, "DUPLICATE_CHURN_EARLY_STOP", float("inf"), best_score)
            break

        print(f"\n--- {domain.upper()} Iteration {iteration}/{epochs} ---")
        hints_str = "\n".join(f"- {h}" for h in config["hints"])
        incumbent_diagnostics = diagnostic_prompt_context(data)
        loop_feedback = loop_feedback_context(loop_feedback_events, duplicate_streak)
        prompt = f"""We are fitting a mathematical model for {domain}.
State vector: {config['state_desc']}.
Time request: {config['time_desc']}.
Current {label}: {best_score:.6f}
Current ABC-SMC median distance: {best_loss:.6f}
Previous Run Evolutionary Directions (CRITICAL to utilize if present):
{dynamic_hints}

Current Dynamics:
```python
{current_logic}
```
Incumbent Diagnostics:
```json
{incumbent_diagnostics}
```
Loop Waste Feedback:
{loop_feedback}

Hints:
{hints_str}

Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`.
The `metadata` variable MUST be a list of dicts, each specifying a parameter name and its search range, matching the size of `args` in order. Example:
metadata = [
    {{'name': 'p1', 'range': (0.1, 10.0)}},
    {{'name': 'p2', 'range': (0.01, 1.0)}}
]
IMPORTANT: DO NOT include any import statements (e.g., 'import jax.numpy as jnp'). These modules are already imported for you in the execution sandbox.
CRITICAL SAFETY BOUNDS: You MUST wrap the final calculated derivatives returned from your `dynamics` function using `jnp.clip(jnp.array([...]), -1e6, 1e6)` to prevent severe solver explosion failures!
        """
        llm_kwargs = {}
        if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
            llm_kwargs["timeout_seconds"] = CANONICAL_CANDIDATE_TIMEOUT_SECONDS
            llm_kwargs["chat_template_kwargs"] = dict(CANONICAL_CHAT_TEMPLATE_KWARGS)
        if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
            reply, response_metadata = invoke_llm_with_metadata(
                prompt,
                model_id,
                llm_endpoint,
                max_tokens=max_tokens,
                **llm_kwargs,
            )
        else:
            # Keep legacy monkeypatches and callers on the original call_llm
            # signature/return path; provenance is canonical-only.
            reply = call_llm(prompt, model_id, llm_endpoint, max_tokens=max_tokens, **llm_kwargs)
            response_metadata = None
        if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
            append_prompt_receipt(
                prompt_receipts,
                iteration=iteration,
                stage="proposal",
                prompt=prompt,
                response=reply,
                protocol=evaluation_protocol,
                response_metadata=response_metadata,
            )
            write_prompt_receipts(PROMPT_RECEIPTS_PATH, prompt_receipts)
        raw_extracted = extract_code(reply)
        repair_source_code = raw_extracted.code if raw_extracted else None
        error_msg = "No code extracted"
        try:
            extracted = extract_and_validate(reply, normalize=True)
            new_logic = extracted.code if extracted else None
            thoughts = extracted.thoughts if extracted else ""
            if extracted and extracted.normalization_applied:
                print("Applied harmless generated-code normalization before validation.")
        except GeneratedCodeError as e:
            new_logic = None
            thoughts = raw_extracted.thoughts if raw_extracted else ""
            error_msg = str(e)
            print(f"Generated code rejected by validator: {e}")

        new_loss = float("inf")
        eval_data = None
        duplicate_of = None

        if new_logic:
            _, duplicate_of = register_proposal_fingerprint(
                proposal_fingerprint_ledger,
                known_proposal_fingerprints,
                iteration=iteration,
                stage="candidate",
                code=new_logic,
            )
            write_proposal_fingerprint_ledger(PROPOSAL_FINGERPRINTS_PATH, proposal_fingerprint_ledger)
            if duplicate_of:
                duplicate_streak += 1
                error_msg = f"DuplicateProposal: candidate duplicates {duplicate_of}"
                print(f"Duplicate proposal suppressed before evaluation: {duplicate_of}")
                loop_feedback_events.append(
                    f"Iteration {iteration} candidate duplicated `{duplicate_of}` and was not evaluated."
                )
                new_logic = None
                repair_source_code = None
            else:
                duplicate_streak = 0

        if new_logic:
            new_loss, eval_data, error_msg = run_evaluation(
                new_logic,
                domain,
                held_out,
                target_samples,
                generations,
                initial_particles,
                run_seed,
                MODELS_DIR,
                inference_strategy=inference_strategy,
                evaluation_protocol=evaluation_protocol,
                protocol_manifest=protocol_manifest,
                timeout_seconds=(CANONICAL_CANDIDATE_TIMEOUT_SECONDS if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else 1200),
            )
            persist_diagnostic_packet(MODELS_DIR, f"iteration_{iteration:03d}_candidate", eval_data)
            if eval_data is not None:
                candidate_evaluation_records.append({"iteration": iteration, "stage": "candidate", "accepted": False})
            if warning := metric_failure_feedback(
                eval_data,
                metrics_key=("validation_metrics" if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else "test_metrics"),
                metric_label=("validation" if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else "held-out"),
            ):
                loop_feedback_events.append(f"Iteration {iteration}: {warning}")

        # Agentic Repair Loop
        repair_attempts = CANONICAL_MAX_REPAIR_ATTEMPTS if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else 2
        for attempt in range(1, repair_attempts + 1):
            if new_logic and new_loss < float("inf"):
                break
            if not repair_source_code:
                break
            if (
                evaluation_protocol == CANONICAL_PROTOCOL_VERSION
                and canonical_repairs_used >= CANONICAL_MAX_REPAIR_ATTEMPTS
            ):
                break
            print(f"  [Debugger Agent] Logic failure detected in iteration {iteration}. Attempting repair {attempt}/2...")
            repair_prompt = build_repair_prompt(domain, repair_source_code, error_msg)
            repair_kwargs = {}
            if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
                canonical_repairs_used += 1
                repair_kwargs["timeout_seconds"] = CANONICAL_CANDIDATE_TIMEOUT_SECONDS
                repair_kwargs["chat_template_kwargs"] = dict(CANONICAL_CHAT_TEMPLATE_KWARGS)
            repair_system_prompt = (
                "You are a senior debugging agent. "
                "Repair the mathematical code and output ONLY a Python code block with "
                "`dynamics(t, y, args)` and `metadata`."
            )
            if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
                repair_reply, repair_response_metadata = invoke_llm_with_metadata(
                    repair_prompt,
                    model_id,
                    llm_endpoint,
                    max_tokens=max_tokens,
                    system_prompt=repair_system_prompt,
                    **repair_kwargs,
                )
            else:
                repair_reply = call_llm(
                    repair_prompt,
                    model_id,
                    llm_endpoint,
                    max_tokens=max_tokens,
                    system_prompt=repair_system_prompt,
                    **repair_kwargs,
                )
                repair_response_metadata = None
            if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
                append_prompt_receipt(
                    prompt_receipts,
                    iteration=iteration,
                    stage=f"repair_{attempt}",
                    prompt=repair_prompt,
                    response=repair_reply,
                    protocol=evaluation_protocol,
                    response_metadata=repair_response_metadata,
                )
                write_prompt_receipts(PROMPT_RECEIPTS_PATH, prompt_receipts)
            raw_repair = extract_code(repair_reply)
            if raw_repair:
                repair_source_code = raw_repair.code
                thoughts = raw_repair.thoughts or thoughts
            try:
                repaired = extract_and_validate(repair_reply, normalize=True)
                new_logic = repaired.code if repaired else None
                if repaired and repaired.normalization_applied:
                    print("Applied harmless generated-code normalization to repair before validation.")
            except GeneratedCodeError as e:
                error_msg = str(e)
                print(f"Generated repair rejected by validator: {e}")
                new_logic = None
                continue

            if not new_logic:
                error_msg = "No code extracted"
                continue

            _, duplicate_of = register_proposal_fingerprint(
                proposal_fingerprint_ledger,
                known_proposal_fingerprints,
                iteration=iteration,
                stage=f"repair_{attempt}",
                code=new_logic,
            )
            write_proposal_fingerprint_ledger(PROPOSAL_FINGERPRINTS_PATH, proposal_fingerprint_ledger)
            if duplicate_of:
                duplicate_streak += 1
                error_msg = f"DuplicateProposal: repair duplicates {duplicate_of}"
                print(f"Duplicate repair proposal suppressed before evaluation: {duplicate_of}")
                loop_feedback_events.append(
                    f"Iteration {iteration} repair {attempt} duplicated `{duplicate_of}` and was not evaluated."
                )
                new_logic = None
                continue
            duplicate_streak = 0

            new_loss, eval_data, error_msg = run_evaluation(
                new_logic,
                domain,
                held_out,
                target_samples,
                generations,
                initial_particles,
                run_seed,
                MODELS_DIR,
                inference_strategy=inference_strategy,
                evaluation_protocol=evaluation_protocol,
                protocol_manifest=protocol_manifest,
                timeout_seconds=(CANONICAL_CANDIDATE_TIMEOUT_SECONDS if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else 1200),
            )
            persist_diagnostic_packet(
                MODELS_DIR,
                f"iteration_{iteration:03d}_repair_{attempt}",
                eval_data,
            )
            if eval_data is not None:
                candidate_evaluation_records.append({"iteration": iteration, "stage": f"repair_{attempt}", "accepted": False})
            if warning := metric_failure_feedback(
                eval_data,
                metrics_key=("validation_metrics" if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else "test_metrics"),
                metric_label=("validation" if evaluation_protocol == CANONICAL_PROTOCOL_VERSION else "held-out"),
            ):
                loop_feedback_events.append(f"Iteration {iteration} repair {attempt}: {warning}")

        if not new_logic or new_loss == float("inf"):
            log_action(iteration, f"ERROR_FAILED_REPAIR: {error_msg[:30]}", float("inf"), best_score)
            write_proposal_waste_summary(
                PROPOSAL_WASTE_PATH,
                summarize_proposal_waste(
                    proposal_fingerprint_ledger=proposal_fingerprint_ledger,
                    duplicate_early_stop=duplicate_early_stop,
                    adaptive_yield_early_stop=adaptive_yield_early_stop,
                    early_stop_reason=early_stop_reason,
                    held_out_history=held_out_history,
                    models_dir=MODELS_DIR,
                    candidate_evaluation_records=candidate_evaluation_records,
                ),
            )
            continue

        new_score = selection_score(new_loss, eval_data, held_out, evaluation_protocol)
        incumbent_eval_for_feedback = data
        if candidate_evaluation_records:
            candidate_evaluation_records[-1].update({
                "score": new_score,
                "median_distance": new_loss,
                "evaluation_protocol": evaluation_protocol,
            })

        # Scaffold proposed module
        scaff_module = f"""import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{new_logic}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        sol = diffeqsolve(ODETerm(dynamics), Tsit5(), t0=float(time_points[0]), t1=float(time_points[-1]), dt0={config['dt0']}, y0=y0, args=params, saveat=saveat, max_steps={config['max_steps']})
        return sol.ys
    def get_parameter_metadata(self):
        return metadata
    def get_initial_conditions(self):
        return jnp.array({config['y0']})
    def get_latex(self):
        return "{family.upper()} Generated {domain}"
"""

        with open(f"{MODELS_DIR}/proposed_{iteration}.py", "w") as f:
            f.write(scaff_module)

        if thoughts:
            with open(f"{MODELS_DIR}/proposed_{iteration}_thoughts.md", "w") as f:
                f.write(thoughts)

        print(f"Proposed ABC-SMC median distance: {new_loss:.6f} vs Best: {best_loss:.6f}")
        if held_out or evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
            print(f"Proposed {label}: {new_score:.6f} vs Best: {best_score:.6f}")

        if new_score < best_score:
            print(">>> UPDATE ACCEPTED! We found a better hypothesis. <<<")
            log_action(iteration, "ACCEPTED", new_score, best_score)
            best_loss = new_loss
            best_score = new_score
            data = eval_data
            current_logic = new_logic
            if candidate_evaluation_records:
                candidate_evaluation_records[-1]["accepted"] = True
            with open(BEST_LOGIC_PATH, "w") as f:
                f.write(scaff_module)
            accepted_diag_path = persist_diagnostic_packet(
                MODELS_DIR,
                f"iteration_{iteration:03d}_accepted",
                eval_data,
            )
            if accepted_diag_path:
                with open(accepted_diag_path, "r") as src, open(BEST_DIAGNOSTICS_PATH, "w") as dst:
                    dst.write(src.read())
                
            if held_out and eval_data and eval_data.get("train_metrics"):
                held_out_history.append({
                    "iteration": iteration,
                    "train": eval_data["train_metrics"],
                    "test": eval_data["test_metrics"],
                })
                with open(held_out_metrics_path, "w") as f:
                    json.dump(held_out_history, f, indent=2)
            if evaluation_protocol == CANONICAL_PROTOCOL_VERSION and eval_data and eval_data.get("train_metrics"):
                validation_history.append({
                    "iteration": iteration,
                    "train": eval_data.get("train_metrics"),
                    "validation": eval_data.get("validation_metrics"),
                    "validation_mse": new_score,
                })
                with open(f"{MODELS_DIR}/validation_metrics.json", "w") as f:
                    json.dump(validation_history, f, indent=2)
        else:
            log_action(iteration, "REJECTED", new_score, best_score)
            if held_out:
                if feedback := held_out_regression_feedback(
                    incumbent_eval_for_feedback,
                    eval_data,
                    incumbent_score=best_score,
                    candidate_score=new_score,
                ):
                    loop_feedback_events.append(f"Iteration {iteration}: {feedback}")

        write_proposal_waste_summary(
            PROPOSAL_WASTE_PATH,
            summarize_proposal_waste(
                proposal_fingerprint_ledger=proposal_fingerprint_ledger,
                duplicate_early_stop=duplicate_early_stop,
                adaptive_yield_early_stop=adaptive_yield_early_stop,
                early_stop_reason=early_stop_reason,
                held_out_history=held_out_history,
                models_dir=MODELS_DIR,
                candidate_evaluation_records=candidate_evaluation_records,
            ),
        )

    if evaluation_protocol == CANONICAL_PROTOCOL_VERSION:
        write_canonical_freeze(
            models_dir=MODELS_DIR,
            domain=domain,
            domain_config=config,
            candidate_path=FROZEN_CANDIDATE_PATH,
            selection_path=FROZEN_SELECTION_PATH,
            search_receipt_path=SEARCH_RECEIPT_PATH,
            prompt_receipts_path=PROMPT_RECEIPTS_PATH,
            protocol_manifest=protocol_manifest,
            current_logic=current_logic,
            best_loss=best_loss,
            best_score=best_score,
            data=data,
            proposal_fingerprint_ledger=proposal_fingerprint_ledger,
            candidate_evaluation_records=candidate_evaluation_records,
            prompt_receipts=prompt_receipts,
        )
    print("Orchestration loop completed successfully.")
    write_proposal_waste_summary(
        PROPOSAL_WASTE_PATH,
        summarize_proposal_waste(
            proposal_fingerprint_ledger=proposal_fingerprint_ledger,
            duplicate_early_stop=duplicate_early_stop,
            adaptive_yield_early_stop=adaptive_yield_early_stop,
            early_stop_reason=early_stop_reason,
            held_out_history=held_out_history,
            models_dir=MODELS_DIR,
            candidate_evaluation_records=candidate_evaluation_records,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified config-driven domain orchestrator.")
    parser.add_argument("domain", help="Domain name, e.g. ecology or real_nile.")
    parser.add_argument("--family", default="qwen36", choices=["qwen36", "qwen3_coder_next", "gemma4"], help="Orchestrator family.")
    parser.add_argument("--model", help="LLM Model ID override.")
    parser.add_argument("--tag-prefix", help="Experiment tag prefix override; domain is appended automatically.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of evolutionary iterations to run.")
    parser.add_argument("--held-out", action="store_true", help="Enable 80/20 train/test split trajectory evaluation.")
    parser.add_argument(
        "--evaluation-protocol",
        default="legacy",
        choices=("legacy", CANONICAL_PROTOCOL_VERSION),
        help="Opt in to canonical train/validation/sealed-final development semantics.",
    )
    parser.add_argument(
        "--protocol-manifest",
        help="Canonical development manifest containing train and validation arrays only.",
    )
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--endpoint", help="LLM endpoint override.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum output tokens for LM Studio chat completions.")
    parser.add_argument(
        "--inference-strategy",
        default="gaussian_weighted",
        help="ABC-SMC transition strategy for candidate evaluation, e.g. gaussian_weighted or bdss.",
    )
    parser.add_argument("--target-samples", type=int, help="Override domain ABC-SMC target samples.")
    parser.add_argument("--generations", type=int, help="Override domain ABC-SMC generations.")
    parser.add_argument("--initial-particles", type=int, help="Override domain ABC-SMC initial particles.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved run configuration without evaluating.")
    args = parser.parse_args()

    return run_domain_main(
        domain=args.domain,
        family=args.family,
        model=args.model,
        epochs=args.epochs,
        held_out=args.held_out,
        seed=args.seed,
        endpoint=args.endpoint,
        tag_prefix=args.tag_prefix,
        max_tokens=args.max_tokens,
        inference_strategy=args.inference_strategy,
        target_samples_override=args.target_samples,
        generations_override=args.generations,
        initial_particles_override=args.initial_particles,
        evaluation_protocol=args.evaluation_protocol,
        protocol_manifest=args.protocol_manifest,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())

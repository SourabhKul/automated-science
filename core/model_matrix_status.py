from __future__ import annotations

from typing import Any

from core.domain_quarantine import quarantine_domains_from_records


NON_TERMINAL_MODEL_STATUSES = {"starting", "loading", "preflighting", "running"}
RANKABLE_MODEL_STATUSES = {"completed", "completed_with_domain_failures", "timeout_with_metrics"}
TERMINAL_MODEL_STATUSES = RANKABLE_MODEL_STATUSES | {
    "failed_no_metrics",
    "load_failed",
    "preflight_failed",
}

MODEL_UNAVAILABLE_STATUS_CODES = {404, 409, 422, 424, 429, 500, 503}


def domain_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = model.get("metrics") or {}
    return list(metrics.get("rows") or [])


def valid_domain_count(model: dict[str, Any]) -> int:
    metrics = model.get("metrics") or {}
    reported = metrics.get("valid_domains")
    if reported is not None:
        return int(reported)
    return sum(1 for row in domain_rows(model) if row.get("metrics_path"))


def quarantined_domains(model: dict[str, Any]) -> list[str]:
    metrics = model.get("metrics") or {}
    reported = metrics.get("quarantined_domains")
    if reported is not None:
        return quarantine_domains_from_records(reported)
    return [
        str(row["domain"])
        for row in domain_rows(model)
        if row.get("domain") and row.get("domain_status") == "quarantined"
    ]


def failed_domains(model: dict[str, Any]) -> list[str]:
    metrics = model.get("metrics") or {}
    reported = metrics.get("failed_domains")
    if reported is not None:
        return [str(domain) for domain in reported]
    quarantined = set(quarantined_domains(model))
    return [
        str(row["domain"])
        for row in domain_rows(model)
        if row.get("domain") and not row.get("metrics_path") and row.get("domain") not in quarantined
    ]


def preflight_failure_category(preflight: dict[str, Any] | None) -> str | None:
    if not preflight or preflight.get("ok") is not False:
        return None
    category = preflight.get("category")
    if category:
        return str(category)

    detail = " ".join(
        str(part)
        for part in (
            preflight.get("detail"),
            preflight.get("raw_response_preview"),
        )
        if part
    ).lower()
    status_code = preflight.get("status_code")
    output_length = preflight.get("output_length")

    if output_length in (None, 0) and not detail.strip():
        return "chat_empty"
    if "timeout" in detail:
        return "timeout"
    if (
        status_code in MODEL_UNAVAILABLE_STATUS_CODES
        or "model unavailable" in detail
        or "model is not loaded" in detail
        or "model not loaded" in detail
        or "model not found" in detail
        or "no such model" in detail
        or "does not exist" in detail
    ):
        return "model_unavailable"
    if (
        output_length == 0
        or "empty content" in detail
        or "empty response" in detail
    ):
        return "chat_empty"
    if (
        "syntax error" in detail
        or "missing dynamics" in detail
        or "missing metadata" in detail
        or "imports are not allowed" in detail
        or "only a dynamics function may be defined" in detail
        or "unable to extract" in detail
    ):
        return "invalid_code"
    return "request_error"


def classify_model_status(model: dict[str, Any], requested_domain_count: int | None = None) -> str:
    raw_status = model.get("status", "unknown")
    if raw_status in NON_TERMINAL_MODEL_STATUSES:
        return raw_status

    load = model.get("load") or {}
    if raw_status == "load_failed" or (load and load.get("ok") is False):
        return "load_failed"

    preflight = model.get("preflight") or {}
    if raw_status == "preflight_failed" or (preflight and preflight.get("ok") is False):
        return "preflight_failed"

    valid_domains = valid_domain_count(model)
    if raw_status in {"timeout", "timeout_with_metrics"}:
        return "timeout_with_metrics" if valid_domains > 0 else "failed_no_metrics"

    if valid_domains <= 0:
        return "failed_no_metrics"

    if requested_domain_count is None:
        requested_domain_count = len(domain_rows(model)) or None

    if requested_domain_count and valid_domains < requested_domain_count:
        return "completed_with_domain_failures"

    return "completed"

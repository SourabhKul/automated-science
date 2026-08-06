import json
from pathlib import Path

from core.model_matrix_status import (
    classify_model_status,
    failed_domains,
    preflight_failure_category,
    quarantined_domains,
)
from scripts.analyze_model_matrix import model_rows, ranking, status_counts


def synthetic_model(*, status: str, valid_domains: int, requested_domains: int = 3, load_ok: bool = True, preflight_ok: bool = True):
    rows = []
    for index in range(requested_domains):
        metrics_path = f"models/run/domain_{index}/held_out_metrics.json" if index < valid_domains else None
        rows.append({
            "domain": f"domain_{index}",
            "metrics_path": metrics_path,
            "final_test_mse": float(index + 1) if metrics_path else None,
            "relative_improvement_vs_seed": 0.1 if metrics_path else None,
        })
    return {
        "model_id": f"model_{status}_{valid_domains}",
        "status": status,
        "load": {"ok": load_ok},
        "preflight": {"ok": preflight_ok},
        "metrics": {
            "valid_domains": valid_domains,
            "accepted_domains": 0,
            "mean_final_test_mse": 1.0 if valid_domains else None,
            "median_final_test_mse": 1.0 if valid_domains else None,
            "rows": rows,
        },
    }


def main():
    assert classify_model_status(synthetic_model(status="completed", valid_domains=3), requested_domain_count=3) == "completed"
    assert classify_model_status(synthetic_model(status="failed", valid_domains=2), requested_domain_count=3) == "completed_with_domain_failures"
    assert classify_model_status(synthetic_model(status="timeout", valid_domains=1), requested_domain_count=3) == "timeout_with_metrics"
    assert classify_model_status(synthetic_model(status="failed", valid_domains=0), requested_domain_count=3) == "failed_no_metrics"
    assert classify_model_status(synthetic_model(status="load_failed", valid_domains=0, load_ok=False), requested_domain_count=3) == "load_failed"
    assert classify_model_status(synthetic_model(status="preflight_failed", valid_domains=0, preflight_ok=False), requested_domain_count=3) == "preflight_failed"
    assert preflight_failure_category({"ok": False, "detail": "", "output_length": 0}) == "chat_empty"
    assert preflight_failure_category({"ok": False, "detail": "missing metadata assignment", "output_length": 27}) == "invalid_code"
    assert preflight_failure_category({"ok": False, "detail": "ReadTimeout: timed out"}) == "timeout"
    assert preflight_failure_category({"ok": False, "detail": "HTTP 503 model unavailable", "status_code": 503}) == "model_unavailable"

    quarantined_model = {
        "model_id": "model_quarantined",
        "status": "completed",
        "metrics": {
            "valid_domains": 2,
            "accepted_domains": 0,
            "failed_domains": [],
            "quarantined_domains": [{"domain": "bz_chem", "reason": "seed eval unstable"}],
            "rows": [
                {"domain": "ecology", "metrics_path": "models/run/ecology/held_out_metrics.json", "final_test_mse": 1.0},
                {"domain": "cardio", "metrics_path": "models/run/cardio/held_out_metrics.json", "final_test_mse": 2.0},
                {"domain": "bz_chem", "domain_status": "quarantined", "metrics_path": None},
            ],
        },
    }
    assert classify_model_status(quarantined_model, requested_domain_count=2) == "completed"
    assert failed_domains(quarantined_model) == []
    assert quarantined_domains(quarantined_model) == ["bz_chem"]

    synthetic_matrix = {
        "domains": ["ecology", "cardio", "bz_chem"],
        "stable_domain_count": 2,
        "quarantined_domains": [{"domain": "bz_chem", "reason": "seed eval unstable"}],
        "models": [quarantined_model],
    }
    synthetic_rows = model_rows(synthetic_matrix)
    assert synthetic_rows[0]["status"] == "completed"
    assert synthetic_rows[0]["failed_domain_count"] == 0
    assert synthetic_rows[0]["quarantined_domain_count"] == 1

    canonical = json.loads(Path("artifacts/model_matrix/canon_big_20260703_104532/summary.json").read_text())
    rows = model_rows(canonical)
    statuses = {row["model_id"]: row["status"] for row in rows}
    counts = status_counts(rows)
    ranked_ids = [row["model_id"] for row in ranking(rows)]

    assert statuses["qwen/qwen3-coder-next"] == "completed_with_domain_failures"
    assert statuses["liquid/lfm2-24b-a2b"] == "completed_with_domain_failures"
    assert statuses["gemma-4-31b-it"] == "timeout_with_metrics"
    assert statuses["nvidia/nemotron-3-nano-omni"] == "preflight_failed"
    assert counts["completed_with_domain_failures"] >= 2
    assert counts["timeout_with_metrics"] >= 1
    assert counts["preflight_failed"] >= 1
    assert "qwen/qwen3-coder-next" in ranked_ids
    assert "liquid/lfm2-24b-a2b" in ranked_ids
    assert "gemma-4-31b-it" in ranked_ids
    canonical_by_model = {row["model_id"]: row for row in rows}
    assert canonical_by_model["nvidia/nemotron-3-nano-omni"]["preflight_category"] == "chat_empty"

    print("SUCCESS: model matrix statuses preserve partial-domain runs and reinterpret the canonical summary")


if __name__ == "__main__":
    main()

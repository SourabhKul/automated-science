from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.domain_quarantine import quarantine_domains_from_records
from core.model_matrix_status import (
    RANKABLE_MODEL_STATUSES,
    classify_model_status,
    failed_domains,
    preflight_failure_category,
    quarantined_domains,
)

BASELINE_RESULT_CANDIDATES = (
    Path("artifacts/baselines/baseline_results_full_20260702.json"),
    Path("artifacts/baselines/baseline_results.json"),
    Path("artifacts/baselines/baseline_smoke.json"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_summary() -> Path | None:
    candidates = [Path(p) for p in glob.glob("artifacts/model_matrix/*/summary.json")]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def load_baseline_results() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for path in BASELINE_RESULT_CANDIDATES:
        if not path.exists():
            continue
        payload = read_json(path)
        for domain, methods in payload.items():
            if isinstance(methods, dict) and domain not in results:
                results[domain] = methods
    return results


def best_baseline_test_mse(baseline_results: dict[str, Any], domain: str) -> float | None:
    methods = baseline_results.get(domain)
    if not isinstance(methods, dict):
        return None
    scores = [
        value
        for method in methods.values()
        if isinstance(method, dict)
        and (value := finite_number(method.get("test_mse"))) is not None
    ]
    return min(scores) if scores else None


def timeout_penalty(status: str) -> float:
    return 1.0 if status == "timeout_with_metrics" else 0.0


def accepted_update_effect(domain_rows: list[dict[str, Any]]) -> float | None:
    return safe_mean([
        improvement
        for row in domain_rows
        if int(row.get("accepted_updates") or 0) > 0
        and (improvement := finite_number(row.get("relative_improvement_vs_seed"))) is not None
    ])


def mean_normalized_rmse(domain_rows: list[dict[str, Any]]) -> float | None:
    scores = []
    for row in domain_rows:
        seed_test = finite_number(row.get("seed_test_mse"))
        final_test = finite_number(row.get("final_test_mse"))
        if seed_test is None or final_test is None or seed_test <= 0 or final_test < 0:
            continue
        scores.append(math.sqrt(final_test / seed_test))
    return safe_mean(scores)


def mean_baseline_relative_score(domain_rows: list[dict[str, Any]], baseline_results: dict[str, Any]) -> float | None:
    scores = []
    for row in domain_rows:
        final_test = finite_number(row.get("final_test_mse"))
        if final_test is None:
            continue
        baseline_test = best_baseline_test_mse(baseline_results, str(row.get("domain")))
        if baseline_test is None or baseline_test <= 0:
            continue
        # Bound each comparator to [-1, 1] so hard domains do not swamp the
        # aggregate signal while still preserving whether the model beat the baseline.
        scores.append((baseline_test - final_test) / max(baseline_test, final_test))
    return safe_mean(scores)


def baseline_comparator_count(domain_rows: list[dict[str, Any]], baseline_results: dict[str, Any]) -> int:
    count = 0
    for row in domain_rows:
        final_test = finite_number(row.get("final_test_mse"))
        if final_test is None:
            continue
        baseline_test = best_baseline_test_mse(baseline_results, str(row.get("domain")))
        if baseline_test is None or baseline_test <= 0:
            continue
        count += 1
    return count


def load_domain_waste_summary(row: dict[str, Any]) -> dict[str, Any]:
    metrics_path = row.get("metrics_path")
    if not metrics_path:
        return {}
    waste_path = Path(str(metrics_path)).parent / "proposal_waste_summary.json"
    if not waste_path.exists():
        return {}
    try:
        payload = read_json(waste_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def aggregate_waste_metrics(domain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "proposal_records": 0,
        "duplicate_proposals": 0,
        "repair_proposals": 0,
        "candidate_evaluations": 0,
        "repair_evaluations": 0,
        "nonfinite_candidate_metrics": 0,
        "extreme_candidate_metrics": 0,
        "duplicate_early_stop_count": 0,
    }
    domains_with_waste = 0
    for row in domain_rows:
        summary = load_domain_waste_summary(row)
        if not summary:
            continue
        domains_with_waste += 1
        for key in totals:
            if key == "duplicate_early_stop_count":
                totals[key] += 1 if summary.get("duplicate_early_stop") else 0
            else:
                totals[key] += int(summary.get(key) or 0)
    proposal_records = totals["proposal_records"]
    accepted_updates = sum(int(row.get("accepted_updates") or 0) for row in domain_rows)
    totals.update({
        "domains_with_waste_metrics": domains_with_waste,
        "duplicate_rate": (totals["duplicate_proposals"] / proposal_records) if proposal_records else 0.0,
        "repair_rate": (totals["repair_proposals"] / proposal_records) if proposal_records else 0.0,
        "accepted_update_efficiency": (accepted_updates / proposal_records) if proposal_records else None,
    })
    return totals


def composite_score(row: dict[str, Any]) -> float:
    coverage_rate = finite_number(row.get("coverage_rate")) or 0.0
    normalized_rmse = finite_number(row.get("mean_normalized_rmse"))
    mean_relative_improvement = clamp(
        finite_number(row.get("mean_relative_improvement")) or 0.0,
        -1.0,
        1.0,
    )
    accepted_effect = clamp(
        finite_number(row.get("accepted_update_effect_size")) or 0.0,
        -1.0,
        1.0,
    )
    baseline_relative_score = finite_number(row.get("baseline_relative_score"))
    baseline_term = clamp(baseline_relative_score, -1.0, 1.0) if baseline_relative_score is not None else 0.0
    normalized_rmse_score = 0.0 if normalized_rmse is None else 1.0 / (1.0 + normalized_rmse)
    failure_penalty = finite_number(row.get("domain_failure_penalty")) or 0.0
    timed_out_penalty = finite_number(row.get("timeout_penalty")) or 0.0
    return (
        (0.35 * coverage_rate)
        + (0.25 * normalized_rmse_score)
        + (0.20 * mean_relative_improvement)
        + (0.15 * accepted_effect)
        + (0.05 * baseline_term)
        - (0.15 * failure_penalty)
        - (0.15 * timed_out_penalty)
    )


def model_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    requested_domains = int(matrix.get("stable_domain_count") or len(matrix.get("domains") or []))
    baseline_results = matrix.get("_baseline_results") or {}
    for model in matrix.get("models", []):
        metrics = model.get("metrics") or {}
        domain_rows = metrics.get("rows") or []
        finite_tests = [
            value
            for row in domain_rows
            if (value := finite_number(row.get("final_test_mse"))) is not None
        ]
        improvements = [
            value
            for row in domain_rows
            if (value := finite_number(row.get("relative_improvement_vs_seed"))) is not None
        ]
        status = classify_model_status(model, requested_domain_count=requested_domains)
        failed = failed_domains(model)
        quarantined = quarantined_domains(model)
        valid_domains = int(metrics.get("valid_domains", 0) or 0)
        accepted_domains = int(metrics.get("accepted_domains", 0) or 0)
        accepted_updates = sum(int(row.get("accepted_updates") or 0) for row in domain_rows)
        coverage_rate = (valid_domains / requested_domains) if requested_domains else 0.0
        mean_norm_rmse = mean_normalized_rmse(domain_rows)
        accepted_effect = accepted_update_effect(domain_rows)
        baseline_relative = mean_baseline_relative_score(domain_rows, baseline_results)
        baseline_count = baseline_comparator_count(domain_rows, baseline_results)
        waste_metrics = aggregate_waste_metrics(domain_rows)
        row = {
            "model_id": model.get("model_id"),
            "status": status,
            "recorded_status": model.get("status", "unknown"),
            "duration_seconds": finite_number(model.get("duration_seconds")),
            "valid_domains": valid_domains,
            "accepted_domains": accepted_domains,
            "accepted_update_count": accepted_updates,
            "failed_domains": failed,
            "failed_domain_count": len(failed),
            "quarantined_domains": quarantined,
            "quarantined_domain_count": len(quarantined),
            "coverage_rate": coverage_rate,
            "mean_final_test_mse": finite_number(metrics.get("mean_final_test_mse")),
            "median_final_test_mse": finite_number(metrics.get("median_final_test_mse")),
            "best_domain_mse": min(finite_tests) if finite_tests else None,
            "worst_domain_mse": max(finite_tests) if finite_tests else None,
            "mean_relative_improvement": safe_mean(improvements),
            "mean_normalized_rmse": mean_norm_rmse,
            "accepted_update_effect_size": accepted_effect,
            "baseline_relative_score": baseline_relative,
            "baseline_comparator_count": baseline_count,
            "baseline_coverage_rate": (baseline_count / valid_domains) if valid_domains else 0.0,
            "domain_failure_penalty": (len(failed) / requested_domains) if requested_domains else 0.0,
            "timeout_penalty": timeout_penalty(status),
            "provisional": status == "timeout_with_metrics",
            "gauntlet_summary": model.get("gauntlet_summary"),
            "tag_prefix": model.get("tag_prefix"),
            "return_code": model.get("return_code"),
            "preflight": model.get("preflight"),
            "preflight_category": preflight_failure_category(model.get("preflight")),
            "waste_metrics": waste_metrics,
            "proposal_record_count": waste_metrics["proposal_records"],
            "duplicate_proposal_count": waste_metrics["duplicate_proposals"],
            "duplicate_proposal_rate": waste_metrics["duplicate_rate"],
            "repair_proposal_count": waste_metrics["repair_proposals"],
            "repair_proposal_rate": waste_metrics["repair_rate"],
            "nonfinite_candidate_metric_count": waste_metrics["nonfinite_candidate_metrics"],
            "extreme_candidate_metric_count": waste_metrics["extreme_candidate_metrics"],
            "accepted_update_efficiency": waste_metrics["accepted_update_efficiency"],
        }
        row["composite_score"] = composite_score(row)
        rows.append(row)
    return rows


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [
        row for row in rows
        if row["status"] in RANKABLE_MODEL_STATUSES
        and row["median_final_test_mse"] is not None
        and row["composite_score"] is not None
    ]
    return sorted(
        completed,
        key=lambda row: (
            -(row["composite_score"]),
            -(row["coverage_rate"]),
            row["median_final_test_mse"],
        ),
    )


def infer_learnings(rows: list[dict[str, Any]], matrix: dict[str, Any]) -> list[str]:
    learnings = []
    counts = status_counts(rows)
    matrix_quarantines = quarantine_domains_from_records(matrix.get("quarantined_domains"))
    if matrix_quarantines:
        learnings.append(
            "Known unstable domains are quarantined from default matrix runs until seed evaluation is repaired: "
            + ", ".join(matrix_quarantines)
            + "."
        )
    if counts.get("preflight_failed"):
        categories = sorted({
            row["preflight_category"]
            for row in rows
            if row["status"] == "preflight_failed" and row.get("preflight_category")
        })
        category_suffix = f" Categories observed: {', '.join(categories)}." if categories else ""
        learnings.append(
            "Some LM Studio models fail the tiny code-generation preflight; "
            "model availability and loadability need to be recorded separately "
            f"from scientific quality.{category_suffix}"
        )
    if counts.get("timeout_with_metrics"):
        learnings.append("At least one model hit the wall-clock budget; architecture comparisons need timeout-aware scoring instead of treating missing domains as ordinary failures.")
    completed = [row for row in rows if row["status"] in {"completed", "completed_with_domain_failures"}]
    if completed:
        if any((row.get("baseline_comparator_count") or 0) == 0 for row in completed):
            learnings.append("Some completed model runs still lack baseline comparators for every valid domain, so the composite score should keep baseline-relative signal as a minor term until baseline coverage is fuller.")
        accepted = [row for row in completed if int(row.get("accepted_domains") or 0) > 0]
        if len(accepted) < len(completed):
            learnings.append("Accepted structural replacement is sparse for some models; proposal novelty and diagnostic feedback remain first-class bottlenecks.")
        valid_counts = [int(row.get("valid_domains") or 0) for row in completed]
        requested_domains = int(matrix.get("stable_domain_count") or len(matrix.get("domains") or []))
        if requested_domains and min(valid_counts) < requested_domains:
            learnings.append("Some completed model runs have missing domain metrics; seed/evaluation robustness should be separated from LLM proposal quality.")
        improvements = [row["mean_relative_improvement"] for row in completed if row["mean_relative_improvement"] is not None]
        if improvements and max(improvements) <= 0:
            learnings.append("No completed model is improving over seeds on average yet; the next architecture iteration should emphasize residual diagnostics and targeted model-edit prompts.")
        duplicate_rates = [
            row["duplicate_proposal_rate"]
            for row in completed
            if int(row.get("proposal_record_count") or 0) > 0
        ]
        if duplicate_rates and max(duplicate_rates) >= 0.4:
            learnings.append(
                "Duplicate proposal churn is high in at least one completed run; enforce diversification or early-stop controls before more broad hero runs."
            )
        repair_counts = [
            int(row.get("repair_proposal_count") or 0)
            for row in completed
        ]
        if repair_counts and max(repair_counts) > 0:
            learnings.append(
                "Repair churn is measurable; model/domain pairs with repeated repairs should receive stricter proposal constraints before retrying."
            )
        bad_metric_counts = [
            int(row.get("nonfinite_candidate_metric_count") or 0)
            + int(row.get("extreme_candidate_metric_count") or 0)
            for row in completed
        ]
        if bad_metric_counts and max(bad_metric_counts) > 0:
            learnings.append(
                "Some candidates produce non-finite or extreme held-out metrics; feed those numerical failure modes back into the next proposal prompt."
            )
    if any(row["status"] in {"running", "preflighting", "starting"} for row in rows):
        learnings.append("The matrix is still in progress; rankings are provisional until the active model finishes or times out.")
    unload_issues = []
    for model in matrix.get("models", []):
        for key in ("pre_unload", "post_unload"):
            unload = model.get(key) or {}
            if unload and not unload.get("ok"):
                unload_issues.append((model.get("model_id"), key, unload.get("after")))
    if unload_issues:
        learnings.append("At least one LM Studio unload check left instances resident; the next run should pause or abort when memory eviction is incomplete.")
    return learnings


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.1f} min"
    return f"{minutes / 60:.2f} h"


def write_markdown(path: Path, matrix_path: Path, matrix: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ranked = ranking(rows)
    counts = status_counts(rows)
    learnings = infer_learnings(rows, matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Matrix Learnings",
        "",
        f"Updated: {utc_now()}",
        f"Matrix summary: `{matrix_path}`",
        f"Run id: `{matrix.get('run_id')}`",
        f"Domains: {', '.join(matrix.get('domains') or [])}",
        f"Stable domains: {', '.join(matrix.get('stable_domains') or matrix.get('domains') or [])}",
        f"Quarantined domains: {', '.join(quarantine_domains_from_records(matrix.get('quarantined_domains'))) or 'none'}",
        f"Epochs: {matrix.get('epochs')} | Held-out: {matrix.get('held_out')} | Base URL: `{matrix.get('base_url')}`",
        "",
        "## Status",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    if not counts:
        lines.append("- no model records yet")

    lines.extend(["", "## Current Ranking", ""])
    if ranked:
        lines.append("| Rank | Model | Effective status | Composite score | Coverage | Mean norm RMSE | Mean improvement | Median held-out MSE | Duplicate rate | Repair count | Failure penalty | Duration |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for index, row in enumerate(ranked, start=1):
            lines.append(
                "| "
                f"{index} | `{row['model_id']}` | "
                f"`{row['status']}{' (provisional)' if row['provisional'] else ''}` | "
                f"{row['composite_score']:.3f} | "
                f"{row['coverage_rate']:.0%} | "
                f"{row['mean_normalized_rmse']:.3f} | "
                f"{row['mean_relative_improvement']:.3f} | "
                f"{row['median_final_test_mse']:.6g} | "
                f"{row['duplicate_proposal_rate']:.0%} | "
                f"{row['repair_proposal_count']} | "
                f"{row['domain_failure_penalty'] + row['timeout_penalty']:.3f} | "
                f"{format_duration(row['duration_seconds'])} |"
            )
    else:
        lines.append("No terminal models with finite held-out metrics yet.")

    lines.extend(["", "## Model Notes", ""])
    for row in rows:
        matching = next((model for model in matrix.get("models", []) if model.get("model_id") == row["model_id"]), {})
        pre_ok = (matching.get("pre_unload") or {}).get("ok")
        post_ok = (matching.get("post_unload") or {}).get("ok")
        lines.append(
            f"- `{row['model_id']}`: {row['status']}; "
            f"valid domains {row['valid_domains']}; "
            f"failed domains {row['failed_domain_count']}; "
            f"quarantined domains {row['quarantined_domain_count']}; "
            f"accepted domains {row['accepted_domains']}; "
            f"accepted updates {row['accepted_update_count']}; "
            f"coverage {row['coverage_rate']:.0%}; "
            f"composite {row['composite_score']:.3f}; "
            f"duration {format_duration(row['duration_seconds'])}; "
            f"duplicates {row['duplicate_proposal_count']}/{row['proposal_record_count']} ({row['duplicate_proposal_rate']:.0%}); "
            f"repairs {row['repair_proposal_count']}; "
            f"bad metrics {row['nonfinite_candidate_metric_count']} non-finite, {row['extreme_candidate_metric_count']} extreme; "
            f"pre-unload {pre_ok}; post-unload {post_ok}; "
            f"artifact `{row['gauntlet_summary']}`"
        )
        if row["recorded_status"] != row["status"]:
            lines[-1] += f"; recorded as `{row['recorded_status']}`"
        if row["provisional"]:
            lines[-1] += "; provisional rank due to timeout"
        if row["mean_normalized_rmse"] is not None:
            lines[-1] += f"; mean normalized RMSE {row['mean_normalized_rmse']:.3f}"
        if row["mean_relative_improvement"] is not None:
            lines[-1] += f"; mean improvement {row['mean_relative_improvement']:.3f}"
        if row["accepted_update_effect_size"] is not None:
            lines[-1] += f"; accepted-update effect {row['accepted_update_effect_size']:.3f}"
        if row["baseline_relative_score"] is not None:
            lines[-1] += f"; baseline-relative score {row['baseline_relative_score']:.3f}"
        if row["baseline_comparator_count"]:
            lines[-1] += (
                f"; baseline comparators {row['baseline_comparator_count']}"
                f" ({row['baseline_coverage_rate']:.0%} of valid domains)"
            )
        if row["preflight_category"]:
            lines[-1] += f"; preflight category `{row['preflight_category']}`"
        preflight = row.get("preflight") or {}
        if preflight:
            finish_reason = preflight.get("finish_reason")
            output_length = preflight.get("output_length")
            if finish_reason is not None:
                lines[-1] += f"; preflight finish `{finish_reason}`"
            if output_length is not None:
                lines[-1] += f"; preflight output chars {output_length}"
        if row["failed_domains"]:
            lines[-1] += f"; failed domains {', '.join(row['failed_domains'])}"
        if row["quarantined_domains"]:
            lines[-1] += f"; quarantined domains {', '.join(row['quarantined_domains'])}"

    lines.extend(["", "## Architecture Learnings", ""])
    if learnings:
        for item in learnings:
            lines.append(f"- {item}")
    else:
        lines.append("- No architecture-level conclusions yet; wait for at least one completed model.")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a canonical LM Studio model matrix run.")
    parser.add_argument("--summary", help="Path to matrix summary JSON. Defaults to latest artifacts/model_matrix/*/summary.json.")
    parser.add_argument("--output", default="artifacts/evaluations/model_matrix_learnings.md")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary) if args.summary else latest_summary()
    if summary_path is None or not summary_path.exists():
        print("No model matrix summary found.")
        return 1

    matrix = read_json(summary_path)
    matrix["_baseline_results"] = load_baseline_results()
    rows = model_rows(matrix)
    write_markdown(Path(args.output), summary_path, matrix, rows)

    payload = {
        "summary": str(summary_path),
        "updated_at": utc_now(),
        "status_counts": status_counts(rows),
        "ranking": ranking(rows),
        "learnings": infer_learnings(rows, matrix),
    }
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote model-matrix learning report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

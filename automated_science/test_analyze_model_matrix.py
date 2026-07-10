import json
import tempfile
from pathlib import Path

from scripts.analyze_model_matrix import model_rows, ranking


def synthetic_domain_row(domain: str, final_test_mse: float | None, *, seed_test_mse: float = 10.0, accepted_updates: int = 0):
    return {
        "domain": domain,
        "metrics_path": f"models/demo/{domain}/held_out_metrics.json" if final_test_mse is not None else None,
        "accepted_updates": accepted_updates,
        "seed_test_mse": seed_test_mse if final_test_mse is not None else None,
        "final_test_mse": final_test_mse,
        "relative_improvement_vs_seed": (
            (seed_test_mse - final_test_mse) / seed_test_mse if final_test_mse is not None else None
        ),
    }


def synthetic_model(model_id: str, status: str, rows: list[dict], *, valid_domains: int, accepted_domains: int, median_mse: float, mean_mse: float):
    return {
        "model_id": model_id,
        "status": status,
        "load": {"ok": True},
        "preflight": {"ok": True},
        "metrics": {
            "valid_domains": valid_domains,
            "accepted_domains": accepted_domains,
            "median_final_test_mse": median_mse,
            "mean_final_test_mse": mean_mse,
            "rows": rows,
        },
    }


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        metric_dir = Path(tmpdir) / "stable" / "d0"
        metric_dir.mkdir(parents=True)
        metrics_path = metric_dir / "held_out_metrics.json"
        metrics_path.write_text("[]")
        (metric_dir / "proposal_waste_summary.json").write_text(json.dumps({
            "proposal_records": 10,
            "duplicate_proposals": 4,
            "repair_proposals": 2,
            "candidate_evaluations": 6,
            "repair_evaluations": 1,
            "nonfinite_candidate_metrics": 1,
            "extreme_candidate_metrics": 1,
            "duplicate_early_stop": True,
        }))
        stable_rows = [
            synthetic_domain_row("d0", 5.0, accepted_updates=1),
            synthetic_domain_row("d1", 5.0, accepted_updates=1),
            synthetic_domain_row("d2", 5.0, accepted_updates=1),
            synthetic_domain_row("d3", 5.0, accepted_updates=1),
        ]
        stable_rows[0]["metrics_path"] = str(metrics_path)
        synthetic_matrix = {
            "domains": ["d0", "d1", "d2", "d3"],
            "stable_domain_count": 4,
            "_baseline_results": {
                "d0": {"SINDy": {"test_mse": 8.0}},
                "d1": {"SINDy": {"test_mse": 8.0}},
                "d2": {"SINDy": {"test_mse": 8.0}},
                "d3": {"SINDy": {"test_mse": 8.0}},
            },
            "models": [
                synthetic_model(
                    "stable-complete",
                    "completed",
                    stable_rows,
                    valid_domains=4,
                    accepted_domains=4,
                    median_mse=5.0,
                    mean_mse=5.0,
                ),
                synthetic_model(
                    "raw-best-but-timeout",
                    "timeout",
                    [
                        synthetic_domain_row("d0", 1.0, accepted_updates=1),
                        synthetic_domain_row("d1", 1.0, accepted_updates=1),
                        synthetic_domain_row("d2", 1.0, accepted_updates=1),
                        synthetic_domain_row("d3", None),
                    ],
                    valid_domains=3,
                    accepted_domains=3,
                    median_mse=1.0,
                    mean_mse=1.0,
                ),
            ],
        }
        synthetic_rows = model_rows(synthetic_matrix)
    synthetic_by_model = {row["model_id"]: row for row in synthetic_rows}
    ranked_ids = [row["model_id"] for row in ranking(synthetic_rows)]

    assert synthetic_by_model["stable-complete"]["coverage_rate"] == 1.0
    assert synthetic_by_model["stable-complete"]["accepted_update_count"] == 4
    assert synthetic_by_model["stable-complete"]["baseline_relative_score"] == 0.375
    assert synthetic_by_model["stable-complete"]["baseline_comparator_count"] == 4
    assert synthetic_by_model["stable-complete"]["baseline_coverage_rate"] == 1.0
    assert synthetic_by_model["stable-complete"]["duplicate_proposal_count"] == 4
    assert synthetic_by_model["stable-complete"]["duplicate_proposal_rate"] == 0.4
    assert synthetic_by_model["stable-complete"]["repair_proposal_count"] == 2
    assert synthetic_by_model["stable-complete"]["nonfinite_candidate_metric_count"] == 1
    assert synthetic_by_model["stable-complete"]["extreme_candidate_metric_count"] == 1
    assert synthetic_by_model["stable-complete"]["accepted_update_efficiency"] == 0.4
    assert synthetic_by_model["raw-best-but-timeout"]["status"] == "timeout_with_metrics"
    assert synthetic_by_model["raw-best-but-timeout"]["provisional"] is True
    assert synthetic_by_model["raw-best-but-timeout"]["timeout_penalty"] == 1.0
    assert ranked_ids[0] == "stable-complete"

    canonical = json.loads(Path("artifacts/model_matrix/canon_big_20260703_104532/summary.json").read_text())
    canonical["_baseline_results"] = json.loads(
        Path("artifacts/baselines/baseline_results_full_20260702.json").read_text()
    )
    canonical_rows = model_rows(canonical)
    canonical_by_model = {row["model_id"]: row for row in canonical_rows}
    canonical_ranked_ids = [row["model_id"] for row in ranking(canonical_rows)]

    qwen = canonical_by_model["qwen/qwen3-coder-next"]
    liquid = canonical_by_model["liquid/lfm2-24b-a2b"]
    gemma = canonical_by_model["gemma-4-31b-it"]

    assert abs(qwen["coverage_rate"] - (23 / 25)) < 1e-9
    assert qwen["accepted_update_count"] == 14
    assert qwen["mean_normalized_rmse"] is not None
    assert qwen["baseline_comparator_count"] >= 20
    assert qwen["baseline_relative_score"] is not None
    assert -1.0 <= qwen["baseline_relative_score"] <= 1.0
    assert qwen["composite_score"] > liquid["composite_score"]
    assert gemma["provisional"] is True
    assert gemma["timeout_penalty"] == 1.0
    assert canonical_ranked_ids[0] == "qwen/qwen3-coder-next"

    print("SUCCESS: model-matrix analysis computes normalized metrics and penalty-aware ranking")


if __name__ == "__main__":
    main()

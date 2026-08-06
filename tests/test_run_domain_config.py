import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from scripts.run_domain import (
    build_repair_prompt,
    adaptive_yield_stop_reason,
    diagnostic_prompt_context,
    held_out_regression_feedback,
    loop_feedback_context,
    metric_failure_feedback,
    persist_diagnostic_packet,
    register_proposal_fingerprint,
    resolve_family,
    run_domain_main,
    score_label,
    selection_score,
    summarize_proposal_waste,
    write_proposal_waste_summary,
    write_proposal_fingerprint_ledger,
)


def main():
    model, endpoint, tag = resolve_family("qwen36", "ecology")
    assert model == "qwen3.6-27b-nvfp4"
    assert endpoint == "http://localhost:1234/v1/chat/completions"
    assert tag == "qwen36_27b_ecology"

    model, endpoint, tag = resolve_family("qwen3_coder_next", "ecology")
    assert model == "qwen/qwen3-coder-next"
    assert endpoint == "http://localhost:1234/v1/chat/completions"
    assert tag == "qwen3_coder_next_ecology"

    model, endpoint, tag = resolve_family(
        "qwen3_coder_next",
        "ecology",
        model="custom/model",
        endpoint="http://localhost:9999/v1/chat/completions",
    )
    assert model == "custom/model"
    assert endpoint == "http://localhost:9999/v1/chat/completions"
    assert tag == "qwen3_coder_next_ecology"

    assert selection_score(10.0, {"test_metrics": {"mse": 2.5}}, held_out=True) == 2.5
    assert selection_score(10.0, {"test_metrics": {"mse": 2.5}}, held_out=False) == 10.0
    assert selection_score(10.0, None, held_out=True) == 10.0
    assert selection_score(10.0, {"test_metrics": {"mse": float("nan")}}, held_out=True) == float("inf")
    assert score_label(True) == "Held-out test MSE"
    assert score_label(False) == "ABC-SMC median distance"
    assert diagnostic_prompt_context(None) == "No incumbent diagnostics available."

    dry_run_out = io.StringIO()
    with redirect_stdout(dry_run_out):
        code = run_domain_main(
            "ecology",
            family="qwen3_coder_next",
            dry_run=True,
            inference_strategy="bdss",
            target_samples_override=123,
            generations_override=7,
            initial_particles_override=50000,
        )
    dry_config = json.loads(dry_run_out.getvalue())
    assert code == 0
    assert dry_config["inference_strategy"] == "bdss"
    assert dry_config["target_samples"] == 123
    assert dry_config["generations"] == 7
    assert dry_config["initial_particles"] == 50000
    assert dry_config["held_out"] is False

    missing_dynamics_prompt = build_repair_prompt(
        "ecology",
        "metadata = [{'name': 'k', 'range': (0.1, 1.0)}]",
        "missing dynamics(t, y, args)",
    )
    assert "Add that exact function signature" in missing_dynamics_prompt
    assert "Do not include any import statements" in missing_dynamics_prompt

    missing_metadata_prompt = build_repair_prompt(
        "ecology",
        "def dynamics(t, y, args):\n    return jnp.array([0.0])",
        "missing metadata assignment",
    )
    assert "Add `metadata = [...]`" in missing_metadata_prompt
    assert "matching parameter entry in order" in missing_metadata_prompt

    forbidden_import_prompt = build_repair_prompt(
        "ecology",
        "import os\n\ndef dynamics(t, y, args):\n    return jnp.array([0.0])",
        "imports are not allowed in generated math code",
    )
    assert "Remove forbidden imports" in forbidden_import_prompt
    assert "jnp" in forbidden_import_prompt

    assert "non-finite held-out metrics" in metric_failure_feedback({"test_metrics": {"mse": float("nan")}})
    assert "extreme held-out MSE" in metric_failure_feedback({"test_metrics": {"mse": 2e8}})
    assert metric_failure_feedback({"test_metrics": {"mse": 2.0}}) is None
    regression_feedback = held_out_regression_feedback(
        {"train_metrics": {"mse": 10.0}},
        {"train_metrics": {"mse": 5.0}},
        incumbent_score=1.0,
        candidate_score=2.0,
    )
    assert regression_feedback is not None
    assert "improved train MSE" in regression_feedback
    assert "Duplicate proposal streak is 2" in loop_feedback_context([], 2)
    assert "No loop waste feedback yet" in loop_feedback_context([], 0)
    assert adaptive_yield_stop_reason([{"accepted": False}] * 11) is None
    stalled = adaptive_yield_stop_reason([{"accepted": False}] * 12)
    assert stalled is not None and "12 distinct evaluated candidates" in stalled
    assert adaptive_yield_stop_reason([{"accepted": True}] + [{"accepted": False}] * 11) is None

    eval_data = {
        "status": "success",
        "median_distance": 1.5,
        "min_distance": 0.75,
        "train_metrics": {"mse": 0.4, "rmse": 0.632},
        "test_metrics": {"mse": 0.5, "rmse": 0.707},
        "diagnostics": {
            "schema_version": 1,
            "fit": {"median_distance": 1.5, "min_distance": 0.75, "accepted_particle_count": 3},
            "best_parameters": {"alpha": 0.4},
            "residual_summary": {"global": {"mae": 0.1}, "segments": []},
            "warnings": [],
            "failure_modes": [],
        },
    }
    context = diagnostic_prompt_context(eval_data)
    assert '"best_parameters": {' in context
    assert '"accepted_particle_count": 3' in context

    with tempfile.TemporaryDirectory() as tmpdir:
        path = persist_diagnostic_packet(tmpdir, "seed", eval_data)
        assert path is not None
        payload = json.loads(Path(path).read_text())
        assert payload["artifact_name"] == "seed"
        assert payload["diagnostics"]["schema_version"] == 1

        ledger = []
        known = {}
        first_fp, first_duplicate = register_proposal_fingerprint(
            ledger,
            known,
            iteration=1,
            stage="candidate",
            code="def dynamics(t, y, args):\n    return jnp.array([0.0])\n\nmetadata = [{'name': 'k', 'range': (0.1, 1.0)}]",
        )
        second_fp, second_duplicate = register_proposal_fingerprint(
            ledger,
            known,
            iteration=2,
            stage="candidate",
            code='"doc"\n\nimport jax.numpy as jnp\n\ndef dynamics(t, y, args):\n    return jnp.array([0.0])\n\nmetadata = [{\'name\': \'k\', \'range\': (0.1, 1.0)}]',
        )
        assert first_fp == second_fp
        assert first_duplicate is None
        assert second_duplicate == "1:candidate"
        assert ledger[1]["is_duplicate"] is True

        ledger_path = str(Path(tmpdir) / "proposal_fingerprints.json")
        write_proposal_fingerprint_ledger(ledger_path, ledger)
        ledger_payload = json.loads(Path(ledger_path).read_text())
        assert ledger_payload["proposals"][1]["duplicate_of"] == "1:candidate"

        diagnostics = Path(tmpdir) / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        (diagnostics / "iteration_001_candidate.json").write_text(json.dumps({
            "test_metrics": {"mse": float("nan")}
        }))
        (diagnostics / "iteration_002_repair_1.json").write_text(json.dumps({
            "test_metrics": {"mse": 2e8}
        }))
        waste = summarize_proposal_waste(
            proposal_fingerprint_ledger=ledger,
            duplicate_early_stop=True,
            adaptive_yield_early_stop=True,
            early_stop_reason="duplicate churn",
            held_out_history=[{"iteration": 0}, {"iteration": 1}],
            models_dir=tmpdir,
            candidate_evaluation_records=[{"accepted": False}] * 12,
        )
        assert waste["proposal_records"] == 2
        assert waste["duplicate_proposals"] == 1
        assert waste["duplicate_rate"] == 0.5
        assert waste["duplicate_early_stop"] is True
        assert waste["accepted_updates"] == 1
        assert waste["candidate_evaluations"] == 1
        assert waste["repair_evaluations"] == 1
        assert waste["nonfinite_candidate_metrics"] == 1
        assert waste["extreme_candidate_metrics"] == 1
        assert waste["distinct_evaluated_candidates"] == 12
        assert waste["accepted_update_yield_over_last_12"] == 0.0
        assert waste["adaptive_yield_early_stop"] is True
        waste_path = str(Path(tmpdir) / "proposal_waste_summary.json")
        write_proposal_waste_summary(waste_path, waste)
        assert json.loads(Path(waste_path).read_text())["early_stop_reason"] == "duplicate churn"

    print("SUCCESS: run-domain family segregation is stable")


if __name__ == "__main__":
    main()

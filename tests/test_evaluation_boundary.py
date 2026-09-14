import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from core.evaluation_boundary import (
    CANONICAL_PROTOCOL_VERSION,
    CanonicalTrajectoryProtocol,
    TrajectoryPartition,
    build_final_evaluation_receipt,
    frozen_parameters_hash,
    load_partition_manifest,
    parameter_semantics_from_code,
    sha256_text,
    sha256_json,
    split_trajectory,
    write_protocol_manifests,
)
from scripts import run_domain


def _candidate_code(marker: float = 0.0) -> str:
    return f"""def dynamics(t, y, args):
    r, a, b, m = args
    return jnp.clip(jnp.array([r * 0.0 + {marker}, m * 0.0]), -1e6, 1e6)

metadata = [
    {{'name': 'r', 'range': (0.1, 2.0)}},
    {{'name': 'a', 'range': (0.01, 0.5)}},
    {{'name': 'b', 'range': (0.01, 0.5)}},
    {{'name': 'm', 'range': (0.1, 2.0)}}
]"""


def _fake_eval(code, domain, held_out, target_samples, generations, initial_particles, seed, models_dir, **kwargs):
    assert kwargs["evaluation_protocol"] == CANONICAL_PROTOCOL_VERSION
    assert kwargs["timeout_seconds"] == 120
    manifest = load_partition_manifest(
        kwargs["protocol_manifest"],
        required_roles=("train", "validation"),
        forbidden_roles=("final",),
    )
    # This fake evaluator is deliberately based only on the declared
    # development arrays. A final sentinel is never available here.
    marker = 1.0 if "0.25" in code else 2.0
    return (
        1.0,
        {
            "status": "success",
            "median_distance": 1.0,
            "min_distance": 0.5,
            "train_metrics": {"mse": marker + 1.0, "rmse": float(np.sqrt(marker + 1.0))},
            "validation_metrics": {"mse": marker, "rmse": float(np.sqrt(marker))},
            "test_metrics": None,
            "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
            "development_data_receipt": manifest["receipt"],
            "diagnostics": {
                "schema_version": 1,
                "fit": {"median_distance": 1.0, "min_distance": 0.5, "accepted_particle_count": 2},
                "best_parameters": {"r": 0.2, "a": 0.1, "b": 0.1, "m": 0.2},
                "residual_summary": {"global": {"rmse": float(np.sqrt(marker))}, "segments": []},
                "warnings": [],
                "failure_modes": [],
            },
        },
        None,
    )


def test_development_manifest_excludes_final_and_final_sentinel_changes_only_final_receipt():
    times = np.arange(15, dtype=float)
    observations = np.column_stack([times, times * 2.0])
    protocol_a = split_trajectory(times, observations)
    protocol_b = split_trajectory(times, observations.copy())
    protocol_b.final.observations.setflags(write=True)
    protocol_b.final.observations[:] = 999999.0
    protocol_b.final.observations.setflags(write=False)

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        dev_a, final_a = write_protocol_manifests(protocol_a, first)
        dev_b, final_b = write_protocol_manifests(protocol_b, second)
        loaded_a = load_partition_manifest(dev_a, required_roles=("train", "validation"), forbidden_roles=("final",))
        loaded_b = load_partition_manifest(dev_b, required_roles=("train", "validation"), forbidden_roles=("final",))
        sealed_a = load_partition_manifest(final_a, required_roles=("final",), forbidden_roles=("train", "validation"))
        sealed_b = load_partition_manifest(final_b, required_roles=("final",), forbidden_roles=("train", "validation"))

        assert set(loaded_a["roles"]) == {"train", "validation"}
        assert set(loaded_b["roles"]) == {"train", "validation"}
        assert loaded_a["receipt"]["roles"] == loaded_b["receipt"]["roles"]
        assert sealed_a["receipt"]["roles"]["final"]["observations_sha256"] != sealed_b["receipt"]["roles"]["final"]["observations_sha256"]


def test_canonical_run_domain_forwards_protocol_and_freezes_without_final_hash():
    original_eval = run_domain.run_evaluation
    original_llm = run_domain.call_llm
    run_domain.run_evaluation = _fake_eval
    calls = []

    def fake_llm(prompt, model_id, endpoint, max_tokens=8192, system_prompt=None, **kwargs):
        assert kwargs["timeout_seconds"] == 120
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
        calls.append(prompt)
        return _candidate_code(0.25 if len(calls) == 1 else 0.5)

    run_domain.call_llm = fake_llm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                protocol = split_trajectory(
                    np.arange(15, dtype=float),
                    np.column_stack([np.arange(15, dtype=float), np.arange(15, dtype=float)]),
                )
                development_manifest, _ = write_protocol_manifests(protocol, Path(tmp) / "inputs")
                code = run_domain.run_domain_main(
                    "ecology",
                    model="Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
                    endpoint="http://127.0.0.1:8000/v1/chat/completions",
                    tag_prefix="boundary_test",
                    epochs=1,
                    seed=7,
                    evaluation_protocol=CANONICAL_PROTOCOL_VERSION,
                    protocol_manifest=str(development_manifest),
                    target_samples_override=2,
                    generations_override=1,
                    initial_particles_override=4,
                )
                assert code == 0
                assert len(calls) == 1
                selection_path = Path("models/boundary_test_ecology/frozen_selection.json")
                search_path = Path("models/boundary_test_ecology/search_receipt.json")
                prompts_path = Path("models/boundary_test_ecology/proposal_prompt_receipts.json")
                assert selection_path.exists() and search_path.exists() and prompts_path.exists()
                selection = json.loads(selection_path.read_text())
                search = json.loads(search_path.read_text())
                prompts = json.loads(prompts_path.read_text())
                assert selection["selected_metric_role"] == "development_validation_mse"
                assert selection["final_outcomes_available_to_search"] is False
                assert selection["development_receipt_sha256"]
                assert search["final_data_hash"] is None
                assert all(item["final_outcomes_in_context"] is False for item in prompts)
                assert all(item["chat_template_kwargs"] == {"enable_thinking": False} for item in prompts)
                assert all("999999" not in item["prompt"] for item in prompts)
            finally:
                os.chdir(old_cwd)
    finally:
        run_domain.run_evaluation = original_eval
        run_domain.call_llm = original_llm


def test_canonical_repair_budget_is_global():
    original_eval = run_domain.run_evaluation
    original_llm = run_domain.call_llm
    run_domain.run_evaluation = _fake_eval
    calls = []
    invalid_reply = """```python
def dynamics(t, y, args):
    return [

metadata = [{'name': 'k', 'range': (0.1, 1.0)}]
```"""

    def fake_llm(prompt, model_id, endpoint, max_tokens=8192, system_prompt=None, **kwargs):
        assert kwargs["timeout_seconds"] == 120
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
        calls.append(prompt)
        return invalid_reply

    run_domain.call_llm = fake_llm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                protocol = split_trajectory(
                    np.arange(15, dtype=float),
                    np.column_stack([np.arange(15, dtype=float), np.arange(15, dtype=float)]),
                )
                development_manifest, _ = write_protocol_manifests(protocol, Path(tmp) / "inputs")
                code = run_domain.run_domain_main(
                    "ecology",
                    model="Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
                    endpoint="http://127.0.0.1:8000/v1/chat/completions",
                    tag_prefix="global_repair_budget",
                    epochs=2,
                    seed=7,
                    evaluation_protocol=CANONICAL_PROTOCOL_VERSION,
                    protocol_manifest=str(development_manifest),
                    target_samples_override=2,
                    generations_override=1,
                    initial_particles_override=4,
                )
                assert code == 0
                receipts = json.loads(Path("models/global_repair_budget_ecology/proposal_prompt_receipts.json").read_text())
                assert len(calls) == 3
                assert sum(item["stage"].startswith("repair_") for item in receipts) == 1
                assert [item["stage"] for item in receipts] == ["proposal", "repair_1", "proposal"]
            finally:
                os.chdir(old_cwd)
    finally:
        run_domain.run_evaluation = original_eval
        run_domain.call_llm = original_llm


def test_frozen_final_evaluator_writes_terminal_success_and_nonfinite_failure():
    try:
        from scripts.run_final_evaluation import run_frozen_final_evaluation
    except ModuleNotFoundError as exc:
        if exc.name in {"jax", "diffrax"}:
            print("SKIP: final evaluator integration requires the isolated JAX runtime")
            return
        raise

    times = np.linspace(0.0, 1.0, 5)
    observations = np.zeros((5, 2))
    protocol = split_trajectory(times, observations)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        development_manifest, final_manifest = write_protocol_manifests(protocol, root / "inputs")
        development_receipt = load_partition_manifest(
            development_manifest,
            required_roles=("train", "validation"),
            forbidden_roles=("final",),
        )["receipt"]
        candidate = root / "candidate.py"
        candidate.write_text(
            """def dynamics(t, y, args):
    k = args[0]
    return jnp.array([k * 0.0, k * 0.0])

metadata = [{'name': 'k', 'range': (0.1, 1.0)}]
"""
        )
        selection = root / "selection.json"
        selection.write_text(json.dumps({
            "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
            "domain": "ecology",
            "protocol_manifest": str(development_manifest),
            "candidate_code_path": str(candidate),
            "candidate_code_sha256": sha256_text(candidate.read_text()),
            "development_receipt_sha256": development_receipt["development_receipt_sha256"],
            "development_split_metadata": development_receipt["metadata"],
            "parameter_semantics": parameter_semantics_from_code(candidate.read_text()),
            "parameter_semantics_sha256": sha256_json(parameter_semantics_from_code(candidate.read_text())),
            "parameters": {"k": 0.5},
            "parameters_sha256": frozen_parameters_hash({"k": 0.5}),
            "model_contract": {
                "initial_conditions": [40.0, 9.0],
                "solver_dt0": 0.5,
                "solver_max_steps": 10000,
                "time_origin": 0.0,
                "time_origin_policy": "declared_manifest_time_points",
            },
            "final_evaluation_required": True,
        }))
        success_receipt = run_frozen_final_evaluation(
            selection_path=selection,
            final_manifest_path=final_manifest,
            domain="ecology",
            output_path=root / "success.json",
        )
        assert success_receipt["status"] == "success"
        assert success_receipt["final_data"]["observations_sha256"]
        assert success_receipt["prediction"]["finite"] is True
        assert (root / "success.json").exists()

        # A final manifest from a different development protocol is rejected
        # even when its split sizes and final shape look compatible.
        cross_protocol = CanonicalTrajectoryProtocol(
            train=TrajectoryPartition("train", np.array([0.0, 0.25]), np.ones((2, 2))),
            validation=TrajectoryPartition("validation", np.array([0.5, 0.75]), np.zeros((2, 2))),
            final=TrajectoryPartition("final", np.array([1.0]), np.zeros((1, 2))),
        )
        _, cross_final_manifest = write_protocol_manifests(cross_protocol, root / "cross_inputs")
        cross_receipt = run_frozen_final_evaluation(
            selection_path=selection,
            final_manifest_path=cross_final_manifest,
            domain="ecology",
            output_path=root / "cross.json",
        )
        assert cross_receipt["status"] == "failed"
        assert "bound to the frozen development receipt" in cross_receipt["failure"]

        # A one-point final partition must be integrated from frozen t0/y0.
        decay_code = root / "decay_candidate.py"
        decay_code.write_text(
            """def dynamics(t, y, args):
    k = args[0]
    return jnp.array([-k * y[0], -k * y[1]])

metadata = [{'name': 'k', 'range': (0.1, 1.0)}]
"""
        )
        decay_protocol = CanonicalTrajectoryProtocol(
            train=TrajectoryPartition("train", np.array([0.0, 0.25]), np.zeros((2, 2))),
            validation=TrajectoryPartition("validation", np.array([0.5, 0.75]), np.zeros((2, 2))),
            final=TrajectoryPartition(
                "final",
                np.array([1.0]),
                np.array([[40.0 * np.exp(-0.5), 9.0 * np.exp(-0.5)]]),
            ),
        )
        decay_development_manifest, decay_final_manifest = write_protocol_manifests(decay_protocol, root / "decay_inputs")
        decay_development_receipt = load_partition_manifest(
            decay_development_manifest,
            required_roles=("train", "validation"),
            forbidden_roles=("final",),
        )["receipt"]
        decay_selection = root / "decay_selection.json"
        decay_selection.write_text(json.dumps({
            "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
            "domain": "ecology",
            "protocol_manifest": str(decay_development_manifest),
            "candidate_code_path": str(decay_code),
            "candidate_code_sha256": sha256_text(decay_code.read_text()),
            "development_receipt_sha256": decay_development_receipt["development_receipt_sha256"],
            "development_split_metadata": decay_development_receipt["metadata"],
            "parameter_semantics": parameter_semantics_from_code(decay_code.read_text()),
            "parameter_semantics_sha256": sha256_json(parameter_semantics_from_code(decay_code.read_text())),
            "parameters": {"k": 0.5},
            "parameters_sha256": frozen_parameters_hash({"k": 0.5}),
            "model_contract": {
                "initial_conditions": [40.0, 9.0],
                "solver_dt0": 0.5,
                "solver_max_steps": 10000,
                "time_origin": 0.0,
                "time_origin_policy": "declared_manifest_time_points",
            },
            "final_evaluation_required": True,
        }))
        decay_receipt = run_frozen_final_evaluation(
            selection_path=decay_selection,
            final_manifest_path=decay_final_manifest,
            domain="ecology",
            output_path=root / "decay.json",
        )
        assert decay_receipt["status"] == "success", decay_receipt
        assert decay_receipt["metrics"]["mse"] < 0.01, decay_receipt

        bad_candidate = root / "bad_candidate.py"
        bad_candidate.write_text(
            """def dynamics(t, y, args):
    return jnp.array([jnp.nan, jnp.nan])

metadata = [{'name': 'k', 'range': (0.1, 1.0)}]
"""
        )
        bad_selection = root / "bad_selection.json"
        bad_selection.write_text(json.dumps({
            "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
            "domain": "ecology",
            "protocol_manifest": str(development_manifest),
            "candidate_code_path": str(bad_candidate),
            "candidate_code_sha256": sha256_text(bad_candidate.read_text()),
            "development_receipt_sha256": development_receipt["development_receipt_sha256"],
            "development_split_metadata": development_receipt["metadata"],
            "parameter_semantics": parameter_semantics_from_code(bad_candidate.read_text()),
            "parameter_semantics_sha256": sha256_json(parameter_semantics_from_code(bad_candidate.read_text())),
            "parameters": {"k": 0.5},
            "parameters_sha256": frozen_parameters_hash({"k": 0.5}),
            "final_evaluation_required": True,
        }))
        failed_receipt = run_frozen_final_evaluation(
            selection_path=bad_selection,
            final_manifest_path=final_manifest,
            domain="ecology",
            output_path=root / "failed.json",
        )
        assert failed_receipt["status"] == "failed"
        assert failed_receipt["failure"]
        assert (root / "failed.json").exists()

        direct_failed = build_final_evaluation_receipt(
            protocol_receipt={"protocol_version": CANONICAL_PROTOCOL_VERSION, "roles": {}},
            model_code_sha256="code",
            parameters_sha256="params",
            prediction=np.zeros((1, 2)),
            metrics={"mse": float("inf"), "rmse": float("nan")},
            status="success",
        )
        assert direct_failed["status"] == "failed"
        assert direct_failed["metrics"] == {"mse": None, "rmse": None}
        assert "non-finite final metrics" in direct_failed["failure"]

        direct_prediction_failed = build_final_evaluation_receipt(
            protocol_receipt={"protocol_version": CANONICAL_PROTOCOL_VERSION, "roles": {}},
            model_code_sha256="code",
            parameters_sha256="params",
            prediction=np.array([[np.nan, 0.0]]),
            metrics={"mse": 0.0},
            status="success",
        )
        assert direct_prediction_failed["status"] == "failed"
        assert "non-finite final prediction" in direct_prediction_failed["failure"]

        overflow_protocol = CanonicalTrajectoryProtocol(
            train=TrajectoryPartition("train", np.array([0.0, 0.25]), np.zeros((2, 2))),
            validation=TrajectoryPartition("validation", np.array([0.5, 0.75]), np.zeros((2, 2))),
            final=TrajectoryPartition("final", np.array([1.0]), np.full((1, 2), 1e308)),
        )
        overflow_dev, overflow_final = write_protocol_manifests(overflow_protocol, root / "overflow_inputs")
        overflow_dev_receipt = load_partition_manifest(
            overflow_dev,
            required_roles=("train", "validation"),
            forbidden_roles=("final",),
        )["receipt"]
        overflow_selection = root / "overflow_selection.json"
        overflow_selection.write_text(json.dumps({
            "evaluation_protocol": CANONICAL_PROTOCOL_VERSION,
            "domain": "ecology",
            "protocol_manifest": str(overflow_dev),
            "candidate_code_path": str(candidate),
            "candidate_code_sha256": sha256_text(candidate.read_text()),
            "development_receipt_sha256": overflow_dev_receipt["development_receipt_sha256"],
            "development_split_metadata": overflow_dev_receipt["metadata"],
            "parameter_semantics": parameter_semantics_from_code(candidate.read_text()),
            "parameter_semantics_sha256": sha256_json(parameter_semantics_from_code(candidate.read_text())),
            "parameters": {"k": 0.5},
            "parameters_sha256": frozen_parameters_hash({"k": 0.5}),
            "model_contract": {
                "initial_conditions": [40.0, 9.0],
                "solver_dt0": 0.5,
                "solver_max_steps": 10000,
                "time_origin": 0.0,
                "time_origin_policy": "declared_manifest_time_points",
            },
            "final_evaluation_required": True,
        }))
        overflow_receipt = run_frozen_final_evaluation(
            selection_path=overflow_selection,
            final_manifest_path=overflow_final,
            domain="ecology",
            output_path=root / "overflow.json",
        )
        assert overflow_receipt["status"] == "failed"
        assert overflow_receipt["metrics"]["mse"] is None
        assert "non-finite final metrics" in overflow_receipt["failure"]
        assert (root / "overflow.json").exists()


def test_canonical_sandbox_uses_authoritative_development_manifest():
    try:
        import jax  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP: canonical sandbox integration requires the isolated JAX runtime")
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        protocol = split_trajectory(
            np.arange(12, dtype=float) * 0.25,
            np.zeros((12, 2), dtype=float),
        )
        development_manifest, _ = write_protocol_manifests(protocol, root / "inputs")
        candidate = root / "candidate.py"
        candidate.write_text(
            """def dynamics(t, y, args):
    k = args[0]
    return jnp.array([k * 0.0, k * 0.0])

metadata = [{'name': 'k', 'range': (0.1, 1.0)}]
"""
        )
        output = root / "development.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(candidate),
                "--domain",
                "ecology",
                "--output",
                str(output),
                "--evaluation-protocol",
                CANONICAL_PROTOCOL_VERSION,
                "--protocol-manifest",
                str(development_manifest),
                "--target-samples",
                "2",
                "--generations",
                "1",
                "--initial-particles",
                "8",
                "--seed",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        payload = json.loads(output.read_text())
        assert payload["status"] == "success"
        assert payload["evaluation_protocol"] == CANONICAL_PROTOCOL_VERSION
        assert payload["test_metrics"] is None
        assert payload["validation_metrics"]["observed_count"] > 0
        assert set(payload["development_data_receipt"]["roles"]) == {"train", "validation"}


def main():
    test_development_manifest_excludes_final_and_final_sentinel_changes_only_final_receipt()
    test_canonical_run_domain_forwards_protocol_and_freezes_without_final_hash()
    test_canonical_repair_budget_is_global()
    test_canonical_sandbox_uses_authoritative_development_manifest()
    test_frozen_final_evaluator_writes_terminal_success_and_nonfinite_failure()
    print("SUCCESS: canonical train/validation/sealed-final boundary is isolated and receipted")


if __name__ == "__main__":
    main()

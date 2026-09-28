"""Synthetic fixtures for the one-use, training-only ABC6 campaign seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_training as training
from core.real_data.cascaded_tanks_abc6_cases import (
    ABC6TrainingBundle,
    ABC6TrainingCaseData,
    TRAINING_INPUT_L,
    TRAINING_INPUT_S,
)
from core.real_data.cascaded_tanks_models import TankState
from core.real_data.cascaded_tanks_pattern_search import (
    MatchedCostBudget,
    PatternSearchEvaluation,
    PatternSearchResult,
)


def _write_manifest(path: Path, *, mutate=None) -> str:
    payload = campaign._manifest_identity()
    payload["source_hashes"] = campaign._current_source_hashes()
    payload["runtime_fingerprint"] = campaign._current_runtime_fingerprint()
    payload["execution_contract"] = {
        "wall_clock_limit_seconds": 900,
        "runner_tree_rss_limit_bytes": 2 * 1024**3,
        "projected_artifact_bytes": 512 * 1024**2,
        "worker_count": 1,
        "watchdog_enforcement": "external",
        "independent_manifest_approval_required": True,
        "prospective_targets_before_receipts": False,
    }
    if mutate is not None:
        mutate(payload)
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fake_result(data, *, fit_status="complete", baseline_status="complete"):
    case = data.case
    complete = fit_status == "complete"
    abc_result = {
        "complete": complete,
        "posterior": (
            {
                "parameter_order": cases.PARAMETER_ORDER,
                "free_parameter_values": {
                    name: np.asarray([0.5], dtype=np.float64)
                    for name in cases.PARAMETER_ORDER
                },
                "weights": np.asarray([1.0], dtype=np.float64),
            }
            if complete
            else None
        ),
        "reference_evidence": {"populations": ()},
    }
    baseline_evaluation = PatternSearchEvaluation(
        evaluation_index=0,
        coordinates=(0.5,) * 6,
        objective=0.25,
        failure_category=None,
        failure_message=None,
        completed_steps=case.input_length,
    )
    baseline = PatternSearchResult(
        dimension=6,
        call_budget=1,
        callback_calls=1,
        candidate_attempts=1,
        cache_hits=0,
        evaluations=(baseline_evaluation,),
        initial_design=(),
        initial_design_complete=True,
        initial_design_blocked_position=None,
        start_evaluation_indices=(0,),
        polls=(),
        restarts=(),
        best_evaluation_index=0,
        stop_reason="all_starts_terminated",
    )
    return training.ABC6TrainingResult(
        case_index=data.case.case_index,
        case_id=data.case.case_id,
        model=data.case.fit_model.value,
        fit_length=case.input_length,
        status=(
            "complete"
            if complete and baseline_status == "complete"
            else "incomplete"
        ),
        abc_status=fit_status,
        baseline_status=baseline_status,
        calibration=training.ABC6CalibrationEvidence(
            seed=case.calibration_seed,
            prior_points=((0.5,) * 6,),
            discrepancies=(0.25,),
            failure_categories_by_draw=(None,),
            failure_category_counts=(),
            finite_count=1,
            failed_count=0,
            q25_index=0,
            q10_index=0,
            epsilon_0=0.25,
            epsilon_1=0.25,
            status="calibrated",
            unresolved_reason=None,
            simulator_calls=1,
            completed_steps=case.input_length,
        ),
        abc_result=abc_result,
        matched_cost_budget=MatchedCostBudget(
            calibration_calls=1,
            populations=(),
            population_calls=0,
            max_calls=1,
            fit_length=case.input_length,
            reserved_step_evaluations=case.input_length,
        ),
        baseline_result=baseline,
        baseline_failure_category_counts=(),
        baseline_completed_steps=case.input_length,
        simulator_call_counts=(("calibration", 1), ("abc", 0), ("baseline", 1)),
        training_input_sha256="1" * 64,
        training_output_sha256="2" * 64,
        production_controls=(("fixture", True),),
        production_controls_sha256="3" * 64,
        execution_controls=(("calibration_draws", 1),),
        production_controls_used=False,
    )


def _test_claim_path(tmp_path: Path) -> Path:
    return tmp_path / "claim-registry" / f"{campaign.RUN_ID}.claim"


def _use_private_test_identity(monkeypatch) -> None:
    protocol_id = "private-test-abc6-protocol"
    run_id = "private-test-abc6-run"
    monkeypatch.setattr(campaign, "PROTOCOL_ID", protocol_id)
    monkeypatch.setattr(campaign, "RUN_ID", run_id)
    monkeypatch.setattr(cases, "PROTOCOL_ID", protocol_id)
    monkeypatch.setattr(cases, "RUN_ID", run_id)


@pytest.fixture(autouse=True)
def _private_run_identity_for_every_test(monkeypatch) -> None:
    _use_private_test_identity(monkeypatch)


def _fake_training_bundle() -> ABC6TrainingBundle:
    data = tuple(
        ABC6TrainingCaseData(
            case=case,
            inputs=(
                TRAINING_INPUT_S
                if case.input_window == "S"
                else TRAINING_INPUT_L
            ),
            observed_outputs=(float(case.case_index),) * case.input_length,
        )
        for case in cases.CASE_ROSTER
    )
    states = (
        ("A", TankState(0.5, 0.5)),
        ("B", TankState(0.25, 0.75)),
        ("N", TankState(0.5, 0.5)),
    )
    return ABC6TrainingBundle(data, states)


def _run_fake_evidence_campaign(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "private-evidence-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = tmp_path / "private-evidence-output"
    output.mkdir()
    bundle = _fake_training_bundle()
    monkeypatch.setattr(campaign, "build_synthetic_training_bundle", lambda: bundle)
    monkeypatch.setattr(
        cases,
        "open_deferred_abc6_target_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("training evidence fixture must not open target gate")
        ),
    )
    execution = campaign._run_campaign_with_fit_callable_and_evidence_for_test(
        manifest,
        manifest_sha256,
        output,
        fit_callable=_fake_result,
        claim_registry_path=_test_claim_path(tmp_path),
    )
    return execution, output


def test_campaign_runs_exact_ordered_prefix_views_and_writes_gate_receipts(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "immutable-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = tmp_path / "run-output"
    output.mkdir()
    seen = []

    def fake_fit(data):
        seen.append(
            (
                data.case.case_index,
                data.case.case_id,
                data.case.fit_model.value,
                len(data.inputs),
                len(data.observed_outputs),
            )
        )
        if data.case.case_index == 4:
            return _fake_result(
                data, fit_status="incomplete", baseline_status="incomplete"
            )
        return _fake_result(data)

    def forbidden_target_gate(*_args, **_kwargs):
        raise AssertionError("training campaign opened the prospective-target gate")

    monkeypatch.setattr(cases, "open_deferred_abc6_target_gate", forbidden_target_gate)
    result = campaign._run_campaign_with_fit_callable_for_test(
        manifest,
        manifest_sha256,
        output,
        fit_callable=fake_fit,
        claim_registry_path=_test_claim_path(tmp_path),
    )

    assert tuple(item[0] for item in seen) == tuple(range(24))
    assert tuple(item[1] for item in seen) == tuple(
        case.case_id for case in cases.CASE_ROSTER
    )
    assert tuple(item[2] for item in seen) == tuple(
        case.fit_model.value for case in cases.CASE_ROSTER
    )
    assert tuple(item[3:] for item in seen) == tuple(
        (case.input_length, case.input_length) for case in cases.CASE_ROSTER
    )
    assert result.status == "incomplete"
    assert result.prospective_targets_generated is False
    assert result.forecasts_run is False
    assert result.manifest_sha256 == manifest_sha256
    status_files = sorted(output.glob("case-*-status.json"))
    assert len(status_files) == 48
    assert (output / campaign.CLAIM_FILENAME).is_file()
    assert result.summary_path.is_file()

    summary = json.loads(result.summary_path.read_text(encoding="ascii"))
    assert summary["training_only"] is True
    assert summary["prospective_targets_generated"] is False
    assert summary["forecasts_run"] is False
    assert summary["postfit_target_gate_opened"] is False
    assert summary["status"] == "incomplete"
    assert summary["case_count"] == 24
    assert summary["claim_sha256"] == result.claim_sha256

    second_output = tmp_path / "different-run-output"
    second_output.mkdir()
    second_calls = []
    with pytest.raises(campaign.ABC6CampaignAlreadyClaimedError):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            second_output,
            fit_callable=lambda data: second_calls.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )
    assert second_calls == []
    assert tuple(second_output.iterdir()) == ()


def test_evidence_campaign_returns_once_built_bundle_and_receipt_linked_results(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "private-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = tmp_path / "private-run-output"
    output.mkdir()
    bundle = _fake_training_bundle()
    bundle_calls = []

    def build_once():
        bundle_calls.append("build")
        return bundle

    monkeypatch.setattr(campaign, "build_synthetic_training_bundle", build_once)
    monkeypatch.setattr(
        cases,
        "open_deferred_abc6_target_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("training campaign must not open the target gate")
        ),
    )

    claim_registry = _test_claim_path(tmp_path)
    real_claim = campaign._claim_run
    claim_calls = []

    def count_claim(*args, **kwargs):
        claim_calls.append(args[2])
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(campaign, "_claim_run", count_claim)
    fit_data = []
    detailed_results = []

    def fake_fit(data):
        fit_data.append(data)
        result = _fake_result(
            data,
            fit_status="incomplete" if data.case.case_index == 4 else "complete",
            baseline_status="incomplete" if data.case.case_index == 4 else "complete",
        )
        detailed_results.append(result)
        return result

    execution = campaign._run_campaign_with_fit_callable_and_evidence_for_test(
        manifest,
        manifest_sha256,
        output,
        fit_callable=fake_fit,
        claim_registry_path=claim_registry,
    )

    assert bundle_calls == ["build"]
    assert execution._training_bundle is bundle
    assert tuple(fit_data) == tuple(bundle.data_for_case(i) for i in range(24))
    assert not hasattr(execution, "training_results")
    assert not hasattr(execution, "training_bundle")
    verified_evidence = execution.load_verified_training_evidence()
    assert verified_evidence.training_bundle is not bundle
    assert len(verified_evidence.training_results) == cases.CASE_COUNT
    assert all(
        captured is not expected
        and captured.case_index == expected.case_index
        for captured, expected in zip(
            verified_evidence.training_results, detailed_results, strict=True
        )
    )
    assert not verified_evidence.training_results[0].abc_result[
        "posterior"
    ]["free_parameter_values"]["a"].flags.writeable
    with pytest.raises(ValueError):
        verified_evidence.training_results[0].abc_result["posterior"][
            "free_parameter_values"
        ]["a"].flags.writeable = True
    assert tuple(
        (status.fit_status, status.baseline_status)
        for status in execution.campaign_result.case_statuses
    ) == tuple(
        (result.abc_status, result.baseline_status)
        for result in detailed_results
    )
    assert execution.campaign_result.case_statuses[4].fit_status == "incomplete"
    assert execution.campaign_result.status == "incomplete"

    # The private registry contains exactly one O_EXCL claim, mirrored into
    # the receipt directory before the fit loop begins.
    assert claim_calls == [claim_registry]
    assert tuple(claim_registry.parent.iterdir()) == (claim_registry,)
    assert claim_registry.read_bytes() == (
        output / campaign.CLAIM_FILENAME
    ).read_bytes()

    # All 48 immutable statuses precede the successful return and each digest
    # in the returned case table points to the exact persisted receipt bytes.
    verified = cases._verify_all_status_receipts(output)
    assert len(verified) == 48
    assert len(tuple(output.glob("case-*-status.json"))) == 48
    for status in execution.campaign_result.case_statuses:
        for component in ("fit", "baseline"):
            path = output / f"case-{status.case_index:02d}.{component}-status.json"
            expected_digest = getattr(status, f"{component}_receipt_sha256")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest
    assert execution.campaign_result.summary_path.is_file()
    summary = json.loads(execution.campaign_result.summary_path.read_text("ascii"))
    assert summary["run_id"] == "private-test-abc6-run"
    assert summary["case_count"] == 24
    assert summary["status"] == "incomplete"
    assert summary["prospective_targets_generated"] is False
    assert summary["forecasts_run"] is False
    evidence_manifest = json.loads(
        execution.evidence_manifest_path.read_text(encoding="ascii")
    )
    assert execution.evidence_manifest_path.name == campaign.EVIDENCE_MANIFEST_FILENAME
    assert len(evidence_manifest["status_receipts"]) == 48
    assert len(evidence_manifest["case_artifacts"]) == 24
    assert evidence_manifest["training_summary_sha256"] == (
        execution.campaign_result.summary_sha256
    )


def test_public_status_only_entrypoint_keeps_compact_result_contract(
    tmp_path, monkeypatch
) -> None:
    expected = campaign.ABC6CampaignResult(
        protocol_id="private-test-abc6-protocol",
        run_id="private-test-abc6-run",
        manifest_sha256="a" * 64,
        claim_sha256="b" * 64,
        status="incomplete",
        case_statuses=(),
        summary_path=tmp_path / "summary.json",
        summary_sha256="c" * 64,
    )
    observed = {}

    def fake_execute(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(campaign, "_execute_campaign", fake_execute)
    claim_path = tmp_path / "private-claim-registry" / "claim"
    monkeypatch.setattr(campaign, "_PROJECT_RUN_CLAIM_PATH", claim_path)

    result = campaign.run_abc6_training_campaign(
        tmp_path / "private-manifest.json", "d" * 64, tmp_path / "private-output"
    )

    assert result is expected
    assert type(result) is campaign.ABC6CampaignResult
    assert observed["kwargs"]["capture_training_evidence"] is False
    assert observed["kwargs"]["fit_callable"] is campaign.training.run_abc6_training_case
    assert observed["kwargs"]["require_reviewed_runtime"] is True
    assert observed["kwargs"]["claim_registry_path"] == claim_path


def test_public_evidence_entrypoint_uses_reviewed_claimed_execution_path(
    tmp_path, monkeypatch
) -> None:
    bundle = _fake_training_bundle()
    results = tuple(
        _fake_result(bundle.data_for_case(index)) for index in range(cases.CASE_COUNT)
    )
    statuses = tuple(
        campaign.ABC6CaseStatus(
            case_index=index,
            case_id=case.case_id,
            fit_status="complete",
            baseline_status="complete",
            fit_receipt_sha256="a" * 64,
            baseline_receipt_sha256="b" * 64,
        )
        for index, case in enumerate(cases.CASE_ROSTER)
    )
    expected = campaign.ABC6TrainingCampaignExecution(
        campaign_result=campaign.ABC6CampaignResult(
            protocol_id=campaign.PROTOCOL_ID,
            run_id=campaign.RUN_ID,
            manifest_sha256="c" * 64,
            claim_sha256="d" * 64,
            status="complete",
            case_statuses=statuses,
            summary_path=tmp_path / "summary.json",
            summary_sha256="e" * 64,
        ),
        evidence_manifest_path=tmp_path / campaign.EVIDENCE_MANIFEST_FILENAME,
        evidence_manifest_sha256="f" * 64,
        _training_bundle=bundle,
        _training_results=results,
    )
    observed = {}

    def fake_execute(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(campaign, "_execute_campaign", fake_execute)
    claim_path = tmp_path / "private-claim-registry" / "claim"
    monkeypatch.setattr(campaign, "_PROJECT_RUN_CLAIM_PATH", claim_path)

    execution = campaign.run_abc6_training_campaign_with_evidence(
        tmp_path / "private-manifest.json", "f" * 64, tmp_path / "private-output"
    )

    assert execution is expected
    assert execution._training_bundle is bundle
    assert all(
        captured is expected
        for captured, expected in zip(
            execution._training_results, results, strict=True
        )
    )
    assert not hasattr(execution, "training_results")
    assert observed["kwargs"]["capture_training_evidence"] is True
    assert observed["kwargs"]["fit_callable"] is campaign.training.run_abc6_training_case
    assert observed["kwargs"]["require_reviewed_runtime"] is True
    assert observed["kwargs"]["claim_registry_path"] == claim_path


@pytest.mark.parametrize("field", ("posterior_a", "posterior_weight", "baseline"))
def test_verified_evidence_rejects_mutated_in_memory_forecast_values(
    tmp_path, monkeypatch, field
) -> None:
    execution, _output = _run_fake_evidence_campaign(tmp_path, monkeypatch)
    result = execution._training_results[0]
    if field == "posterior_a":
        result.abc_result["posterior"]["free_parameter_values"]["a"][0] = 0.59
    elif field == "posterior_weight":
        result.abc_result["posterior"]["weights"][0] = 0.5
    else:
        best = result.baseline_result.best_evaluation
        object.__setattr__(best, "coordinates", (0.59,) + best.coordinates[1:])

    with pytest.raises(ValueError, match="in-memory case result differs"):
        execution.load_verified_training_evidence()


def test_verified_evidence_rejects_durable_case_artifact_tamper(
    tmp_path, monkeypatch
) -> None:
    execution, output = _run_fake_evidence_campaign(tmp_path, monkeypatch)
    evidence_path = (
        output
        / campaign.EVIDENCE_DIRECTORY_NAME
        / "case-00.training-evidence.json"
    )
    original = evidence_path.read_bytes()
    evidence_path.write_bytes(original + b" ")

    with pytest.raises(ValueError, match="not canonical|artifact digest"):
        execution.load_verified_training_evidence()


def test_evidence_publication_failure_is_terminal_and_returns_no_execution(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "private-failure-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = tmp_path / "private-failure-output"
    output.mkdir()
    monkeypatch.setattr(
        campaign, "build_synthetic_training_bundle", _fake_training_bundle
    )
    real_publish = campaign._write_exclusive_durable

    def fail_first_case_evidence(path, payload):
        if Path(path).name == "case-00.training-evidence.json":
            raise OSError("private evidence write failure")
        return real_publish(path, payload)

    monkeypatch.setattr(campaign, "_write_exclusive_durable", fail_first_case_evidence)
    with pytest.raises(campaign.ABC6CampaignExecutionError, match="publication failed"):
        campaign._run_campaign_with_fit_callable_and_evidence_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=_fake_result,
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert len(tuple(output.glob("case-*-status.json"))) == 48
    assert (output / campaign.SUMMARY_FILENAME).is_file()
    assert not (output / campaign.EVIDENCE_MANIFEST_FILENAME).exists()
    failure = json.loads((output / campaign.FAILURE_FILENAME).read_text("ascii"))
    assert failure["completed_case_count"] == cases.CASE_COUNT
    assert failure["stop_reason"] == "training_evidence_publication_failed"
    assert failure["resume_allowed"] is False


def test_non_finite_evidence_values_have_explicit_canonical_tags() -> None:
    encoded_scalar = campaign._canonical_evidence_value(float("inf"))
    encoded_array = campaign._canonical_evidence_value(
        np.asarray([1.0, np.nan, -np.inf], dtype=np.float64)
    )

    assert encoded_scalar["classification"] == "positive_infinity"
    assert encoded_array["$ndarray"]["non_finite"] == [
        {
            "flat_index": 1,
            "classification": "nan",
            "bits": np.asarray(np.nan, dtype=np.float64).tobytes().hex(),
        },
        {
            "flat_index": 2,
            "classification": "negative_infinity",
            "bits": np.asarray(-np.inf, dtype=np.float64).tobytes().hex(),
        },
    ]


def test_interruption_seals_failure_and_run_id_cannot_resume(tmp_path) -> None:
    manifest = tmp_path / "immutable-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = tmp_path / "run-output"
    output.mkdir()
    called = []

    def interrupted_fit(data):
        called.append(data.case.case_index)
        if data.case.case_index == 3:
            raise RuntimeError("bounded fixture interruption")
        return _fake_result(data)

    with pytest.raises(campaign.ABC6CampaignExecutionError, match="terminal"):
        campaign._run_campaign_with_fit_callable_and_evidence_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=interrupted_fit,
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert called == [0, 1, 2, 3]
    failure = json.loads((output / campaign.FAILURE_FILENAME).read_text("ascii"))
    assert failure["case_index"] == 3
    assert failure["completed_case_count"] == 3
    assert failure["stop_reason"] == "one_case_training_callable_raised"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["resume_allowed"] is False
    claim_sha256 = hashlib.sha256(
        (output / campaign.CLAIM_FILENAME).read_bytes()
    ).hexdigest()
    assert failure["claim_sha256"] == claim_sha256
    assert (output / "case-03.fit-status.json").is_file()
    assert (output / "case-03.baseline-status.json").is_file()
    assert len(tuple(output.glob("case-*-status.json"))) == 8

    retried_calls = []
    with pytest.raises(campaign.ABC6CampaignAlreadyClaimedError):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=lambda data: retried_calls.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )
    assert retried_calls == []


def test_manifest_hash_and_roster_checks_fail_before_claim(tmp_path) -> None:
    manifest = tmp_path / "immutable-manifest.json"
    _write_manifest(manifest)
    output = tmp_path / "run-output"
    output.mkdir()
    called = []

    with pytest.raises(campaign.ABC6CampaignPreflightError, match="SHA-256"):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            "0" * 64,
            output,
            fit_callable=lambda data: called.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )
    assert tuple(output.iterdir()) == ()
    assert called == []

    def reorder_cases(payload):
        payload["ordered_cases"][0], payload["ordered_cases"][1] = (
            payload["ordered_cases"][1],
            payload["ordered_cases"][0],
        )

    bad_manifest = tmp_path / "wrong-roster.json"
    bad_sha256 = _write_manifest(bad_manifest, mutate=reorder_cases)
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="roster mismatch"):
        campaign._run_campaign_with_fit_callable_for_test(
            bad_manifest,
            bad_sha256,
            output,
            fit_callable=lambda data: called.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )
    assert tuple(output.iterdir()) == ()
    assert called == []


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("schema_version", True),
        lambda payload: payload["ordered_cases"][1].__setitem__("case_index", True),
        lambda payload: payload["execution_contract"].__setitem__("worker_count", True),
        lambda payload: payload.__setitem__("unreviewed_extra_field", 1),
        lambda payload: payload["source_hashes"].__setitem__("unreviewed.py", "0" * 64),
        lambda payload: payload["source_hashes"].pop(
            "core/real_data/cascaded_tanks_models.py"
        ),
    ),
    ids=(
        "bool-schema",
        "bool-roster-index",
        "bool-worker-count",
        "extra-key",
        "extra-source-hash",
        "missing-source-hash",
    ),
)
def test_manifest_strict_schema_rejects_bad_types_and_source_hash_sets(
    tmp_path, mutate
) -> None:
    manifest = tmp_path / "invalid-schema.json"
    manifest_sha256 = _write_manifest(manifest, mutate=mutate)
    output = tmp_path / "run-output"
    output.mkdir()
    called = []

    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="schema/type validation",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=lambda data: called.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )
    assert tuple(output.iterdir()) == ()
    assert called == []
    assert not _test_claim_path(tmp_path).exists()


def test_atomic_case_status_publication_is_immutable_and_case_gate_compatible(
    tmp_path,
) -> None:
    directory = tmp_path / "receipts"
    directory.mkdir()
    fit_path, fit_sha = campaign._write_case_status(directory, 0, "fit", "complete")
    baseline_path, baseline_sha = campaign._write_case_status(
        directory, 0, "baseline", "incomplete"
    )

    assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == fit_sha
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_sha
    fit_bytes = fit_path.read_bytes()
    with pytest.raises(FileExistsError):
        campaign._write_case_status(directory, 0, "fit", "failed")
    assert fit_path.read_bytes() == fit_bytes

    for case_index in range(1, cases.CASE_COUNT):
        cases.write_status_receipt(directory, case_index, "fit", "complete")
        cases.write_status_receipt(directory, case_index, "baseline", "complete")
    # Reading every receipt through the target gate proves schema compatibility.
    # No target is generated by this training orchestration test.
    gate = cases.open_deferred_abc6_target_gate(directory)
    assert gate is not None

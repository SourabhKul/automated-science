"""Synthetic fixtures for the one-use, training-only ABC6 campaign seam."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
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


def _anchored_claim_test_path(tmp_path: Path) -> Path:
    return (
        tmp_path
        / campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{campaign.RUN_ID}.claim"
    )


def _receipt_root(tmp_path: Path) -> Path:
    return tmp_path.resolve() / campaign.RECEIPT_ROOT_RELATIVE


def _test_receipt_identity(tmp_path: Path) -> campaign.ABC6ReceiptRootIdentity:
    root = tmp_path.resolve()
    root_fd = campaign._open_matching_checkout_root(str(root))
    try:
        return campaign._open_receipt_root_identity(
            root_fd,
            str(root),
            campaign.RECEIPT_ROOT_RELATIVE,
            str(_receipt_root(tmp_path)),
        )
    finally:
        os.close(root_fd)


def _use_private_test_identity(monkeypatch) -> None:
    protocol_id = "private-test-abc6-protocol"
    run_id = "private-test-abc6-run"
    monkeypatch.setattr(campaign, "PROTOCOL_ID", protocol_id)
    monkeypatch.setattr(campaign, "RUN_ID", run_id)
    monkeypatch.setattr(cases, "PROTOCOL_ID", protocol_id)
    monkeypatch.setattr(cases, "RUN_ID", run_id)


@pytest.fixture(autouse=True)
def _private_run_identity_for_every_test(monkeypatch, tmp_path) -> None:
    _use_private_test_identity(monkeypatch)
    monkeypatch.setattr(
        campaign, "_active_checkout_root_path", lambda: tmp_path.resolve()
    )
    monkeypatch.setattr(campaign, "_current_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        campaign, "_REVIEWED_SOURCE_SHA256", campaign._current_source_hashes()
    )


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
    output = _receipt_root(tmp_path)
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
    output = _receipt_root(tmp_path)
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

    saved_receipts = tmp_path / "saved-first-run-receipts"
    output.rename(saved_receipts)
    second_manifest = tmp_path / "second-private-manifest.json"
    second_manifest_sha256 = _write_manifest(second_manifest)
    second_output = _receipt_root(tmp_path)
    second_calls = []
    with pytest.raises(campaign.ABC6CampaignAlreadyClaimedError):
        campaign._run_campaign_with_fit_callable_for_test(
            second_manifest,
            second_manifest_sha256,
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
    output = _receipt_root(tmp_path)
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
    assert execution.receipt_root_identity is not None
    execution.receipt_root_identity.verify()
    assert execution.receipt_root_identity.receipt_root_path == output
    assert execution.receipt_root_identity.runtime_identity == {
        "repository_root_device": output.parents[3].stat().st_dev,
        "repository_root_inode": output.parents[3].stat().st_ino,
        "receipt_root_device": output.stat().st_dev,
        "receipt_root_inode": output.stat().st_ino,
    }
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
    output = _receipt_root(tmp_path)
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
            summary_path=output / campaign.SUMMARY_FILENAME,
            summary_sha256="e" * 64,
        ),
        evidence_manifest_path=output / campaign.EVIDENCE_MANIFEST_FILENAME,
        evidence_manifest_sha256="f" * 64,
        receipt_root_identity=_test_receipt_identity(tmp_path),
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


def test_verified_execution_requires_anchored_receipt_identity(
    tmp_path, monkeypatch
) -> None:
    execution, _output = _run_fake_evidence_campaign(tmp_path, monkeypatch)
    identity = execution.receipt_root_identity

    with pytest.raises(TypeError, match="receipt_root_identity"):
        replace(execution, receipt_root_identity=None)

    object.__setattr__(execution, "receipt_root_identity", None)
    try:
        with pytest.raises(ValueError, match="no anchored receipt identity"):
            execution.load_verified_training_evidence()
    finally:
        object.__setattr__(execution, "receipt_root_identity", identity)
        execution.close()


@pytest.mark.parametrize("redirect", ("summary", "manifest"))
def test_verified_execution_rejects_copied_receipt_tree_path_redirect(
    tmp_path, monkeypatch, redirect
) -> None:
    execution, output = _run_fake_evidence_campaign(tmp_path, monkeypatch)
    copied_receipts = tmp_path / "copied-receipts"
    shutil.copytree(output, copied_receipts)
    original_summary_path = execution.campaign_result.summary_path
    original_manifest_path = execution.evidence_manifest_path
    if redirect == "summary":
        object.__setattr__(
            execution.campaign_result,
            "summary_path",
            copied_receipts / campaign.SUMMARY_FILENAME,
        )
    else:
        object.__setattr__(
            execution,
            "evidence_manifest_path",
            copied_receipts / campaign.EVIDENCE_MANIFEST_FILENAME,
        )
    try:
        with pytest.raises(ValueError, match="outside the anchored receipt root"):
            execution.load_verified_training_evidence()
    finally:
        object.__setattr__(
            execution.campaign_result, "summary_path", original_summary_path
        )
        object.__setattr__(execution, "evidence_manifest_path", original_manifest_path)
        execution.close()


def test_verified_evidence_rejects_status_receipt_symlink(
    tmp_path, monkeypatch
) -> None:
    execution, output = _run_fake_evidence_campaign(tmp_path, monkeypatch)
    status_path = output / "case-00.fit-status.json"
    saved_status_path = output / "case-00.fit-status.saved"
    status_path.rename(saved_status_path)
    status_path.symlink_to(saved_status_path)
    try:
        with pytest.raises(ValueError, match="status receipt is missing or unsafe"):
            execution.load_verified_training_evidence()
    finally:
        status_path.unlink()
        saved_status_path.rename(status_path)
        execution.close()


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


def test_execution_root_anchor_rejects_post_open_ancestor_substitution(
    tmp_path, monkeypatch
) -> None:
    execution, _output = _run_fake_evidence_campaign(tmp_path, monkeypatch)
    artifacts = tmp_path / "artifacts"
    saved_artifacts = tmp_path / "artifacts-opened-before-swap"
    artifacts.rename(saved_artifacts)
    artifacts.symlink_to(saved_artifacts, target_is_directory=True)
    try:
        with pytest.raises(
            campaign.ABC6CampaignPreflightError,
            match="symlinked path component",
        ):
            execution.receipt_root_identity.verify()
    finally:
        artifacts.unlink()
        saved_artifacts.rename(artifacts)
        execution.close()


def test_evidence_publication_failure_is_terminal_and_returns_no_execution(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "private-failure-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = _receipt_root(tmp_path)
    monkeypatch.setattr(
        campaign, "build_synthetic_training_bundle", _fake_training_bundle
    )
    real_publish = campaign._write_exclusive_durable_at

    def fail_first_case_evidence(directory_fd, filename, payload):
        if filename == "case-00.training-evidence.json":
            raise OSError("private evidence write failure")
        return real_publish(directory_fd, filename, payload)

    monkeypatch.setattr(
        campaign, "_write_exclusive_durable_at", fail_first_case_evidence
    )
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
    output = _receipt_root(tmp_path)
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
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="must be empty"):
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
    output = _receipt_root(tmp_path)
    called = []

    with pytest.raises(campaign.ABC6CampaignPreflightError, match="SHA-256"):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            "0" * 64,
            output,
            fit_callable=lambda data: called.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )
    assert not output.exists()
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
    assert not output.exists()
    assert called == []


def test_missing_receipt_io_source_pin_fails_before_claim(tmp_path) -> None:
    manifest = tmp_path / "missing-receipt-io-pin.json"
    helper_path = "core/real_data/cascaded_tanks_abc6_receipt_io.py"
    manifest_sha256 = _write_manifest(
        manifest,
        mutate=lambda payload: payload["source_hashes"].pop(helper_path),
    )
    output = _receipt_root(tmp_path)
    called = []

    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="manifest schema/type validation",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=lambda data: called.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert called == []
    assert not output.exists()
    assert not _test_claim_path(tmp_path).exists()


def test_modified_receipt_io_source_bytes_fail_before_claim(
    tmp_path, monkeypatch
) -> None:
    helper_path = "core/real_data/cascaded_tanks_abc6_receipt_io.py"
    fake_source_root = tmp_path / "fake-source-root"
    for relative_path in campaign._REQUIRED_SOURCE_PATHS:
        source = campaign._REPO_ROOT / relative_path
        target = fake_source_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    altered_helper = fake_source_root / helper_path
    altered_helper.write_bytes(altered_helper.read_bytes() + b"\\n# private mutation")
    monkeypatch.setattr(campaign, "_REPO_ROOT", fake_source_root)

    manifest = tmp_path / "modified-receipt-io-source.json"
    manifest_sha256 = _write_manifest(manifest)
    output = _receipt_root(tmp_path)
    called = []

    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="source changed",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=lambda data: called.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert called == []
    assert not output.exists()
    assert not _test_claim_path(tmp_path).exists()


def test_manifest_v2_binds_canonical_root_relative_path_and_reviewed_head(
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest-v2.json"
    _write_manifest(manifest)
    payload = json.loads(manifest.read_text("ascii"))

    assert payload["schema_version"] == 2
    assert payload["repository_root_realpath"] == str(tmp_path.resolve())
    assert payload["receipt_root_relative"] == campaign.RECEIPT_ROOT_RELATIVE
    assert payload["reviewed_git_head"] == "a" * 40
    assert campaign._strict_manifest_schema(payload)


def test_active_head_mismatch_fails_before_receipt_directory_or_claim(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "private-head-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    receipt_directory = _receipt_root(tmp_path)
    fit_calls = []
    monkeypatch.setattr(campaign, "_current_git_head", lambda: "b" * 40)

    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="differs from the manifest reviewed_git_head",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            receipt_directory,
            fit_callable=lambda data: fit_calls.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert fit_calls == []
    assert not receipt_directory.exists()
    assert not _test_claim_path(tmp_path).exists()


def test_caller_receipt_path_must_equal_manifest_root_join_before_claim(
    tmp_path,
) -> None:
    manifest = tmp_path / "private-path-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    fit_calls = []
    alternate_directory = tmp_path / "caller-selected-receipts"
    with pytest.raises(
        campaign.ABC6CampaignPreflightError, match="must equal the manifest"
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            alternate_directory,
            fit_callable=lambda data: fit_calls.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert fit_calls == []
    assert not alternate_directory.exists()
    assert not _receipt_root(tmp_path).exists()
    assert not _test_claim_path(tmp_path).exists()


def test_preexisting_copied_artifacts_symlink_fails_before_campaign_claim(
    tmp_path,
) -> None:
    manifest = tmp_path / "private-symlink-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    copied_artifacts = tmp_path / "copied-artifacts"
    copied_artifacts.mkdir()
    (tmp_path / "artifacts").symlink_to(copied_artifacts, target_is_directory=True)
    receipt_directory = _receipt_root(tmp_path)
    fit_calls = []

    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="symlinked path component",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            receipt_directory,
            fit_callable=lambda data: fit_calls.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert fit_calls == []
    assert tuple(copied_artifacts.iterdir()) == ()
    assert not _test_claim_path(tmp_path).exists()


def test_preexisting_global_claim_ancestor_symlink_fails_without_decoy_claim(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "private-claim-parent-symlink-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    decoy_evaluations = tmp_path / "copied-evaluations"
    decoy_claims = (
        decoy_evaluations
        / "cascaded_tanks_abc6_campaign_fit"
        / "claims"
    )
    decoy_claims.mkdir(parents=True)
    (artifacts / "evaluations").symlink_to(
        decoy_evaluations, target_is_directory=True
    )
    output = _receipt_root(tmp_path)
    fit_calls = []
    bundle_calls = []
    monkeypatch.setattr(
        campaign,
        "build_synthetic_training_bundle",
        lambda: bundle_calls.append("built"),
    )

    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="campaign global claim parent.*symlinked path component",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=lambda data: fit_calls.append(data.case.case_index),
            claim_registry_path=_anchored_claim_test_path(tmp_path),
        )

    assert fit_calls == []
    assert bundle_calls == []
    assert output.is_dir()
    assert tuple(output.iterdir()) == ()
    assert tuple(decoy_claims.iterdir()) == ()
    assert not (decoy_claims / f"{campaign.RUN_ID}.claim").exists()


def test_post_identity_ancestor_swap_seals_failure_in_anchored_receipt_tree(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "private-post-claim-swap-manifest.json"
    manifest_sha256 = _write_manifest(manifest)
    output = _receipt_root(tmp_path)
    decoy_artifacts = tmp_path / "copied-artifacts"
    decoy_receipt_root = (
        decoy_artifacts
        / "cascaded_tanks_abc6_synthetic"
        / "ct-abc6-20260928-v1"
        / "receipts"
    )
    decoy_claim_parent = (
        decoy_artifacts
        / "evaluations"
        / "cascaded_tanks_abc6_campaign_fit"
        / "claims"
    )
    decoy_receipt_root.mkdir(parents=True)
    decoy_claim_parent.mkdir(parents=True)
    saved_artifacts = tmp_path / "artifacts-opened-before-swap"
    artifacts = tmp_path / "artifacts"
    fit_calls = []
    monkeypatch.setattr(
        campaign, "build_synthetic_training_bundle", _fake_training_bundle
    )

    def swap_ancestor_during_fit(data):
        fit_calls.append(data.case.case_index)
        if data.case.case_index == 0:
            artifacts.rename(saved_artifacts)
            artifacts.symlink_to(decoy_artifacts, target_is_directory=True)
        return _fake_result(data)

    with pytest.raises(
        campaign.ABC6CampaignExecutionError,
        match="campaign path changed during case 0",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            output,
            fit_callable=swap_ancestor_during_fit,
            claim_registry_path=_anchored_claim_test_path(tmp_path),
        )

    saved_receipt_root = (
        saved_artifacts
        / "cascaded_tanks_abc6_synthetic"
        / "ct-abc6-20260928-v1"
        / "receipts"
    )
    saved_claim_parent = (
        saved_artifacts
        / "evaluations"
        / "cascaded_tanks_abc6_campaign_fit"
        / "claims"
    )
    assert fit_calls == [0]
    assert (saved_claim_parent / f"{campaign.RUN_ID}.claim").is_file()
    assert (saved_receipt_root / campaign.CLAIM_FILENAME).is_file()
    for component in ("fit", "baseline"):
        status = json.loads(
            (
                saved_receipt_root
                / f"case-00.{component}-status.json"
            ).read_text("ascii")
        )
        assert status["status"] == "failed"
    failure = json.loads(
        (saved_receipt_root / campaign.FAILURE_FILENAME).read_text("ascii")
    )
    assert failure["stop_reason"] == "campaign_path_identity_changed_during_case"
    assert failure["completed_case_count"] == 0
    assert failure["case_index"] == 0
    assert failure["resume_allowed"] is False
    assert tuple(decoy_receipt_root.iterdir()) == ()
    assert tuple(decoy_claim_parent.iterdir()) == ()


def test_manifest_root_symlink_alias_fails_before_claim(tmp_path) -> None:
    manifest = tmp_path / "private-root-alias-manifest.json"
    root_alias = tmp_path / "checkout-alias"
    root_alias.symlink_to(tmp_path, target_is_directory=True)

    def use_alias(payload):
        payload["repository_root_realpath"] = str(root_alias)

    manifest_sha256 = _write_manifest(manifest, mutate=use_alias)
    fit_calls = []
    with pytest.raises(
        campaign.ABC6CampaignPreflightError,
        match="manifest checkout root differs from the active source checkout root",
    ):
        campaign._run_campaign_with_fit_callable_for_test(
            manifest,
            manifest_sha256,
            _receipt_root(tmp_path),
            fit_callable=lambda data: fit_calls.append(data.case.case_index),
            claim_registry_path=_test_claim_path(tmp_path),
        )

    assert fit_calls == []
    assert not _receipt_root(tmp_path).exists()
    assert not _test_claim_path(tmp_path).exists()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("schema_version", True),
        lambda payload: payload.__setitem__("receipt_root_relative", "../receipts"),
        lambda payload: payload.__setitem__("repository_root_realpath", "relative/root"),
        lambda payload: payload.__setitem__("reviewed_git_head", "not-a-commit"),
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
        "traversal-receipt-path",
        "relative-checkout-root",
        "invalid-reviewed-head",
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
    output = _receipt_root(tmp_path)
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
    assert not output.exists()
    assert called == []
    assert not _test_claim_path(tmp_path).exists()


def test_atomic_case_status_publication_is_immutable_and_case_gate_compatible(
    tmp_path,
) -> None:
    directory = tmp_path / cases.RECEIPT_ROOT_RELATIVE
    directory.mkdir(parents=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fit_path, fit_sha = campaign._write_case_status(
            directory_fd, directory, 0, "fit", "complete"
        )
        baseline_path, baseline_sha = campaign._write_case_status(
            directory_fd, directory, 0, "baseline", "incomplete"
        )
        with pytest.raises(FileExistsError):
            campaign._write_case_status(
                directory_fd, directory, 0, "fit", "failed"
            )
    finally:
        os.close(directory_fd)

    assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == fit_sha
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_sha
    fit_bytes = fit_path.read_bytes()
    assert fit_path.read_bytes() == fit_bytes

    for case_index in range(1, cases.CASE_COUNT):
        cases.write_status_receipt(directory, case_index, "fit", "complete")
        cases.write_status_receipt(directory, case_index, "baseline", "complete")
    # Reading every receipt through the target gate proves schema compatibility.
    # No target is generated by this training orchestration test.
    root_fd = campaign._open_directory_nofollow(
        tmp_path, label="private fake checkout root"
    )
    receipt_fd = campaign._open_relative_directory_nofollow(
        root_fd,
        campaign.RECEIPT_ROOT_RELATIVE,
        label="private fake receipt root",
    )
    identity = campaign.ABC6ReceiptRootIdentity(
        repository_root_realpath=str(tmp_path),
        receipt_root_relative=campaign.RECEIPT_ROOT_RELATIVE,
        repository_root_fd=root_fd,
        receipt_root_fd=receipt_fd,
    )
    try:
        gate = cases.open_deferred_abc6_target_gate(identity)
    finally:
        identity.close()
    assert gate is not None

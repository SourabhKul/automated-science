"""Fake-only tests for read-only ABC6 deferred-score evidence replay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import numpy as np

from core.real_data import (
    cascaded_tanks_abc6_campaign_fit as campaign_fit,
    cascaded_tanks_abc6_cases as cases,
    cascaded_tanks_abc6_receipt_io as receipt_io,
    cascaded_tanks_abc6_replay as replay,
    cascaded_tanks_abc6_scoring as scoring,
)
from scripts import run_cascaded_tanks_abc6_synthetic as runner


def _load_fixture(name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(name)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign_fixtures = _load_fixture(
    "test_cascaded_tanks_abc6_campaign_fit.py", "abc6_replay_campaign_fixtures"
)
scoring_fixtures = _load_fixture(
    "test_cascaded_tanks_abc6_scoring.py", "abc6_replay_scoring_fixtures"
)
runner_fixtures = _load_fixture(
    "test_run_cascaded_tanks_abc6_synthetic.py", "abc6_replay_runner_fixtures"
)

_FAKE_PROTOCOL = "private-fake-abc6-synthetic-protocol"
_FAKE_RUN = "private-fake-abc6-runner-tests"
_APPROVAL_SHA256 = "b" * 64
_HEAD = "a" * 40
_SCORER_CLAIMS_RELATIVE = "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims"


def _private_ids(monkeypatch) -> None:
    for module in (campaign_fit, cases, scoring):
        monkeypatch.setattr(module, "PROTOCOL_ID", _FAKE_PROTOCOL)
        monkeypatch.setattr(module, "RUN_ID", _FAKE_RUN)


def _private_root_fixture(tmp_path: Path, monkeypatch):
    _private_ids(monkeypatch)
    (
        root,
        receipts,
        training,
        manifest,
        manifest_sha256,
        _unused_claim_registry,
        marker,
        source_hashes,
        test_authority,
    ) = runner_fixtures._private_runner_fixture(tmp_path, monkeypatch)
    claim_registry = (
        root
        / campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{_FAKE_RUN}.claim"
    )
    return (
        root,
        receipts,
        training,
        manifest,
        manifest_sha256,
        claim_registry,
        marker,
        source_hashes,
        test_authority,
    )


def _fake_forecast(data, result, *, simulator=None):
    index = result.case_index
    incomplete = result.abc_status != "complete"
    fixture_result = scoring_fixtures._result(index, incomplete=incomplete)
    forecast = scoring_fixtures._forecast(
        index,
        fixture_result,
        incomplete=incomplete,
        particle_failure=index == 16,
    )
    if incomplete or cases.CASE_ROSTER[index].truth_id == "N":
        return forecast
    rows, weights = scoring._validated_posterior(result, index)
    particles = tuple(
        replace(
            particle,
            parameter_values=rows[particle.particle_index],
            weight=weights[particle.particle_index],
        )
        for particle in forecast.particles
    )
    trajectories = forecast.particle_trajectories
    has_particle_failure = any(path is None for path in trajectories)
    summaries = {
        "effective_sample_size": 1.0 / sum(weight * weight for weight in weights),
    }
    if not has_particle_failure:
        matrix = np.asarray(trajectories, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        summaries.update(
            pointwise_weighted_mean=tuple(
                float(value)
                for value in np.sum(matrix * weight_array[:, None], axis=0)
            ),
            pointwise_weighted_median=tuple(
                scoring.weighted_left_inverse_quantile(matrix[:, time], weights, 0.50)
                for time in range(cases.PROSPECTIVE_LENGTH)
            ),
            pointwise_q05=tuple(
                scoring.weighted_left_inverse_quantile(matrix[:, time], weights, 0.05)
                for time in range(cases.PROSPECTIVE_LENGTH)
            ),
            pointwise_q95=tuple(
                scoring.weighted_left_inverse_quantile(matrix[:, time], weights, 0.95)
                for time in range(cases.PROSPECTIVE_LENGTH)
            ),
        )
    return replace(
        forecast,
        particles=particles,
        weights=weights,
        **summaries,
    )


def _run_fake_score(
    tmp_path: Path,
    monkeypatch,
    *,
    bad_target_hash: bool = False,
    create_terminal: bool = True,
    intervention_before_score: bool = False,
    intervention_after_score: bool = False,
    intervention_same_as_score: bool = False,
    terminal_status_override: str | None = None,
    intervention_utc: str | None = "2026-09-30T00:00:00.000000Z",
):
    (
        root,
        receipts,
        _training,
        manifest,
        manifest_sha256,
        claim_registry,
        marker,
        source_hashes,
        test_authority,
    ) = _private_root_fixture(tmp_path, monkeypatch)
    executions = []

    def fake_campaign(manifest_path, digest, receipt_path):
        execution = runner_fixtures._execute_private_training(
            manifest_path, digest, receipt_path, claim_registry
        )
        executions.append(execution)
        return execution

    monkeypatch.setattr(
        campaign_fit, "run_abc6_training_campaign_with_evidence", fake_campaign
    )
    monkeypatch.setattr(
        runner.forecast_module,
        "forecast_abc6_posterior_and_baseline",
        _fake_forecast,
    )
    monkeypatch.setattr(
        scoring, "forecast_abc6_posterior_and_baseline", _fake_forecast
    )
    materializer_calls = []

    def fake_materializer(_training, *, simulator, _scoring_handoff):
        materializer_calls.append("fake-only")
        return scoring_fixtures._fake_targets(bad_hash=bad_target_hash)

    monkeypatch.setattr(cases, "_materialize_prospective_targets", fake_materializer)
    score_callable = scoring.score_deferred_abc6_synthetic

    def fake_score(execution, frozen, artifact_path, artifact_sha256):
        return score_callable(
            execution,
            frozen,
            artifact_path,
            artifact_sha256,
            simulator=scoring_fixtures._fake_simulator,
        )

    monkeypatch.setattr(scoring, "score_deferred_abc6_synthetic", fake_score)
    consumed_error = None
    run_result = None
    try:
        run_result = runner.run_cascaded_tanks_abc6_synthetic(
            manifest, manifest_sha256, receipts, launch_authority=test_authority
        )
    except scoring.ABC6DeferredScoreConsumedError as error:
        consumed_error = error
    assert len(executions) == 1
    assert executions[0].receipt_root_identity is not None
    assert materializer_calls == ["fake-only"]
    if bad_target_hash:
        assert consumed_error is not None
        assert consumed_error.condition_consumed is True
        assert consumed_error.retry_forbidden is True
    else:
        assert consumed_error is None
        assert run_result is not None
    watchdog_claim_sha256 = None
    if create_terminal:
        watchdog_claim_sha256 = _publish_fake_watchdog_terminal(
            root,
            receipts,
            executions[0],
            source_hashes,
            terminal_status=terminal_status_override or ("failed" if bad_target_hash else (
                "capped"
                if intervention_before_score
                or intervention_after_score
                or intervention_same_as_score
                else "completed"
            )),
            intervention_before_score=intervention_before_score,
            intervention_after_score=intervention_after_score,
            intervention_same_as_score=intervention_same_as_score,
            intervention_utc=intervention_utc,
        )
    frozen = replay.ABC6FrozenReplayIdentity(
        receipt_root_identity=executions[0].receipt_root_identity,
        protocol_id=_FAKE_PROTOCOL,
        run_id=_FAKE_RUN,
        manifest_sha256=manifest_sha256,
        approval_record_sha256=_APPROVAL_SHA256,
        reviewed_git_head=_HEAD,
        watchdog_claim_sha256=(watchdog_claim_sha256 or "c" * 64),
        integrated_source_hashes=source_hashes,
    )
    return root, receipts, marker, frozen, run_result, consumed_error, materializer_calls


def _publish_fake_watchdog_terminal(
    root: Path,
    receipts: Path,
    execution,
    source_hashes,
    *,
    terminal_status: str,
    intervention_before_score: bool = False,
    intervention_after_score: bool = False,
    intervention_same_as_score: bool = False,
    intervention_utc: str | None = "2026-09-30T00:00:00.000000Z",
) -> str:
    identity = execution.receipt_root_identity
    identity.verify()
    root_anchor, receipt_anchor = cases._open_execution_directory_anchors(identity)
    claim_parent = cases._open_relative_directory_anchor(
        root_anchor,
        campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE,
        label="private replay watchdog claim parent",
        create=True,
    )
    scorer_parent = cases._open_relative_directory_anchor(
        root_anchor,
        _SCORER_CLAIMS_RELATIVE,
        label="private replay scorer claim parent",
        create=True,
    )
    try:
        root_binding = {
            "repository_root_realpath": str(root_anchor.path),
            "repository_root_device": root_anchor.device,
            "repository_root_inode": root_anchor.inode,
            "receipt_root_relative": campaign_fit.RECEIPT_ROOT_RELATIVE,
            "receipt_root_device": receipt_anchor.device,
            "receipt_root_inode": receipt_anchor.inode,
        }
        claim_parent_binding = {
            "watchdog_claim_parent_device": claim_parent.device,
            "watchdog_claim_parent_inode": claim_parent.inode,
            "scorer_claim_parent_device": scorer_parent.device,
            "scorer_claim_parent_inode": scorer_parent.inode,
        }
        watchdog_claim = {
            "protocol_id": _FAKE_PROTOCOL,
            "run_id": _FAKE_RUN,
            "manifest_sha256": execution.campaign_result.manifest_sha256,
            "approval_record_sha256": _APPROVAL_SHA256,
            "reviewed_git_head": _HEAD,
            **root_binding,
            **claim_parent_binding,
            "claim_semantics": "consumed_once_no_resume",
        }
        watchdog_claim_bytes = campaign_fit._canonical_json(watchdog_claim)
        watchdog_claim_sha256 = hashlib.sha256(watchdog_claim_bytes).hexdigest()
        watchdog_claim_filename = f"{_FAKE_RUN}.watchdog.claim"
        campaign_fit._write_exclusive_durable_at(
            claim_parent.descriptor, watchdog_claim_filename, watchdog_claim_bytes
        )

        statuses = []
        for case in cases.CASE_ROSTER:
            for component in ("fit", "baseline"):
                filename = f"case-{case.case_index:02d}.{component}-status.json"
                raw = cases._read_regular_file_at(
                    receipt_anchor.descriptor,
                    filename,
                    maximum_bytes=receipt_io.MAX_STATUS_RECEIPT_BYTES,
                    label="fake terminal status",
                )
                status, digest = receipt_io._validate_status_receipt_bytes(
                    raw,
                    filename,
                    case_index=case.case_index,
                    case_id=case.case_id,
                    component=component,
                    protocol_id=_FAKE_PROTOCOL,
                    run_id=_FAKE_RUN,
                )
                statuses.append(
                    {
                        "case_index": case.case_index,
                        "case_id": case.case_id,
                        "component": component,
                        "status": status,
                        "filename": filename,
                        "path": str(receipts / filename),
                        "sha256": digest,
                    }
                )
        summary_sha = execution.campaign_result.summary_sha256
        evidence_sha = execution.evidence_manifest_sha256
        forecast_sha = hashlib.sha256(
            (receipts / runner.FORECAST_ARTIFACT_FILENAME).read_bytes()
        ).hexdigest()
        score_receipt = json.loads(
            (receipts / scoring.SCORE_RECEIPT_FILENAME).read_text("ascii")
        )
        intervention = None
        if (
            intervention_before_score
            or intervention_after_score
            or intervention_same_as_score
        ):
            intervention = {
                "type": "budget_cap" if terminal_status == "capped" else "operator_stop",
                "stage": (
                    "wall_clock_limit_exceeded"
                    if terminal_status == "capped"
                    else "watchdog_interrupted"
                ),
                "monotonic_ns": (
                    score_receipt["score_event"]["monotonic_ns"]
                    + (
                        1
                        if intervention_after_score
                        else (-1 if intervention_before_score else 0)
                    )
                ),
                "occurred_at_utc": intervention_utc,
                "clock": "host-local-monotonic-ns",
            }
        claim_path = root / campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE / watchdog_claim_filename
        declared_vector = ["/private/fake-python3.14", "-c", "private fake child"]
        observed_path = "/private/fake-Python.app/Contents/MacOS/Python"
        observed_sha = "e" * 64
        observed_utc = "2026-09-30T00:00:00.000000Z"
        terminal = {
            "schema_version": 2,
            "protocol_id": _FAKE_PROTOCOL,
            "run_id": _FAKE_RUN,
            "manifest_sha256": execution.campaign_result.manifest_sha256,
            "approval_record_sha256": _APPROVAL_SHA256,
            "reviewed_git_head": _HEAD,
            **root_binding,
            **claim_parent_binding,
            "watchdog_claim_path": str(claim_path),
            "watchdog_claim_sha256": watchdog_claim_sha256,
            "status": terminal_status,
            "stop_reason": (
                "wall_clock_limit_exceeded"
                if terminal_status == "capped"
                else (
                    "watchdog_interrupted"
                    if intervention is not None
                    else "child_exited"
                )
            ),
            "process_return_code": (
                -9 if terminal_status == "capped"
                else (
                    -15
                    if intervention is not None
                    else (1 if terminal_status == "failed" else 0)
                )
            ),
            "intervention_capture": (
                {
                    "status": "not_attempted",
                    "attempted_type": None,
                    "attempted_stage": None,
                    "failure_kind": None,
                }
                if intervention is None
                else {
                    "status": "recorded",
                    "attempted_type": intervention["type"],
                    "attempted_stage": intervention["stage"],
                    "failure_kind": None,
                }
            ),
            "watchdog_intervention": intervention,
            "monitor_error": None,
            "child_launch": {
                "declared_launch_vector": declared_vector,
                "declared_executable_path": declared_vector[0],
                "declared_executable_sha256": "d" * 64,
                "observed_process": {
                    "observed_live_argv": declared_vector,
                    "observed_executable_path": observed_path,
                    "observed_executable_sha256": observed_sha,
                    "verified_image_role": "observed_python_app_image",
                    "observed_at_utc": observed_utc,
                },
                "image_observations": [
                    {
                        "timestamp_utc": observed_utc,
                        "path": observed_path,
                        "sha256": observed_sha,
                        "phase": "observed_python_app_image",
                    }
                ],
            },
            "kill_and_reap": {
                "process_group_id": 12345,
                "membership_verification": "verified_empty",
                "membership_enumeration_errors": [],
                "unreaped_process_pids": [],
                "child_reaped": True,
                "tracked_process_group_reaped": True,
            },
            "fit_phase_gate": {
                "status_receipt_count": 48,
                "status_receipts": statuses,
                "summary_sha256": summary_sha,
                "evidence_manifest_sha256": evidence_sha,
                "target_free_forecast_sha256": forecast_sha,
                "pre_score_artifact_chain_valid": True,
                "all_48_status_receipts_present_and_linked": True,
                "training_evidence_chain_valid": True,
                "target_free_forecast_chain_valid": True,
                "problems": [],
            },
        }
        terminal_bytes = campaign_fit._canonical_json(terminal)
        campaign_fit._write_exclusive_durable_at(
            receipt_anchor.descriptor, replay.TERMINAL_FILENAME, terminal_bytes
        )
        terminal_info = os.stat(
            replay.TERMINAL_FILENAME,
            dir_fd=receipt_anchor.descriptor,
            follow_symlinks=False,
        )
        acknowledgement = {
            "schema_version": 1,
            "kind": "abc6-watchdog-terminal-readback-acknowledgment",
            "protocol_id": _FAKE_PROTOCOL,
            "run_id": _FAKE_RUN,
            "manifest_sha256": execution.campaign_result.manifest_sha256,
            "watchdog_claim_path": str(claim_path),
            "watchdog_claim_sha256": watchdog_claim_sha256,
            "root_binding": root_binding,
            "terminal_receipt": {
                "leaf_name": replay.TERMINAL_FILENAME,
                "sha256": hashlib.sha256(terminal_bytes).hexdigest(),
                "file_identity": {
                    "device": terminal_info.st_dev,
                    "inode": terminal_info.st_ino,
                    "size": terminal_info.st_size,
                    "mtime_ns": terminal_info.st_mtime_ns,
                    "ctime_ns": terminal_info.st_ctime_ns,
                },
            },
        }
        campaign_fit._write_exclusive_durable_at(
            receipt_anchor.descriptor,
            replay.TERMINAL_ACK_FILENAME,
            campaign_fit._canonical_json(acknowledgement),
        )
        return watchdog_claim_sha256
    finally:
        scorer_parent.close()
        claim_parent.close()
        receipt_anchor.close()
        root_anchor.close()


def _rewrite_fake_terminal(receipts: Path, edit) -> dict[str, object]:
    terminal_path = receipts / replay.TERMINAL_FILENAME
    ack_path = receipts / replay.TERMINAL_ACK_FILENAME
    terminal = json.loads(terminal_path.read_text("ascii"))
    acknowledgement = json.loads(ack_path.read_text("ascii"))
    edit(terminal)
    terminal_raw = campaign_fit._canonical_json(terminal)
    terminal_path.write_bytes(terminal_raw)
    terminal_info = terminal_path.stat()
    acknowledgement["terminal_receipt"] = {
        "leaf_name": replay.TERMINAL_FILENAME,
        "sha256": hashlib.sha256(terminal_raw).hexdigest(),
        "file_identity": {
            "device": terminal_info.st_dev,
            "inode": terminal_info.st_ino,
            "size": terminal_info.st_size,
            "mtime_ns": terminal_info.st_mtime_ns,
            "ctime_ns": terminal_info.st_ctime_ns,
        },
    }
    ack_path.write_bytes(campaign_fit._canonical_json(acknowledgement))
    return terminal


def _frozen_for_execution(execution, source_hashes):
    return replay.ABC6FrozenReplayIdentity(
        receipt_root_identity=execution.receipt_root_identity,
        protocol_id=_FAKE_PROTOCOL,
        run_id=_FAKE_RUN,
        manifest_sha256=execution.campaign_result.manifest_sha256,
        approval_record_sha256=_APPROVAL_SHA256,
        reviewed_git_head=_HEAD,
        watchdog_claim_sha256="c" * 64,
        integrated_source_hashes=source_hashes,
    )


def _run_fake_pre_campaign_terminal(
    tmp_path: Path,
    monkeypatch,
    *,
    child: bool = False,
    outcome: str = "prechild_failure",
    bootstrap_grant: bool = True,
):
    (
        root,
        receipts,
        _training,
        manifest,
        manifest_sha256,
        _claim_registry,
        marker,
        source_hashes,
        _test_authority,
    ) = _private_root_fixture(tmp_path, monkeypatch)
    root_fd = campaign_fit._open_matching_checkout_root(str(root))
    try:
        identity = campaign_fit._open_receipt_root_identity(
            root_fd,
            str(root),
            campaign_fit.RECEIPT_ROOT_RELATIVE,
            str(receipts),
        )
    finally:
        os.close(root_fd)
    root_anchor, receipt_anchor = cases._open_execution_directory_anchors(identity)
    claim_parent = cases._open_relative_directory_anchor(
        root_anchor,
        campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE,
        label="fake pre-campaign watchdog claim parent",
        create=True,
    )
    scorer_parent = cases._open_relative_directory_anchor(
        root_anchor,
        _SCORER_CLAIMS_RELATIVE,
        label="fake pre-campaign scorer claim parent",
        create=True,
    )
    try:
        root_binding = {
            "repository_root_realpath": str(root_anchor.path),
            "repository_root_device": root_anchor.device,
            "repository_root_inode": root_anchor.inode,
            "receipt_root_relative": campaign_fit.RECEIPT_ROOT_RELATIVE,
            "receipt_root_device": receipt_anchor.device,
            "receipt_root_inode": receipt_anchor.inode,
        }
        claim_parent_binding = {
            "watchdog_claim_parent_device": claim_parent.device,
            "watchdog_claim_parent_inode": claim_parent.inode,
            "scorer_claim_parent_device": scorer_parent.device,
            "scorer_claim_parent_inode": scorer_parent.inode,
        }
        watchdog_claim = {
            "protocol_id": _FAKE_PROTOCOL,
            "run_id": _FAKE_RUN,
            "manifest_sha256": manifest_sha256,
            "approval_record_sha256": _APPROVAL_SHA256,
            "reviewed_git_head": _HEAD,
            **root_binding,
            **claim_parent_binding,
            "claimed_at_utc": "2026-09-30T00:00:00.000000Z",
            "claim_semantics": "consumed_once_no_resume",
        }
        claim_bytes = campaign_fit._canonical_json(watchdog_claim)
        claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
        campaign_fit._write_exclusive_durable_at(
            claim_parent.descriptor,
            f"{_FAKE_RUN}.watchdog.claim",
            claim_bytes,
        )

        declared_vector = ["/private/fake-python3.14", "-c", "private fake child"]
        observed_path = "/private/fake-Python.app/Contents/MacOS/Python"
        observed_sha = "e" * 64
        observed_utc = "2026-09-30T00:00:00.000000Z"
        child_launch = {
            "declared_launch_vector": declared_vector,
            "declared_executable_path": declared_vector[0],
            "declared_executable_sha256": "d" * 64,
            "observed_process": (
                {
                    "observed_live_argv": declared_vector,
                    "observed_executable_path": observed_path,
                    "observed_executable_sha256": observed_sha,
                    "verified_image_role": "observed_python_app_image",
                    "observed_at_utc": observed_utc,
                }
                if child
                else None
            ),
            "image_observations": (
                [
                    {
                        "timestamp_utc": observed_utc,
                        "path": observed_path,
                        "sha256": observed_sha,
                        "phase": "observed_python_app_image",
                    }
                ]
                if child
                else []
            ),
        }
        intervention = None
        if outcome in {"prechild_cap", "child_cap", "prechild_operator"}:
            is_operator = outcome == "prechild_operator"
            intervention = {
                "type": "operator_stop" if is_operator else "budget_cap",
                "stage": "watchdog_interrupted" if is_operator else "wall_clock_limit_exceeded",
                "monotonic_ns": 123456789,
                "occurred_at_utc": "2026-09-30T00:00:01.000000Z",
                "clock": "host-local-monotonic-ns",
            }
        if outcome == "prechild_unavailable":
            capture = {
                "status": "unavailable",
                "attempted_type": "watchdog_stop",
                "attempted_stage": "watchdog_monitoring_error",
                "failure_kind": "monotonic_unavailable",
            }
        elif intervention is None:
            capture = {
                "status": "not_attempted",
                "attempted_type": None,
                "attempted_stage": None,
                "failure_kind": None,
            }
        else:
            capture = {
                "status": "recorded",
                "attempted_type": intervention["type"],
                "attempted_stage": intervention["stage"],
                "failure_kind": None,
            }
        if outcome in {"prechild_cap", "child_cap"}:
            terminal_status = "capped"
            stop_reason = "wall_clock_limit_exceeded"
            return_code = -9 if child else None
        elif outcome == "prechild_operator":
            terminal_status = "failed"
            stop_reason = "watchdog_interrupted"
            return_code = None
        elif child:
            terminal_status = "failed"
            stop_reason = "runner_failed_before_campaign_claim"
            return_code = 1
        else:
            terminal_status = "failed"
            stop_reason = "watchdog_monitoring_error"
            return_code = None
        terminal = {
            "schema_version": 2,
            "protocol_id": _FAKE_PROTOCOL,
            "run_id": _FAKE_RUN,
            "manifest_sha256": manifest_sha256,
            "approval_record_sha256": _APPROVAL_SHA256,
            "reviewed_git_head": _HEAD,
            **root_binding,
            **claim_parent_binding,
            "watchdog_claim_path": str(
                root
                / campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE
                / f"{_FAKE_RUN}.watchdog.claim"
            ),
            "watchdog_claim_sha256": claim_sha256,
            "bootstrap_grant": (
                {
                    "transport": "inherited-anonymous-unix-stream-socket",
                    "environment_locator": "ABC6_GRANT_FD",
                    "descriptor_fd": 10,
                    "grant_sha256": None,
                    "digest_receipt_leaf": replay.GRANT_RECORD_FILENAME,
                }
                if bootstrap_grant and not child
                else None
            ),
            "status": terminal_status,
            "stop_reason": stop_reason,
            "monitor_error": (
                "KeyboardInterrupt: "
                if outcome == "prechild_operator"
                else (
                    "RuntimeError: fake prechild setup failure"
                    if not child and intervention is None
                    else None
                )
            ),
            "intervention_capture": capture,
            "watchdog_intervention": intervention,
            "process_return_code": return_code,
            "child_launch": child_launch,
            "kill_and_reap": (
                {
                    "process_group_id": 12345,
                    "membership_verification": "verified_empty",
                    "membership_enumeration_errors": [],
                    "unreaped_process_pids": [],
                    "child_reaped": True,
                    "tracked_process_group_reaped": True,
                }
                if child
                else None
            ),
            "fit_phase_gate": {
                "status_receipt_count": 0,
                "status_receipts": [],
                "summary_sha256": None,
                "evidence_manifest_sha256": None,
                "target_free_forecast_sha256": None,
                "pre_score_artifact_chain_valid": False,
                "all_48_status_receipts_present_and_linked": False,
                "training_evidence_chain_valid": False,
                "target_free_forecast_chain_valid": False,
                "problems": ["campaign has no claim or status receipts"],
            },
        }
        terminal_bytes = campaign_fit._canonical_json(terminal)
        campaign_fit._write_exclusive_durable_at(
            receipt_anchor.descriptor, replay.TERMINAL_FILENAME, terminal_bytes
        )
        terminal_info = os.stat(
            replay.TERMINAL_FILENAME,
            dir_fd=receipt_anchor.descriptor,
            follow_symlinks=False,
        )
        acknowledgement = {
            "schema_version": 1,
            "kind": "abc6-watchdog-terminal-readback-acknowledgment",
            "protocol_id": _FAKE_PROTOCOL,
            "run_id": _FAKE_RUN,
            "manifest_sha256": manifest_sha256,
            "watchdog_claim_path": terminal["watchdog_claim_path"],
            "watchdog_claim_sha256": claim_sha256,
            "root_binding": root_binding,
            "terminal_receipt": {
                "leaf_name": replay.TERMINAL_FILENAME,
                "sha256": hashlib.sha256(terminal_bytes).hexdigest(),
                "file_identity": {
                    "device": terminal_info.st_dev,
                    "inode": terminal_info.st_ino,
                    "size": terminal_info.st_size,
                    "mtime_ns": terminal_info.st_mtime_ns,
                    "ctime_ns": terminal_info.st_ctime_ns,
                },
            },
        }
        campaign_fit._write_exclusive_durable_at(
            receipt_anchor.descriptor,
            replay.TERMINAL_ACK_FILENAME,
            campaign_fit._canonical_json(acknowledgement),
        )
    finally:
        scorer_parent.close()
        claim_parent.close()
        receipt_anchor.close()
        root_anchor.close()
    frozen = replay.ABC6FrozenReplayIdentity(
        receipt_root_identity=identity,
        protocol_id=_FAKE_PROTOCOL,
        run_id=_FAKE_RUN,
        manifest_sha256=manifest_sha256,
        approval_record_sha256=_APPROVAL_SHA256,
        reviewed_git_head=_HEAD,
        watchdog_claim_sha256=claim_sha256,
        integrated_source_hashes=source_hashes,
    )
    return root, receipts, marker, frozen


def test_full_fake_score_chain_replays_complete_but_not_scientifically_ready(
    tmp_path, monkeypatch
):
    _root, _receipts, _marker, frozen, result, error, materializer_calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    assert error is None
    assert result is not None
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "complete", gate
    assert gate.marker_present is True
    assert gate.score_outcome == "complete"
    assert gate.score_receipt_sha256 == result.deferred_score.score_receipt_sha256
    assert gate.status_receipt_count == 48
    assert gate.all_statuses_complete is False
    assert gate.scientifically_ready is False
    assert tuple(name for name, _ in gate.verified_target_hashes) == ("A", "B", "M")
    assert materializer_calls == ["fake-only"]


@pytest.mark.parametrize("outcome", ["prechild_cap", "prechild_operator"])
def test_recorded_prechild_budget_or_operator_stop_is_incomplete(
    tmp_path, monkeypatch, outcome
):
    _root, receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, outcome=outcome
    )

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "incomplete", gate
    assert gate.marker_present is False
    assert gate.status_receipt_count == 0
    assert gate.all_statuses_complete is False
    assert gate.watchdog_intervention_capture_status == "recorded"
    assert gate.watchdog_intervention_attempted_type in {"budget_cap", "operator_stop"}
    assert sorted(path.name for path in receipts.iterdir()) == sorted(
        [replay.TERMINAL_FILENAME, replay.TERMINAL_ACK_FILENAME]
    )


def test_not_attempted_prechild_failure_is_unreplayable(tmp_path, monkeypatch):
    _root, _receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch
    )

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "prechild_failure_unstructured"
    assert gate.watchdog_intervention_capture_status == "not_attempted"


def test_unavailable_capture_without_score_is_unreplayable(tmp_path, monkeypatch):
    _root, _receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, outcome="prechild_unavailable"
    )

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "terminal_intervention_unavailable"
    assert gate.watchdog_intervention_capture_status == "unavailable"


def test_attested_nonzero_child_without_campaign_claim_is_failed(tmp_path, monkeypatch):
    _root, _receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, child=True, outcome="child_failure"
    )

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "failed", gate
    assert gate.failure_stage == "runner_failed_before_campaign_claim"
    assert gate.marker_present is False
    assert gate.terminal_status == "failed"


def test_attested_child_budget_stop_before_campaign_claim_is_incomplete(
    tmp_path, monkeypatch
):
    _root, _receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, child=True, outcome="child_cap"
    )

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "incomplete", gate
    assert gate.watchdog_intervention_attempted_type == "budget_cap"


@pytest.mark.parametrize("extra_kind", ["symlink_status", "extra_leaf", "partial_status"])
def test_pre_campaign_artifact_presence_is_unreplayable(
    tmp_path, monkeypatch, extra_kind
):
    _root, receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, outcome="prechild_cap"
    )
    status_path = receipts / "case-00.fit-status.json"
    if extra_kind == "symlink_status":
        target = tmp_path.resolve() / "prechild-status-decoy"
        target.write_bytes(b"private status decoy")
        status_path.symlink_to(target)
    elif extra_kind == "partial_status":
        status_path.write_bytes(b"partial stage receipt")
    else:
        (receipts / "unexpected.extra").write_bytes(b"private extra artifact")

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage in {"campaign_status_absence", "campaign_artifact_absence"}


def test_tampered_watchdog_claim_cannot_support_pre_campaign_result(
    tmp_path, monkeypatch
):
    root, _receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, outcome="prechild_cap"
    )
    claim_path = (
        root
        / campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{_FAKE_RUN}.watchdog.claim"
    )
    claim = json.loads(claim_path.read_text("ascii"))
    claim["reviewed_git_head"] = "f" * 40
    claim_path.write_bytes(campaign_fit._canonical_json(claim))

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "watchdog_claim"


def test_observed_child_without_verified_group_cleanup_is_unreplayable(
    tmp_path, monkeypatch
):
    _root, receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, child=True, outcome="child_failure"
    )
    _rewrite_fake_terminal(
        receipts,
        lambda terminal: terminal["kill_and_reap"].update(
            tracked_process_group_reaped=False
        ),
    )

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "terminal_child_reap"


def test_child_grant_record_symlink_is_unreplayable(tmp_path, monkeypatch):
    _root, receipts, _marker, frozen = _run_fake_pre_campaign_terminal(
        tmp_path, monkeypatch, child=True, outcome="child_failure"
    )
    target = tmp_path.resolve() / "pre-campaign-grant-decoy"
    target.write_bytes(b"private grant decoy")
    (receipts / replay.GRANT_RECORD_FILENAME).symlink_to(target)

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "child_grant"


def test_postmarker_failure_is_replayed_as_failed_and_consumed(tmp_path, monkeypatch):
    _root, receipts, marker, frozen, _result, error, calls = _run_fake_score(
        tmp_path, monkeypatch, bad_target_hash=True
    )
    assert error is not None and error.condition_consumed is True
    assert marker.is_file()
    score_receipt = json.loads(
        (receipts / scoring.SCORE_RECEIPT_FILENAME).read_text("ascii")
    )
    assert score_receipt["outcome"] == "failed"
    assert score_receipt["failure_checkpoint"]["condition_consumed"] is True
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "failed", gate
    assert gate.marker_present is True
    assert gate.score_outcome == "failed"
    assert gate.failure_stage == "target_hash_validation"
    assert gate.verified_target_hashes == ()
    assert calls == ["fake-only"]


def test_failure_event_before_watchdog_stop_remains_failed(tmp_path, monkeypatch):
    _root, receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path,
        monkeypatch,
        bad_target_hash=True,
        intervention_after_score=True,
        terminal_status_override="capped",
    )
    assert error is not None and error.condition_consumed is True
    terminal = json.loads(
        (receipts / replay.TERMINAL_FILENAME).read_text("ascii")
    )
    assert terminal["status"] == "capped"
    assert terminal["watchdog_intervention"]["type"] == "budget_cap"
    assert terminal["watchdog_intervention"]["stage"] == "wall_clock_limit_exceeded"

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "failed", gate
    assert gate.score_outcome == "failed"
    assert gate.failure_stage == "target_hash_validation"
    assert gate.watchdog_intervention_monotonic_ns > gate.score_event_monotonic_ns


def test_equal_failure_and_cap_timestamps_are_unreplayable_with_raw_events(
    tmp_path, monkeypatch
):
    _root, receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path,
        monkeypatch,
        bad_target_hash=True,
        intervention_same_as_score=True,
        terminal_status_override="capped",
    )
    assert error is not None and error.condition_consumed is True

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "ambiguous_event_order"
    assert gate.score_outcome == "failed"
    assert gate.score_event_stage == "target_hash_validation"
    assert gate.score_event_monotonic_ns == gate.watchdog_intervention_monotonic_ns
    assert gate.score_event_utc is not None
    assert gate.score_event_clock == "host-local-monotonic-ns"
    assert gate.watchdog_intervention_type == "budget_cap"
    assert gate.watchdog_intervention_stage == "wall_clock_limit_exceeded"
    assert gate.watchdog_intervention_utc is not None
    assert gate.watchdog_intervention_clock == "host-local-monotonic-ns"


@pytest.mark.parametrize("score_case", ["complete", "failed", "none"])
def test_unavailable_intervention_capture_is_unreplayable_before_score_ordering(
    tmp_path, monkeypatch, score_case
):
    _root, receipts, marker, frozen, _result, _error, _calls = _run_fake_score(
        tmp_path,
        monkeypatch,
        bad_target_hash=score_case == "failed",
        intervention_before_score=True,
        terminal_status_override="capped",
    )

    def make_unavailable(terminal):
        terminal["intervention_capture"].update(
            status="unavailable", failure_kind="monotonic_unavailable"
        )
        terminal["watchdog_intervention"] = None

    _rewrite_fake_terminal(receipts, make_unavailable)
    if score_case == "none":
        marker.unlink()
        (receipts / scoring.SCORE_RECEIPT_FILENAME).unlink()
        (receipts / scoring.TARGET_ARRAYS_ARTIFACT_FILENAME).unlink()

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "terminal_intervention_unavailable"
    assert gate.watchdog_intervention_capture_status == "unavailable"
    assert gate.watchdog_intervention_attempted_type == "budget_cap"
    assert gate.watchdog_intervention_attempted_stage == "wall_clock_limit_exceeded"
    assert gate.watchdog_intervention_failure_kind == "monotonic_unavailable"


def test_monotonic_order_replays_utc_null_and_postscore_cap_as_complete(
    tmp_path, monkeypatch
):
    _root, receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path,
        monkeypatch,
        intervention_after_score=True,
        intervention_utc=None,
    )
    assert error is None

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "complete", gate
    assert gate.watchdog_intervention_monotonic_ns > gate.score_event_monotonic_ns
    assert gate.watchdog_intervention_utc is None
    terminal = json.loads((receipts / replay.TERMINAL_FILENAME).read_text("ascii"))
    assert terminal["intervention_capture"]["status"] == "recorded"


def test_contradictory_intervention_capture_fields_are_unreplayable(
    tmp_path, monkeypatch
):
    _root, receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch, intervention_before_score=True
    )
    assert error is None

    def mismatch_cause(terminal):
        terminal["intervention_capture"]["attempted_stage"] = (
            "sampled_rss_limit_exceeded"
        )

    _rewrite_fake_terminal(receipts, mismatch_cause)
    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "terminal_intervention"


def test_terminal_v1_is_rejected_after_acknowledged_readback(tmp_path, monkeypatch):
    _root, receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    assert error is None
    _rewrite_fake_terminal(receipts, lambda terminal: terminal.update(schema_version=1))

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "terminal_receipt"


def test_intervention_before_score_is_incomplete_even_with_48_statuses(
    tmp_path, monkeypatch
):
    _root, _receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch, intervention_before_score=True
    )
    assert error is None
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "incomplete", gate
    assert gate.status_receipt_count == 48
    assert gate.score_outcome == "complete"
    assert gate.all_statuses_complete is False


def test_consumed_watchdog_terminal_without_postscore_files_is_not_premarker_absent(
    tmp_path, monkeypatch
):
    _root, receipts, marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch, intervention_before_score=True
    )
    assert error is None
    marker.unlink()
    (receipts / scoring.SCORE_RECEIPT_FILENAME).unlink()
    (receipts / scoring.TARGET_ARRAYS_ARTIFACT_FILENAME).unlink()

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "incomplete", gate
    assert gate.marker_present is False
    assert gate.terminal_status == "capped"
    assert gate.status_receipt_count == 48
    assert gate.failure_stage == "wall_clock_limit_exceeded"

    status_leaf = receipts / "case-00.fit-status.json"
    status_bytes = status_leaf.read_bytes()
    status_leaf.unlink()
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "status_receipts"
    status_leaf.write_bytes(status_bytes)


def test_capped_terminal_with_marker_but_no_score_remains_unreplayable(
    tmp_path, monkeypatch
):
    _root, receipts, marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch, intervention_before_score=True
    )
    assert error is None
    assert marker.is_file()
    (receipts / scoring.SCORE_RECEIPT_FILENAME).unlink()
    (receipts / scoring.TARGET_ARRAYS_ARTIFACT_FILENAME).unlink()

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.marker_present is True
    assert gate.score_outcome is None
    assert gate.terminal_status == "capped"
    assert gate.failure_stage == "post_marker_chain"


def test_failed_terminal_cannot_be_overridden_by_complete_score_chain(tmp_path, monkeypatch):
    _root, _receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch, terminal_status_override="failed"
    )
    assert error is None

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.score_outcome == "complete"
    assert gate.terminal_status == "failed"
    assert gate.failure_stage == "terminal_receipt"


def test_terminal_without_valid_readback_ack_is_unreplayable(tmp_path, monkeypatch):
    _root, receipts, _marker, frozen, _result, error, _calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    assert error is None
    (receipts / replay.TERMINAL_ACK_FILENAME).unlink()

    gate = replay.verify_abc6_postscore_evidence(frozen)

    assert gate.classification == "unreplayable", gate
    assert gate.failure_stage == "terminal_acknowledgment"


def test_campaign_claim_without_watchdog_is_unreplayable(tmp_path, monkeypatch):
    (
        root,
        receipts,
        _training,
        manifest,
        manifest_sha256,
        claim_registry,
        _marker,
        source_hashes,
        _test_authority,
    ) = _private_root_fixture(tmp_path, monkeypatch)
    execution = runner_fixtures._execute_private_training(
        manifest, manifest_sha256, receipts, claim_registry
    )
    frozen = _frozen_for_execution(execution, source_hashes)
    materializer_calls = []
    monkeypatch.setattr(
        cases,
        "_materialize_prospective_targets",
        lambda *args, **kwargs: materializer_calls.append((args, kwargs)),
    )
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "unreplayable", gate
    assert gate.marker_present is False
    assert gate.failure_stage == "watchdog_claim"
    assert materializer_calls == []
    assert not (receipts / scoring.SCORE_RECEIPT_FILENAME).exists()
    assert not (receipts / scoring.TARGET_ARRAYS_ARTIFACT_FILENAME).exists()

    (receipts / campaign_fit.CLAIM_FILENAME).unlink()
    claim_registry.unlink()
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "premarker_absent", gate
    assert gate.marker_present is False
    assert materializer_calls == []


def test_missing_status_or_altered_score_receipt_is_unreplayable(tmp_path, monkeypatch):
    _root, receipts, _marker, frozen, _result, _error, _calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    status_leaf = receipts / "case-00.fit-status.json"
    saved_status = status_leaf.read_bytes()
    status_leaf.unlink()
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "unreplayable"
    assert gate.failure_stage == "status_receipts"
    status_leaf.write_bytes(saved_status)

    score_leaf = receipts / scoring.SCORE_RECEIPT_FILENAME
    saved_score = score_leaf.read_bytes()
    score_leaf.write_bytes(saved_score + b" ")
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "unreplayable"
    assert gate.failure_stage == "deferred_score_receipt"
    score_leaf.write_bytes(saved_score)


def test_fifo_status_score_and_marker_fail_without_blocking(tmp_path, monkeypatch):
    _root, receipts, marker, frozen, _result, _error, _calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    for leaf, expected_stage in (
        (receipts / "case-00.fit-status.json", "status_receipts"),
        (receipts / scoring.SCORE_RECEIPT_FILENAME, "deferred_score_receipt"),
        (marker, "reveal_marker"),
    ):
        saved = leaf.read_bytes()
        leaf.unlink()
        os.mkfifo(leaf)
        gate = replay.verify_abc6_postscore_evidence(frozen)
        assert gate.classification == "unreplayable"
        assert gate.failure_stage == expected_stage
        leaf.unlink()
        leaf.write_bytes(saved)


def test_preopen_copied_artifacts_ancestor_is_rejected(tmp_path, monkeypatch):
    root, _receipts, _marker, frozen, _result, _error, _calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    artifacts = root / "artifacts"
    saved = root / "artifacts.saved"
    decoy = tmp_path.resolve() / "copied-artifacts"
    decoy.mkdir()
    artifacts.rename(saved)
    artifacts.symlink_to(decoy, target_is_directory=True)
    gate = replay.verify_abc6_postscore_evidence(frozen)
    assert gate.classification == "unreplayable"
    assert gate.failure_stage == "receipt_root_identity"
    artifacts.unlink()
    saved.rename(artifacts)


def test_postread_leaf_symlink_swaps_are_caught_across_pre_score_and_score_chain(
    tmp_path, monkeypatch
):
    root, receipts, marker, frozen, _result, _error, _calls = _run_fake_score(
        tmp_path, monkeypatch
    )
    decoy = tmp_path.resolve() / "swap-decoy"
    decoy.write_bytes(b"private replay swap sentinel")
    targets = (
        (receipts, campaign_fit.CLAIM_FILENAME),
        (receipts, "case-00.fit-status.json"),
        (receipts, campaign_fit.SUMMARY_FILENAME),
        (receipts, campaign_fit.EVIDENCE_MANIFEST_FILENAME),
        (receipts / campaign_fit.EVIDENCE_DIRECTORY_NAME, "case-00.training-evidence.json"),
        (receipts, runner.FORECAST_ARTIFACT_FILENAME),
        (receipts, scoring.TARGET_ARRAYS_ARTIFACT_FILENAME),
        (receipts, scoring.SCORE_RECEIPT_FILENAME),
        (marker.parent, marker.name),
    )
    original_read = replay._ReadSession.read
    for parent, filename in targets:
        leaf = parent / filename
        original_bytes = leaf.read_bytes()
        swapped = []
        seen_reads = []

        def swap_after_read(self, actual_parent, actual_filename, **kwargs):
            value = original_read(self, actual_parent, actual_filename, **kwargs)
            seen_reads.append((actual_parent.path, actual_filename))
            if (
                not swapped
                and actual_parent.path == parent
                and actual_filename == filename
            ):
                target = actual_parent.path / actual_filename
                target.unlink()
                target.symlink_to(decoy)
                swapped.append(True)
            return value

        with monkeypatch.context() as local:
            local.setattr(replay._ReadSession, "read", swap_after_read)
            gate = replay.verify_abc6_postscore_evidence(frozen)
        assert swapped == [True], (
            f"target={parent / filename}; observed="
            f"gate={gate.failure_stage}:{gate.detail}; observed="
            f"{[(str(path), name) for path, name in seen_reads]}"
        )
        assert gate.classification == "unreplayable", (parent, filename, gate)
        assert gate.failure_stage in {
            "final_identity_revalidation",
            "terminal_acknowledgment",
            "terminal_receipt",
        }
        leaf.unlink()
        leaf.write_bytes(original_bytes)

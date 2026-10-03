"""Fake-root tests for the Stage B1 runner-to-scorer permit chain."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import stat
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from core.real_data import cascaded_tanks_abc6_authority as authority
from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_scoring as scoring

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "abc6_scoring_permit_fake_fixtures",
    Path(__file__).with_name("test_cascaded_tanks_abc6_scoring.py"),
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
fixtures = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(fixtures)


def _role_frame(function, root: Path, role: str):
    _, relative_path, function_name = authority.ABC6_ROLE_CONTRACT[
        {"runner": 0, "campaign": 1, "case": 2, "scorer": 3}[role]
    ]
    function.__code__ = function.__code__.replace(
        co_name=function_name,
        co_filename=str(root / relative_path),
    )
    return function


def _register_frame(root: Path):
    def register(authority_value, execution):
        authority_value.register_training_execution(execution)

    return _role_frame(register, root, "campaign")


def _create_handoff_frame(root: Path):
    def create(authority_value, execution, statuses, summary_sha256, forecast_sha256):
        return authority_value.create_runner_handoff(
            execution, statuses, summary_sha256, forecast_sha256
        )

    return _role_frame(create, root, "runner")


def _activate_frame(root: Path):
    def activate(authority_value, handoff):
        return authority_value.activate_scoring(handoff)

    return _role_frame(activate, root, "runner")


def _helper_handoff_frame(root: Path):
    def helper(authority_value, execution, statuses, summary_sha256, forecast_sha256):
        return authority_value.create_runner_handoff(
            execution, statuses, summary_sha256, forecast_sha256
        )

    helper.__code__ = helper.__code__.replace(
        co_name="handoff_helper",
        co_filename=str(root / authority.ABC6_ROLE_CONTRACT[0][1]),
    )
    return helper


def _digest(value: str) -> str:
    return value


def _new_authority(execution, *, grant_sha256: str, request):
    identity = execution.receipt_root_identity
    root_stat = Path(identity.repository_root_realpath).stat()
    receipt_stat = identity.receipt_root_path.stat()
    source_hashes = tuple(sorted(
        (
            relative_path,
            hashlib.sha256(
                (Path(identity.repository_root_realpath) / relative_path).read_bytes()
            ).hexdigest(),
        )
        for relative_path in authority.ABC6_MANIFEST_SOURCE_PATHS
    ))
    runtime_sha256 = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    binding = authority.ABC6AuthorityBindings(
        protocol_id=execution.campaign_result.protocol_id,
        run_id=execution.campaign_result.run_id,
        manifest_sha256=execution.campaign_result.manifest_sha256,
        approval_sha256="d" * 64,
        review_sha256="e" * 64,
        physical_root=identity.repository_root_realpath,
        receipt_root_relative=identity.receipt_root_relative,
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        receipt_device=receipt_stat.st_dev,
        receipt_inode=receipt_stat.st_ino,
        reviewed_git_head="0" * 40,
        watchdog_claim_sha256="f" * 64,
        watchdog_pid=os.getpid(),
        watchdog_start="fake-watchdog-start",
        child_pid=os.getpid(),
        child_start="fake-child-start",
        child_vector=(sys.executable, "fake-child.py"),
        child_launch_image_path=sys.executable,
        child_launch_image_sha256=runtime_sha256,
        child_image_path=sys.executable,
        child_image_sha256=runtime_sha256,
        source_hashes=source_hashes,
        runtime_hashes=tuple(
            sorted(
                (
                    ("child_image_sha256", runtime_sha256),
                    ("child_launch_image_sha256", runtime_sha256),
                    ("python_version_sha256", hashlib.sha256(sys.version.encode()).hexdigest()),
                )
            )
        ),
        campaign_claim_relative=(
            f"artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/"
            f"{execution.campaign_result.run_id}.claim"
        ),
        runner_source_path=authority.ABC6_ROLE_CONTRACT[0][1],
        runner_function=authority.ABC6_ROLE_CONTRACT[0][2],
        campaign_source_path=authority.ABC6_ROLE_CONTRACT[1][1],
        campaign_function=authority.ABC6_ROLE_CONTRACT[1][2],
        case_source_path=authority.ABC6_ROLE_CONTRACT[2][1],
        case_function=authority.ABC6_ROLE_CONTRACT[2][2],
        scorer_source_path=authority.ABC6_ROLE_CONTRACT[3][1],
        scorer_function=authority.ABC6_ROLE_CONTRACT[3][2],
    )
    root_fd = os.open(
        identity.repository_root_realpath,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    receipt_fd = os.open(
        identity.receipt_root_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    request.addfinalizer(lambda fd=root_fd: os.close(fd))
    request.addfinalizer(lambda fd=receipt_fd: os.close(fd))
    record = authority._ReceivedGrantRecord(
        record_sha256="9" * 64,
        file_identity=authority._FileId(1, 2, 3, 4, 5),
        file_mode=0o600,
        regular_file=True,
    )
    value = object.__new__(authority.ABC6LaunchAuthority)
    for name, item in {
        "_seal": authority._AUTHORITY_SEAL,
        "_bindings": binding,
        "_grant_digest": grant_sha256,
        "_grant_fd": 7,
        "_grant_record": record,
        "_root_fd": root_fd,
        "_receipt_fd": receipt_fd,
        "_identity": None,
        "_state": authority._State(),
    }.items():
        object.__setattr__(value, name, item)
    value._state.training_started = True
    value._state.issued = set(range(cases.CASE_COUNT))
    value._state.consumed = set(range(cases.CASE_COUNT))
    value._state.training_bundle = execution._training_bundle
    return value


def _prepare_supervised_case(
    tmp_path,
    monkeypatch,
    request,
    *,
    bad_target_hash=False,
    create_handoff=True,
):
    training, results, forecasts = fixtures._fixture_rosters()
    receipts = fixtures._private_receipt_root(tmp_path)
    fixtures._write_receipts(receipts, results)
    project_root, marker = fixtures._fake_marker_paths(tmp_path)
    materializer_calls = fixtures._install_fake_gate(
        monkeypatch,
        marker,
        project_root,
        bad_hash=bad_target_hash,
    )
    execution, frozen, artifact_path, artifact_sha256 = (
        fixtures._prepare_private_execution_call(
            monkeypatch,
            receipts,
            training,
            results,
            forecasts,
        )
    )
    root = execution.receipt_root_identity.repository_root_realpath
    root_path = Path(root)

    monkeypatch.setattr(authority.ABC6LaunchAuthority, "_assert_live", lambda _self: None)

    grant_sha256 = "c" * 64
    authority_value = _new_authority(
        execution, grant_sha256=grant_sha256, request=request
    )
    object.__setattr__(execution, "_authority", authority_value)
    object.__setattr__(execution, "_grant_sha256", grant_sha256)
    _register_frame(root_path)(authority_value, execution)

    statuses = tuple(
        (index, component, digest)
        for index, status in enumerate(execution.campaign_result.case_statuses)
        for component, digest in (
            ("fit", status.fit_receipt_sha256),
            ("baseline", status.baseline_receipt_sha256),
        )
    )
    summary_sha256 = execution.campaign_result.summary_sha256
    forecast_sha256 = artifact_sha256
    create_frame = _create_handoff_frame(root_path)
    handoff = (
        create_frame(
            authority_value,
            execution,
            statuses,
            summary_sha256,
            forecast_sha256,
        )
        if create_handoff
        else None
    )
    monkeypatch.setattr(
        scoring.score_deferred_abc6_synthetic,
        "__code__",
        scoring.score_deferred_abc6_synthetic.__code__.replace(
            co_filename=str(root_path / authority.ABC6_ROLE_CONTRACT[3][1])
        ),
    )
    return {
        "authority": authority_value,
        "execution": execution,
        "frozen": frozen,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "statuses": statuses,
        "summary_sha256": summary_sha256,
        "forecast_sha256": forecast_sha256,
        "handoff": handoff,
        "activate_frame": _activate_frame(root_path),
        "create_frame": create_frame,
        "root": root_path,
        "receipts": receipts,
        "marker": marker,
        "materializer_calls": materializer_calls,
    }


def _activate(context):
    context["permit"] = context["activate_frame"](
        context["authority"], context["handoff"]
    )
    return context["permit"]


def _score(context, *, execution=None, permit=None, frozen=None):
    return scoring.score_deferred_abc6_synthetic(
        context["execution"] if execution is None else execution,
        context["frozen"] if frozen is None else frozen,
        context["artifact_path"],
        context["artifact_sha256"],
        scoring_permit=context["permit"] if permit is None else permit,
        simulator=fixtures._fake_simulator,
    )


def test_attempt_is_durable_before_marker_and_fake_target_materialization(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)
    attempt_path = context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME
    publish_marker = scoring._claim_reveal_marker
    publish_attempt = scoring._publish_score_attempt
    consumed_contexts = []

    def capture_consumed_context(consumed, event):
        consumed_contexts.append(consumed)
        return publish_attempt(consumed, event)

    def require_attempt_before_marker(*args, **kwargs):
        assert attempt_path.is_file()
        assert not context["marker"].exists()
        assert context["materializer_calls"] == []
        return publish_marker(*args, **kwargs)

    monkeypatch.setattr(scoring, "_claim_reveal_marker", require_attempt_before_marker)
    monkeypatch.setattr(
        scoring, "_publish_score_attempt", capture_consumed_context
    )

    result = _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    consumed = consumed_contexts[0]
    assert type(consumed) is authority.ABC6ConsumedScoringContext
    assert dataclasses.is_dataclass(consumed)
    assert consumed.authority_bindings is context["authority"]._bindings
    assert consumed.grant_record is context["authority"]._grant_record
    assert consumed.campaign_claim_sha256 == context["execution"].campaign_result.claim_sha256
    assert consumed.execution is context["execution"]
    assert consumed.handoff is context["handoff"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        consumed.forecast_artifact_sha256 = "f" * 64
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(consumed)
    attempt_raw = attempt_path.read_bytes()
    attempt = json.loads(attempt_raw.decode("ascii"))
    assert attempt_raw == scoring._canonical_json(attempt)
    assert len(attempt_raw) <= scoring._MAX_SCORE_ATTEMPT_BYTES
    assert stat.S_IMODE(attempt_path.stat().st_mode) == 0o600
    assert set(attempt) == {
        "schema_version",
        "artifact_type",
        "authority_bindings",
        "grant_record_sha256",
        "campaign_claim_sha256",
        "training_evidence_manifest_sha256",
        "training_summary_sha256",
        "status_receipts",
        "forecast_artifact_sha256",
        "reveal_marker_relative",
        "event",
    }
    assert attempt["schema_version"] == 1
    assert attempt["artifact_type"] == "abc6_score_attempt"
    assert attempt["authority_bindings"] == context["authority"]._bindings.payload()
    assert attempt["grant_record_sha256"] == context["authority"]._grant_record.record_sha256
    assert attempt["campaign_claim_sha256"] == context["execution"].campaign_result.claim_sha256
    assert attempt["training_evidence_manifest_sha256"] == context["execution"].evidence_manifest_sha256
    assert attempt["training_summary_sha256"] == context["summary_sha256"]
    assert attempt["status_receipts"] == [
        {"case_index": index, "component": component, "sha256": digest}
        for index, component, digest in context["statuses"]
    ]
    assert attempt["forecast_artifact_sha256"] == context["forecast_sha256"]
    assert attempt["reveal_marker_relative"] == scoring._REVEAL_MARKER_RELATIVE
    fixtures._assert_score_event(attempt["event"], stage="scoring_permit_consumed")
    assert context["marker"].is_file()
    assert context["materializer_calls"] == ["generate"]
    receipt = json.loads(result.score_receipt_path.read_text("ascii"))
    assert receipt["schema_version"] == 1
    assert "watchdog_authority" not in receipt
    forecast_artifact = json.loads(context["artifact_path"].read_text("ascii"))
    assert len(forecast_artifact["integrated_source_hashes"]) == 3
    assert all(
        set(entry) == {"path", "sha256"}
        for entry in forecast_artifact["integrated_source_hashes"]
    )


def test_missing_wrong_cross_authority_and_copied_capabilities_fail_before_marker(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)

    with pytest.raises(TypeError, match="scoring_permit"):
        scoring.score_deferred_abc6_synthetic(
            context["execution"],
            context["frozen"],
            context["artifact_path"],
            context["artifact_sha256"],
            simulator=fixtures._fake_simulator,
        )
    with pytest.raises(TypeError, match="scoring_permit"):
        _score(context, permit=object())

    copied_execution = copy.copy(context["execution"])
    with pytest.raises(scoring.ABC6ScoringError, match="permit was rejected"):
        _score(context, execution=copied_execution, permit=permit)

    second_execution = copy.copy(context["execution"])
    second_authority = _new_authority(
        second_execution,
        grant_sha256="d" * 64,
        request=request,
    )
    object.__setattr__(second_execution, "_authority", second_authority)
    object.__setattr__(second_execution, "_grant_sha256", "d" * 64)
    second_authority._state.training_started = True
    second_authority._state.issued = set(range(cases.CASE_COUNT))
    second_authority._state.consumed = set(range(cases.CASE_COUNT))
    second_authority._state.training_bundle = second_execution._training_bundle
    _register_frame(context["root"])(second_authority, second_execution)
    second_handoff = context["create_frame"](
        second_authority,
        second_execution,
        context["statuses"],
        context["summary_sha256"],
        context["forecast_sha256"],
    )
    second_permit = context["activate_frame"](second_authority, second_handoff)
    with pytest.raises(scoring.ABC6ScoringError, match="permit was rejected"):
        _score(context, permit=second_permit)

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(permit)

    assert context["authority"]._state.score_consumed is False
    assert not (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).exists()
    assert not (context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME).exists()
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


@pytest.mark.parametrize("mutation", ("reordered", "missing", "duplicate"))
def test_bad_order_or_incomplete_runner_receipt_handoff_is_terminal(
    tmp_path, monkeypatch, request, mutation
):
    context = _prepare_supervised_case(
        tmp_path, monkeypatch, request, create_handoff=False
    )
    statuses = context["statuses"]
    if mutation == "reordered":
        statuses = (statuses[1], statuses[0], *statuses[2:])
    elif mutation == "missing":
        statuses = statuses[:-1]
    else:
        statuses = (statuses[0], statuses[0], *statuses[2:])

    with pytest.raises(authority.ABC6AuthorityError):
        context["create_frame"](
            context["authority"],
            context["execution"],
            tuple(statuses),
            context["summary_sha256"],
            context["forecast_sha256"],
        )

    assert context["authority"]._state.handoff_attempted is True
    assert context["authority"]._state.handoff is None
    with pytest.raises(authority.ABC6AuthorityError, match="one-use"):
        context["create_frame"](
            context["authority"],
            context["execution"],
            context["statuses"],
            context["summary_sha256"],
            context["forecast_sha256"],
        )
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


@pytest.mark.parametrize(
    ("filename", "replacement"),
    (
        ("case-00.fit-status.json", b"stale status bytes"),
        ("campaign.training-summary.json", b"stale summary bytes"),
        ("campaign.target-free-forecasts.json", b"stale forecast bytes"),
    ),
)
def test_activation_rechecks_fresh_status_summary_and_forecast_bytes(
    tmp_path, monkeypatch, request, filename, replacement
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    (context["receipts"] / filename).write_bytes(replacement)

    with pytest.raises(authority.ABC6AuthorityError):
        context["activate_frame"](context["authority"], context["handoff"])

    assert context["authority"]._state.activation_attempted is True
    assert context["authority"]._state.activated is False
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []
    with pytest.raises(authority.ABC6AuthorityError, match="one-use"):
        context["activate_frame"](context["authority"], context["handoff"])


def test_helper_frame_failed_activation_and_stale_execution_are_rejected(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(
        tmp_path, monkeypatch, request, create_handoff=False
    )
    with pytest.raises(authority.ABC6AuthorityError, match="attested runner"):
        _helper_handoff_frame(context["root"])(
            context["authority"],
            context["execution"],
            context["statuses"],
            context["summary_sha256"],
            context["forecast_sha256"],
        )
    assert context["authority"]._state.handoff_attempted is False

    with pytest.raises(authority.ABC6AuthorityError):
        context["activate_frame"](context["authority"], object())
    assert context["authority"]._state.activation_attempted is True
    assert context["authority"]._state.activated is False
    with pytest.raises(authority.ABC6AuthorityError, match="one-use"):
        context["activate_frame"](context["authority"], context["handoff"])
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


def test_valid_permit_then_pre_marker_error_is_consumed_without_marker(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)

    with pytest.raises(TypeError, match="frozen_forecasts"):
        _score(context, permit=permit, frozen=object())

    assert context["authority"]._state.score_consumed is True
    assert permit._used is True
    attempt = json.loads(
        (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).read_text("ascii")
    )
    failure_raw = (
        context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME
    ).read_bytes()
    failure = json.loads(failure_raw.decode("ascii"))
    assert failure_raw == scoring._canonical_json(failure)
    assert set(failure) == {
        "schema_version",
        "artifact_type",
        "score_attempt_sha256",
        "authority_bindings_sha256",
        "campaign_claim_sha256",
        "phase",
        "reason_code",
        "event",
    }
    assert failure["schema_version"] == 1
    assert failure["artifact_type"] == "abc6_score_pre_marker_failure"
    assert failure["score_attempt_sha256"] == hashlib.sha256(
        scoring._canonical_json(attempt)
    ).hexdigest()
    assert failure["authority_bindings_sha256"] == hashlib.sha256(
        scoring._canonical_json(attempt["authority_bindings"])
    ).hexdigest()
    assert failure["campaign_claim_sha256"] == attempt["campaign_claim_sha256"]
    assert failure["phase"] == "forecast_revalidation"
    assert failure["reason_code"] == "checked_rejection"
    fixtures._assert_score_event(failure["event"], stage="pre_marker_failure_detected")
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


def test_unexpected_forecast_error_gets_typed_pre_marker_failure(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)

    def fail_target_free_forecast(*_args, **_kwargs):
        raise RuntimeError("private fake forecast failure")

    monkeypatch.setattr(
        scoring, "forecast_abc6_posterior_and_baseline", fail_target_free_forecast
    )
    with pytest.raises(scoring.ABC6ScoringError, match="could not reproduce"):
        _score(context, permit=permit)

    failure = json.loads(
        (
            context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME
        ).read_text("ascii")
    )
    assert failure["phase"] == "forecast_revalidation"
    assert failure["reason_code"] == "unexpected_exception"
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


@pytest.mark.parametrize("claim_copy", ("external", "receipt_root"))
def test_campaign_claim_copy_mismatch_after_consume_is_typed_after_attempt(
    tmp_path, monkeypatch, request, claim_copy
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)
    attempt_path = context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME
    failure_path = context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME
    if claim_copy == "external":
        claim_path = (
            context["root"]
            / context["authority"]._bindings.campaign_claim_relative
        )
    else:
        claim_path = context["receipts"] / campaign_fit.CLAIM_FILENAME
    claim_path.write_bytes(b"mismatched fake claim")
    os.chmod(claim_path, 0o600)
    verify_claim = scoring._verified_campaign_claim_sha256

    def require_attempt_first(consumed, root_anchor, receipt_anchor):
        assert attempt_path.is_file()
        return verify_claim(consumed, root_anchor, receipt_anchor)

    monkeypatch.setattr(
        scoring, "_verified_campaign_claim_sha256", require_attempt_first
    )

    with pytest.raises(scoring.ABC6ScoringError):
        _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    assert permit._used is True
    attempt = json.loads(attempt_path.read_text("ascii"))
    assert attempt["campaign_claim_sha256"] == context["execution"].campaign_result.claim_sha256
    failure = json.loads(failure_path.read_text("ascii"))
    assert failure["campaign_claim_sha256"] == attempt["campaign_claim_sha256"]
    assert failure["phase"] == "source_revalidation"
    assert failure["reason_code"] == "checked_rejection"
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


@pytest.mark.parametrize("claim_copy", ("external", "receipt_root"))
def test_campaign_claim_swap_after_attempt_is_typed_and_stops_before_marker(
    tmp_path, monkeypatch, request, claim_copy
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)
    attempt_path = context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME
    failure_path = context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME
    if claim_copy == "external":
        claim_path = (
            context["root"]
            / context["authority"]._bindings.campaign_claim_relative
        )
    else:
        claim_path = context["receipts"] / campaign_fit.CLAIM_FILENAME
    publish_attempt = scoring._publish_score_attempt

    def publish_then_swap(consumed, event):
        attempt_payload, attempt_sha256 = publish_attempt(consumed, event)
        assert attempt_path.is_file()
        claim_path.write_bytes(b"claim changed after fake attempt publication")
        os.chmod(claim_path, 0o600)
        return attempt_payload, attempt_sha256

    monkeypatch.setattr(scoring, "_publish_score_attempt", publish_then_swap)

    with pytest.raises(scoring.ABC6ScoringError):
        _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    assert permit._used is True
    attempt = json.loads(attempt_path.read_text("ascii"))
    failure_raw = failure_path.read_bytes()
    failure = json.loads(failure_raw.decode("ascii"))
    assert failure_raw == scoring._canonical_json(failure)
    assert failure["score_attempt_sha256"] == hashlib.sha256(
        scoring._canonical_json(attempt)
    ).hexdigest()
    assert failure["campaign_claim_sha256"] == attempt["campaign_claim_sha256"]
    assert failure["phase"] == "source_revalidation"
    assert failure["reason_code"] == "checked_rejection"
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


class _SimulatedScoringCrash(BaseException):
    pass


def test_crash_immediately_after_consume_leaves_no_receipt_or_retry(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)
    snapshot = scoring._score_event_snapshot

    def crash_after_consume(stage):
        snapshot(stage)
        raise _SimulatedScoringCrash()

    monkeypatch.setattr(scoring, "_score_event_snapshot", crash_after_consume)
    with pytest.raises(_SimulatedScoringCrash):
        _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    assert permit._used is True
    assert not (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).exists()
    assert not (context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME).exists()
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


@pytest.mark.parametrize("fault", ("write", "file_fsync", "parent_fsync", "readback"))
def test_ambiguous_attempt_publication_failure_stops_without_retry_or_targets(
    tmp_path, monkeypatch, request, fault
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)

    if fault == "write":
        write = os.write

        def fail_write(descriptor, raw):
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("injected attempt write failure")
            return write(descriptor, raw)

        monkeypatch.setattr(scoring.os, "write", fail_write)
    elif fault in {"file_fsync", "parent_fsync"}:
        fsync = os.fsync

        def fail_fsync(descriptor):
            info = os.fstat(descriptor)
            if fault == "file_fsync" and stat.S_ISREG(info.st_mode):
                raise OSError("injected attempt file fsync failure")
            if (
                fault == "parent_fsync"
                and stat.S_ISDIR(info.st_mode)
                and info.st_ino == context["receipts"].stat().st_ino
            ):
                raise OSError("injected attempt parent fsync failure")
            return fsync(descriptor)

        monkeypatch.setattr(scoring.os, "fsync", fail_fsync)
    else:
        reader = scoring._read_stable_score_file_at

        def fail_readback(directory_fd, filename, *, label):
            if label == "score-attempt receipt readback":
                raise OSError("injected attempt readback failure")
            return reader(directory_fd, filename, label=label)

        monkeypatch.setattr(scoring, "_read_stable_score_file_at", fail_readback)

    with pytest.raises(Exception):
        _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    assert permit._used is True
    # A failed publication may leave a partial or fully durable exclusive leaf;
    # it must never be retried or promoted to a marker/target operation.
    assert (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).exists()
    assert not (context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME).exists()
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []
    failed_leaf = (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).read_bytes()
    with pytest.raises(scoring.ABC6ScoringError, match="permit was rejected"):
        _score(context, permit=permit)
    assert (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).read_bytes() == failed_leaf


def test_marker_publication_error_does_not_emit_pre_marker_failure_receipt(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)

    def fail_marker_publication(*_args, **_kwargs):
        raise OSError("private fake marker publication failure")

    monkeypatch.setattr(scoring, "_claim_reveal_marker", fail_marker_publication)
    with pytest.raises(OSError, match="marker publication"):
        _score(context, permit=permit)

    assert (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).is_file()
    assert not (context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME).exists()
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


@pytest.mark.parametrize("leaf_kind", ("duplicate", "symlink", "fifo"))
def test_score_attempt_leaf_collision_fails_closed(
    tmp_path, monkeypatch, request, leaf_kind
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)
    attempt_path = context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME
    if leaf_kind == "duplicate":
        attempt_path.write_bytes(b"occupied fake leaf")
        os.chmod(attempt_path, 0o600)
    elif leaf_kind == "symlink":
        attempt_path.symlink_to(context["receipts"] / campaign_fit.CLAIM_FILENAME)
    else:
        os.mkfifo(attempt_path, 0o600)

    with pytest.raises(Exception):
        _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    assert not (context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME).exists()
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


def test_copied_receipt_ancestor_after_consume_cannot_receive_attempt(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)
    snapshot = scoring._score_event_snapshot

    def copy_receipt_root_after_consume(stage):
        event = snapshot(stage)
        if stage == "scoring_permit_consumed":
            moved = context["receipts"].with_name("receipts-pinned-original")
            os.rename(context["receipts"], moved)
            shutil.copytree(moved, context["receipts"])
        return event

    monkeypatch.setattr(
        scoring,
        "_score_event_snapshot",
        copy_receipt_root_after_consume,
    )
    with pytest.raises(Exception):
        _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
    assert not (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).exists()
    assert not (
        context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME
    ).exists()
    assert not context["marker"].exists()
    assert context["materializer_calls"] == []


def test_post_marker_error_keeps_one_use_failure_semantics_with_permit(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(
        tmp_path, monkeypatch, request, bad_target_hash=True
    )
    permit = _activate(context)

    with pytest.raises(scoring.ABC6DeferredScoreConsumedError) as failure:
        _score(context, permit=permit)

    assert failure.value.condition_consumed is True
    assert failure.value.retry_forbidden is True
    assert context["authority"]._state.score_consumed is True
    assert context["marker"].is_file()
    assert context["materializer_calls"] == ["generate"]
    assert (context["receipts"] / scoring.SCORE_ATTEMPT_FILENAME).is_file()
    assert not (context["receipts"] / scoring.SCORE_PRE_MARKER_FAILURE_FILENAME).exists()
    receipt = json.loads((context["receipts"] / scoring.SCORE_RECEIPT_FILENAME).read_text("ascii"))
    assert receipt["outcome"] == "failed"

"""Fake-root tests for the Stage B1 runner-to-scorer permit chain."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
    source_hashes = tuple(
        (
            relative_path,
            hashlib.sha256(
                (Path(identity.repository_root_realpath) / relative_path).read_bytes()
            ).hexdigest(),
        )
        for _role, relative_path, _function in authority.ABC6_ROLE_CONTRACT
    )
    binding = SimpleNamespace(
        protocol_id=execution.campaign_result.protocol_id,
        run_id=execution.campaign_result.run_id,
        manifest_sha256=execution.campaign_result.manifest_sha256,
        physical_root=identity.repository_root_realpath,
        receipt_root_relative=identity.receipt_root_relative,
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        receipt_device=receipt_stat.st_dev,
        receipt_inode=receipt_stat.st_ino,
        runner_source_path=authority.ABC6_ROLE_CONTRACT[0][1],
        runner_function=authority.ABC6_ROLE_CONTRACT[0][2],
        campaign_source_path=authority.ABC6_ROLE_CONTRACT[1][1],
        campaign_function=authority.ABC6_ROLE_CONTRACT[1][2],
        case_source_path=authority.ABC6_ROLE_CONTRACT[2][1],
        case_function=authority.ABC6_ROLE_CONTRACT[2][2],
        scorer_source_path=authority.ABC6_ROLE_CONTRACT[3][1],
        scorer_function=authority.ABC6_ROLE_CONTRACT[3][2],
        source_hashes=source_hashes,
    )
    root_fd = os.open(
        identity.repository_root_realpath,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    request.addfinalizer(lambda fd=root_fd: os.close(fd))
    value = object.__new__(authority.ABC6LaunchAuthority)
    for name, item in {
        "_seal": authority._AUTHORITY_SEAL,
        "_bindings": binding,
        "_grant_digest": grant_sha256,
        "_grant_fd": 7,
        "_grant_record": None,
        "_root_fd": root_fd,
        "_receipt_fd": -1,
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

    def fake_receipt_metadata(self):
        return {
            "schema_version": 1,
            "protocol_id": self._bindings.protocol_id,
            "run_id": self._bindings.run_id,
            "grant_sha256": self._grant_digest,
            "manifest_sha256": self._bindings.manifest_sha256,
            "runner_handoff_attempted": self._state.handoff_attempted,
            "scoring_activated": self._state.activated,
            "scoring_consumed": self._state.score_consumed,
        }

    monkeypatch.setattr(
        authority.ABC6LaunchAuthority,
        "receipt_metadata",
        fake_receipt_metadata,
    )
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


def test_permit_consumes_before_materialization_and_records_only_in_score_receipt(
    tmp_path, monkeypatch, request
):
    context = _prepare_supervised_case(tmp_path, monkeypatch, request)
    permit = _activate(context)

    result = _score(context, permit=permit)

    assert context["authority"]._state.score_consumed is True
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
    receipt = json.loads((context["receipts"] / scoring.SCORE_RECEIPT_FILENAME).read_text("ascii"))
    assert receipt["outcome"] == "failed"

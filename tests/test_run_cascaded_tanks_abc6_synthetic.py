"""Fake-only orchestration tests for the one-child ABC6 synthetic runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_scoring as scoring
from scripts import run_cascaded_tanks_abc6_synthetic as runner

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "abc6_scoring_fake_fixtures",
    Path(__file__).with_name("test_cascaded_tanks_abc6_scoring.py"),
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
scoring_fixtures = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(scoring_fixtures)

_CAMPAIGN_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "abc6_campaign_fake_fixtures",
    Path(__file__).with_name("test_cascaded_tanks_abc6_campaign_fit.py"),
)
assert _CAMPAIGN_FIXTURE_SPEC is not None
assert _CAMPAIGN_FIXTURE_SPEC.loader is not None
campaign_fixtures = importlib.util.module_from_spec(_CAMPAIGN_FIXTURE_SPEC)
_CAMPAIGN_FIXTURE_SPEC.loader.exec_module(campaign_fixtures)


_FAKE_PROTOCOL = "private-fake-abc6-synthetic-protocol"
_FAKE_RUN = "private-fake-abc6-runner-tests"


def _private_identity(monkeypatch) -> None:
    for module in (cases, scoring, campaign_fit):
        monkeypatch.setattr(module, "PROTOCOL_ID", _FAKE_PROTOCOL)
        monkeypatch.setattr(module, "RUN_ID", _FAKE_RUN)


def _install_fake_source_pins(
    tmp_path: Path, monkeypatch
) -> tuple[tuple[str, str], ...]:
    root = tmp_path.resolve() / "fake-source-tree"
    hashes = {}
    for index, relative in enumerate(campaign_fit.authority_module.ABC6_MANIFEST_SOURCE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"private fake source {index}\n".encode("ascii")
        path.write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(runner, "_REPO_ROOT", root)
    monkeypatch.setattr(campaign_fit, "_REPO_ROOT", root)
    monkeypatch.setattr(campaign_fit, "_active_checkout_root_path", lambda: root)
    monkeypatch.setattr(campaign_fit, "_current_git_head", lambda: "a" * 40)
    monkeypatch.setattr(scoring, "_SOURCE_ROOT", root)
    monkeypatch.setattr(
        campaign_fit,
        "_REQUIRED_SOURCE_PATHS",
        campaign_fit.authority_module.ABC6_MANIFEST_SOURCE_PATHS,
    )
    monkeypatch.setattr(
        campaign_fit,
        "_REVIEWED_SOURCE_SHA256",
        {
            path: digest
            for path, digest in hashes.items()
            if path != campaign_fit.authority_module.ABC6_CAMPAIGN_SOURCE_PATH
        },
    )
    return tuple((path, hashes[path]) for path in runner._INTEGRATED_SOURCE_PATHS)


def _private_runner_fixture(tmp_path: Path, monkeypatch):
    _private_identity(monkeypatch)
    # These legacy orchestration tests exercise downstream fake campaign
    # behavior. The launch bootstrap is covered separately with a real sealed
    # anonymous-FD grant. The injected object is paired with a no-op startup
    # consumer only inside these downstream-only fake tests.
    test_authority = object.__new__(runner.authority_module.ABC6LaunchAuthority)
    monkeypatch.setattr(
        runner.authority_module.ABC6LaunchAuthority,
        "consume_runner_startup",
        lambda self: None,
    )
    handoff_calls = []

    def fake_create_handoff(self, execution, status_receipts, summary_sha256, forecast_sha256):
        handoff_calls.append(
            (self, execution, status_receipts, summary_sha256, forecast_sha256)
        )
        return object()

    monkeypatch.setattr(
        runner.authority_module.ABC6LaunchAuthority,
        "create_runner_handoff",
        fake_create_handoff,
    )
    monkeypatch.setattr(
        runner.authority_module.ABC6LaunchAuthority,
        "activate_scoring",
        lambda self, handoff: object.__new__(
            runner.authority_module.ABC6ScoringPermit
        ),
    )
    monkeypatch.setattr(
        runner.authority_module.ABC6ScoringPermit,
        "consume",
        lambda _permit, _execution: {"schema_version": 1, "fake_runner_test": True},
    )
    source_hashes = _install_fake_source_pins(tmp_path, monkeypatch)
    root = runner._REPO_ROOT
    receipts = root / campaign_fit.RECEIPT_ROOT_RELATIVE
    training = campaign_fixtures._fake_training_bundle()
    monkeypatch.setattr(
        campaign_fit, "build_synthetic_training_bundle", lambda: training
    )
    manifest = tmp_path.resolve() / "private-manifest.json"
    manifest_sha256 = campaign_fixtures._write_manifest(manifest)
    claim_registry = root / "private-claim-registry" / f"{_FAKE_RUN}.claim"
    marker = (
        root
        / "artifacts"
        / "evaluations"
        / "cascaded_tanks_abc6_scoring"
        / "claims"
        / f"{_FAKE_RUN}.claim"
    )
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    monkeypatch.setattr(cases, "_REVEAL_MARKER_PATH", marker)
    monkeypatch.setattr(scoring, "_PROJECT_ROOT", root)
    monkeypatch.setattr(scoring, "_REVEAL_MARKER_PATH", marker)
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
        handoff_calls,
    )


def _execute_private_training(
    manifest: Path,
    manifest_sha256: str,
    receipts: Path,
    claim_registry: Path,
):
    return campaign_fit._run_campaign_with_fit_callable_and_evidence_for_test(
        manifest,
        manifest_sha256,
        receipts,
        fit_callable=_typed_fake_fit,
        claim_registry_path=claim_registry,
    )


def _typed_fake_fit(data):
    incomplete = data.case.case_index == 14
    result = campaign_fixtures._fake_result(
        data,
        fit_status="incomplete" if incomplete else "complete",
    )
    if incomplete:
        abc_result = {
            key: value
            for key, value in result.abc_result.items()
            if key != "posterior"
        }
        return replace(result, abc_result=abc_result)

    rows = scoring_fixtures._parameter_rows(data.case.case_index)
    columns = {
        name: np.full(48, rows[0][column_index], dtype=np.float64)
        for column_index, name in enumerate(cases.PARAMETER_ORDER)
    }
    posterior = {
        "parameter_order": cases.PARAMETER_ORDER,
        "free_parameter_values": columns,
        "weights": np.full(48, 1.0 / 48.0, dtype=np.float64),
    }
    abc_result = dict(result.abc_result)
    abc_result["posterior"] = posterior
    return replace(result, abc_result=abc_result)


def test_current_campaign_source_pins_block_before_training_claim(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        campaign_fit,
        "run_abc6_training_campaign_with_evidence",
        lambda *args: calls.append(args),
    )

    receipts = (
        tmp_path.resolve() / campaign_fit.RECEIPT_ROOT_RELATIVE
    )
    with pytest.raises(runner.ABC6SyntheticRunnerPreflightError, match="direct runner/API"):
        runner.run_cascaded_tanks_abc6_synthetic(
            tmp_path / "private-manifest.json",
            "d" * 64,
            receipts,
        )

    assert calls == []
    assert not receipts.exists()


def test_integrated_runner_freezes_all_cases_before_scorer_claim_and_generator(
    tmp_path, monkeypatch
):
    (
        root,
        receipts,
        training,
        manifest,
        manifest_sha256,
        claim_registry,
        marker,
        source_hashes,
        test_authority,
        handoff_calls,
    ) = _private_runner_fixture(tmp_path, monkeypatch)
    events = []
    campaign_calls = []

    def fake_campaign(*args, **kwargs):
        assert kwargs == {"launch_authority": test_authority}
        campaign_calls.append(args)
        events.append("campaign")
        return _execute_private_training(
            args[0], args[1], args[2], claim_registry
        )

    real_load_verified = (
        campaign_fit.ABC6TrainingCampaignExecution.load_verified_training_evidence
    )

    def checked_load_verified(self):
        verified = real_load_verified(self)
        events.append("verified-evidence")
        return verified

    monkeypatch.setattr(
        campaign_fit, "run_abc6_training_campaign_with_evidence", fake_campaign
    )
    monkeypatch.setattr(
        campaign_fit.ABC6TrainingCampaignExecution,
        "load_verified_training_evidence",
        checked_load_verified,
    )

    real_status_reader = scoring._read_verified_statuses

    def checked_status_reader(path):
        result = real_status_reader(path)
        events.append("verified-48-statuses")
        return result

    monkeypatch.setattr(scoring, "_read_verified_statuses", checked_status_reader)

    real_forecast = runner.forecast_module.forecast_abc6_posterior_and_baseline

    def checked_forecast(data, result):
        assert "verified-evidence" in events
        assert "verified-48-statuses" in events
        index = result.case_index
        assert data.case == cases.CASE_ROSTER[index]
        events.append(f"forecast-{index}")
        return real_forecast(data, result)

    monkeypatch.setattr(
        runner.forecast_module,
        "forecast_abc6_posterior_and_baseline",
        checked_forecast,
    )
    real_freeze = scoring.freeze_abc6_forecasts

    def checked_freeze(values):
        assert len(values) == cases.CASE_COUNT
        assert len([event for event in events if event.startswith("forecast-")]) == 24
        events.append("frozen")
        return real_freeze(values)

    monkeypatch.setattr(scoring, "freeze_abc6_forecasts", checked_freeze)

    generator_calls = []
    real_deferred_scorer = scoring.score_deferred_abc6_synthetic

    def checked_deferred_scorer(
        execution,
        frozen,
        artifact_path,
        artifact_sha256,
        *,
        scoring_permit,
        simulator=scoring.simulate_cascaded_tanks,
    ):
        assert isinstance(execution, campaign_fit.ABC6TrainingCampaignExecution)
        assert len(frozen.forecasts) == 24
        assert artifact_path == receipts / runner.FORECAST_ARTIFACT_FILENAME
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact_sha256
        events.append("typed-scorer-handoff")
        return real_deferred_scorer(
            execution,
            frozen,
            artifact_path,
            artifact_sha256,
            scoring_permit=scoring_permit,
            simulator=simulator,
        )

    monkeypatch.setattr(
        scoring, "score_deferred_abc6_synthetic", checked_deferred_scorer
    )

    def fake_materialize(_training, *, simulator, _scoring_handoff):
        assert marker.is_file(), "generator ran before one-use reveal marker"
        assert _scoring_handoff._used is True
        artifact = receipts / runner.FORECAST_ARTIFACT_FILENAME
        assert artifact.is_file(), "generator ran before forecast roster was frozen"
        assert "verified-evidence" in events
        assert "verified-48-statuses" in events
        assert "frozen" in events
        assert "typed-scorer-handoff" in events
        document = json.loads(artifact.read_text("ascii"))
        assert document["target_free"] is True
        assert document["prospective_targets_generated_by_runner"] is False
        assert len(document["forecasts"]) == 24
        assert document["status_receipts"] and len(document["status_receipts"]) == 48
        n_records = [document["forecasts"][index] for index in (8, 9)]
        assert all(record["status"] == "abstained_n" for record in n_records)
        assert all(record["prospective_inputs"] is None for record in n_records)
        assert all(record["particles"] == [] for record in n_records)
        assert all(record["particle_trajectories"] == [] for record in n_records)
        incomplete = document["forecasts"][14]
        assert incomplete["status"] == "incomplete_abc_fit"
        assert incomplete["particles"] == []
        assert incomplete["weights"] == []
        assert incomplete["particle_trajectories"] == []
        assert incomplete["pointwise_weighted_mean"] is None
        assert incomplete["pointwise_weighted_median"] is None
        for index, record in enumerate(document["forecasts"]):
            if index not in {8, 9, 14}:
                assert len(record["particles"]) == 48
        generator_calls.append("called-after-marker-and-freeze")
        return scoring_fixtures._fake_targets(bad_hash=True)

    monkeypatch.setattr(cases, "_materialize_prospective_targets", fake_materialize)

    with pytest.raises(scoring.ABC6DeferredScoreConsumedError) as raised:
        runner.run_cascaded_tanks_abc6_synthetic(
            manifest,
            manifest_sha256,
            receipts,
            launch_authority=test_authority,
        )

    assert raised.value.condition_consumed is True
    assert raised.value.retry_forbidden is True
    assert marker.is_file()
    assert len(campaign_calls) == 1
    assert campaign_calls[0][2] == receipts
    assert generator_calls == ["called-after-marker-and-freeze"]
    assert len(handoff_calls) == 1
    assert handoff_calls[0][0] is test_authority
    assert isinstance(
        handoff_calls[0][1], campaign_fit.ABC6TrainingCampaignExecution
    )
    assert len(handoff_calls[0][2]) == 48
    assert handoff_calls[0][3] == hashlib.sha256(
        (receipts / campaign_fit.SUMMARY_FILENAME).read_bytes()
    ).hexdigest()
    assert handoff_calls[0][4] == hashlib.sha256(
        (receipts / runner.FORECAST_ARTIFACT_FILENAME).read_bytes()
    ).hexdigest()
    assert events.index("verified-evidence") < events.index("verified-48-statuses")
    assert events.index("frozen") < events.index("typed-scorer-handoff")
    assert events.index("frozen") < len(events)
    assert source_hashes == runner._source_pin_check()
    failure_receipt_path = receipts / scoring.SCORE_RECEIPT_FILENAME
    assert failure_receipt_path.is_file()
    failure_receipt = json.loads(failure_receipt_path.read_text("ascii"))
    assert failure_receipt["outcome"] == "failed"
    assert failure_receipt["score_event"]["stage"] == "target_hash_validation"
    assert type(failure_receipt["score_event"]["monotonic_ns"]) is int
    assert failure_receipt["score_event"]["clock"] == "host-local-monotonic-ns"
    assert failure_receipt["score_event"]["occurred_at_utc"].endswith("Z")
    assert failure_receipt["failure_checkpoint"]["occurred_at_monotonic_ns"] == (
        failure_receipt["score_event"]["monotonic_ns"]
    )
    assert failure_receipt["protocol_id"] == cases.PROTOCOL_ID
    assert failure_receipt["run_id"] == cases.RUN_ID
    assert len(failure_receipt["training"]["status_receipts"]) == 48
    assert failure_receipt["target_free_forecast"]["artifact_sha256"] == (
        hashlib.sha256(
            (receipts / runner.FORECAST_ARTIFACT_FILENAME).read_bytes()
        ).hexdigest()
    )
    assert failure_receipt["failure_checkpoint"]["stage"] == (
        "target_hash_validation"
    )
    assert failure_receipt["prospective_targets"]["N"][
        "prospective_target_generated"
    ] is False
    assert failure_receipt["prospective_targets"]["target_sha256_by_truth"] == [
        ["A", None],
        ["B", None],
        ["M", None],
    ]


def test_preopen_artifacts_symlink_fails_before_claim_marker_or_materializer(
    tmp_path, monkeypatch
):
    (
        root,
        receipts,
        _training,
        manifest,
        manifest_sha256,
        claim_registry,
        marker,
        _source_hashes,
        test_authority,
        _handoff_calls,
    ) = _private_runner_fixture(tmp_path, monkeypatch)
    copied_artifacts = tmp_path.resolve() / "copied-artifacts"
    copied_artifacts.mkdir()
    (root / "artifacts").symlink_to(copied_artifacts, target_is_directory=True)
    campaign_calls = []
    bundle_builds = []
    materializer_calls = []
    monkeypatch.setattr(
        campaign_fit,
        "build_synthetic_training_bundle",
        lambda: bundle_builds.append("build"),
    )
    monkeypatch.setattr(
        campaign_fit,
        "run_abc6_training_campaign_with_evidence",
        lambda *args, **kwargs: (
            pytest.fail("runner did not pass the received launch authority")
            if kwargs != {"launch_authority": test_authority}
            else (
                campaign_calls.append(args),
                _execute_private_training(*args, claim_registry),
            )[1]
        ),
    )
    monkeypatch.setattr(
        cases,
        "_materialize_prospective_targets",
        lambda *args, **kwargs: materializer_calls.append((args, kwargs)),
    )

    with pytest.raises(
        campaign_fit.ABC6CampaignPreflightError, match="symlinked path component"
    ):
        runner.run_cascaded_tanks_abc6_synthetic(
            manifest,
            manifest_sha256,
            receipts,
            launch_authority=test_authority,
        )

    assert len(campaign_calls) == 1
    assert bundle_builds == []
    assert not claim_registry.exists()
    assert not marker.exists()
    assert materializer_calls == []


def test_postopen_artifacts_swap_cannot_redirect_forecast_readback_or_reveal(
    tmp_path, monkeypatch
):
    (
        root,
        receipts,
        _training,
        manifest,
        manifest_sha256,
        claim_registry,
        marker,
        _source_hashes,
        test_authority,
        _handoff_calls,
    ) = _private_runner_fixture(tmp_path, monkeypatch)
    campaign_calls = []

    def fake_campaign(*args, **kwargs):
        assert kwargs == {"launch_authority": test_authority}
        campaign_calls.append(args)
        return _execute_private_training(
            args[0], args[1], args[2], claim_registry
        )

    monkeypatch.setattr(
        campaign_fit, "run_abc6_training_campaign_with_evidence", fake_campaign
    )
    decoy_artifacts = tmp_path.resolve() / "copied-artifacts-after-open"
    decoy_artifacts.mkdir()
    saved_artifacts = root / "artifacts-opened-before-swap"
    original_read = cases._read_regular_file_at
    swapped = []
    anchored_readback = []

    def swap_then_read(directory_fd, filename, **kwargs):
        if filename == runner.FORECAST_ARTIFACT_FILENAME and not swapped:
            (root / "artifacts").rename(saved_artifacts)
            (root / "artifacts").symlink_to(
                decoy_artifacts, target_is_directory=True
            )
            swapped.append(True)
        raw = original_read(directory_fd, filename, **kwargs)
        if filename == runner.FORECAST_ARTIFACT_FILENAME:
            anchored_readback.append(raw)
        return raw

    monkeypatch.setattr(cases, "_read_regular_file_at", swap_then_read)
    scorer_calls = []
    materializer_calls = []
    monkeypatch.setattr(
        scoring,
        "score_deferred_abc6_synthetic",
        lambda *args, **kwargs: scorer_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        cases,
        "_materialize_prospective_targets",
        lambda *args, **kwargs: materializer_calls.append((args, kwargs)),
    )

    with pytest.raises(
        runner.ABC6SyntheticRunnerIntegrityError,
        match="pinned receipt root",
    ):
        runner.run_cascaded_tanks_abc6_synthetic(
            manifest,
            manifest_sha256,
            receipts,
            launch_authority=test_authority,
        )

    saved_receipt_root = saved_artifacts / Path(
        campaign_fit.RECEIPT_ROOT_RELATIVE
    ).relative_to("artifacts")
    decoy_receipt_root = decoy_artifacts / Path(
        campaign_fit.RECEIPT_ROOT_RELATIVE
    ).relative_to("artifacts")
    assert len(campaign_calls) == 1
    assert swapped == [True]
    assert len(anchored_readback) == 1
    assert json.loads(anchored_readback[0].decode("ascii"))["target_free"] is True
    assert (saved_receipt_root / runner.FORECAST_ARTIFACT_FILENAME).is_file()
    assert not (decoy_receipt_root / runner.FORECAST_ARTIFACT_FILENAME).exists()
    assert not marker.exists()
    assert scorer_calls == []
    assert materializer_calls == []


def test_missing_incomplete_case_receipt_fails_before_scorer_or_materializer(
    tmp_path, monkeypatch
):
    (
        _root,
        receipts,
        _training,
        manifest,
        manifest_sha256,
        claim_registry,
        marker,
        _source_hashes,
        test_authority,
        _handoff_calls,
    ) = _private_runner_fixture(tmp_path, monkeypatch)
    campaign_calls = []

    def fake_campaign(*args, **kwargs):
        assert kwargs == {"launch_authority": test_authority}
        campaign_calls.append(args)
        execution = _execute_private_training(
            args[0], args[1], args[2], claim_registry
        )
        # Case 14 has an intentional final incomplete fit; removing its receipt
        # makes the all-candidate durable gate unverifiable.
        (receipts / "case-14.fit-status.json").unlink()
        return execution

    monkeypatch.setattr(
        campaign_fit, "run_abc6_training_campaign_with_evidence", fake_campaign
    )
    scorer_calls = []
    materializer_calls = []
    monkeypatch.setattr(
        scoring,
        "score_deferred_abc6_synthetic",
        lambda *args, **kwargs: scorer_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        cases,
        "_materialize_prospective_targets",
        lambda *args, **kwargs: materializer_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="receipt|training evidence"):
        runner.run_cascaded_tanks_abc6_synthetic(
            manifest,
            manifest_sha256,
            receipts,
            launch_authority=test_authority,
        )

    assert len(campaign_calls) == 1
    assert not marker.exists()
    assert scorer_calls == []
    assert materializer_calls == []


def test_withheld_s_suffix_mutation_leaves_short_fit_view_unchanged():
    baseline = cases.build_synthetic_training_bundle(
        simulator=scoring_fixtures._fake_simulator
    )

    def mutate_suffix(truth_id, values):
        if truth_id != "A":
            return values
        return values[: cases.SHORT_LENGTH] + tuple(
            value + 123.0 for value in values[cases.SHORT_LENGTH :]
        )

    changed = cases.build_synthetic_training_bundle(
        simulator=scoring_fixtures._fake_simulator,
        training_output_mutator=mutate_suffix,
    )

    assert baseline.data_for_case(0).observed_outputs == changed.data_for_case(
        0
    ).observed_outputs
    assert baseline.data_for_case(2).observed_outputs != changed.data_for_case(
        2
    ).observed_outputs

"""Synthetic fixtures for the one-use, training-only ABC6 campaign seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign
from core.real_data import cascaded_tanks_abc6_cases as cases


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
    return SimpleNamespace(
        case_index=data.case.case_index,
        case_id=data.case.case_id,
        model=data.case.fit_model.value,
        abc_status=fit_status,
        baseline_status=baseline_status,
    )


def _test_claim_path(tmp_path: Path) -> Path:
    return tmp_path / "claim-registry" / f"{campaign.RUN_ID}.claim"


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
        campaign._run_campaign_with_fit_callable_for_test(
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

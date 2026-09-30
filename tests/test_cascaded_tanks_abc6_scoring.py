"""Private synthetic-only fixtures for the deferred ABC6 scoring seam."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_scoring as scoring
from core.real_data.cascaded_tanks_abc6_cases import (
    CASE_ROSTER,
    CROSSING_PRIOR_BOUNDS,
    NO_CROSSING_PRIOR_BOUNDS,
    PARAMETER_ORDER,
    TRAINING_INPUT_L,
    TRAINING_INPUT_S,
    ABC6ProspectiveTargets,
    ABC6TrainingBundle,
    ABC6TrainingCaseData,
    write_status_receipt,
)
from core.real_data.cascaded_tanks_abc6_forecast import (
    ABC6ForecastFailure,
    ABC6ParticleForecast,
    ABC6PosteriorBaselineForecast,
)
from core.real_data.cascaded_tanks_models import (
    TankSimulationSuccess,
    TankState,
    TankStep,
)

_WEIGHTS = (0.25, 0.75)
_FAKE_TARGETS = {"A": (1.0,) * 60, "B": (2.0,) * 60, "M": (3.0,) * 60}


def _parameter_rows(case_index: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    case = CASE_ROSTER[case_index]
    bounds = NO_CROSSING_PRIOR_BOUNDS if case.truth_id == "N" else CROSSING_PRIOR_BOUNDS
    center = tuple((lower + upper) / 2.0 for lower, upper in bounds)
    if case.truth_id in {"A", "B"} and case.fit_model.value == (
        "O2" if case.truth_id == "A" else "C2"
    ):
        center = (
            (0.50, 0.40, 0.50, 0.50, 0.50, 3.0)
            if case.truth_id == "A"
            else (0.54, 0.36, 0.46, 0.25, 0.75, 3.2)
        )
    if case.truth_id == "N":
        center = (0.50, 0.40, 0.50, 0.50, 0.50, 950.0)
        second = (0.50, 0.40, 0.50, 0.50, 0.50, 1050.0)
    else:
        second_values = list(center)
        second_values[0] = min(bounds[0][1], center[0] + 0.01)
        second = tuple(second_values)
    return center, second


def _result(case_index: int, *, incomplete: bool = False):
    case = CASE_ROSTER[case_index]
    abc_status = "incomplete" if incomplete else "complete"
    if incomplete:
        abc_result = {"complete": False}
    else:
        first, second = _parameter_rows(case_index)
        rows = (first, second) + (first,) * 46
        weights = np.asarray((_WEIGHTS[0], _WEIGHTS[1]) + (0.0,) * 46, dtype=float)
        columns = {
            name: np.asarray(tuple(row[index] for row in rows), dtype=float)
            for index, name in enumerate(PARAMETER_ORDER)
        }
        abc_result = {
            "complete": True,
            "posterior": {
                "parameter_order": PARAMETER_ORDER,
                "free_parameter_values": columns,
                "weights": weights,
            },
        }
    status = "incomplete" if incomplete else "complete"
    return SimpleNamespace(
        case_index=case_index,
        case_id=case.case_id,
        model=case.fit_model.value,
        fit_length=case.input_length,
        status=status,
        abc_status=abc_status,
        baseline_status="complete",
        abc_result=abc_result,
    )


def _forecast(
    case_index: int, result, *, incomplete: bool = False, particle_failure: bool = False
):
    case = CASE_ROSTER[case_index]
    common = {
        "case_index": case_index,
        "case_id": case.case_id,
        "fit_window": case.input_window,
        "parameter_order": PARAMETER_ORDER,
        "quantile_convention": "weighted left-inverse empirical CDF",
        "pointwise_summaries_are_coherent_trajectories": False,
        "baseline_parameter_values": None,
        "baseline_common_time_state": None,
        "baseline_trajectory": None,
        "baseline_failure": None,
    }
    if case.truth_id == "N":
        return ABC6PosteriorBaselineForecast(
            status="abstained_n",
            prospective_inputs=None,
            common_state_index=None,
            particles=(),
            weights=(),
            particle_trajectories=(),
            aggregate_status="unavailable",
            pointwise_weighted_mean=None,
            pointwise_weighted_median=None,
            pointwise_q05=None,
            pointwise_q95=None,
            effective_sample_size=None,
            **common,
        )
    if incomplete:
        return ABC6PosteriorBaselineForecast(
            status="incomplete_abc_fit",
            prospective_inputs=None,
            common_state_index=None,
            particles=(),
            weights=(),
            particle_trajectories=(),
            aggregate_status="unavailable",
            pointwise_weighted_mean=None,
            pointwise_weighted_median=None,
            pointwise_q05=None,
            pointwise_q95=None,
            effective_sample_size=None,
            **common,
        )

    first, second = _parameter_rows(case_index)
    rows = (first, second) + (first,) * 46
    weights = _WEIGHTS + (0.0,) * 46
    base = 0.1 * case_index
    first_path = (base,) * 60
    second_path = None if particle_failure else (base + 1.0,) * 60
    particles = (
        ABC6ParticleForecast(
            particle_index=0,
            parameter_values=rows[0],
            weight=weights[0],
            common_time_state=TankState(0.5, 0.5),
            trajectory=first_path,
            failure=None,
        ),
        ABC6ParticleForecast(
            particle_index=1,
            parameter_values=rows[1],
            weight=weights[1],
            common_time_state=TankState(0.5, 0.5),
            trajectory=second_path,
            failure=(
                ABC6ForecastFailure("forecast", "fixture_failure", "private fixture")
                if particle_failure
                else None
            ),
        ),
    )
    particles += tuple(
        ABC6ParticleForecast(
            particle_index=index,
            parameter_values=rows[index],
            weight=weights[index],
            common_time_state=TankState(0.5, 0.5),
            trajectory=first_path,
            failure=None,
        )
        for index in range(2, 48)
    )
    baseline_values = rows[0]
    baseline_path = (base + 2.0,) * 60
    if particle_failure:
        aggregate_status = "suppressed_particle_failure"
        mean = median = q05 = q95 = None
    else:
        aggregate_status = "complete"
        mean = (base + 0.75,) * 60
        median = (base + 1.0,) * 60
        q05 = (base,) * 60
        q95 = (base + 1.0,) * 60
    return ABC6PosteriorBaselineForecast(
        status="complete",
        prospective_inputs=tuple(cases.PROSPECTIVE_INPUT),
        common_state_index=204,
        particles=particles,
        weights=weights,
        particle_trajectories=(first_path, second_path) + (first_path,) * 46,
        aggregate_status=aggregate_status,
        pointwise_weighted_mean=mean,
        pointwise_weighted_median=median,
        pointwise_q05=q05,
        pointwise_q95=q95,
        effective_sample_size=1.6,
        baseline_parameter_values=baseline_values,
        baseline_common_time_state=TankState(0.5, 0.5),
        baseline_trajectory=baseline_path,
        baseline_failure=None,
        case_index=case_index,
        case_id=case.case_id,
        fit_window=case.input_window,
        parameter_order=PARAMETER_ORDER,
        quantile_convention="weighted left-inverse empirical CDF",
        pointwise_summaries_are_coherent_trajectories=False,
    )


def _fixture_rosters():
    data = []
    results = []
    forecasts = []
    for case in CASE_ROSTER:
        inputs = TRAINING_INPUT_S if case.input_window == "S" else TRAINING_INPUT_L
        observed = (
            (5.0,) * 24 + (2.0,) * 180
            if case.truth_id == "M"
            else (0.0,) * case.input_length
        )
        data.append(ABC6TrainingCaseData(case, tuple(inputs), tuple(observed)))
        incomplete = case.case_index == 14
        particle_failure = case.case_index == 16
        result = _result(case.case_index, incomplete=incomplete)
        results.append(result)
        forecasts.append(
            _forecast(
                case.case_index,
                result,
                incomplete=incomplete,
                particle_failure=particle_failure,
            )
        )
    training = ABC6TrainingBundle(
        tuple(data),
        (
            ("A", TankState(0.5, 0.5)),
            ("B", TankState(0.5, 0.5)),
            ("N", TankState(0.5, 0.5)),
        ),
    )
    return training, tuple(results), tuple(forecasts)


def _write_receipts(directory: Path, results) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for case, result in zip(CASE_ROSTER, results, strict=True):
        write_status_receipt(directory, case.case_index, "fit", result.abc_status)
        write_status_receipt(
            directory, case.case_index, "baseline", result.baseline_status
        )
    _write_training_summary(directory)


def _private_receipt_root(tmp_path: Path) -> Path:
    return tmp_path.resolve() / campaign_fit.RECEIPT_ROOT_RELATIVE


def _private_receipt_identity(receipts: Path) -> campaign_fit.ABC6ReceiptRootIdentity:
    root = receipts.parents[3]
    if receipts != root / campaign_fit.RECEIPT_ROOT_RELATIVE:
        raise AssertionError("fake receipts are outside the fixed private receipt root")
    root_fd = campaign_fit._open_directory_nofollow(
        root, label="private fake checkout root"
    )
    try:
        receipt_fd = campaign_fit._open_relative_directory_nofollow(
            root_fd,
            campaign_fit.RECEIPT_ROOT_RELATIVE,
            label="private fake receipt root",
        )
    except BaseException:
        os.close(root_fd)
        raise
    try:
        identity = campaign_fit.ABC6ReceiptRootIdentity(
            repository_root_realpath=str(root),
            receipt_root_relative=campaign_fit.RECEIPT_ROOT_RELATIVE,
            repository_root_fd=root_fd,
            receipt_root_fd=receipt_fd,
        )
    except BaseException:
        os.close(root_fd)
        os.close(receipt_fd)
        raise
    identity.verify()
    return identity


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _write_training_summary(directory: Path, *, mutate=None) -> Path:
    case_rows = []
    fit_counts = {}
    baseline_counts = {}
    for case in CASE_ROSTER:
        fit_path = directory / f"case-{case.case_index:02d}.fit-status.json"
        baseline_path = directory / f"case-{case.case_index:02d}.baseline-status.json"
        fit_receipt = json.loads(fit_path.read_text("ascii"))
        baseline_receipt = json.loads(baseline_path.read_text("ascii"))
        fit_status = fit_receipt["status"]
        baseline_status = baseline_receipt["status"]
        fit_counts[fit_status] = fit_counts.get(fit_status, 0) + 1
        baseline_counts[baseline_status] = baseline_counts.get(baseline_status, 0) + 1
        case_rows.append(
            {
                "case_index": case.case_index,
                "case_id": case.case_id,
                "fit_status": fit_status,
                "baseline_status": baseline_status,
                "fit_receipt_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                "baseline_receipt_sha256": hashlib.sha256(
                    baseline_path.read_bytes()
                ).hexdigest(),
            }
        )
    status = (
        "complete"
        if all(
            row["fit_status"] == "complete" and row["baseline_status"] == "complete"
            for row in case_rows
        )
        else "incomplete"
    )
    payload = {
        "schema_version": 1,
        "protocol_id": cases.PROTOCOL_ID,
        "run_id": cases.RUN_ID,
        "manifest_sha256": "a" * 64,
        "claim_sha256": "b" * 64,
        "written_at_utc": "2026-09-27T12:00:00Z",
        "status": status,
        "case_count": 24,
        "fit_status_counts": fit_counts,
        "baseline_status_counts": baseline_counts,
        "case_statuses": case_rows,
        "training_only": True,
        "prospective_targets_generated": False,
        "forecasts_run": False,
        "postfit_target_gate_opened": False,
        "external_watchdog_enforced_here": False,
        "independent_manifest_approval_enforced_here": False,
        "resume_allowed": False,
    }
    if mutate is not None:
        mutate(payload)
    payload["payload_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    summary_path = directory / "campaign.training-summary.json"
    summary_path.write_bytes(_canonical_json(payload))
    return summary_path


def _rewrite_status_receipt(
    directory: Path, case_index: int, component: str, updates
) -> None:
    path = directory / f"case-{case_index:02d}.{component}-status.json"
    payload = json.loads(path.read_text("ascii"))
    payload.update(updates)
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    path.write_bytes(_canonical_json(payload))


def _fake_targets(*, bad_hash: bool = False) -> ABC6ProspectiveTargets:
    targets = tuple((name, _FAKE_TARGETS[name]) for name in ("A", "B", "M"))
    hashes = tuple(
        (
            name,
            "0" * 64
            if bad_hash and name == "B"
            else hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest(),
        )
        for name, values in targets
    )
    return ABC6ProspectiveTargets(targets, hashes)


def _fake_simulator(inputs, parameters, initial_state, *, model, ceiling, limits):
    state = TankState(initial_state.x1, initial_state.x2)
    observation_base = parameters.a + 0.1 * initial_state.x1 + (ceiling or 0.0) * 1.0e-4
    trace = []
    for index, input_value in enumerate(inputs):
        value = observation_base + 0.01 * float(input_value)
        trace.append(
            TankStep(
                index=index,
                input_u=float(input_value),
                state=state,
                observation_y=value,
                q12=0.0,
                q2=0.0,
                x1_next=state.x1,
                x2_raw=state.x2,
                next_state=state,
            )
        )
    return TankSimulationSuccess(
        model=model,
        parameters=parameters,
        initial_state=initial_state,
        ceiling=ceiling,
        sample_interval_seconds=4.0,
        trace=tuple(trace),
        terminal_state=state,
    )


def _install_fake_gate(
    monkeypatch,
    marker_path: Path,
    project_root: Path,
    *,
    bad_hash: bool = False,
):
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(scoring, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(scoring, "_REVEAL_MARKER_PATH", marker_path)
    monkeypatch.setattr(cases, "_REVEAL_MARKER_PATH", marker_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", project_root)
    calls = []

    def fake_materialize(training, *, simulator, _scoring_handoff):
        assert marker_path.is_file(), (
            "target generator ran before durable reveal marker"
        )
        assert _scoring_handoff._used is True
        calls.append("generate")
        return _fake_targets(bad_hash=bad_hash)

    monkeypatch.setattr(cases, "_materialize_prospective_targets", fake_materialize)
    return calls


def _install_fake_source_contract(
    monkeypatch, source_root: Path
) -> tuple[tuple[str, str], ...]:
    integrated = scoring._INTEGRATED_SOURCE_PATHS
    required = tuple(dict.fromkeys((*campaign_fit._REQUIRED_SOURCE_PATHS, *integrated)))
    source_root.mkdir(parents=True, exist_ok=True)
    digests = {}
    for relative_path in required:
        source_file = source_root / relative_path
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(
            f"private fake source: {relative_path}\n", encoding="ascii"
        )
        digests[relative_path] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    monkeypatch.setattr(campaign_fit, "_REQUIRED_SOURCE_PATHS", required)
    monkeypatch.setattr(campaign_fit, "_REVIEWED_SOURCE_SHA256", digests)
    monkeypatch.setattr(campaign_fit, "_REPO_ROOT", source_root)
    monkeypatch.setattr(scoring, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(scoring, "_PROJECT_ROOT", source_root)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", source_root)
    monkeypatch.setattr(
        campaign_fit,
        "_current_source_hashes",
        lambda: {
            path: digests.get(path, hashlib.sha256(path.encode()).hexdigest())
            for path in required
        },
    )
    return tuple((path, digests[path]) for path in integrated)


def test_reveal_claim_rejects_preexisting_ancestor_symlink_before_leaf_creation(
    tmp_path, monkeypatch,
) -> None:
    project_root = tmp_path / "private-claim-project"
    project_root.mkdir()
    marker_parent = (
        project_root
        / "artifacts"
        / "evaluations"
        / "cascaded_tanks_abc6_scoring"
        / "claims"
    )
    marker_parent.parent.mkdir(parents=True)
    copied_parent = tmp_path / "copied-private-claims"
    copied_parent.mkdir()
    marker_parent.symlink_to(copied_parent, target_is_directory=True)
    marker_path = marker_parent / f"{cases.RUN_ID}.claim"
    monkeypatch.setattr(cases, "_REVEAL_MARKER_PATH", marker_path)
    root_anchor = cases._open_directory_anchor(
        project_root, label="private fake checkout root"
    )

    try:
        with pytest.raises(scoring.ABC6ScoringError, match="ancestry.*symlink"):
            scoring._claim_reveal_marker(
                marker_path,
                {"run_id": "private-fake-run"},
                root_anchor=root_anchor,
            )
    finally:
        root_anchor.close()

    assert not marker_path.exists()
    assert not (copied_parent / marker_path.name).exists()


def _fake_execution(receipts: Path, training, results):
    statuses = []
    for case, result in zip(CASE_ROSTER, results, strict=True):
        fit_path = receipts / f"case-{case.case_index:02d}.fit-status.json"
        baseline_path = receipts / f"case-{case.case_index:02d}.baseline-status.json"
        statuses.append(
            campaign_fit.ABC6CaseStatus(
                case_index=case.case_index,
                case_id=case.case_id,
                fit_status=result.abc_status,
                baseline_status=result.baseline_status,
                fit_receipt_sha256=(
                    hashlib.sha256(fit_path.read_bytes()).hexdigest()
                    if fit_path.exists()
                    else "0" * 64
                ),
                baseline_receipt_sha256=(
                    hashlib.sha256(baseline_path.read_bytes()).hexdigest()
                    if baseline_path.exists()
                    else "1" * 64
                ),
            )
        )
    summary_path = receipts / "campaign.training-summary.json"
    summary_sha256 = (
        hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if summary_path.exists()
        else "2" * 64
    )
    overall_status = (
        "complete"
        if all(
            result.abc_status == "complete" and result.baseline_status == "complete"
            for result in results
        )
        else "incomplete"
    )
    campaign_result = campaign_fit.ABC6CampaignResult(
        protocol_id=cases.PROTOCOL_ID,
        run_id=cases.RUN_ID,
        manifest_sha256="a" * 64,
        claim_sha256="b" * 64,
        status=overall_status,
        case_statuses=tuple(statuses),
        summary_path=summary_path,
        summary_sha256=summary_sha256,
    )
    return campaign_fit.ABC6TrainingCampaignExecution(
        campaign_result=campaign_result,
        evidence_manifest_path=(receipts / campaign_fit.EVIDENCE_MANIFEST_FILENAME),
        evidence_manifest_sha256="c" * 64,
        receipt_root_identity=_private_receipt_identity(receipts),
        _training_bundle=training,
        _training_results=tuple(results),
    )


def _prepare_private_execution_call(
    monkeypatch,
    receipts: Path,
    training,
    results,
    forecasts,
    *,
    frozen=None,
):
    source_root = receipts.parents[len(Path(campaign_fit.RECEIPT_ROOT_RELATIVE).parts) - 1]
    source_hashes = _install_fake_source_contract(monkeypatch, source_root)
    execution = _fake_execution(receipts, training, results)

    def fake_load_verified(self):
        assert self is execution
        return campaign_fit.ABC6VerifiedTrainingEvidence(
            training_bundle=training,
            training_results=tuple(results),
        )

    monkeypatch.setattr(
        campaign_fit.ABC6TrainingCampaignExecution,
        "load_verified_training_evidence",
        fake_load_verified,
    )
    monkeypatch.setattr(
        scoring,
        "forecast_abc6_posterior_and_baseline",
        lambda data, result, *, simulator: forecasts[data.case.case_index],
    )
    frozen_roster = (
        scoring.freeze_abc6_forecasts(forecasts) if frozen is None else frozen
    )
    receipt_hashes = tuple(
        (f"case-{case.case_index:02d}.{component}-status.json", digest)
        for case, status in zip(CASE_ROSTER, execution.campaign_result.case_statuses)
        for component, digest in (
            ("fit", status.fit_receipt_sha256),
            ("baseline", status.baseline_receipt_sha256),
        )
    )
    summary_path = execution.campaign_result.summary_path
    summary_sha256 = (
        hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if summary_path.exists()
        else execution.campaign_result.summary_sha256
    )
    payload = scoring._expected_forecast_artifact_payload(
        campaign_result=execution.campaign_result,
        evidence_manifest_sha256=execution.evidence_manifest_sha256,
        source_hashes=source_hashes,
        receipt_hashes=receipt_hashes,
        summary_sha256=summary_sha256,
        frozen=frozen_roster,
    )
    artifact_path = receipts / scoring._FORECAST_ARTIFACT_FILENAME
    receipts.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(payload)
    return execution, frozen_roster, artifact_path, hashlib.sha256(payload).hexdigest()


def _private_score(
    monkeypatch,
    receipts,
    training,
    results,
    forecasts,
    *,
    frozen=None,
    simulator=_fake_simulator,
):
    prepared = _prepare_private_execution_call(
        monkeypatch,
        receipts,
        training,
        results,
        forecasts,
        frozen=frozen,
    )
    return scoring.score_deferred_abc6_synthetic(
        *prepared,
        simulator=simulator,
    )


def _fake_marker_paths(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path.resolve()
    marker_path = (
        project_root
        / "artifacts"
        / "evaluations"
        / "cascaded_tanks_abc6_scoring"
        / "claims"
        / f"{cases.RUN_ID}.claim"
    )
    return project_root, marker_path


def test_pre_marker_failure_never_calls_target_generator_or_claims_marker(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    frozen = scoring.freeze_abc6_forecasts(forecasts)
    malformed = replace(
        frozen,
        case_sha256=("0" * 64,) + frozen.case_sha256[1:],
    )

    with pytest.raises(scoring.ABC6ScoringError, match="forecast hash"):
        _private_score(
            monkeypatch,
            receipts,
            training,
            results,
            forecasts,
            frozen=malformed,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_bare_mutated_posterior_and_matching_forecast_are_rejected_pre_marker(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    # Reproduce the former bypass: alter posterior a from .50 to .59 and make
    # a self-consistent forecast that agrees with that caller-owned posterior.
    changed_results = list(results)
    changed_result = SimpleNamespace(**vars(results[0]))
    changed_abc = dict(changed_result.abc_result)
    changed_posterior = dict(changed_abc["posterior"])
    changed_columns = dict(changed_posterior["free_parameter_values"])
    changed_a = np.asarray(changed_columns["a"], dtype=float).copy()
    assert changed_a[0] == 0.50
    changed_a[0] = 0.59
    changed_columns["a"] = changed_a
    changed_posterior["free_parameter_values"] = changed_columns
    changed_abc["posterior"] = changed_posterior
    changed_result.abc_result = changed_abc
    changed_results[0] = changed_result

    changed_forecasts = list(forecasts)
    changed_forecast = changed_forecasts[0]
    changed_particles = list(changed_forecast.particles)
    changed_particle = changed_particles[0]
    changed_particles[0] = replace(
        changed_particle,
        parameter_values=(0.59, *changed_particle.parameter_values[1:]),
    )
    changed_forecasts[0] = replace(
        changed_forecast,
        particles=tuple(changed_particles),
        baseline_parameter_values=(
            0.59,
            *changed_forecast.baseline_parameter_values[1:],
        ),
    )
    frozen = scoring.freeze_abc6_forecasts(changed_forecasts)

    # The old receipt/bundle/results/forecast call shape is now rejected at
    # the public boundary, before loading evidence, claiming, or revealing.
    with pytest.raises(
        TypeError, match="execution must be an ABC6TrainingCampaignExecution"
    ):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            tuple(changed_results),
            frozen,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_current_campaign_source_allowlist_fails_closed_before_loading_or_reveal(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    execution = _fake_execution(receipts, training, results)
    frozen = scoring.freeze_abc6_forecasts(forecasts)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    loader_calls = []
    monkeypatch.setattr(
        campaign_fit.ABC6TrainingCampaignExecution,
        "load_verified_training_evidence",
        lambda self: loader_calls.append(self),
    )

    with pytest.raises(scoring.ABC6ScoringError, match="must allowlist and review"):
        scoring.score_deferred_abc6_synthetic(
            execution,
            frozen,
            receipts / scoring._FORECAST_ARTIFACT_FILENAME,
            "d" * 64,
            simulator=_fake_simulator,
        )

    assert loader_calls == []
    assert calls == []
    assert not marker.exists()


def test_mismatched_checkout_identity_fails_before_marker_or_target_generation(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    execution, frozen, artifact_path, artifact_sha256 = _prepare_private_execution_call(
        monkeypatch, receipts, training, results, forecasts
    )
    original_identity = execution.receipt_root_identity
    foreign_root = tmp_path / "copied-checkout-root"
    foreign_receipts = foreign_root / campaign_fit.RECEIPT_ROOT_RELATIVE
    foreign_receipts.mkdir(parents=True)
    foreign_identity = _private_receipt_identity(foreign_receipts)
    object.__setattr__(execution, "receipt_root_identity", foreign_identity)

    try:
        with pytest.raises(
            scoring.ABC6ScoringError, match="differs from the typed campaign root"
        ):
            scoring.score_deferred_abc6_synthetic(
                execution,
                frozen,
                artifact_path,
                artifact_sha256,
                simulator=_fake_simulator,
            )
    finally:
        original_identity.close()
        foreign_identity.close()

    assert calls == []
    assert not marker.exists()


def test_missing_status_roster_fails_before_marker_or_target_generation(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    missing_receipts = _private_receipt_root(tmp_path)
    missing_receipts.mkdir(parents=True)

    with pytest.raises(cases.ABC6StatusReceiptError, match="missing durable receipt"):
        _private_score(
            monkeypatch,
            missing_receipts,
            training,
            results,
            forecasts,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_success_scores_fake_targets_and_emits_protocol_diagnostics(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    output = _private_score(
        monkeypatch,
        receipts,
        training,
        results,
        forecasts,
        simulator=_fake_simulator,
    )

    assert marker.is_file()
    assert calls == ["generate"]
    assert output.protocol_id == cases.PROTOCOL_ID
    assert output.run_id == cases.RUN_ID
    assert tuple(name for name, _digest in output.target_sha256_by_truth) == (
        "A",
        "B",
        "M",
    )
    assert len(output.case_scores) == 24
    assert len(output.paired_horizon_contrasts) == 8
    assert len(output.parameter_inclusion_counts) == 24
    assert len(output.no_crossing_diagnostics) == 2
    assert len(output.m_training_residual_means) == 2
    assert output.model_choice is None
    assert output.bayes_factors is None

    n_scores = [score for score in output.case_scores if score.truth_id == "N"]
    assert len(n_scores) == 2
    assert all(
        score.abc_weighted_mean_rmse is None
        and score.abc_weighted_median_rmse is None
        and score.baseline_coherent_rmse is None
        and score.abc_q05_q95_envelope_inclusion_fraction is None
        for score in n_scores
    )
    assert [item.mechanism_abstention for item in output.no_crossing_diagnostics] == [
        True,
        True,
    ]
    assert [item.ceiling_q05 for item in output.no_crossing_diagnostics] == [
        950.0,
        950.0,
    ]
    assert [item.ceiling_q50 for item in output.no_crossing_diagnostics] == [
        1050.0,
        1050.0,
    ]
    assert [item.ceiling_q95 for item in output.no_crossing_diagnostics] == [
        1050.0,
        1050.0,
    ]

    failed_pair = next(
        item
        for item in output.paired_horizon_contrasts
        if item.truth_id == "A" and item.replicate == 2
    )
    assert failed_pair.abc_weighted_mean_rmse_s_minus_l is None
    assert failed_pair.abc_weighted_median_rmse_s_minus_l is None
    assert failed_pair.baseline_coherent_rmse_s_minus_l is None
    particle_failure_pair = next(
        item
        for item in output.paired_horizon_contrasts
        if item.truth_id == "A" and item.replicate == 3
    )
    assert particle_failure_pair.abc_weighted_mean_rmse_s_minus_l is None
    assert particle_failure_pair.abc_weighted_median_rmse_s_minus_l is None
    assert particle_failure_pair.baseline_coherent_rmse_s_minus_l is not None

    a_short_count = next(
        item
        for item in output.parameter_inclusion_counts
        if item.truth_id == "A" and item.input_window == "S" and item.parameter == "a"
    )
    assert (a_short_count.included_count, a_short_count.eligible_fit_count) == (3, 3)
    assert a_short_count.interpretation == "descriptive_not_calibrated_coverage"
    assert all(
        item.truth_inclusion_claim is False for item in output.m_training_residual_means
    )
    assert all(
        value is not None
        for item in output.m_training_residual_means
        for value in (
            item.baseline_u8_first_24_mean,
            item.baseline_u0_last_180_mean,
            item.abc_weighted_mean_u8_first_24_mean,
            item.abc_weighted_mean_u0_last_180_mean,
        )
    )


def test_durable_forecast_artifact_source_tampering_fails_before_marker(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    prepared = _prepare_private_execution_call(
        monkeypatch, receipts, training, results, forecasts
    )
    execution, frozen, artifact_path, _digest = prepared
    payload = json.loads(artifact_path.read_text("ascii"))
    payload["integrated_source_hashes"][0]["sha256"] = "f" * 64
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = scoring._sha256(scoring._canonical_json(body))
    tampered = scoring._canonical_json(payload)
    artifact_path.write_bytes(tampered)
    tampered_digest = scoring._sha256(tampered)

    with pytest.raises(scoring.ABC6ScoringError, match="does not bind"):
        scoring.score_deferred_abc6_synthetic(
            execution,
            frozen,
            artifact_path,
            tampered_digest,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_post_marker_failure_is_consumed_and_second_call_cannot_regenerate(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root, bad_hash=True)
    frozen = scoring.freeze_abc6_forecasts(forecasts)
    prepared = _prepare_private_execution_call(
        monkeypatch, receipts, training, results, forecasts, frozen=frozen
    )

    with pytest.raises(scoring.ABC6DeferredScoreConsumedError) as first:
        scoring.score_deferred_abc6_synthetic(
            *prepared,
            simulator=_fake_simulator,
        )
    assert first.value.condition_consumed is True
    assert first.value.retry_forbidden is True
    assert marker.is_file()
    assert calls == ["generate"]
    with pytest.raises(
        scoring.ABC6RevealAlreadyConsumedError, match="cannot be retried"
    ):
        scoring.score_deferred_abc6_synthetic(
            *prepared,
            simulator=_fake_simulator,
        )
    assert calls == ["generate"]


def test_fifo_marker_readback_is_bounded_nonblocking_and_consumes_claim(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    materializer_calls = _install_fake_gate(monkeypatch, marker, project_root)
    prepared = _prepare_private_execution_call(
        monkeypatch, receipts, training, results, forecasts
    )
    original_open = os.open
    readback_flags = []

    def replace_marker_with_fifo(path, flags, mode=0o777, *, dir_fd=None):
        safe_flags = flags
        if (
            path == marker.name
            and dir_fd is not None
            and not flags & os.O_CREAT
            and not flags & os.O_WRONLY
        ):
            readback_flags.append(flags)
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, mode=0o600, dir_fd=dir_fd)
            # Keep a regression without O_NONBLOCK from hanging the test
            # process. The assertion below still verifies the production flag.
            safe_flags |= os.O_NONBLOCK
        return original_open(path, safe_flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(scoring.os, "open", replace_marker_with_fifo)
    with pytest.raises(
        scoring.ABC6ScoringError, match="could not read back the durable reveal marker"
    ):
        scoring.score_deferred_abc6_synthetic(*prepared, simulator=_fake_simulator)

    assert readback_flags
    assert all(flags & os.O_NONBLOCK for flags in readback_flags)
    assert stat.S_ISFIFO(marker.lstat().st_mode)
    assert materializer_calls == []

    with pytest.raises(
        scoring.ABC6RevealAlreadyConsumedError, match="cannot be retried"
    ):
        scoring.score_deferred_abc6_synthetic(*prepared, simulator=_fake_simulator)
    assert materializer_calls == []


def test_claim_parent_swap_after_marker_is_terminal_and_never_materializes(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    original_claim = scoring._claim_reveal_marker
    saved_parent = marker.parent.with_name("claims-before-swap")
    copied_parent = tmp_path / "copied-claims-parent"

    def claim_then_redirect(path, payload, *, root_anchor):
        claim = original_claim(path, payload, root_anchor=root_anchor)
        shutil.copytree(marker.parent, copied_parent)
        marker.parent.rename(saved_parent)
        marker.parent.symlink_to(copied_parent, target_is_directory=True)
        return claim

    monkeypatch.setattr(scoring, "_claim_reveal_marker", claim_then_redirect)
    with pytest.raises(scoring.ABC6DeferredScoreConsumedError) as first:
        _private_score(
            monkeypatch,
            receipts,
            training,
            results,
            forecasts,
            simulator=_fake_simulator,
        )

    assert first.value.condition_consumed is True
    assert first.value.retry_forbidden is True
    assert calls == []
    assert (saved_parent / marker.name).is_file()
    assert (copied_parent / marker.name).is_file()


def test_forecast_identity_and_weights_are_checked_before_marker(tmp_path, monkeypatch):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    altered = list(forecasts)
    altered[0] = replace(altered[0], weights=(0.5, 0.5))
    # Freeze an altered target-free forecast. Its self-hash is valid, but its
    # weights no longer agree with the durable fit result and must fail closed.
    frozen = scoring.freeze_abc6_forecasts(altered)

    with pytest.raises(scoring.ABC6ScoringError, match="weights"):
        _private_score(
            monkeypatch,
            receipts,
            training,
            results,
            forecasts,
            frozen=frozen,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"case_index": True}, "integer fields"),
        ({"case_id": "wrong-case-id"}, "identity mismatch"),
    ),
)
def test_status_receipt_exact_integer_types_and_roster_ids_fail_closed(
    tmp_path, monkeypatch, updates, message
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    _rewrite_status_receipt(receipts, 23, "baseline", updates)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    with pytest.raises(cases.ABC6StatusReceiptError, match=message):
        _private_score(
            monkeypatch,
            receipts,
            training,
            results,
            forecasts,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda summary: summary.update(case_count=True), "case_count"),
        (
            lambda summary: summary["case_statuses"][7].update(case_id="case-06-wrong"),
            "row 7 identity",
        ),
    ),
)
def test_training_summary_exact_types_and_ordered_case_ids_fail_before_marker(
    tmp_path, monkeypatch, mutate, message
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    _write_training_summary(receipts, mutate=mutate)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    with pytest.raises(scoring.ABC6ScoringError, match=message):
        _private_score(
            monkeypatch,
            receipts,
            training,
            results,
            forecasts,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_symlinked_fixed_root_ancestor_is_rejected_before_exclusive_claim(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    prepared = _prepare_private_execution_call(
        monkeypatch, receipts, training, results, forecasts
    )
    project_root.mkdir(parents=True, exist_ok=True)
    artifacts = project_root / "artifacts"
    redirect = tmp_path / "redirected-artifacts"
    shutil.copytree(artifacts, redirect)
    saved_artifacts = tmp_path / "original-artifacts"
    artifacts.rename(saved_artifacts)
    artifacts.symlink_to(redirect, target_is_directory=True)

    with pytest.raises(scoring.ABC6ScoringError, match="identity changed"):
        scoring.score_deferred_abc6_synthetic(
            *prepared,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()
    assert not (redirect / "evaluations").exists()


@pytest.mark.parametrize(
    ("identity_path", "message"),
    (
        ("result.case_index", "training result case_index"),
        ("forecast.case_index", "forecast case_index"),
        ("particle.particle_index", "particle_index"),
    ),
)
def test_boolean_case_and_particle_ids_are_rejected_before_marker(
    tmp_path, monkeypatch, identity_path, message
):
    training, results, forecasts = _fixture_rosters()
    original_results = results
    receipts = _private_receipt_root(tmp_path)
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    frozen = scoring.freeze_abc6_forecasts(forecasts)

    if identity_path == "result.case_index":
        changed_results = list(results)
        changed_results[1] = SimpleNamespace(**vars(results[1]))
        changed_results[1].case_index = True
        results = tuple(changed_results)
    elif identity_path == "forecast.case_index":
        changed_forecasts = list(frozen.forecasts)
        changed_forecasts[1] = replace(changed_forecasts[1], case_index=True)
        frozen = replace(frozen, forecasts=tuple(changed_forecasts))
    else:
        changed_forecasts = list(forecasts)
        target = changed_forecasts[1]
        changed_particles = tuple(
            replace(particle, particle_index=True)
            if particle.particle_index == 1
            else particle
            for particle in target.particles
        )
        changed_forecasts[1] = replace(target, particles=changed_particles)
        frozen = scoring.freeze_abc6_forecasts(changed_forecasts)

    if identity_path == "result.case_index":
        prepared = _prepare_private_execution_call(
            monkeypatch, receipts, training, original_results, forecasts, frozen=frozen
        )
        monkeypatch.setattr(
            campaign_fit.ABC6TrainingCampaignExecution,
            "load_verified_training_evidence",
            lambda _self: campaign_fit.ABC6VerifiedTrainingEvidence(
                training_bundle=training,
                training_results=results,
            ),
        )
        with pytest.raises(scoring.ABC6ScoringError, match=message):
            scoring.score_deferred_abc6_synthetic(
                *prepared,
                simulator=_fake_simulator,
            )
    else:
        with pytest.raises(scoring.ABC6ScoringError, match=message):
            _private_score(
                monkeypatch,
                receipts,
                training,
                results,
                forecasts,
                frozen=frozen,
                simulator=_fake_simulator,
            )

    assert calls == []
    assert not marker.exists()

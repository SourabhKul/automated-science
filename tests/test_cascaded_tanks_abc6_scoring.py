"""Private synthetic-only fixtures for the deferred ABC6 scoring seam."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
    project_root.mkdir(exist_ok=True)
    monkeypatch.setattr(scoring, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(scoring, "_REVEAL_MARKER_PATH", marker_path)
    calls = []

    def fake_materialize(training, *, simulator):
        assert marker_path.is_file(), (
            "target generator ran before durable reveal marker"
        )
        calls.append("generate")
        return _fake_targets(bad_hash=bad_hash)

    monkeypatch.setattr(cases, "_materialize_prospective_targets", fake_materialize)
    return calls


def _fake_marker_paths(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "private-fake-project"
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
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    frozen = scoring.freeze_abc6_forecasts(forecasts)
    malformed = replace(
        frozen,
        case_sha256=("0" * 64,) + frozen.case_sha256[1:],
    )

    with pytest.raises(scoring.ABC6ScoringError, match="forecast hash"):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            malformed,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_missing_status_roster_fails_before_marker_or_target_generation(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    with pytest.raises(cases.ABC6StatusReceiptError, match="does not exist"):
        scoring.score_deferred_abc6_synthetic(
            tmp_path / "missing-receipts",
            training,
            results,
            scoring.freeze_abc6_forecasts(forecasts),
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_success_scores_fake_targets_and_emits_protocol_diagnostics(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    output = scoring.score_deferred_abc6_synthetic(
        receipts,
        training,
        results,
        scoring.freeze_abc6_forecasts(forecasts),
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


def test_post_marker_failure_is_consumed_and_second_call_cannot_regenerate(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root, bad_hash=True)
    frozen = scoring.freeze_abc6_forecasts(forecasts)

    with pytest.raises(scoring.ABC6DeferredScoreConsumedError) as first:
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            frozen,
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
            receipts,
            training,
            results,
            frozen,
            simulator=_fake_simulator,
        )
    assert calls == ["generate"]


def test_forecast_identity_and_weights_are_checked_before_marker(tmp_path, monkeypatch):
    training, results, forecasts = _fixture_rosters()
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)
    altered = list(forecasts)
    altered[0] = replace(altered[0], weights=(0.5, 0.5))
    # Freeze an altered target-free forecast. Its self-hash is valid, but its
    # weights no longer agree with the durable fit result and must fail closed.
    frozen = scoring.freeze_abc6_forecasts(altered)

    with pytest.raises(scoring.ABC6ScoringError, match="weights"):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            frozen,
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
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    _rewrite_status_receipt(receipts, 23, "baseline", updates)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    with pytest.raises(cases.ABC6StatusReceiptError, match=message):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            scoring.freeze_abc6_forecasts(forecasts),
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
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    _write_training_summary(receipts, mutate=mutate)
    project_root, marker = _fake_marker_paths(tmp_path)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    with pytest.raises(scoring.ABC6ScoringError, match=message):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            scoring.freeze_abc6_forecasts(forecasts),
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()


def test_symlinked_fixed_root_ancestor_is_rejected_before_exclusive_claim(
    tmp_path, monkeypatch
):
    training, results, forecasts = _fixture_rosters()
    receipts = tmp_path / "receipts"
    _write_receipts(receipts, results)
    project_root, marker = _fake_marker_paths(tmp_path)
    project_root.mkdir()
    redirect = tmp_path / "redirected-artifacts"
    redirect.mkdir()
    (project_root / "artifacts").symlink_to(redirect, target_is_directory=True)
    calls = _install_fake_gate(monkeypatch, marker, project_root)

    with pytest.raises(scoring.ABC6ScoringError, match="ancestry.*symlink"):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            scoring.freeze_abc6_forecasts(forecasts),
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
    receipts = tmp_path / "receipts"
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

    with pytest.raises(scoring.ABC6ScoringError, match=message):
        scoring.score_deferred_abc6_synthetic(
            receipts,
            training,
            results,
            frozen,
            simulator=_fake_simulator,
        )

    assert calls == []
    assert not marker.exists()

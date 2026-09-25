from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from core.real_data.silverbox_controlled import (
    SAMPLE_TIME_SECONDS,
    TRAIN_SOURCE_START,
    TRAIN_WINDOW_LENGTH,
    TRAIN_WINDOW_RELATIVE_STARTS,
    VALIDATION_SOURCE_START,
    VALIDATION_SOURCE_STOP,
    SilverboxControlledSeries,
)
import core.real_data.silverbox_first_fit as first_fit


def _train_windows() -> tuple[SilverboxControlledSeries, ...]:
    windows = []
    for index, relative_start in enumerate(TRAIN_WINDOW_RELATIVE_STARTS):
        start = TRAIN_SOURCE_START + relative_start
        samples = np.arange(TRAIN_WINDOW_LENGTH, dtype=float)
        input_u = 0.015 + 0.002 * index + 0.00001 * samples
        observed_y = 0.04 + 0.004 * index + 0.00002 * samples
        windows.append(
            SilverboxControlledSeries(
                role="train",
                source_start=start,
                source_stop=start + TRAIN_WINDOW_LENGTH,
                input_u=input_u,
                initialization_y=observed_y[:50],
                target_y=observed_y[50:],
                sampling_time=SAMPLE_TIME_SECONDS,
            )
        )
    return tuple(windows)


def _validation_series(target_value: float = 0.47) -> SilverboxControlledSeries:
    count = VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START
    input_u = np.full(count, 0.02, dtype=float)
    observed_y = np.concatenate((np.full(50, 0.10), np.full(count - 50, target_value)))
    return SilverboxControlledSeries(
        role="validation",
        source_start=VALIDATION_SOURCE_START,
        source_stop=VALIDATION_SOURCE_STOP,
        input_u=input_u,
        initialization_y=observed_y[:50],
        target_y=observed_y[50:],
        sampling_time=SAMPLE_TIME_SECONDS,
    )


def _complete_fake_abc(prior_sampler, simulator, discrepancy, **kwargs):
    """Return two complete populations and exercise the callbacks/simulator."""
    expected = {
        "target_samples": 64,
        "max_attempts_per_population": 512,
        "covariance_scale": 2.0,
        "lambda_noise": 0.01,
        "nugget": 1e-9,
    }
    for key, value in expected.items():
        assert kwargs[key] == value
    dimension = len(kwargs["bounds"])
    expected_bounds = first_fit.LINEAR_BOUNDS if dimension == 4 else first_fit.NONLINEAR_BOUNDS
    np.testing.assert_array_equal(kwargs["bounds"], expected_bounds)
    assert kwargs["seed"] == (26092511 if dimension == 4 else 26092512)
    assert len(kwargs["epsilon_schedule"]) == 2
    assert kwargs["epsilon_schedule"][0] >= kwargs["epsilon_schedule"][1]
    assert kwargs["prior_logpdf"](np.mean(np.asarray(expected_bounds), axis=1)) > float("-inf")
    assert kwargs["prior_logpdf"](np.full(dimension, 100.0)) == float("-inf")

    probe_rng = np.random.default_rng(42)
    probe = np.asarray(prior_sampler(probe_rng), dtype=float)
    discrepancy(simulator(probe, probe_rng))

    populations = []
    bounds = np.asarray(kwargs["bounds"], dtype=float)
    accepted = np.tile(np.mean(bounds, axis=1), (64, 1))
    weights = np.full(64, 1.0 / 64.0)
    for generation, epsilon in enumerate(kwargs["epsilon_schedule"]):
        kwargs["population_event"]("start", generation, None)
        population = {
            "generation": generation,
            "epsilon": float(epsilon),
            "accepted_params": accepted.copy(),
            "distances": np.full(64, float(epsilon)),
            "weights": weights.copy(),
            "log_weights": np.log(weights),
            "log_prior_density": np.full(64, np.nan),
            "log_proposal_mixture_density": np.full(64, np.nan),
            "proposal_covariance": None if generation == 0 else np.eye(dimension),
            "effective_sample_size": 64.0,
            "diagnostics": {
                "proposed": 64,
                "simulated": 64,
                "accepted": 64,
                "out_of_support": 0,
                "failed_prior_draws": 0,
                "failed_proposals": 0,
                "failed_simulations": 0,
                "failed_discrepancies": 0,
                "weight_failures": 0,
                "max_attempts": 512,
                "complete": True,
                "termination_reason": "target_reached",
                "ancestor_indices": [None] * 64 if generation == 0 else [0] * 64,
            },
        }
        kwargs["population_event"]("end", generation, population)
        populations.append(population)
    return {
        "status": "complete",
        "complete": True,
        "termination_reason": "completed",
        "reference_path": "gaussian_abc_smc_reference_opt_in",
        "target_samples": 64,
        "epsilon_schedule": kwargs["epsilon_schedule"],
        "max_attempts_per_population": [512, 512],
        "seed": kwargs["seed"],
        "populations": populations,
    }


def test_train_only_scalers_use_fixed_target_and_observed_lag_indices():
    windows = _train_windows()
    scales = first_fit.train_only_scalers(windows, "y_cubed")
    all_targets = np.concatenate([window.target_y for window in windows])
    expected_sy = np.sqrt(np.mean(all_targets**2))
    observed_cubes = []
    for window in windows:
        observed = np.concatenate((window.initialization_y, window.target_y))
        observed_cubes.append(observed[49:255] ** 3)
    expected_sf = np.sqrt(np.mean(np.concatenate(observed_cubes) ** 2))
    assert scales["training_target_count"] == 824
    assert scales["s_y"] == pytest.approx(expected_sy)
    assert scales["s_feature"] == pytest.approx(expected_sf)
    assert scales["divergence_bound"] == pytest.approx(1.0)


def test_recurrence_uses_k_minus_one_input_and_free_run_output_history():
    u = np.full(52, 0.25)
    initializer = np.zeros(50)
    initializer[48:50] = [3.0, 2.0]
    linear = first_fit.simulate_controlled_ar2(
        u,
        initializer,
        [1.0, 1.0, 2.0, 0.5],
        hypothesis="linear",
        term_id="u_y_product",
        s_y=2.0,
        s_feature=4.0,
        divergence_bound=100.0,
    )
    # k=50 uses y[49]=2, y[48]=3 and u[49]=0.25.
    assert linear[0] == pytest.approx(2.0 + 3.0 + 0.5 + 0.5)
    # k=51 uses the generated prediction at k=50 and the recorded u[50].
    assert linear[1] == pytest.approx(linear[0] + 2.0 + 0.5 + 0.5)

    nonlinear = first_fit.simulate_controlled_ar2(
        u,
        initializer,
        [1.0, 1.0, 2.0, 0.5, 1.0],
        hypothesis="nonlinear",
        term_id="u_y_product",
        s_y=2.0,
        s_feature=4.0,
        divergence_bound=100.0,
    )
    expected_extra = 1.0 * 2.0 * (0.25 * 2.0) / 4.0
    assert nonlinear[0] == pytest.approx(linear[0] + expected_extra)


def test_simulator_failure_is_explicit_and_never_clips():
    with pytest.raises(first_fit.SimulationFailure, match="wrong_parameter_shape"):
        first_fit.simulate_controlled_ar2(
            np.zeros(51), np.zeros(50), [1.0], hypothesis="linear", term_id="y_cubed",
            s_y=1.0, s_feature=1.0, divergence_bound=1.0,
        )
    with pytest.raises(first_fit.SimulationFailure, match="divergence_bound_exceeded"):
        first_fit.simulate_controlled_ar2(
            np.zeros(51), np.ones(50), [1.0, 1.0, 0.0, 0.0], hypothesis="linear", term_id="y_cubed",
            s_y=1.0, s_feature=1.0, divergence_bound=1.0,
        )


def test_pointwise_weighted_median_uses_stable_first_cumulative_half():
    trajectories = np.asarray([[0.0, 3.0], [2.0, 1.0], [4.0, 2.0]])
    weights = np.asarray([0.25, 0.50, 0.25])
    # At the first time, the middle particle lands exactly on cumulative 0.5;
    # at the second time, the first sorted particle carries exactly 0.5.
    np.testing.assert_array_equal(first_fit.pointwise_weighted_median(trajectories, weights), [2.0, 1.0])
    tie_paths = np.asarray([[1.0], [1.0], [9.0]])
    np.testing.assert_array_equal(first_fit.pointwise_weighted_median(tie_paths, [0.25, 0.25, 0.5]), [1.0])


def test_fit_api_is_explicitly_train_only_and_requires_upstream_term_id():
    parameters = inspect.signature(first_fit.fit_silverbox_development).parameters
    assert "development" not in parameters
    assert "validation_series" not in parameters
    assert parameters["term_id"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="no fallback"):
        first_fit.fit_silverbox_development(
            _train_windows(), "", source_sha256="a" * 64, proposal_receipt_sha256="b" * 64,
            run_id="invalid", receipt_dir="unused"
        )
    with pytest.raises(TypeError, match="proposal_receipt_sha256"):
        first_fit.fit_silverbox_development(
            _train_windows(), "y_cubed", source_sha256="a" * 64, run_id="missing-provenance", receipt_dir="unused"
        )
    with pytest.raises(ValueError, match="lowercase"):
        first_fit.fit_silverbox_development(
            _train_windows(), "y_cubed", source_sha256="a" * 64, proposal_receipt_sha256="B" * 64,
            run_id="invalid-provenance", receipt_dir="unused"
        )


def test_calibration_and_reference_call_contract_and_separate_validation_selection(monkeypatch, tmp_path):
    windows = _train_windows()

    def fake_simulator(input_u, initialization_y, parameters, *, hypothesis, term_id, s_y, s_feature, divergence_bound):
        value = 0.50 if hypothesis == "linear" else 0.47
        return np.full(len(input_u) - 50, value)

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", fake_simulator)
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", _complete_fake_abc)
    fit = first_fit.fit_silverbox_development(
        windows,
        "y_cubed",
        source_sha256=hashlib.sha256(b"synthetic-only").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-proposal-receipt").hexdigest(),
        run_id="synthetic-complete",
        receipt_dir=tmp_path,
    )
    assert fit.status == "complete"
    fit_payload = json.loads(fit.receipt_path.read_text())
    assert fit_payload["proposal_receipt_sha256"] == fit.proposal_receipt_sha256
    assert fit_payload["selection"]["validation_accessed"] is False
    assert "validation_target_sha256" not in fit_payload
    for hypothesis in ("linear", "nonlinear"):
        calibration = fit_payload["hypotheses"][hypothesis]["calibration"]
        assert calibration["draws_completed"] == 256
        assert calibration["finite_count"] == 256
        assert calibration["quantile_method"] == "linear"
        finite = np.asarray([row["discrepancy"] for row in calibration["calibration_draws"]])
        expected = np.quantile(finite, [0.5, 0.2], method="linear")
        np.testing.assert_allclose(calibration["epsilon_schedule"], expected)
        assert len(fit_payload["hypotheses"][hypothesis]["abc_result"]["populations"]) == 2
        for generation in range(2):
            assert (fit.receipt_path.parent / f"{hypothesis}_population_{generation:02d}.json").exists()

    selection = first_fit.select_silverbox_development(fit, _validation_series())
    assert selection.status == "selected"
    assert selection.selected_hypothesis == "nonlinear"
    selection_payload = json.loads(selection.receipt_path.read_text())
    assert selection_payload["validation_accessed"] is True
    assert set(selection_payload["forecasts"]) == {"linear", "nonlinear"}
    assert selection_payload["scores"]["rmse"]["nonlinear"] == pytest.approx(0.0)
    assert selection_payload["scores"]["rmse"]["linear"] > 0.0
    assert selection_payload["scores"]["persistence_rmse"] > 0.0


def test_incomplete_fit_writes_unresolved_receipt_without_reading_validation(monkeypatch, tmp_path):
    def fake_incomplete_abc(prior_sampler, simulator, discrepancy, **kwargs):
        return {
            "status": "incomplete",
            "complete": False,
            "termination_reason": "attempt_budget_exhausted",
            "populations": [],
        }

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", lambda input_u, initialization_y, parameters, **kwargs: np.zeros(len(input_u) - 50))
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", fake_incomplete_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "u_cubed",
        source_sha256=hashlib.sha256(b"synthetic-incomplete").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-incomplete-proposal").hexdigest(),
        run_id="synthetic-incomplete",
        receipt_dir=tmp_path,
    )
    assert fit.status == "incomplete"
    result = first_fit.select_silverbox_development(fit, object())  # must return before touching it
    assert result.status == "unresolved"
    payload = json.loads(result.receipt_path.read_text())
    assert payload["reason"] == "fit_incomplete"
    assert payload["validation_accessed"] is False
    assert payload["scores"] is None


def test_calibration_failure_count_and_finite_draw_gate_are_retained(monkeypatch, tmp_path):
    def failed_simulator(*args, **kwargs):
        raise first_fit.SimulationFailure("divergence_bound_exceeded", "synthetic divergence")

    def forbidden_abc(*args, **kwargs):
        raise AssertionError("ABC must not run after the finite-calibration gate fails")

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", failed_simulator)
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", forbidden_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "y_cubed",
        source_sha256=hashlib.sha256(b"synthetic-calibration-failure").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-calibration-proposal").hexdigest(),
        run_id="synthetic-calibration-failure",
        receipt_dir=tmp_path,
    )
    assert fit.status == "incomplete"
    receipt = json.loads(fit.receipt_path.read_text())
    for family_name in ("linear", "nonlinear"):
        family = receipt["hypotheses"][family_name]
        calibration = family["calibration"]
        assert family["status"] == "incomplete"
        assert calibration["finite_count"] == 0
        assert calibration["failure_count"] == 256
        assert calibration["epsilon_schedule"] is None
        assert calibration["failure_modes"]["divergence_bound_exceeded"] == 256
        assert all(row["discrepancy"] == "Infinity" for row in calibration["calibration_draws"])


def test_no_sealed_scorer_or_qwen_client_is_exported():
    assert "load_silverbox_for_scoring" not in first_fit.__dict__
    assert not any("score" in name.lower() and "selection" not in name.lower() for name in first_fit.__all__)
    assert "request_silverbox_term_id" not in first_fit.__dict__

"""Focused information-flow and abstention checks for simulator controls."""

from __future__ import annotations

import numpy as np

from core.observation_mismatch_controls import (
    BASELINE_KIND,
    ControlDataset,
    canonical_json,
    discriminating_measurement_rule,
    generate_nested_mechanism_dataset,
    generate_observation_mismatch_dataset,
    nested_mechanism_decision,
    observation_mismatch_selection,
    predeclared_control_cells,
    run_nested_mechanism_abstention_control,
    run_observation_mismatch_control,
    sha256_json,
)


def test_experiment_a_freezes_observed_scale_map_and_roles():
    dataset = generate_observation_mismatch_dataset(11)
    expected_latent = np.exp(-0.65 * dataset.time_points)
    expected_observed_mean = expected_latent / (0.35 + expected_latent)
    assert np.allclose(dataset.latent, expected_latent)
    assert not np.allclose(dataset.observed, dataset.latent)
    assert np.isclose(expected_observed_mean[0], 1.0 / 1.35)
    assert len(dataset.train_observed) == 16
    assert len(dataset.validation_observed) == 8
    assert len(dataset.sealed_observed) == 9
    development = dataset.development_input_receipt()
    assert development["sealed_outcomes_available_to_development"] is False
    assert "sealed_observed_sha256" not in development


def test_experiment_a_selects_correct_map_and_persists_grid_audit():
    result = run_observation_mismatch_control(
        generate_observation_mismatch_dataset(11),
        predictive_draw_count=32,
    )
    assert result["status"] == "complete"
    assert result["complete"] is True
    assert result["baseline_kind"] == BASELINE_KIND
    assert result["selection"]["selected_model"] == "correct_saturating_observation"
    assert result["selection"]["separated"] is True
    assert result["final_evaluation"]["evaluation_count"] == 1
    assert result["final_evaluation"]["posterior_summary"] is False
    for model_name, fit in result["model_fits"].items():
        assert fit["status"] == "complete"
        assert fit["termination_reason"] == "grid_exhausted_complete"
        assert fit["epsilon_schedule"] == []
        assert len(fit["population_receipts"]) == 1
        assert np.allclose(fit["population_receipts"][0]["weights"], fit["weights"])
        assert fit["seeds"]["proposal_seed"] is None
        assert fit["seeds"]["simulator_noise_seed"] is None
        weights = np.asarray(fit["weights"], dtype=float)
        assert len(weights) == fit["parameter_grid_count"]
        assert np.all(np.isfinite(weights))
        assert np.isclose(np.sum(weights), 1.0)
        assert np.isclose(fit["effective_sample_size"], 1.0 / np.sum(weights**2))
        assert fit["inference_diagnostics"]["epsilon"] is None
        assert fit["inference_diagnostics"]["simulator_calls"] == 0
        assert fit["observation_model"]["name"] in {
            "saturating_observation",
            "direct_observed_exponential",
        }
        draws = np.asarray(fit["posterior_predictive_draws"], dtype=float)
        assert draws.shape[0] == 32
        assert np.all(np.isfinite(draws))


def test_experiment_a_sealed_sentinel_cannot_change_selection_or_development_receipt():
    dataset = generate_observation_mismatch_dataset(11)
    ordinary = run_observation_mismatch_control(dataset, predictive_draw_count=24)
    sentinel = run_observation_mismatch_control(
        dataset.with_sealed_observations(np.full(len(dataset.sealed_observed), 1e2)),
        predictive_draw_count=24,
    )
    assert ordinary["selection"] == sentinel["selection"]
    assert (
        ordinary["development_receipt"]["development_receipt_sha256"]
        == sentinel["development_receipt"]["development_receipt_sha256"]
    )
    assert ordinary["final_evaluation"]["sealed_rmse"] != sentinel["final_evaluation"]["sealed_rmse"]
    payload = {key: value for key, value in ordinary.items() if key != "receipt_sha256"}
    assert sha256_json(payload) == ordinary["receipt_sha256"]
    assert canonical_json(ordinary)[0] == "{"


def test_experiment_a_insufficient_margin_abstains_without_sealed_score():
    result = run_observation_mismatch_control(
        generate_observation_mismatch_dataset(29),
        predictive_draw_count=16,
    )
    assert result["selection"]["decision"] == "unresolved_retain_both"
    assert result["selection"]["selected_model"] is None
    assert result["final_evaluation"]["status"] == "not_run_selection_unresolved"
    assert "sealed_rmse" not in result["final_evaluation"]
    assert set(result["development_receipt"]["forecasts"]) == {
        "correct_saturating_observation",
        "direct_observed_exponential_surrogate",
    }


def test_nested_mocked_negative_control_requires_sealed_reproduction():
    decision = nested_mechanism_decision(
        simple_validation_rmse=0.10,
        flexible_validation_rmse=0.08,
        simple_sealed_rmse=0.10,
        flexible_sealed_rmse=0.095,
        practical_margin=0.01,
    )
    assert decision["validation_margin_pass"] is True
    assert decision["sealed_advantage_reproduced"] is False
    assert decision["promoted"] is False
    assert decision["decision"] == "unresolved_retain_simpler"
    assert decision["selected_model"] == "one_rate_exponential"


def test_frozen_margin_boundaries_and_promotion_branch_are_explicit():
    at_margin = observation_mismatch_selection(0.10, 0.11, 0.009999)
    below_margin = observation_mismatch_selection(0.10, 0.11, 0.010001)
    assert at_margin["decision"] == "selected_correct_observation_model"
    assert below_margin["decision"] == "unresolved_retain_both"
    promoted = nested_mechanism_decision(0.10, 0.08, 0.10, 0.08, 0.01)
    assert promoted["promoted"] is True
    assert promoted["decision"] == "promote_two_rate"


def test_discriminating_measurement_rule_covers_both_threshold_sides_without_search():
    candidate_times = [8.5, 9.0, 9.5]
    no_measurement = discriminating_measurement_rule(
        [0.10, 0.08, 0.06],
        [0.11, 0.09, 0.05],
        candidate_times,
        0.05,
    )
    measurement = discriminating_measurement_rule(
        [0.10, 0.08, 0.06],
        [0.11, 0.20, 0.05],
        candidate_times,
        0.05,
    )
    first_qualifying = discriminating_measurement_rule(
        [0.00, 0.00, 0.00],
        [0.11, 0.50, 0.00],
        candidate_times,
        0.05,
    )
    assert no_measurement["recommendation"]["status"] == "no_informative_measurement_under_this_protocol"
    assert measurement["recommendation"]["status"] == "informative_condition_available"
    assert measurement["recommendation"]["time"] == 9.0
    # The later disagreement is larger, but the fixed candidate order still
    # selects the first qualifying time.
    assert first_qualifying["recommendation"]["time"] == 8.5
    assert first_qualifying["recommendation"]["selection_order"] == "first_qualifying_predeclared_time"
    assert no_measurement["post_hoc_time_search"] is False
    assert measurement["post_hoc_time_search"] is False


def test_three_independent_data_seeds_are_predeclared_for_each_experiment():
    cells = predeclared_control_cells()
    assert len(cells) == 6
    for experiment in ("observation_mismatch", "nested_mechanism"):
        experiment_cells = [cell for cell in cells if cell["experiment"] == experiment]
        assert [cell["data_seed"] for cell in experiment_cells] == [11, 29, 47]
        if experiment == "observation_mismatch":
            datasets = [generate_observation_mismatch_dataset(cell["data_seed"]) for cell in experiment_cells]
        else:
            datasets = [generate_nested_mechanism_dataset(cell["data_seed"]) for cell in experiment_cells]
        assert len({dataset.full_data_receipt()["observed_sha256"] for dataset in datasets}) == 3


def test_nested_generated_negative_control_abstains_and_declares_measurement_rule():
    result = run_nested_mechanism_abstention_control(
        generate_nested_mechanism_dataset(11),
        predictive_draw_count=32,
    )
    assert result["status"] == "complete"
    assert result["selection"]["decision"] == "unresolved_retain_simpler"
    assert result["selection"]["selected_model"] == "one_rate_exponential"
    assert result["selection"]["promoted"] is False
    measurement = result["discriminating_measurement"]
    assert measurement["post_hoc_time_search"] is False
    assert measurement["threshold"] == 2.0 * generate_nested_mechanism_dataset(11).sigma
    assert measurement["recommendation"]["status"] in {
        "no_informative_measurement_under_this_protocol",
        "informative_condition_available",
    }
    assert result["development_receipt"]["selection"]["sealed_advantage_reproduced"] is None
    assert result["final_evaluation"]["evaluation_count"] == 1


def test_latent_array_is_not_an_inference_input():
    dataset = generate_observation_mismatch_dataset(47)
    altered_latent = np.full_like(dataset.latent, 123.0)
    altered = ControlDataset(
        experiment=dataset.experiment,
        data_seed=dataset.data_seed,
        time_points=dataset.time_points,
        latent=altered_latent,
        observed=dataset.observed,
        sigma=dataset.sigma,
        true_parameters=dataset.true_parameters,
        train_count=dataset.train_count,
        validation_count=dataset.validation_count,
    )
    ordinary = run_observation_mismatch_control(dataset, predictive_draw_count=16)
    changed_latent = run_observation_mismatch_control(altered, predictive_draw_count=16)
    assert ordinary["development_receipt"]["development_receipt_sha256"] == changed_latent["development_receipt"]["development_receipt_sha256"]
    assert ordinary["selection"] == changed_latent["selection"]


def test_incomplete_fit_gates_selection_and_summaries():
    result = run_observation_mismatch_control(
        generate_observation_mismatch_dataset(11),
        predictive_draw_count=16,
        max_grid_points=10,
    )
    assert result["status"] == "incomplete"
    assert result["complete"] is False
    assert result["selection"]["posterior_summary"] is False
    assert result["final_evaluation"]["status"] == "not_run"
    for fit in result["model_fits"].values():
        assert fit["status"] == "incomplete"
        assert fit["observation_model"]["name"] in {
            "saturating_observation",
            "direct_observed_exponential",
        }
        assert fit["solver_config"]["method"] == "analytic_exp_decay"
        assert fit["posterior_predictive_status"] == "unavailable_incomplete_fit"
        assert fit["predictive_summary"]["status"] == "unavailable"
        assert fit["posterior_predictive_draws"] == []

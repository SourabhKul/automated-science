"""Synthetic-only integration controls for cascaded-tanks ABC-SMC."""

from __future__ import annotations

import math

import numpy as np

from core.real_data.cascaded_tanks_models import (
    TankFailureCategory,
    TankModel,
    TankSimulationLimits,
)
from core.real_data.cascaded_tanks_synthetic_abc import (
    CascadedTanksSyntheticABCConfig,
    run_cascaded_tanks_synthetic_abc_smc,
)


def _scalar_config(*, target_samples: int = 20, epsilons=(0.2, 0.1), seed: int = 81):
    return CascadedTanksSyntheticABCConfig(
        model=TankModel.S0,
        fixed_parameters={"a": 0.5, "c": 0.5, "x1_0": 0.0, "x2_0": 0.0},
        free_parameter_bounds={"p": (0.6, 1.4)},
        target_samples=target_samples,
        epsilon_schedule=epsilons,
        max_attempts_per_population=10_000,
        seed=seed,
        output_units="synthetic voltage-equivalent units",
    )


def _independent_s0_outputs(inputs: np.ndarray, *, p: float) -> np.ndarray:
    """Test-side equation loop, independent of the production simulator."""

    x1, x2 = 0.0, 0.0
    result = []
    for u in inputs:
        result.append(x2)
        old_x1, old_x2 = x1, x2
        x1 = max(0.0, old_x1 + p * float(u) - 0.5 * math.sqrt(old_x1))
        x2 = max(0.0, old_x2 + math.sqrt(old_x1) - 0.5 * math.sqrt(old_x2))
    return np.asarray(result, dtype=float)


def _manual_scalar_log_mixture(point, means, weights, variance):
    component_logs = np.array(
        [
            -0.5 * (math.log(2.0 * math.pi * variance) + (point - mean) ** 2 / variance)
            for mean in means
        ]
    )
    largest = float(np.max(component_logs))
    return largest + math.log(float(np.sum(weights * np.exp(component_logs - largest))))


def test_scalar_known_parameter_glue_control_and_independent_reference_weights():
    inputs = np.array([4.0, 0.0, 0.0, 0.0, 0.0])
    observed = _independent_s0_outputs(inputs, p=1.0)
    assert np.allclose(
        observed[:4], [0.0, 0.0, 2.0, 2.0 + math.sqrt(3.0) - 0.5 * math.sqrt(2.0)]
    )

    result = run_cascaded_tanks_synthetic_abc_smc(inputs, observed, _scalar_config())

    assert result["status"] == "complete"
    assert result["posterior"] is not None
    assert result["provenance"]["has_fixed_parameters"] is True
    assert (
        result["provenance"]["evidence_scope"] == "intended_synthetic_glue_control_only"
    )
    assert result["provenance"]["data_scope"] == "caller_supplied_unverified_arrays"
    assert result["provenance"]["synthetic_origin_verified"] is False
    posterior = result["posterior"]
    p_values = posterior["free_parameter_values"]["p"]
    weights = posterior["weights"]
    assert np.isclose(np.sum(weights), 1.0)
    assert abs(float(np.sum(p_values * weights)) - 1.0) < 0.15
    assert np.all(posterior["distances"] <= 0.1)

    # Independently reconstruct generation-one prior-over-mixture weights.
    first, second = result["reference_evidence"]["populations"]
    means = np.asarray(first["accepted_params"], dtype=float)[:, 0]
    previous_weights = np.asarray(first["weights"], dtype=float)
    variance = float(np.asarray(second["proposal_covariance"])[0, 0])
    second_points = np.asarray(second["accepted_params"], dtype=float)[:, 0]
    assert np.all(np.asarray(second["log_prior_density"]) == 0.0)
    assert np.all((second_points >= 0.0) & (second_points <= 1.0))
    manual_log_weights = np.array(
        [
            -_manual_scalar_log_mixture(point, means, previous_weights, variance)
            for point in second_points
        ]
    )
    manual_weights = np.exp(manual_log_weights - np.max(manual_log_weights))
    manual_weights /= np.sum(manual_weights)
    assert np.allclose(second["weights"], manual_weights, rtol=1e-10, atol=1e-12)
    assert np.isclose(
        posterior["effective_sample_size"], 1.0 / np.sum(manual_weights**2)
    )


def test_forced_simulator_failure_exhausts_budget_without_terminal_posterior():
    inputs = np.array([1.0, 1.0, 1.0])
    observed = np.array([0.0, 0.0, 0.0])
    config = CascadedTanksSyntheticABCConfig(
        model=TankModel.S0,
        fixed_parameters={"a": 0.5, "c": 0.5, "p": 1.0, "x2_0": 0.0},
        free_parameter_bounds={"x1_0": (0.1, 0.2)},
        target_samples=3,
        epsilon_schedule=[100.0],
        max_attempts_per_population=5,
        seed=17,
        limits=TankSimulationLimits(max_steps=2, max_magnitude=1.0e12),
    )

    result = run_cascaded_tanks_synthetic_abc_smc(inputs, observed, config)

    assert result["status"] == "incomplete"
    assert result["posterior"] is None
    assert result["failure_categories"] == {TankFailureCategory.TOO_MANY_STEPS.value: 5}
    diagnostics = result["reference_evidence"]["diagnostics"]
    assert diagnostics["proposed"] == 5
    assert diagnostics["failed_simulations"] == 5
    assert diagnostics["accepted"] == 0
    assert diagnostics["termination_reason"] == "attempt_budget_exhausted"


def test_seeded_synthetic_run_replays_exactly():
    inputs = np.array([4.0, 0.0, 0.0, 0.0])
    observed = _independent_s0_outputs(inputs, p=1.0)
    config = _scalar_config(target_samples=8, epsilons=(1.0,), seed=222)

    first = run_cascaded_tanks_synthetic_abc_smc(inputs, observed, config)
    second = run_cascaded_tanks_synthetic_abc_smc(inputs, observed, config)

    assert first["status"] == second["status"] == "complete"
    assert np.array_equal(
        first["posterior"]["unit_parameters"], second["posterior"]["unit_parameters"]
    )
    assert np.array_equal(first["posterior"]["weights"], second["posterior"]["weights"])
    assert first["provenance"] == second["provenance"]

"""One-case source-free ABC6 training seam and consumer-boundary sentinels."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_training as training
from core.real_data import cascaded_tanks_synthetic_abc as synthetic_abc
from core.real_data.cascaded_tanks_abc6_cases import (
    SHORT_LENGTH,
    TRAINING_INPUT_L,
    build_synthetic_training_bundle,
)
from core.real_data.cascaded_tanks_models import (
    TankFailureCategory,
    TankSimulationFailure,
    simulate_cascaded_tanks,
)


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
        return
    if isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float) and math.isnan(right)
        return
    assert left == right


def test_production_controls_are_frozen_and_manifest_checkable() -> None:
    assert training.CALIBRATION_DRAW_COUNT == 256
    assert training.PARTICLE_COUNT == 48
    assert training.POPULATION_COUNT == 2
    assert training.MAX_PROPOSALS_PER_POPULATION == 4096
    assert training.CALIBRATION_QUANTILES == (0.25, 0.10)
    manifest_bytes = json.dumps(
        dict(training.TRAINING_CONTROL_MANIFEST),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        training.TRAINING_CONTROL_MANIFEST_SHA256
    )
    assert training.TRAINING_CONTROL_MANIFEST["minimum_finite_calibration_draws"] == 256
    assert "unverified" in training.TRAINING_CONTROL_MANIFEST["data_origin"]


def test_public_entry_point_uses_only_frozen_production_controls(monkeypatch) -> None:
    bundle = build_synthetic_training_bundle()
    observed: list[tuple[int, int, int]] = []
    sentinel = object()

    def capture_controls(_data, *, controls):
        observed.append(
            (
                controls.calibration_draws,
                controls.target_samples,
                controls.max_attempts_per_population,
            )
        )
        return sentinel

    monkeypatch.setattr(training, "_run_one_case", capture_controls)
    result = training.run_abc6_training_case(bundle.data_for_case(0))

    assert result is sentinel
    assert observed == [(256, 48, 4096)]


def test_calibration_uses_declared_left_inverse_indices() -> None:
    assert training._left_inverse_indices(256) == (63, 25)
    assert training._left_inverse_indices(11) == (2, 1)


def test_mutating_withheld_short_case_suffix_preserves_real_consumers(
    monkeypatch,
) -> None:
    """Calibration, ABC, and baseline see the same actual short-case calls."""

    def mutate_only_withheld_a_suffix(truth_id, outputs):
        if truth_id != "A":
            return outputs
        return outputs[:SHORT_LENGTH] + tuple(
            value + 75.0 + index for index, value in enumerate(outputs[SHORT_LENGTH:])
        )

    ordinary = build_synthetic_training_bundle()
    mutated = build_synthetic_training_bundle(
        training_output_mutator=mutate_only_withheld_a_suffix
    )
    ordinary_case = ordinary.data_for_case(0)
    mutated_case = mutated.data_for_case(0)
    assert (
        ordinary_case.inputs == mutated_case.inputs == TRAINING_INPUT_L[:SHORT_LENGTH]
    )
    assert ordinary_case.observed_outputs == mutated_case.observed_outputs
    assert (
        ordinary.data_for_case(2).observed_outputs[:SHORT_LENGTH]
        == (mutated.data_for_case(2).observed_outputs[:SHORT_LENGTH])
    )
    assert (
        ordinary.data_for_case(2).observed_outputs[SHORT_LENGTH:]
        != (mutated.data_for_case(2).observed_outputs[SHORT_LENGTH:])
    )

    # The short-case consumer cannot materialize or request prospective truth.
    def forbidden_future_target(*_args, **_kwargs):
        raise AssertionError("training path accessed deferred prospective targets")

    monkeypatch.setattr(
        cases, "_materialize_prospective_targets", forbidden_future_target
    )
    monkeypatch.setattr(
        cases.DeferredABC6TargetGate,
        "generate_targets",
        forbidden_future_target,
    )

    phase = {"name": "calibration"}
    traces: dict[str, list[tuple[object, ...]]] = {"first": [], "second": []}
    active_run = {"name": "first"}
    original_simulator = simulate_cascaded_tanks

    def recording_simulator(inputs, parameters, initial_state, **kwargs):
        outcome = original_simulator(inputs, parameters, initial_state, **kwargs)
        input_tuple = tuple(float(value) for value in inputs)
        if isinstance(outcome, TankSimulationFailure):
            outcome_signature = (
                "failure",
                outcome.category.value,
                outcome.step_index,
                len(outcome.trace),
            )
        else:
            outcome_signature = (
                "success",
                tuple(outcome.observations),
                outcome.terminal_state.x1,
                outcome.terminal_state.x2,
            )
        traces[active_run["name"]].append(
            (
                phase["name"],
                input_tuple,
                parameters.a,
                parameters.c,
                parameters.p,
                initial_state.x1,
                initial_state.x2,
                kwargs["model"].value,
                kwargs["ceiling"],
                kwargs["limits"],
                outcome_signature,
            )
        )
        return outcome

    monkeypatch.setattr(training, "simulate_cascaded_tanks", recording_simulator)
    monkeypatch.setattr(synthetic_abc, "simulate_cascaded_tanks", recording_simulator)

    original_calibration = training._calibrate

    def recording_calibration(*args, **kwargs):
        phase["name"] = "calibration"
        return original_calibration(*args, **kwargs)

    monkeypatch.setattr(training, "_calibrate", recording_calibration)

    original_abc_runner = synthetic_abc.run_cascaded_tanks_synthetic_abc_smc

    def recording_abc_runner(*args, **kwargs):
        phase["name"] = "abc"
        return original_abc_runner(*args, **kwargs)

    monkeypatch.setattr(
        synthetic_abc,
        "run_cascaded_tanks_synthetic_abc_smc",
        recording_abc_runner,
    )

    original_pattern_search = training.run_normalized_cube_pattern_search

    def recording_pattern_search(*args, **kwargs):
        phase["name"] = "baseline"
        return original_pattern_search(*args, **kwargs)

    monkeypatch.setattr(
        training, "run_normalized_cube_pattern_search", recording_pattern_search
    )

    controls = {
        "calibration_draws": 64,
        "target_samples": 4,
        "max_attempts_per_population": 96,
    }
    first = training._run_one_case_with_bounded_controls_for_test(
        ordinary_case, **controls
    )
    first_counts = tuple(
        sum(call[0] == stage for call in traces["first"])
        for stage in ("calibration", "abc", "baseline")
    )
    active_run["name"] = "second"
    second = training._run_one_case_with_bounded_controls_for_test(
        mutated_case, **controls
    )
    second_counts = tuple(
        sum(call[0] == stage for call in traces["second"])
        for stage in ("calibration", "abc", "baseline")
    )

    assert traces["first"] == traces["second"]
    assert first_counts == second_counts
    assert first_counts[0] == second_counts[0] == controls["calibration_draws"]
    assert (
        first_counts[1]
        == second_counts[1]
        == sum(
            population["diagnostics"]["simulated"]
            + population["diagnostics"]["failed_simulations"]
            for population in first.abc_result["reference_evidence"]["populations"]
        )
    )
    assert first_counts[2] == second_counts[2] == first.baseline_result.callback_calls
    assert all(call[1] == TRAINING_INPUT_L[:SHORT_LENGTH] for call in traces["first"])
    assert first.calibration == second.calibration
    assert first.matched_cost_budget == second.matched_cost_budget
    assert first.baseline_result == second.baseline_result
    assert first.simulator_call_counts == second.simulator_call_counts
    assert first.training_input_sha256 == second.training_input_sha256
    assert first.training_output_sha256 == second.training_output_sha256
    assert first.production_controls_used is second.production_controls_used is False
    assert first.abc_result is not None and second.abc_result is not None
    _assert_nested_equal(first.abc_result, second.abc_result)
    assert len(first.abc_result["reference_evidence"]["populations"]) == 2
    assert first.matched_cost_budget.max_calls == (
        controls["calibration_draws"] + first_counts[1]
    )
    assert first.baseline_result.call_budget == first.matched_cost_budget.max_calls
    assert first.baseline_result.callback_calls <= first.matched_cost_budget.max_calls
    assert first.synthetic_origin_verified is False


def test_failed_calibration_draws_resolve_to_diagnostic_only(monkeypatch) -> None:
    bundle = build_synthetic_training_bundle()

    def always_fail(*_args, **_kwargs):
        return TankSimulationFailure(
            category=TankFailureCategory.MAGNITUDE_LIMIT,
            message="sentinel failure",
            step_index=0,
            trace=(),
            terminal_state=None,
        )

    monkeypatch.setattr(training, "simulate_cascaded_tanks", always_fail)
    result = training._run_one_case_with_bounded_controls_for_test(
        bundle.data_for_case(0),
        calibration_draws=8,
        target_samples=2,
        max_attempts_per_population=8,
    )

    assert result.status == "unresolved"
    assert result.abc_status == "unresolved"
    assert result.abc_result is None
    assert result.calibration.finite_count == 0
    assert result.calibration.failed_count == 8
    assert result.calibration.status == "unresolved"
    assert result.calibration.unresolved_reason == (
        "fewer_than_required_finite_calibration_draws"
    )
    assert result.calibration.failure_category_counts == (("magnitude_limit", 8),)
    assert result.matched_cost_budget.calibration_calls == 8
    assert result.matched_cost_budget.population_calls == 0
    assert result.matched_cost_budget.max_calls == 8
    assert result.baseline_result.failure_count == 8
    assert result.baseline_failure_category_counts == (("magnitude_limit", 8),)
    assert result.baseline_result.callback_calls == 8

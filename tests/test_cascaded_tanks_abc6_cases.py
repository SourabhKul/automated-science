"""Source-free roster and deferred-target boundary controls for ABC6."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from core.real_data.cascaded_tanks_abc6_cases import (
    CASE_COUNT,
    CASE_ROSTER,
    CROSSING_PRIOR_BOUNDS,
    MISSING_FEEDTHROUGH,
    NO_CROSSING_PRIOR_BOUNDS,
    PARAMETER_ORDER,
    PROSPECTIVE_INPUT,
    PROSPECTIVE_LENGTH,
    SHORT_LENGTH,
    TRAINING_INPUT_L,
    TRAINING_INPUT_S,
    TRAINING_LENGTH,
    ABC6StatusReceiptError,
    build_synthetic_training_bundle,
    open_deferred_abc6_target_gate,
    write_status_receipt,
)
from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)


def _write_complete_receipt_roster(directory) -> None:
    for case_index in range(CASE_COUNT):
        fit_status = "unresolved" if case_index == 2 else "complete"
        baseline_status = "incomplete" if case_index == 1 else "complete"
        if case_index == 3:
            baseline_status = "failed"
        write_status_receipt(directory, case_index, "fit", fit_status)
        write_status_receipt(directory, case_index, "baseline", baseline_status)


def test_exact_24_case_roster_seeds_models_and_prior_order() -> None:
    assert len(CASE_ROSTER) == CASE_COUNT == 24
    assert tuple(case.case_index for case in CASE_ROSTER) == tuple(range(24))
    assert tuple(
        (case.truth_id, case.input_window, case.replicate, case.fit_model.value)
        for case in CASE_ROSTER
    ) == (
        ("A", "S", 0, "O2"),
        ("A", "S", 0, "C2"),
        ("A", "L", 0, "O2"),
        ("A", "L", 0, "C2"),
        ("B", "S", 0, "C2"),
        ("B", "S", 0, "O2"),
        ("B", "L", 0, "C2"),
        ("B", "L", 0, "O2"),
        ("N", "L", None, "O2"),
        ("N", "L", None, "C2"),
        ("M", "L", None, "C2"),
        ("M", "L", None, "O2"),
        ("A", "S", 1, "O2"),
        ("A", "L", 1, "O2"),
        ("A", "S", 2, "O2"),
        ("A", "L", 2, "O2"),
        ("A", "S", 3, "O2"),
        ("A", "L", 3, "O2"),
        ("B", "S", 1, "C2"),
        ("B", "L", 1, "C2"),
        ("B", "S", 2, "C2"),
        ("B", "L", 2, "C2"),
        ("B", "S", 3, "C2"),
        ("B", "L", 3, "C2"),
    )
    assert tuple(case.calibration_seed for case in CASE_ROSTER) == tuple(
        9000 + index for index in range(CASE_COUNT)
    )
    assert tuple(case.abc_seed for case in CASE_ROSTER) == tuple(
        10000 + index for index in range(CASE_COUNT)
    )
    assert PARAMETER_ORDER == ("a", "c", "p", "x1_0", "x2_0", "ceiling")
    assert CASE_ROSTER[0].prior_bounds == CROSSING_PRIOR_BOUNDS
    assert CASE_ROSTER[10].prior_bounds == CROSSING_PRIOR_BOUNDS
    assert CASE_ROSTER[8].prior_bounds == NO_CROSSING_PRIOR_BOUNDS
    assert tuple(case.input_length for case in CASE_ROSTER[:8]) == (
        78,
        78,
        204,
        204,
        78,
        78,
        204,
        204,
    )


def test_training_builder_constructs_three_paths_and_paired_noise_prefixes() -> None:
    simulator_calls = []
    truth_outputs = []

    def recording_simulator(inputs, parameters, initial_state, **kwargs):
        simulator_calls.append((tuple(inputs), kwargs["model"], kwargs["ceiling"]))
        outcome = simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)
        assert isinstance(outcome, TankSimulationSuccess)
        truth_outputs.append(outcome.observations)
        return outcome

    bundle = build_synthetic_training_bundle(simulator=recording_simulator)

    assert len(simulator_calls) == 3  # A, B and N; M reuses B plus feedthrough.
    assert all(call[0] == TRAINING_INPUT_L for call in simulator_calls)
    assert tuple(call[1] for call in simulator_calls) == (
        TankModel.O2,
        TankModel.C2,
        TankModel.C2,
    )
    assert tuple(call[2] for call in simulator_calls) == (3.0, 3.2, 1000.0)
    assert TRAINING_INPUT_S == TRAINING_INPUT_L[:SHORT_LENGTH]
    assert len(TRAINING_INPUT_L) == TRAINING_LENGTH == 204
    assert len(PROSPECTIVE_INPUT) == PROSPECTIVE_LENGTH == 60
    assert PROSPECTIVE_INPUT == (3.0,) * 12 + (8.0,) * 12 + (0.0,) * 36

    data_a_s = bundle.data_for_case(0)
    data_a_l = bundle.data_for_case(2)
    data_b_s = bundle.data_for_case(4)
    data_b_l = bundle.data_for_case(6)
    assert len(data_a_s.inputs) == len(data_a_s.observed_outputs) == SHORT_LENGTH
    assert len(data_a_l.inputs) == len(data_a_l.observed_outputs) == TRAINING_LENGTH
    assert data_a_l.inputs[:SHORT_LENGTH] == data_a_s.inputs
    assert data_a_l.observed_outputs[:SHORT_LENGTH] == data_a_s.observed_outputs
    assert data_b_l.inputs[:SHORT_LENGTH] == data_b_s.inputs
    assert data_b_l.observed_outputs[:SHORT_LENGTH] == data_b_s.observed_outputs

    for short_index, long_index, base_path_index, noise_seed in (
        (0, 2, 0, 8100),
        (4, 6, 1, 8200),
    ):
        noise = np.random.default_rng(noise_seed).normal(
            0.0, 0.05, size=TRAINING_LENGTH
        )
        expected = tuple(
            float(value + added)
            for value, added in zip(truth_outputs[base_path_index], noise, strict=True)
        )
        assert (
            bundle.data_for_case(short_index).observed_outputs
            == expected[:SHORT_LENGTH]
        )
        assert bundle.data_for_case(long_index).observed_outputs == expected

    n_data = bundle.data_for_case(8)
    m_data = bundle.data_for_case(10)
    assert len(n_data.observed_outputs) == TRAINING_LENGTH
    assert len(m_data.observed_outputs) == TRAINING_LENGTH
    # M is exactly the deterministic B simulator observation plus 0.15*u.
    b_noiseless = simulate_cascaded_tanks(
        TRAINING_INPUT_L,
        TankParameters(0.54, 0.36, 0.46),
        TankState(0.25, 0.75),
        model=TankModel.C2,
        ceiling=3.2,
    )
    assert isinstance(b_noiseless, TankSimulationSuccess)
    expected_m = tuple(
        value + MISSING_FEEDTHROUGH * input_value
        for value, input_value in zip(
            b_noiseless.observations, TRAINING_INPUT_L, strict=True
        )
    )
    assert np.allclose(m_data.observed_outputs, expected_m, rtol=0.0, atol=0.0)
    assert not hasattr(bundle, "prospective_targets")


def test_short_case_view_is_invariant_to_mutations_after_index_77() -> None:
    def mutate_withheld_suffix(truth_id: str, outputs: tuple[float, ...]):
        if truth_id not in {"A", "B"}:
            return outputs
        return outputs[:SHORT_LENGTH] + tuple(
            value + 100.0 + index for index, value in enumerate(outputs[SHORT_LENGTH:])
        )

    ordinary = build_synthetic_training_bundle()
    suffix_mutated = build_synthetic_training_bundle(
        training_output_mutator=mutate_withheld_suffix
    )

    for short_index, long_index in ((0, 2), (4, 6), (12, 13), (18, 19)):
        ordinary_short = ordinary.data_for_case(short_index)
        mutated_short = suffix_mutated.data_for_case(short_index)
        assert ordinary_short.inputs == mutated_short.inputs
        assert ordinary_short.observed_outputs == mutated_short.observed_outputs

        ordinary_long = ordinary.data_for_case(long_index)
        mutated_long = suffix_mutated.data_for_case(long_index)
        assert (
            ordinary_long.observed_outputs[:SHORT_LENGTH]
            == mutated_long.observed_outputs[:SHORT_LENGTH]
        )
        assert (
            ordinary_long.observed_outputs[SHORT_LENGTH:]
            != mutated_long.observed_outputs[SHORT_LENGTH:]
        )

    # This proves only case-view prefix invariance. Actual calibration, ABC,
    # and baseline call-boundary sentinels remain a runner integration gate.


def test_future_targets_are_unavailable_until_all_48_durable_status_receipts(
    tmp_path,
) -> None:
    training = build_synthetic_training_bundle()
    calls = []

    def recording_future_simulator(inputs, parameters, initial_state, **kwargs):
        calls.append((tuple(inputs), kwargs["model"], kwargs["ceiling"]))
        return simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)

    with pytest.raises(ABC6StatusReceiptError, match="missing"):
        open_deferred_abc6_target_gate(tmp_path)
    assert calls == []

    for case_index in range(CASE_COUNT):
        for component in ("fit", "baseline"):
            if (case_index, component) == (23, "baseline"):
                continue
            write_status_receipt(tmp_path, case_index, component, "complete")
    with pytest.raises(ABC6StatusReceiptError, match="missing"):
        open_deferred_abc6_target_gate(tmp_path)
    assert calls == []

    write_status_receipt(tmp_path, 23, "baseline", "failed")
    gate = open_deferred_abc6_target_gate(tmp_path)
    assert calls == []  # verifying receipts itself never simulates a future.
    targets = gate.generate_targets(training, simulator=recording_future_simulator)

    assert len(calls) == 2
    assert all(call[0] == PROSPECTIVE_INPUT for call in calls)
    assert tuple(call[1] for call in calls) == (TankModel.O2, TankModel.C2)
    assert targets.truth_ids == ("A", "B", "M")
    assert all(
        len(targets.for_truth(truth)) == PROSPECTIVE_LENGTH
        for truth in targets.truth_ids
    )
    expected_m = tuple(
        value + MISSING_FEEDTHROUGH * input_value
        for value, input_value in zip(
            targets.for_truth("B"), PROSPECTIVE_INPUT, strict=True
        )
    )
    assert targets.for_truth("M") == expected_m
    assert dict(targets.sha256_by_truth) == {
        truth: sha256(
            np.asarray(targets.for_truth(truth), dtype="<f8").tobytes()
        ).hexdigest()
        for truth in targets.truth_ids
    }
    with pytest.raises(KeyError, match="N"):
        targets.for_truth("N")


def test_future_output_mutation_only_occurs_after_gate_and_changes_target_hashes(
    tmp_path,
) -> None:
    ordinary = build_synthetic_training_bundle()
    suffix_mutated = build_synthetic_training_bundle(
        training_output_mutator=lambda truth, outputs: (
            outputs[:SHORT_LENGTH]
            + tuple(value + 100.0 for value in outputs[SHORT_LENGTH:])
            if truth in {"A", "B"}
            else outputs
        )
    )
    for index in (0, 4, 12, 18):
        assert ordinary.data_for_case(index) == suffix_mutated.data_for_case(index)

    calls = []

    def changed_future_simulator(inputs, parameters, initial_state, **kwargs):
        calls.append(tuple(inputs))
        outcome = simulate_cascaded_tanks(inputs, parameters, initial_state, **kwargs)
        assert isinstance(outcome, TankSimulationSuccess)
        changed_trace = tuple(
            replace(step, observation_y=step.observation_y + 0.25)
            for step in outcome.trace
        )
        return replace(outcome, trace=changed_trace)

    assert calls == []
    _write_complete_receipt_roster(tmp_path)
    gate = open_deferred_abc6_target_gate(tmp_path)
    assert calls == []
    ordinary_targets = gate.generate_targets(ordinary)
    changed_targets = gate.generate_targets(
        suffix_mutated, simulator=changed_future_simulator
    )

    assert len(calls) == 2
    assert calls == [PROSPECTIVE_INPUT, PROSPECTIVE_INPUT]
    assert ordinary_targets.for_truth("A") != changed_targets.for_truth("A")
    assert ordinary_targets.for_truth("B") != changed_targets.for_truth("B")
    assert ordinary_targets.sha256_by_truth != changed_targets.sha256_by_truth
    # Mutating the prospective outputs never changes the already exposed short
    # fitting observations.
    for index in (0, 4, 12, 18):
        assert ordinary.data_for_case(index) == suffix_mutated.data_for_case(index)


def test_gate_revalidates_receipt_bytes_before_any_future_simulation(tmp_path) -> None:
    training = build_synthetic_training_bundle()
    _write_complete_receipt_roster(tmp_path)
    gate = open_deferred_abc6_target_gate(tmp_path)
    receipt_path = tmp_path / "case-00.fit-status.json"
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("future simulator must remain behind receipt revalidation")

    with pytest.raises(ABC6StatusReceiptError, match="hash or encoding"):
        gate.generate_targets(training, simulator=forbidden_simulator)
    assert calls == []


def test_status_receipts_are_exclusive_and_require_final_status(tmp_path) -> None:
    path = write_status_receipt(tmp_path, 0, "fit", "complete")
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_status_receipt(tmp_path, 0, "fit", "complete")
    with pytest.raises(ValueError, match="final"):
        write_status_receipt(tmp_path, 0, "baseline", "pending")

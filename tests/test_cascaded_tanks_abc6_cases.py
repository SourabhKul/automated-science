"""Source-free roster and deferred-target boundary controls for ABC6."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit
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
    TankStep,
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


def _fake_future_simulator(inputs, parameters, initial_state, *, model, ceiling, limits):
    del limits
    state = TankState(initial_state.x1, initial_state.x2)
    trace = tuple(
        TankStep(
            index=index,
            input_u=float(input_value),
            state=state,
            observation_y=float(parameters.a + index * 0.01 + state.x2),
            q12=0.0,
            q2=0.0,
            x1_next=state.x1,
            x2_raw=state.x2,
            next_state=state,
        )
        for index, input_value in enumerate(inputs)
    )
    return TankSimulationSuccess(
        model=model,
        parameters=parameters,
        initial_state=initial_state,
        ceiling=ceiling,
        sample_interval_seconds=4.0,
        trace=trace,
        terminal_state=state,
    )


def _write_private_scoring_handoff(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(cases, "PROTOCOL_ID", "private-fake-abc6-protocol")
    monkeypatch.setattr(cases, "RUN_ID", "private-fake-abc6-run")
    source_root = gate._root_anchor.path
    monkeypatch.setattr(cases, "_PROJECT_ROOT", source_root)
    marker_path = (
        source_root
        / "artifacts"
        / "evaluations"
        / "cascaded_tanks_abc6_scoring"
        / "claims"
        / f"{cases.RUN_ID}.claim"
    )
    monkeypatch.setattr(cases, "_REVEAL_MARKER_PATH", marker_path)
    for relative_path in cases._SCORING_SOURCE_PATHS:
        source_file = source_root / relative_path
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(f"private fixture source: {relative_path}\n")
    summary = gate._receipt_directory / cases._TRAINING_SUMMARY_FILENAME
    summary_bytes = b"private fake summary\n"
    summary.write_bytes(summary_bytes)
    artifact = gate._receipt_directory / cases._FORECAST_ARTIFACT_FILENAME
    artifact_bytes = b"private fake target-free forecast artifact\n"
    artifact.write_bytes(artifact_bytes)
    source_hashes = [
        [
            relative_path,
            sha256((cases._PROJECT_ROOT / relative_path).read_bytes()).hexdigest(),
        ]
        for relative_path in cases._SCORING_SOURCE_PATHS
    ]
    marker = {
        "schema": "cascaded-tanks-abc6-synthetic-reveal-v1",
        "protocol_id": cases.PROTOCOL_ID,
        "run_id": cases.RUN_ID,
        "condition": "synthetic-prospective-target-confirmation-v1",
        "semantics": "consumed-on-create; success-or-failure; no-retry",
        "forecast_roster_sha256": "a" * 64,
        "forecast_artifact_sha256": sha256(artifact_bytes).hexdigest(),
        "training_manifest_sha256": "b" * 64,
        "training_evidence_manifest_sha256": "c" * 64,
        "integrated_source_hashes": source_hashes,
        "status_receipt_sha256": [list(pair) for pair in gate._verified_receipts],
        "training_summary_sha256": sha256(summary_bytes).hexdigest(),
        "created_at_utc": "2026-09-28T00:00:00+00:00",
    }
    raw = json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    marker_parent = cases._open_relative_directory_anchor(
        gate._root_anchor,
        "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims",
        label="private marker parent",
        create=True,
    )
    try:
        marker_fd = os.open(
            marker_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=marker_parent.descriptor,
        )
        try:
            os.write(marker_fd, raw)
            os.fsync(marker_fd)
        finally:
            os.close(marker_fd)
        os.fsync(marker_parent.descriptor)
    except BaseException:
        marker_parent.close()
        raise
    handoff = cases._issue_scoring_target_handoff(
        gate, sha256(raw).hexdigest(), marker_anchor=marker_parent
    )
    return marker_path, handoff


def _replace_directory_with_symlink(directory: Path, link_target: Path) -> None:
    saved = directory.with_name(f"{directory.name}.saved")
    shutil.copytree(directory, link_target)
    directory.rename(saved)
    directory.symlink_to(link_target, target_is_directory=True)


def _private_receipt_root(tmp_path):
    root = tmp_path / "private-checkout"
    root.mkdir(parents=True, exist_ok=True)
    receipt_directory = root / cases.RECEIPT_ROOT_RELATIVE
    receipt_directory.mkdir(parents=True, exist_ok=True)
    return root, receipt_directory


def _private_receipt_identity(receipt_directory):
    root = receipt_directory.parents[len(Path(cases.RECEIPT_ROOT_RELATIVE).parts) - 1]
    root_fd = campaign_fit._open_directory_nofollow(
        root, label="private fake checkout root"
    )
    try:
        receipt_fd = campaign_fit._open_relative_directory_nofollow(
            root_fd,
            cases.RECEIPT_ROOT_RELATIVE,
            label="private fake receipt root",
        )
    except BaseException:
        os.close(root_fd)
        raise
    return campaign_fit.ABC6ReceiptRootIdentity(
        repository_root_realpath=str(root),
        receipt_root_relative=cases.RECEIPT_ROOT_RELATIVE,
        repository_root_fd=root_fd,
        receipt_root_fd=receipt_fd,
    )


def _open_private_gate(receipt_directory):
    identity = _private_receipt_identity(receipt_directory)
    try:
        return open_deferred_abc6_target_gate(identity)
    finally:
        identity.close()


def _private_gate_with_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "PROTOCOL_ID", "private-fake-abc6-protocol")
    monkeypatch.setattr(cases, "RUN_ID", "private-fake-abc6-run")
    training = build_synthetic_training_bundle()
    root, directory = _private_receipt_root(tmp_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    _write_complete_receipt_roster(directory)
    gate = _open_private_gate(directory)
    marker_path, handoff = _write_private_scoring_handoff(
        tmp_path, monkeypatch, gate
    )
    return training, gate, marker_path, handoff


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
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cases, "PROTOCOL_ID", "private-fake-abc6-protocol")
    monkeypatch.setattr(cases, "RUN_ID", "private-fake-abc6-run")
    training = build_synthetic_training_bundle()
    root, receipt_directory = _private_receipt_root(tmp_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    calls = []

    def recording_future_simulator(inputs, parameters, initial_state, **kwargs):
        calls.append((tuple(inputs), kwargs["model"], kwargs["ceiling"]))
        return _fake_future_simulator(
            inputs, parameters, initial_state, **kwargs
        )

    with pytest.raises(ABC6StatusReceiptError, match="missing"):
        _open_private_gate(receipt_directory)
    assert calls == []

    for case_index in range(CASE_COUNT):
        for component in ("fit", "baseline"):
            if (case_index, component) == (23, "baseline"):
                continue
            write_status_receipt(receipt_directory, case_index, component, "complete")
    with pytest.raises(ABC6StatusReceiptError, match="missing"):
        _open_private_gate(receipt_directory)
    assert calls == []

    write_status_receipt(receipt_directory, 23, "baseline", "failed")
    gate = _open_private_gate(receipt_directory)
    assert calls == []  # verifying receipts itself never simulates a future.
    with pytest.raises(ABC6StatusReceiptError, match="scorer's durable reveal handoff"):
        gate.generate_targets(training, simulator=recording_future_simulator)
    assert calls == []

    targets = cases._materialize_prospective_targets_for_test(
        training, simulator=recording_future_simulator
    )

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


def test_fifo_status_receipt_fails_without_blocking_or_materializing_targets(
    tmp_path, monkeypatch
) -> None:
    root, receipt_directory = _private_receipt_root(tmp_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    _write_complete_receipt_roster(receipt_directory)
    fifo_path = receipt_directory / "case-00.fit-status.json"
    fifo_path.unlink()
    os.mkfifo(fifo_path)
    identity = _private_receipt_identity(receipt_directory)
    receipt_fd = identity.duplicate_receipt_root_fd()
    materializer_calls = []
    monkeypatch.setattr(
        cases,
        "_materialize_prospective_targets",
        lambda *args, **kwargs: materializer_calls.append((args, kwargs)),
    )
    original_open = os.open
    observed_flags = []

    def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == fifo_path.name and dir_fd is not None:
            parent_stat = os.fstat(dir_fd)
            receipt_stat = os.fstat(receipt_fd)
            if (parent_stat.st_dev, parent_stat.st_ino) == (
                receipt_stat.st_dev,
                receipt_stat.st_ino,
            ):
                observed_flags.append(flags)
                assert flags & os.O_NONBLOCK, "FIFO receipt open must not block"
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(cases.os, "open", guarded_open)
    try:
        with pytest.raises(ABC6StatusReceiptError, match="bounded regular file"):
            open_deferred_abc6_target_gate(identity)
    finally:
        os.close(receipt_fd)
        identity.close()

    assert observed_flags and all(flags & os.O_NONBLOCK for flags in observed_flags)
    assert materializer_calls == []


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
        outcome = _fake_future_simulator(inputs, parameters, initial_state, **kwargs)
        changed_trace = tuple(
            replace(step, observation_y=step.observation_y + 0.25)
            for step in outcome.trace
        )
        return replace(outcome, trace=changed_trace)

    assert calls == []
    ordinary_targets = cases._materialize_prospective_targets_for_test(
        ordinary, simulator=_fake_future_simulator
    )
    changed_targets = cases._materialize_prospective_targets_for_test(
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


def test_public_gate_rejects_receipts_only_and_fake_marker_tampering_fails_closed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cases, "PROTOCOL_ID", "private-fake-abc6-protocol")
    monkeypatch.setattr(cases, "RUN_ID", "private-fake-abc6-run")
    training = build_synthetic_training_bundle()
    root, receipt_directory = _private_receipt_root(tmp_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    _write_complete_receipt_roster(receipt_directory)
    with pytest.raises(
        ABC6StatusReceiptError, match="typed campaign receipt-root identity"
    ):
        open_deferred_abc6_target_gate(receipt_directory)
    gate = _open_private_gate(receipt_directory)
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("future simulator must remain behind scorer handoff")

    with pytest.raises(ABC6StatusReceiptError, match="scorer's durable reveal handoff"):
        gate.generate_targets(training, simulator=forbidden_simulator)
    assert calls == []

    marker_path, handoff = _write_private_scoring_handoff(
        tmp_path, monkeypatch, gate
    )
    receipt_path = receipt_directory / "case-00.fit-status.json"
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    with pytest.raises(ABC6StatusReceiptError, match="receipt hash or encoding mismatch"):
        gate.generate_targets(
            training,
            simulator=forbidden_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == []
    assert marker_path.is_file()


def test_reveal_handoff_is_bound_one_use_and_reentrant_calls_fail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cases, "PROTOCOL_ID", "private-fake-abc6-protocol")
    monkeypatch.setattr(cases, "RUN_ID", "private-fake-abc6-run")
    training = build_synthetic_training_bundle()
    root, receipt_directory = _private_receipt_root(tmp_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    _write_complete_receipt_roster(receipt_directory)
    gate = _open_private_gate(receipt_directory)
    _marker, handoff = _write_private_scoring_handoff(tmp_path, monkeypatch, gate)
    calls = []
    reentrant_calls = []

    def reentrant_fake_simulator(inputs, parameters, initial_state, **kwargs):
        calls.append(tuple(inputs))
        with pytest.raises(ABC6StatusReceiptError, match="generation already started"):
            gate.generate_targets(
                training,
                simulator=_fake_future_simulator,
                _scoring_handoff=handoff,
            )
        reentrant_calls.append("rejected")
        return _fake_future_simulator(inputs, parameters, initial_state, **kwargs)

    targets = gate.generate_targets(
        training,
        simulator=reentrant_fake_simulator,
        _scoring_handoff=handoff,
    )
    assert targets.truth_ids == ("A", "B", "M")
    assert calls == [PROSPECTIVE_INPUT, PROSPECTIVE_INPUT]
    assert reentrant_calls == ["rejected", "rejected"]
    with pytest.raises(ABC6StatusReceiptError, match="generation already started"):
        gate.generate_targets(
            training,
            simulator=reentrant_fake_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == [PROSPECTIVE_INPUT, PROSPECTIVE_INPUT]


def test_second_scoring_handoff_cannot_be_issued_for_one_gate(
    tmp_path, monkeypatch
) -> None:
    training, gate, marker_path, first_handoff = _private_gate_with_handoff(
        tmp_path, monkeypatch
    )
    marker_parent = cases._open_relative_directory_anchor(
        gate._root_anchor,
        "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims",
        label="private second-handoff marker parent",
    )
    marker_digest = sha256(marker_path.read_bytes()).hexdigest()
    calls = []

    with pytest.raises(
        ABC6StatusReceiptError, match="handoff was already issued"
    ):
        cases._issue_scoring_target_handoff(
            gate,
            marker_digest,
            marker_anchor=marker_parent,
        )
    assert calls == []

    targets = gate.generate_targets(
        training,
        simulator=lambda inputs, parameters, state, **kwargs: (
            calls.append(tuple(inputs))
            or _fake_future_simulator(inputs, parameters, state, **kwargs)
        ),
        _scoring_handoff=first_handoff,
    )
    assert targets.truth_ids == ("A", "B", "M")
    assert calls == [PROSPECTIVE_INPUT, PROSPECTIVE_INPUT]


def test_fixed_marker_symlink_after_handoff_never_reaches_materializer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cases, "PROTOCOL_ID", "private-fake-abc6-protocol")
    monkeypatch.setattr(cases, "RUN_ID", "private-fake-abc6-run")
    training = build_synthetic_training_bundle()
    root, receipt_directory = _private_receipt_root(tmp_path)
    monkeypatch.setattr(cases, "_PROJECT_ROOT", root)
    _write_complete_receipt_roster(receipt_directory)
    gate = _open_private_gate(receipt_directory)
    marker_path, handoff = _write_private_scoring_handoff(
        tmp_path, monkeypatch, gate
    )
    replacement = marker_path.with_suffix(".replacement")
    replacement.write_bytes(marker_path.read_bytes())
    marker_path.unlink()
    marker_path.symlink_to(replacement)
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("marker symlink must fail before simulation")

    with pytest.raises(ABC6StatusReceiptError, match="missing or unsafe"):
        gate.generate_targets(
            training,
            simulator=forbidden_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == []


def test_marker_ancestor_symlink_to_copied_tree_fails_before_materializer(
    tmp_path, monkeypatch
) -> None:
    training, gate, marker_path, handoff = _private_gate_with_handoff(
        tmp_path, monkeypatch
    )
    project_directory = marker_path.parent.parent
    _replace_directory_with_symlink(
        project_directory, tmp_path / "copied-marker-project"
    )
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("marker ancestor symlink reached target materializer")

    with pytest.raises(
        ABC6StatusReceiptError, match="reveal marker parent ancestry.*symlink"
    ):
        gate.generate_targets(
            training,
            simulator=forbidden_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == []


def test_receipt_ancestor_symlink_to_copied_tree_fails_before_materializer(
    tmp_path, monkeypatch
) -> None:
    training, gate, _marker, handoff = _private_gate_with_handoff(
        tmp_path, monkeypatch
    )
    receipt_home = gate._root_anchor.path / "artifacts"
    _replace_directory_with_symlink(
        receipt_home, tmp_path / "copied-receipt-home"
    )
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("receipt ancestor symlink reached target materializer")

    with pytest.raises(
        ABC6StatusReceiptError, match="status receipt directory ancestry.*symlink"
    ):
        gate.generate_targets(
            training,
            simulator=forbidden_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == []


@pytest.mark.parametrize("source_parent", ("scripts", "core/real_data"))
def test_integrated_source_parent_symlink_to_copy_fails_before_materializer(
    tmp_path, monkeypatch, source_parent
) -> None:
    training, gate, _marker, handoff = _private_gate_with_handoff(
        tmp_path, monkeypatch
    )
    source_directory = cases._PROJECT_ROOT / source_parent
    copied = cases._PROJECT_ROOT / f"copied-{source_directory.name}"
    _replace_directory_with_symlink(source_directory, copied)
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("source ancestor symlink reached target materializer")

    with pytest.raises(
        ABC6StatusReceiptError,
        match="integrated source parent.*ancestry.*symlink",
    ):
        gate.generate_targets(
            training,
            simulator=forbidden_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == []


@pytest.mark.parametrize(
    "artifact_name",
    (cases._TRAINING_SUMMARY_FILENAME, cases._FORECAST_ARTIFACT_FILENAME),
)
def test_summary_or_forecast_leaf_symlink_fails_before_materializer(
    tmp_path, monkeypatch, artifact_name
) -> None:
    training, gate, _marker, handoff = _private_gate_with_handoff(
        tmp_path, monkeypatch
    )
    artifact = gate._receipt_directory / artifact_name
    saved_artifact = artifact.with_suffix(".saved")
    artifact.rename(saved_artifact)
    artifact.symlink_to(saved_artifact)
    calls = []

    def forbidden_simulator(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("artifact symlink reached target materializer")

    with pytest.raises(ABC6StatusReceiptError, match="missing or unsafe"):
        gate.generate_targets(
            training,
            simulator=forbidden_simulator,
            _scoring_handoff=handoff,
        )
    assert calls == []


def test_status_receipts_are_exclusive_and_require_final_status(tmp_path) -> None:
    path = write_status_receipt(tmp_path, 0, "fit", "complete")
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_status_receipt(tmp_path, 0, "fit", "complete")
    with pytest.raises(ValueError, match="final"):
        write_status_receipt(tmp_path, 0, "baseline", "pending")

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from core.real_data import cascaded_tanks_controlled as source
from core.real_data import cascaded_tanks_forecast_gate as gate
from core.real_data import cascaded_tanks_training_fit as fit
from core.real_data.cascaded_tanks_controlled import (
    CSV_MEMBER_BYTES,
    SYNTHETIC_FIXTURE_CONTRACT_SHA256,
    SYNTHETIC_FIXTURE_TRACK_ID,
    ArchiveExpectation,
)
from core.real_data.cascaded_tanks_models import (
    TankFailureCategory,
    TankSimulationFailure,
)
from tests.test_cascaded_tanks_controlled import (
    _archive_bytes,
    _csv_bytes,
    _numeric_field_width,
)
from tests.test_cascaded_tanks_training_fit import _write_manifest


def _write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _comparison_declaration(
    manifest: dict[str, Any], manifest_raw: bytes
) -> dict[str, Any]:
    return {
        "schema": gate.COMPARISON_SCHEMA,
        "protocol_id": manifest["protocol_id"],
        "run_id": manifest["run_id"],
        "scope": "synthetic-fixture-only",
        "fit_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source": manifest["source"],
        "candidate_roster": [
            {
                "candidate_id": candidate["candidate_id"],
                "model": candidate["model"],
                "expected_particle_count": candidate["target_samples"],
                "simulation_limits": candidate["simulation_limits"],
            }
            for candidate in manifest["candidates"]
        ],
        "forecast": {
            "training_indices": [0, 768],
            "boundary_state_index": 768,
            "forecast_indices": [768, 1024],
            "forecast_length": 256,
            "alignment": gate._ALIGNMENT,
        },
        "code_sha256": gate._current_code_sha256(),
    }


def _make_case(
    tmp_path: Path,
    *,
    archive_data: bytes | None = None,
    candidate_count: int = 1,
) -> tuple[Path, ArchiveExpectation, Path, Path, Path, dict[str, Any], dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive_data = (
        archive_data if archive_data is not None else _archive_bytes(_csv_bytes())
    )
    archive_path, expected_archive, manifest_path = _write_manifest(
        tmp_path, archive_data
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if candidate_count > 1:
        candidates = manifest["candidates"]
        for index in range(1, candidate_count):
            extra = dict(candidates[0])
            extra["candidate_id"] = f"synthetic-s0-{index}"
            extra["seed"] = 101 + index
            candidates.append(extra)
        _write_json(manifest_path, manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_raw = manifest_path.read_bytes()
    fit_receipt = fit._run_synthetic_fixture_training_fit(
        archive_path, manifest_path, expected_archive=expected_archive
    )
    receipt_path = tmp_path / f"{archive_path.stem}.fit-receipt.json"
    _write_json(receipt_path, fit_receipt)
    declaration = _comparison_declaration(manifest, manifest_raw)
    declaration_path = tmp_path / f"{archive_path.stem}.comparison.json"
    _write_json(declaration_path, declaration)
    return (
        archive_path,
        expected_archive,
        manifest_path,
        receipt_path,
        declaration_path,
        fit_receipt,
        declaration,
    )


def _run_case(case):
    (
        archive_path,
        expected_archive,
        manifest_path,
        receipt_path,
        declaration_path,
        *_,
    ) = case
    return gate._run_synthetic_fixture_forecast_gate(
        archive_path,
        manifest_path,
        receipt_path,
        declaration_path,
        expected_archive=expected_archive,
    )


def _assert_gate_hash(outcome) -> None:
    unsigned = {
        key: value for key, value in outcome.receipt.items() if key != "receipt_sha256"
    }
    assert outcome.receipt["receipt_sha256"] == fit._canonical_sha256(unsigned)


def test_complete_synthetic_roster_replays_with_stable_target_free_receipt(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    first = _run_case(case)
    second = _run_case(case)

    assert first.status == second.status == "complete"
    assert first.receipt == second.receipt
    assert first.forecasts == second.forecasts
    assert first.forecasts is not None and len(first.forecasts) == 1
    assert len(first.forecasts[0].values) == 256
    assert first.receipt["development_score_eligible"] is False
    assert first.receipt["target_access"] == {
        "development_y_est_materialized": 0,
        "u_val_materialized": 0,
        "y_val_materialized": 0,
    }
    assert first.receipt["target_sha256"] is None
    assert first.receipt["source"]["track_id"] == SYNTHETIC_FIXTURE_TRACK_ID
    assert (
        first.receipt["source"]["contract_sha256"] == SYNTHETIC_FIXTURE_CONTRACT_SHA256
    )
    assert first.receipt["forecast"]["boundary_state_index"] == 768
    candidate = first.receipt["candidate_results"][0]
    assert candidate["status"] == "complete"
    assert candidate["weighted_median_forecast_sha256"]
    assert len(candidate["particle_results"]) == 3
    assert [row["identity"] for row in candidate["particle_results"]] == [
        {"candidate_id": "synthetic-s0", "final_population_index": index}
        for index in range(3)
    ]
    original_weights = case[5]["candidate_results"][0]["posterior"]["weights"]
    assert candidate["original_weights"] == original_weights
    assert [row["weight"] for row in candidate["particle_results"]] == original_weights
    _assert_gate_hash(first)


@pytest.mark.parametrize("roster_change", ["missing", "reordered"])
def test_comparison_gate_rejects_missing_or_reordered_candidate(
    tmp_path: Path, roster_change: str
) -> None:
    case = _make_case(tmp_path, candidate_count=2)
    declaration_path = case[4]
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    if roster_change == "missing":
        declaration["candidate_roster"].pop()
    else:
        declaration["candidate_roster"].reverse()
    _write_json(declaration_path, declaration)

    outcome = _run_case(case)

    assert outcome.status == "terminal_failure"
    assert outcome.forecasts is None
    assert outcome.receipt["failures"][0]["code"] == "candidate_roster_mismatch"
    assert outcome.receipt["candidate_results"] == []
    _assert_gate_hash(outcome)


@pytest.mark.parametrize("roster_change", ["extra", "duplicate"])
def test_comparison_gate_rejects_extra_or_duplicate_candidate(
    tmp_path: Path, roster_change: str
) -> None:
    case = _make_case(tmp_path, candidate_count=2)
    declaration_path = case[4]
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    roster = declaration["candidate_roster"]
    if roster_change == "extra":
        extra = dict(roster[-1])
        extra["candidate_id"] = "undeclared-candidate"
        roster.append(extra)
    else:
        declaration["candidate_roster"] = [roster[0], dict(roster[0])]
    _write_json(declaration_path, declaration)

    outcome = _run_case(case)

    assert outcome.status == "terminal_failure"
    assert outcome.forecasts is None
    assert outcome.receipt["failures"][0]["code"] == "candidate_roster_mismatch"
    assert outcome.receipt["candidate_results"] == []
    _assert_gate_hash(outcome)


def test_gate_rejects_altered_rehashed_posterior_against_final_population(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    receipt_path = case[3]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["candidate_results"][0]["posterior"]["weights"][0] += 0.01
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = fit._canonical_sha256(unsigned)
    _write_json(receipt_path, receipt)

    outcome = _run_case(case)

    assert outcome.status == "terminal_failure"
    assert outcome.forecasts is None
    assert outcome.receipt["failures"][0]["code"] == "posterior_population_mismatch"
    _assert_gate_hash(outcome)


def test_gate_rejects_receipt_with_invalid_self_hash(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    receipt_path = case[3]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["candidate_results"][0]["posterior"]["weights"][0] += 0.01
    _write_json(receipt_path, receipt)

    outcome = _run_case(case)

    assert outcome.status == "terminal_failure"
    assert outcome.forecasts is None
    assert outcome.receipt["failures"][0]["code"] == "fit_receipt_hash_mismatch"
    _assert_gate_hash(outcome)


def test_later_candidate_particle_failure_suppresses_completed_earlier_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path, candidate_count=2)
    fit_candidates = case[5]["candidate_results"]
    target_p = fit_candidates[1]["posterior"]["free_parameter_values"]["p"][0]
    first_candidate_ps = fit_candidates[0]["posterior"]["free_parameter_values"]["p"]
    assert target_p not in first_candidate_ps
    original_simulate = gate._forecast.simulate_cascaded_tanks
    calls: list[float] = []

    def fail_later_candidate_particle(inputs, parameters, initial_state, **kwargs):
        if len(inputs) == 768 and parameters.p == target_p:
            calls.append(parameters.p)
            return TankSimulationFailure(
                category=TankFailureCategory.NON_FINITE,
                message="synthetic later-candidate particle failure",
                step_index=23,
                trace=(),
                terminal_state=None,
            )
        return original_simulate(inputs, parameters, initial_state, **kwargs)

    monkeypatch.setattr(
        gate._forecast, "simulate_cascaded_tanks", fail_later_candidate_particle
    )

    outcome = _run_case(case)

    assert calls == [target_p]
    assert outcome.status == "terminal_failure"
    assert outcome.forecasts is None
    candidates = outcome.receipt["candidate_results"]
    assert len(candidates) == 2
    assert candidates[0]["candidate_id"] == "synthetic-s0"
    assert candidates[0]["status"] == "complete"
    assert candidates[0]["weighted_median_forecast_sha256"]
    assert candidates[1]["candidate_id"] == "synthetic-s0-1"
    assert candidates[1]["status"] == "terminal_failure"
    assert candidates[1]["weighted_median_forecast_sha256"] is None
    assert [row["status"] for row in candidates[1]["particle_results"]].count(
        "terminal_failure"
    ) == 1
    _assert_gate_hash(outcome)


def test_single_particle_forecast_failure_removes_all_comparison_forecasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    target_p = case[5]["candidate_results"][0]["posterior"]["free_parameter_values"][
        "p"
    ][0]
    original_simulate = gate._forecast.simulate_cascaded_tanks
    calls: list[float] = []

    def fail_one_particle(inputs, parameters, initial_state, **kwargs):
        if len(inputs) == 768 and parameters.p == target_p:
            calls.append(parameters.p)
            return TankSimulationFailure(
                category=TankFailureCategory.NON_FINITE,
                message="synthetic single-particle sentinel failure",
                step_index=19,
                trace=(),
                terminal_state=None,
            )
        return original_simulate(inputs, parameters, initial_state, **kwargs)

    monkeypatch.setattr(gate._forecast, "simulate_cascaded_tanks", fail_one_particle)

    outcome = _run_case(case)

    assert calls == [target_p]
    assert outcome.status == "terminal_failure"
    assert outcome.forecasts is None
    candidate = outcome.receipt["candidate_results"][0]
    assert candidate["status"] == "terminal_failure"
    assert [row["status"] for row in candidate["particle_results"]].count(
        "terminal_failure"
    ) == 1
    assert candidate["weighted_median_forecast_sha256"] is None
    _assert_gate_hash(outcome)


@pytest.mark.parametrize("artifact", ["fit receipt", "comparison declaration"])
def test_receipt_and_declaration_parse_the_original_same_inode_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    case = _make_case(tmp_path)
    (
        archive_path,
        expected_archive,
        manifest_path,
        receipt_path,
        declaration_path,
        *_,
    ) = case
    target_path = receipt_path if artifact == "fit receipt" else declaration_path
    original_bytes = target_path.read_bytes()
    original_inode = target_path.stat().st_ino
    parse = gate._parse_json_document
    mutated = False

    def mutate_after_snapshot(raw: bytes, label: str):
        nonlocal mutated
        if label == artifact and not mutated:
            with target_path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"!" * len(original_bytes))
                stream.flush()
            mutated = True
        return parse(raw, label)

    monkeypatch.setattr(gate, "_parse_json_document", mutate_after_snapshot)
    outcome = gate._run_synthetic_fixture_forecast_gate(
        archive_path,
        manifest_path,
        receipt_path,
        declaration_path,
        expected_archive=expected_archive,
    )

    assert mutated
    assert target_path.stat().st_ino == original_inode
    assert target_path.read_bytes() != original_bytes
    assert outcome.status == "complete"
    expected_key = (
        "fit_receipt_sha256"
        if artifact == "fit receipt"
        else "comparison_declaration_sha256"
    )
    assert outcome.receipt[expected_key] == hashlib.sha256(original_bytes).hexdigest()


def test_gate_forecasts_the_loaded_archive_snapshot_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_archive = _archive_bytes(_csv_bytes())
    case = _make_case(tmp_path, archive_data=original_archive)
    baseline = _run_case(case)
    (
        archive_path,
        expected_archive,
        manifest_path,
        receipt_path,
        declaration_path,
        *_,
    ) = case
    replacement_fields = {
        (index, "uEst"): (
            b"1.1234" if _numeric_field_width(index, 0) == 6 else b"1.123"
        )
        for index in range(768, 1024)
    }
    replacement_archive = _archive_bytes(_csv_bytes(mutate=replacement_fields))
    replacement_path = tmp_path / "replacement.zip"
    replacement_path.write_bytes(replacement_archive)
    original_loader = source._load_fixture_development_data
    replaced = False

    def replace_after_snapshot(path, *, expected_archive):
        nonlocal replaced
        data = original_loader(path, expected_archive=expected_archive)
        os.replace(replacement_path, path)
        replaced = True
        return data

    monkeypatch.setattr(
        source, "_load_fixture_development_data", replace_after_snapshot
    )
    outcome = gate._run_synthetic_fixture_forecast_gate(
        archive_path,
        manifest_path,
        receipt_path,
        declaration_path,
        expected_archive=expected_archive,
    )

    assert replaced
    assert (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != expected_archive.archive_sha256
    )
    assert baseline.status == outcome.status == "complete"
    assert outcome.receipt["archive_sha256"] == expected_archive.archive_sha256
    assert outcome.forecasts == baseline.forecasts


def test_hidden_target_mutation_changes_archive_hash_not_allowed_forecast_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _csv_bytes()
    hidden_changes: dict[tuple[int, str], bytes] = {}
    for index in range(1024):
        hidden_changes[(index, "uVal")] = b"u" * _numeric_field_width(index, 1)
        hidden_changes[(index, "yVal")] = b"v" * _numeric_field_width(index, 3)
        if index >= 768:
            hidden_changes[(index, "yEst")] = b"x" * _numeric_field_width(index, 2)
    changed_csv = _csv_bytes(mutate=hidden_changes)
    original_archive = _archive_bytes(original_csv)
    changed_archive = _archive_bytes(changed_csv)
    assert original_archive != changed_archive

    converted_fields: list[bytes] = []
    parse_number = source._parse_finite_number

    def capture_selected(raw: bytes, label: str) -> float:
        converted_fields.append(raw)
        return parse_number(raw, label)

    monkeypatch.setattr(source, "_parse_finite_number", capture_selected)
    baseline = _run_case(
        _make_case(tmp_path / "baseline", archive_data=original_archive)
    )
    baseline_converted = tuple(converted_fields)
    converted_fields.clear()
    changed = _run_case(_make_case(tmp_path / "changed", archive_data=changed_archive))

    assert baseline.status == changed.status == "complete"
    assert baseline.receipt["archive_sha256"] != changed.receipt["archive_sha256"]
    assert (
        baseline.receipt["source_stage_sha256"]
        == changed.receipt["source_stage_sha256"]
    )
    assert baseline.receipt["candidate_results"] == changed.receipt["candidate_results"]
    assert baseline.forecasts == changed.forecasts
    assert baseline_converted == tuple(converted_fields)
    assert not any(value in (b"u", b"v", b"x") for value in converted_fields)
    assert CSV_MEMBER_BYTES > 0

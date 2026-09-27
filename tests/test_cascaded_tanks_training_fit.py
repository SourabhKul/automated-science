from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.real_data import cascaded_tanks_controlled as source
from core.real_data import cascaded_tanks_training_fit as fit
from core.real_data.cascaded_tanks_controlled import (
    CSV_MEMBER,
    CSV_MEMBER_BYTES,
    SYNTHETIC_FIXTURE_CONTRACT_SHA256,
    SYNTHETIC_FIXTURE_TRACK_ID,
    ArchiveExpectation,
    _load_fixture_development_data,
)
from tests.test_cascaded_tanks_controlled import (
    _archive_bytes,
    _csv_bytes,
    _expectation,
    _numeric_field_width,
)


def _write_manifest(
    tmp_path: Path,
    archive_data: bytes,
    *,
    epsilon: float = 1.0e6,
    candidate_extra: dict[str, object] | None = None,
) -> tuple[Path, ArchiveExpectation, object]:
    archive_path = (
        tmp_path / f"fixture-{hashlib.sha256(archive_data).hexdigest()[:8]}.zip"
    )
    archive_path.write_bytes(archive_data)
    expectation = _expectation(archive_data)
    view = _load_fixture_development_data(archive_path, expected_archive=expectation)
    candidate: dict[str, object] = {
        "candidate_id": "synthetic-s0",
        "model": "S0",
        "fixed_parameters": {"a": 0.5, "c": 0.5, "x1_0": 0.0, "x2_0": 0.0},
        "free_parameter_bounds": {"p": [0.2, 1.2]},
        "target_samples": 3,
        "epsilon_schedule": [epsilon],
        "max_attempts_per_population": [3],
        "seed": 101,
        "output_units": "synthetic voltage-equivalent units",
        "simulation_limits": {"max_steps": 1000, "max_magnitude": 1.0e12},
        "discrepancy": "all_trajectory_rmse",
    }
    if candidate_extra:
        candidate.update(candidate_extra)
    manifest = {
        "schema": fit.FIT_MANIFEST_SCHEMA,
        "protocol_id": "synthetic-fixture-fit-protocol-v1",
        "run_id": f"fixture-{hashlib.sha256(archive_data).hexdigest()[:8]}",
        "scope": "synthetic-fixture-only",
        "source": {
            "track_id": SYNTHETIC_FIXTURE_TRACK_ID,
            "contract_sha256": SYNTHETIC_FIXTURE_CONTRACT_SHA256,
            "archive_sha256": expectation.archive_sha256,
            "archive_bytes": expectation.archive_bytes,
            "csv_member": CSV_MEMBER,
            "csv_member_bytes": CSV_MEMBER_BYTES,
            "training_indices": [0, 768],
            "stage_sha256": {
                "source_visible_sha256": view.source_visible_sha256,
                "training_sha256": view.stage_receipts.training_sha256,
                "forecast_inputs_sha256": view.stage_receipts.forecast_inputs_sha256,
            },
            "source_doi": "synthetic-fixture-only",
            "source_version": "synthetic-fixture-only",
        },
        "runtime": fit._current_runtime(),
        "code_sha256": fit._current_code_sha256(),
        "abc_kernel": {
            "covariance_scale": 2.0,
            "lambda_noise": 0.01,
            "nugget": 1.0e-9,
        },
        "candidates": [candidate],
    }
    manifest_path = tmp_path / f"{archive_path.stem}.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return archive_path, expectation, manifest_path


def _fit_fixture(
    archive_path: Path, manifest_path: Path, expectation: ArchiveExpectation
):
    return fit._run_synthetic_fixture_training_fit(
        archive_path, manifest_path, expected_archive=expectation
    )


def test_hidden_target_mutations_change_archive_identity_but_not_fit_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _csv_bytes()
    hidden_changes: dict[tuple[int, str], bytes] = {}
    for index in range(1024):
        hidden_changes[(index, "uVal")] = b"u" * _numeric_field_width(index, 1)
        hidden_changes[(index, "yVal")] = b"v" * _numeric_field_width(index, 3)
        if index >= 768:
            hidden_changes[(index, "yEst")] = b"x" * _numeric_field_width(index, 2)
    mutated_csv = _csv_bytes(mutate=hidden_changes)
    baseline_archive = _archive_bytes(original_csv)
    mutated_archive = _archive_bytes(mutated_csv)
    assert len(original_csv) == len(mutated_csv) == CSV_MEMBER_BYTES

    # Invalid sentinel bytes prove these excluded fields are never converted
    # by the source scanner. The only selected fields are uEst, training yEst,
    # and the row-zero sample interval.
    converted_fields: list[bytes] = []
    simulator_inputs: list[np.ndarray] = []
    parse_selected = source._parse_finite_number
    simulate = fit.simulate_cascaded_tanks

    def capture_selected(raw: bytes, label: str) -> float:
        converted_fields.append(raw)
        return parse_selected(raw, label)

    def capture_simulator_inputs(inputs, *args, **kwargs):
        simulator_inputs.append(np.asarray(inputs, dtype=np.float64).copy())
        return simulate(inputs, *args, **kwargs)

    monkeypatch.setattr(source, "_parse_finite_number", capture_selected)
    monkeypatch.setattr(fit, "simulate_cascaded_tanks", capture_simulator_inputs)
    baseline_path, baseline_expectation, baseline_manifest = _write_manifest(
        tmp_path, baseline_archive
    )
    baseline = _fit_fixture(baseline_path, baseline_manifest, baseline_expectation)
    baseline_converted = tuple(converted_fields)
    converted_fields.clear()

    mutated_path, mutated_expectation, mutated_manifest = _write_manifest(
        tmp_path, mutated_archive
    )
    changed = _fit_fixture(mutated_path, mutated_manifest, mutated_expectation)
    changed_converted = tuple(converted_fields)

    assert baseline["status"] == changed["status"] == "complete"
    assert baseline["source"]["archive_sha256"] != changed["source"]["archive_sha256"]
    assert baseline["source"]["stage_sha256"] == changed["source"]["stage_sha256"]
    assert baseline["training_fit_payload"] == changed["training_fit_payload"]
    assert (
        baseline["training_fit_payload_sha256"]
        == changed["training_fit_payload_sha256"]
    )
    assert baseline["candidate_results"] == changed["candidate_results"]
    assert baseline["manifest_sha256"] != changed["manifest_sha256"]
    assert baseline["receipt_sha256"] != changed["receipt_sha256"]
    assert (
        b"x" not in changed_converted
        and b"u" not in changed_converted
        and b"v" not in changed_converted
    )
    assert changed_converted == baseline_converted
    assert len(simulator_inputs) == 6
    assert all(values.shape == (768,) for values in simulator_inputs)
    assert all(
        np.array_equal(values, simulator_inputs[0]) for values in simulator_inputs
    )
    assert baseline["target_access"] == {
        "training_y_est_materialized": 768,
        "development_y_est_materialized": 0,
        "u_val_materialized": 0,
        "y_val_materialized": 0,
    }
    assert "development_y_est" not in baseline
    assert "y_val" not in baseline


def test_receipt_hash_covers_full_abc_diagnostics_and_manifest_binding(
    tmp_path: Path,
) -> None:
    archive_path, expectation, manifest_path = _write_manifest(
        tmp_path, _archive_bytes(_csv_bytes())
    )
    result = _fit_fixture(archive_path, manifest_path, expectation)

    assert result["status"] == "complete"
    assert result["complete"] is True
    assert result["all_declared_candidate_fits_complete"] is True
    assert result["candidate_roster_scope"] == "manifest_declared_candidates_only"
    assert result["development_score_eligible"] is False
    assert result["abc_reference_path"] == "gaussian_abc_smc_reference_opt_in"
    assert result["source"]["training_indices"] == [0, 768]
    assert result["source"]["track_id"] == SYNTHETIC_FIXTURE_TRACK_ID
    assert result["runtime_sha256"] == fit._canonical_sha256(result["runtime"])
    assert result["code_sha256"] == fit._current_code_sha256()
    candidate = result["candidate_results"][0]
    assert candidate["status"] == "complete"
    assert candidate["posterior"] is not None
    assert len(candidate["abc_diagnostics"]["populations"]) == 1
    assert candidate["abc_diagnostics"]["diagnostics"]["accepted"] == 3
    unsigned = {key: value for key, value in result.items() if key != "receipt_sha256"}
    assert result["receipt_sha256"] == fit._canonical_sha256(unsigned)


def test_incomplete_population_is_terminal_and_has_no_promotable_posterior(
    tmp_path: Path,
) -> None:
    archive_path, expectation, manifest_path = _write_manifest(
        tmp_path,
        _archive_bytes(_csv_bytes()),
        epsilon=0.0,
    )
    result = _fit_fixture(archive_path, manifest_path, expectation)

    assert result["status"] == "terminal_failure"
    assert result["complete"] is False
    assert result["all_declared_candidate_fits_complete"] is False
    assert result["development_score_eligible"] is False
    candidate = result["candidate_results"][0]
    assert candidate["status"] == "terminal_failure"
    assert candidate["posterior"] is None
    assert candidate["abc_diagnostics"]["status"] == "incomplete"
    assert candidate["abc_diagnostics"]["diagnostics"]["complete"] is False
    assert result["receipt_sha256"] == fit._canonical_sha256(
        {key: value for key, value in result.items() if key != "receipt_sha256"}
    )


def test_one_incomplete_candidate_removes_all_batch_posteriors(tmp_path: Path) -> None:
    archive_path, expectation, manifest_path = _write_manifest(
        tmp_path, _archive_bytes(_csv_bytes())
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    incomplete = dict(manifest["candidates"][0])
    incomplete.update(
        {
            "candidate_id": "synthetic-s0-incomplete",
            "epsilon_schedule": [0.0],
            "max_attempts_per_population": [3],
            "seed": 102,
        }
    )
    manifest["candidates"].append(incomplete)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    result = _fit_fixture(archive_path, manifest_path, expectation)

    assert result["status"] == "terminal_failure"
    assert result["complete"] is False
    assert result["all_declared_candidate_fits_complete"] is False
    assert result["development_score_eligible"] is False
    assert [item["status"] for item in result["candidate_results"]] == [
        "complete",
        "terminal_failure",
    ]
    assert all(item["posterior"] is None for item in result["candidate_results"])
    assert all(
        item["abc_diagnostics"] is not None for item in result["candidate_results"]
    )


def test_fit_entrypoint_rejects_caller_arrays_and_unverified_population_fields(
    tmp_path: Path,
) -> None:
    _archive_path, expectation, manifest_path = _write_manifest(
        tmp_path, _archive_bytes(_csv_bytes())
    )
    with pytest.raises(TypeError, match="never caller arrays"):
        fit.run_cascaded_tanks_training_fit(np.asarray([1.0]), manifest_path)

    bad_path, _, bad_manifest = _write_manifest(
        tmp_path,
        _archive_bytes(_csv_bytes()),
        candidate_extra={"population": [[0.5]]},
    )
    with pytest.raises(
        fit.CascadedTanksTrainingFitError, match="candidate specification"
    ):
        _fit_fixture(bad_path, bad_manifest, expectation)


def test_synthetic_manifest_cannot_enter_production_path(tmp_path: Path) -> None:
    archive_path, _, manifest_path = _write_manifest(
        tmp_path, _archive_bytes(_csv_bytes())
    )
    with pytest.raises(fit.CascadedTanksTrainingFitError, match="official-source"):
        fit.run_cascaded_tanks_training_fit(archive_path, manifest_path)


def test_stage_hash_mismatch_fails_before_abc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path, expectation, manifest_path = _write_manifest(
        tmp_path, _archive_bytes(_csv_bytes())
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["stage_sha256"]["training_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    called = False

    def unexpected_runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("ABC must not run when source stage pins mismatch")

    monkeypatch.setattr(fit, "run_gaussian_abc_smc_reference", unexpected_runner)
    with pytest.raises(fit.CascadedTanksTrainingFitError, match="stage hashes"):
        _fit_fixture(archive_path, manifest_path, expectation)
    assert called is False

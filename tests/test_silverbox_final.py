from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import core.real_data.silverbox_final as silverbox_final
from core import abc_smc_reference
from core.real_data import (
    silverbox_controlled,
    silverbox_first_fit,
    silverbox_proposer,
    silverbox_v2_manifest,
)


def _synthetic_archive(path: Path) -> tuple[Path, bytes, np.ndarray, np.ndarray]:
    """Create an index-coded source; never consult ignored real Silverbox data."""

    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("V1", "V2"))
    indices = np.arange(silverbox_final.SAMPLE_COUNT)
    input_source = indices.astype(float) / 1_000_000.0
    output_source = 0.125 + indices.astype(float) / 2_000_000.0
    writer.writerows(zip(input_source, output_source))
    payload = stream.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SilverboxFiles/SNLS80mV.csv", payload)
    return path, path.read_bytes(), input_source, output_source


def _sha_array(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_content_hashed(path: Path, body: dict) -> tuple[Path, str]:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {**body, "receipt_sha256": digest},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path, digest


def _code_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_receipts(tmp_path: Path, archive_bytes: bytes, *, selected: str = "nonlinear"):
    run_id = "synthetic-final"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    source_sha = hashlib.sha256(archive_bytes).hexdigest()
    orchestration_dir = tmp_path / "_orchestration" / run_id
    proposal_dir = orchestration_dir / "proposal_receipts"
    proposal_dir.mkdir(parents=True)
    source_archive_path = tmp_path / "synthetic-source.zip"
    source_archive_path.write_bytes(archive_bytes)
    run_started_unix = time.time()
    (orchestration_dir / "run_started.json").write_text(
        json.dumps(
            {
                "protocol_id": silverbox_first_fit.PROTOCOL_ID,
                "run_id": run_id,
                "status": "started",
                "started_unix": run_started_unix,
                "started_monotonic": time.monotonic(),
                "wall_limit_seconds": silverbox_first_fit.PILOT_WALL_SECONDS,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    falsifying_prediction = "The selected term should matter at a larger input amplitude."
    response_body = json.dumps(
        {
            "model": silverbox_proposer.MODEL_ID,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"term_id": "y_cubed", "falsifying_prediction": falsifying_prediction},
                            separators=(",", ":"),
                        ),
                    },
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    preflight_body = json.dumps(
        {"data": [{"id": silverbox_proposer.MODEL_ID}]}, separators=(",", ":")
    ).encode("utf-8")
    request_payload = silverbox_proposer._build_request_payload()
    proposal_receipt = {
        "status": "success",
        "parser_outcome": "success",
        "endpoint": silverbox_proposer.ENDPOINT,
        "timeout_seconds": silverbox_proposer.REQUEST_TIMEOUT_SECONDS,
        "chat_dispatched": True,
        "returned_model": silverbox_proposer.MODEL_ID,
        "finish_reason": "stop",
        "http_status": 200,
        "request_payload": request_payload,
        "request_payload_sha256": silverbox_proposer.FROZEN_REQUEST_PAYLOAD_SHA256,
        "raw_response_base64": base64.b64encode(response_body).decode("ascii"),
        "response_sha256": hashlib.sha256(response_body).hexdigest(),
        "model_preflight_status": "success",
        "model_preflight_endpoint": silverbox_proposer.MODELS_ENDPOINT,
        "model_preflight_timeout_seconds": silverbox_proposer.REQUEST_TIMEOUT_SECONDS,
        "model_preflight_http_status": 200,
        "model_preflight_raw_response_base64": base64.b64encode(preflight_body).decode("ascii"),
        "model_preflight_response_sha256": hashlib.sha256(preflight_body).hexdigest(),
        "model_preflight_model_ids": [silverbox_proposer.MODEL_ID],
        "term_id": "y_cubed",
        "falsifying_prediction": falsifying_prediction,
        "falsifying_prediction_status": "unscored_selection_independent",
    }
    proposal_path = proposal_dir / "synthetic-proposal.json"
    proposal_path.write_text(
        json.dumps(proposal_receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    proposal_bytes = proposal_path.read_bytes()
    proposal_sha = hashlib.sha256(proposal_bytes).hexdigest()
    (orchestration_dir / "source_verified.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "verified",
                "archive_path": str(source_archive_path),
                "archive_sha256": source_sha,
            }
        ),
        encoding="utf-8",
    )
    (orchestration_dir / "proposal_succeeded.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "success",
                "model_id": silverbox_proposer.MODEL_ID,
                "term_id": "y_cubed",
                "proposal_receipt_path": str(proposal_path),
                "proposal_receipt_sha256": proposal_sha,
                "request_payload_sha256": silverbox_proposer.FROZEN_REQUEST_PAYLOAD_SHA256,
                "response_sha256": hashlib.sha256(response_body).hexdigest(),
                "falsifying_prediction_status": "unscored_selection_independent",
                "validation_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    protocol_constants = {
        "abc_particles": silverbox_first_fit.ABC_PARTICLES,
        "abc_attempts_per_population": silverbox_first_fit.ABC_ATTEMPTS_PER_POPULATION,
        "linear_parameter_names": list(silverbox_first_fit.LINEAR_PARAMETER_NAMES),
        "nonlinear_parameter_names": list(silverbox_first_fit.NONLINEAR_PARAMETER_NAMES),
        "linear_bounds": [list(row) for row in silverbox_first_fit.LINEAR_BOUNDS],
        "nonlinear_bounds": [list(row) for row in silverbox_first_fit.NONLINEAR_BOUNDS],
    }
    hypotheses = {}
    for hypothesis, names, bounds in (
        ("linear", silverbox_first_fit.LINEAR_PARAMETER_NAMES, silverbox_first_fit.LINEAR_BOUNDS),
        ("nonlinear", silverbox_first_fit.NONLINEAR_PARAMETER_NAMES, silverbox_first_fit.NONLINEAR_BOUNDS),
    ):
        params = np.tile(np.mean(np.asarray(bounds, dtype=float), axis=1), (64, 1))
        params[:, 0] = np.linspace(bounds[0][0], bounds[0][1], 64)
        weights = np.full(64, 1.0 / 64.0)
        populations = []
        for generation in range(2):
            populations.append(
                {
                    "generation": generation,
                    "accepted_params": params.tolist(),
                    "weights": weights.tolist(),
                    "diagnostics": {
                        "complete": True,
                        "max_attempts": silverbox_first_fit.ABC_ATTEMPTS_PER_POPULATION,
                        "proposed": silverbox_first_fit.ABC_PARTICLES,
                    },
                }
            )
        hypotheses[hypothesis] = {
            "status": "complete",
            "parameter_names": list(names),
            "bounds": [list(row) for row in bounds],
            "calibration": {
                "protocol_id": silverbox_first_fit.PROTOCOL_ID,
                "status": "complete",
                "finite_count": 256,
            },
            "abc_result": {
                "status": "complete",
                "complete": True,
                "max_attempts_per_population": [
                    silverbox_first_fit.ABC_ATTEMPTS_PER_POPULATION
                ] * silverbox_first_fit.ABC_POPULATIONS,
                "populations": populations,
            },
        }

    fit_body = {
        "protocol_id": silverbox_first_fit.PROTOCOL_ID,
        "run_id": run_id,
        "status": "complete",
        "source_sha256": source_sha,
        "proposal_receipt_sha256": proposal_sha,
        "term_id": "y_cubed",
        "source_indices": [
            [start, start + silverbox_controlled.TRAIN_WINDOW_LENGTH]
            for start in (
                silverbox_controlled.TRAIN_SOURCE_START + offset
                for offset in silverbox_controlled.TRAIN_WINDOW_RELATIVE_STARTS
            )
        ],
        "sampling_time": 1.0 / 610.35,
        "protocol_constants": protocol_constants,
        "implementation_sha256": _code_hash(Path(silverbox_first_fit.__file__)),
        "controlled_contract_sha256": _code_hash(Path(silverbox_controlled.__file__)),
        "abc_reference_sha256": _code_hash(Path(abc_smc_reference.__file__)),
        "training_scalers": {"s_y": 0.1, "s_feature": 0.05, "divergence_bound": 1.0},
        "hypotheses": hypotheses,
        "budget": {"stop_reason": None, "wall_elapsed_seconds": 1.0},
        "selection": {"validation_accessed": False},
    }
    fit_path, fit_digest = _write_content_hashed(run_dir / "fit.json", fit_body)
    fit = silverbox_first_fit.FrozenSilverboxFit(
        run_id,
        fit_path,
        fit_digest,
        "complete",
        "y_cubed",
        source_sha,
        proposal_sha,
    )
    (orchestration_dir / "fit_completed.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "complete",
                "reason": None,
                "fit_receipt_path": str(fit_path),
                "fit_receipt_sha256": fit_digest,
                "proposal_receipt_sha256": proposal_sha,
                "validation_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    selection_forecasts = {}
    for hypothesis, value in (("linear", 0.20), ("nonlinear", 0.10)):
        trajectory = np.full(1_742, value, dtype=float)
        selection_forecasts[hypothesis] = {
            "status": "success",
            "pointwise_weighted_median": trajectory.tolist(),
            "forecast_sha256": _sha_array(trajectory),
        }
    selection_body = {
        "protocol_id": silverbox_first_fit.PROTOCOL_ID,
        "run_id": run_id,
        "status": "selected",
        "reason": "nonlinear_at_least_five_percent_below_linear",
        "selected_hypothesis": selected,
        "source_sha256": source_sha,
        "proposal_receipt_sha256": proposal_sha,
        "fit_receipt_sha256": fit_digest,
        "validation_accessed": True,
        "validation_source_range": [
            silverbox_controlled.VALIDATION_SOURCE_START,
            silverbox_controlled.VALIDATION_SOURCE_STOP,
        ],
        "validation_target_count": 1_742,
        "validation_input_sha256": hashlib.sha256(b"validation input").hexdigest(),
        "validation_initializer_sha256": hashlib.sha256(b"validation init").hexdigest(),
        "validation_target_sha256": hashlib.sha256(b"validation target").hexdigest(),
        "forecasts": selection_forecasts,
        "scores": {
            "rmse": {"linear": 0.50, "nonlinear": 0.10},
            "persistence_rmse": 0.60,
        },
        "selection_rule": {
            "minimum_margin": silverbox_first_fit.NONLINEAR_PROMOTION_MARGIN,
            "both_models_must_beat_persistence": True,
        },
    }
    selection_path, selection_digest = _write_content_hashed(run_dir / "selection.json", selection_body)
    selection = silverbox_first_fit.SilverboxSelection(
        "selected", selected, selection_path, selection_digest, selection_body["reason"]
    )
    (orchestration_dir / "selection_completed.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "selected",
                "reason": selection_body["reason"],
                "selected_hypothesis": selected,
                "selection_receipt_path": str(selection_path),
                "selection_receipt_sha256": selection_digest,
                "fit_receipt_sha256": fit_digest,
                "validation_accessed": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (orchestration_dir / "run_finished.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "selected",
                "reason": selection_body["reason"],
                "source_sha256": source_sha,
                "proposal_receipt_sha256": proposal_sha,
                "fit_receipt_sha256": fit_digest,
                "selection_receipt_sha256": selection_digest,
                "finished_unix": run_started_unix + 1.0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return fit, selection, fit_body, selection_body


def _convert_receipts_to_synthetic_v2(tmp_path: Path, archive_bytes: bytes, monkeypatch):
    fit, selection, fit_body, selection_body = _complete_receipts(tmp_path, archive_bytes)
    source_sha = hashlib.sha256(archive_bytes).hexdigest()
    protocol_id = silverbox_first_fit.V2_PROTOCOL_ID
    manifest_hashes = {
        name: format(index + 1, "064x")
        for index, name in enumerate(silverbox_v2_manifest.CODE_FILES)
    }
    actual = {
        "reviewed_git_commit": "a" * 40,
        "reviewed_tree_clean": True,
        "runtime": dict(silverbox_v2_manifest.PINNED_RUNTIME),
        "code_sha256": manifest_hashes,
    }
    manifest = {
        "protocol_id": protocol_id,
        "abc_attempts_per_population": 2_048,
        "reviewed_git_commit": actual["reviewed_git_commit"],
        "reviewed_tree_clean": True,
        "source": {"archive_sha256": source_sha, "archive_bytes": len(archive_bytes)},
        "runtime": dict(actual["runtime"]),
        "code_sha256": manifest_hashes,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    monkeypatch.setattr(
        silverbox_final.silverbox_v2_manifest,
        "current_environment",
        lambda _root: actual,
    )

    fit_body.pop("receipt_sha256", None)
    fit_body["protocol_id"] = protocol_id
    fit_body["abc_attempts_per_population"] = 2_048
    fit_body["preflight_manifest_sha256"] = manifest_sha
    fit_body["protocol_constants"]["abc_attempts_per_population"] = 2_048
    for family in fit_body["hypotheses"].values():
        family["calibration"]["protocol_id"] = protocol_id
        family["abc_result"]["max_attempts_per_population"] = [2_048, 2_048]
        for population in family["abc_result"]["populations"]:
            population["diagnostics"]["max_attempts"] = 2_048
    fit_path, fit_sha = _write_content_hashed(fit.receipt_path, fit_body)
    fit = replace(
        fit,
        receipt_path=fit_path,
        receipt_sha256=fit_sha,
        protocol_id=protocol_id,
        preflight_manifest_sha256=manifest_sha,
    )

    selection_body.pop("receipt_sha256", None)
    selection_body.update(
        {
            "protocol_id": protocol_id,
            "abc_attempts_per_population": 2_048,
            "preflight_manifest_sha256": manifest_sha,
            "fit_receipt_sha256": fit_sha,
        }
    )
    selection_path, selection_sha = _write_content_hashed(selection.receipt_path, selection_body)
    selection = replace(
        selection,
        receipt_path=selection_path,
        receipt_sha256=selection_sha,
        protocol_id=protocol_id,
        preflight_manifest_sha256=manifest_sha,
    )

    stage_dir = _orchestration_dir(fit)
    (stage_dir / "preflight_manifest.json").write_bytes(manifest_bytes)
    verified_payload = {
        "run_id": fit.run_id,
        "status": "verified",
        "reviewed_git_commit": actual["reviewed_git_commit"],
        "reviewed_tree_clean": True,
        "runtime_versions": actual["runtime"],
        "code_sha256": actual["code_sha256"],
        "archive_sha256": source_sha,
        "archive_bytes": len(archive_bytes),
        "protocol_id": protocol_id,
        "abc_attempts_per_population": 2_048,
        "preflight_manifest_sha256": manifest_sha,
    }
    verified_bytes = (json.dumps(verified_payload, sort_keys=True, indent=2) + "\n").encode()
    verified_sha = hashlib.sha256(verified_bytes).hexdigest()
    (stage_dir / "preflight_verified.json").write_bytes(verified_bytes)
    stage_values = {
        "run_started.json": {},
        "source_verified.json": {"archive_bytes": len(archive_bytes)},
        "proposal_succeeded.json": {},
        "fit_completed.json": {"fit_receipt_sha256": fit_sha, "fit_receipt_path": str(fit_path)},
        "selection_completed.json": {
            "selection_receipt_sha256": selection_sha,
            "selection_receipt_path": str(selection_path),
            "fit_receipt_sha256": fit_sha,
        },
        "run_finished.json": {
            "fit_receipt_sha256": fit_sha,
            "selection_receipt_sha256": selection_sha,
        },
    }
    for filename, updates in stage_values.items():
        path = stage_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        payload.update(
            {
                "protocol_id": protocol_id,
                "abc_attempts_per_population": 2_048,
                "preflight_manifest_sha256": manifest_sha,
            }
        )
        if filename in {
            "proposal_succeeded.json",
            "fit_completed.json",
            "selection_completed.json",
            "run_finished.json",
        }:
            payload["preflight_verified_sha256"] = verified_sha
        path.write_text(json.dumps(payload), encoding="utf-8")
    return fit, selection, fit_body, selection_body, manifest_sha, stage_dir


def _orchestration_dir(fit: silverbox_first_fit.FrozenSilverboxFit) -> Path:
    return fit.receipt_path.parent.parent / "_orchestration" / fit.run_id


def _assert_denied_before_archive_access(
    monkeypatch,
    fit: silverbox_first_fit.FrozenSilverboxFit,
    selection: silverbox_first_fit.SilverboxSelection,
    archive_path: Path,
    expected_error: str,
) -> None:
    calls = {"archive_read": 0, "archive_parser": 0, "target_suffix": 0}
    original_read_bytes = Path.read_bytes
    archive_resolved = archive_path.resolve()

    def counted_read_bytes(path: Path) -> bytes:
        if path.resolve() == archive_resolved:
            calls["archive_read"] += 1
            raise AssertionError("sealed archive must not be opened after a terminal-gate denial")
        return original_read_bytes(path)

    def forbidden_archive_parser(*args, **kwargs):
        calls["archive_parser"] += 1
        raise AssertionError("sealed multisine parser must not run after a terminal-gate denial")

    def forbidden_target_reader(self):
        calls["target_suffix"] += 1
        raise AssertionError("sealed target suffix must not be read after a terminal-gate denial")

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_archive_parser)
    monkeypatch.setattr(silverbox_final._MultisineFinalRecord, "_read_target_suffix", forbidden_target_reader)
    with pytest.raises(ValueError, match=expected_error):
        silverbox_final.score_silverbox_final(fit, selection, archive_path)
    assert calls == {"archive_read": 0, "archive_parser": 0, "target_suffix": 0}
    assert not (fit.receipt_path.parent / silverbox_final.FINAL_STARTED_NAME).exists()


@pytest.fixture
def synthetic_source(tmp_path):
    return _synthetic_archive(tmp_path / "index-coded-silverbox.zip")


def test_v2_final_gate_requires_matching_fit_selection_stage_manifest_and_cap(
    monkeypatch, tmp_path, synthetic_source
):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _, manifest_sha, stage_dir = _convert_receipts_to_synthetic_v2(
        tmp_path, archive_bytes, monkeypatch
    )
    locked = silverbox_final._verify_locked_inputs(fit, selection)
    assert locked.protocol_id == silverbox_first_fit.V2_PROTOCOL_ID
    assert locked.abc_attempts_per_population == 2_048
    assert locked.preflight_manifest_sha256 == manifest_sha
    assert locked.expected_archive_bytes == len(archive_bytes)
    final_binding = silverbox_final._protocol_receipt_fields(locked)
    assert final_binding["preflight_verified_sha256"] == locked.preflight_verified_sha256
    assert final_binding["reviewed_git_commit"] == "a" * 40
    assert final_binding["runtime_versions"] == silverbox_v2_manifest.PINNED_RUNTIME
    assert final_binding["code_sha256"] == locked.code_sha256
    assert final_binding["archive_bytes"] == len(archive_bytes)

    run_finished_path = stage_dir / "run_finished.json"
    run_finished = json.loads(run_finished_path.read_text(encoding="utf-8"))
    run_finished["abc_attempts_per_population"] = 512
    run_finished_path.write_text(json.dumps(run_finished), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol, cap, and manifest"):
        silverbox_final._verify_locked_inputs(fit, selection)
    assert not (fit.receipt_path.parent / silverbox_final.FINAL_STARTED_NAME).exists()


def test_v1_receipts_are_denied_when_handles_claim_v2_before_archive_access(
    monkeypatch, tmp_path, synthetic_source
):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    manifest_sha = "a" * 64
    mislabeled_fit = replace(
        fit,
        protocol_id=silverbox_first_fit.V2_PROTOCOL_ID,
        preflight_manifest_sha256=manifest_sha,
    )
    mislabeled_selection = replace(
        selection,
        protocol_id=silverbox_first_fit.V2_PROTOCOL_ID,
        preflight_manifest_sha256=manifest_sha,
    )
    with pytest.raises(ValueError, match="fit receipt protocol id"):
        silverbox_final.score_silverbox_final(mislabeled_fit, mislabeled_selection, archive_path)
    assert not (fit.receipt_path.parent / silverbox_final.FINAL_STARTED_NAME).exists()


def test_v2_selection_receipt_with_v1_cap_is_denied_before_archive_access(
    monkeypatch, tmp_path, synthetic_source
):
    _, archive_bytes, _, _ = synthetic_source
    fit, selection, _, selection_body, _, _ = _convert_receipts_to_synthetic_v2(
        tmp_path, archive_bytes, monkeypatch
    )
    selection_body.pop("receipt_sha256", None)
    selection_body["abc_attempts_per_population"] = 512
    path, digest = _write_content_hashed(selection.receipt_path, selection_body)
    selection = replace(selection, receipt_path=path, receipt_sha256=digest)
    with pytest.raises(ValueError, match="v2 selection receipt"):
        silverbox_final._verify_locked_inputs(fit, selection)
    assert not (fit.receipt_path.parent / silverbox_final.FINAL_STARTED_NAME).exists()


def test_final_scorer_uses_only_multisine_and_scores_all_particle_medians(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, input_source, output_source = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    calls = []

    def fake_simulator(input_u, initialization_y, parameters, **kwargs):
        calls.append((input_u, initialization_y, parameters, kwargs))
        assert input_u.shape == (silverbox_final.TEST_SAMPLE_COUNT,)
        assert initialization_y.shape == (50,)
        assert kwargs["hypothesis"] == "nonlinear"
        return np.full(silverbox_final.TEST_PREDICTION_COUNT, parameters[0], dtype=float)

    monkeypatch.setattr(silverbox_first_fit, "simulate_controlled_ar2", fake_simulator)
    result = silverbox_final.score_silverbox_final(fit, selection, archive_path)

    assert result.status == "complete"
    assert result.run_id == fit.run_id
    assert result.selected_hypothesis == "nonlinear"
    assert result.rmse == pytest.approx(
        np.sqrt(np.mean((np.full(silverbox_final.TEST_PREDICTION_COUNT, 1.35 + (1.57 - 1.35) * 31 / 63) - output_source[silverbox_final.TEST_SOURCE_START + 50 : silverbox_final.TEST_SOURCE_STOP]) ** 2))
    )
    assert len(calls) == silverbox_first_fit.ABC_PARTICLES
    assert all(call[0].shape == (silverbox_final.TEST_SAMPLE_COUNT,) for call in calls)
    assert all(call[1].shape == (50,) for call in calls)
    # The simulation-facing view carries one input record and its initializer;
    # no arrow candidate objects or arrays are part of this interface.
    record = silverbox_final._load_multisine_final_record(archive_bytes)
    assert not hasattr(record, "test_arrow_full")
    assert not hasattr(record, "test_arrow_no_extrapolation")
    np.testing.assert_array_equal(record.input_u, input_source[silverbox_final.TEST_SOURCE_START : silverbox_final.TEST_SOURCE_STOP])
    np.testing.assert_array_equal(record.initialization_y, output_source[silverbox_final.TEST_SOURCE_START : silverbox_final.TEST_SOURCE_START + 50])
    assert not hasattr(record, "_target_y")
    np.testing.assert_array_equal(
        record._read_target_suffix(),
        output_source[silverbox_final.TEST_SOURCE_START + 50 : silverbox_final.TEST_SOURCE_STOP],
    )

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["receipt_sha256"] == result.receipt_sha256
    assert receipt["status"] == "complete"
    assert receipt["source_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert receipt["input_sha256"] == _sha_array(input_source[silverbox_final.TEST_SOURCE_START : silverbox_final.TEST_SOURCE_STOP])
    assert receipt["target_sha256"] == _sha_array(output_source[silverbox_final.TEST_SOURCE_START + 50 : silverbox_final.TEST_SOURCE_STOP])
    assert receipt["forecast_sha256"] == _sha_array(
        np.full(silverbox_final.TEST_PREDICTION_COUNT, 1.35 + (1.57 - 1.35) * 31 / 63)
    )
    assert receipt["selection_receipt_sha256"] == selection.receipt_sha256

    with pytest.raises(ValueError, match="already has a final scoring attempt"):
        silverbox_final.score_silverbox_final(fit, selection, tmp_path / "does-not-exist.zip")


def test_unresolved_or_tampered_selection_is_rejected_before_archive_read(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    selection_body = json.loads(selection.receipt_path.read_text(encoding="utf-8"))
    selection_body.pop("receipt_sha256")
    selection_body["status"] = "unresolved"
    selection_body["selected_hypothesis"] = None
    path, digest = _write_content_hashed(selection.receipt_path, selection_body)
    unresolved = silverbox_first_fit.SilverboxSelection(
        "unresolved", None, path, digest, "validation_forecast_failure"
    )

    def forbidden_read(*args, **kwargs):
        raise AssertionError("archive must not be read after an unresolved selection")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="complete fit and selected development outcome"):
        silverbox_final.score_silverbox_final(fit, unresolved, archive_path)


def test_receipt_tampering_is_rejected_before_suffix_load(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    tampered = selection.receipt_path.read_text(encoding="utf-8").replace(
        "nonlinear_at_least_five_percent_below_linear", "tampered_selection_outcome"
    )
    selection.receipt_path.write_text(tampered, encoding="utf-8")

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("source parser must not run after receipt tampering")

    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_loader)
    with pytest.raises(ValueError, match="frozen receipt content hash mismatch"):
        silverbox_final.score_silverbox_final(fit, selection, archive_path)


def test_proposal_receipt_tampering_is_rejected_before_suffix_load(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    run_dir = fit.receipt_path.parent.parent / "_orchestration" / fit.run_id
    proposal_stage = json.loads((run_dir / "proposal_succeeded.json").read_text(encoding="utf-8"))
    proposal_path = Path(proposal_stage["proposal_receipt_path"])
    proposal_path.write_bytes(proposal_path.read_bytes() + b" ")

    parser_calls = 0
    target_reader_calls = 0

    def forbidden_loader(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        raise AssertionError("source parser must not run after proposal receipt tampering")

    def forbidden_target_reader(self):
        nonlocal target_reader_calls
        target_reader_calls += 1
        raise AssertionError("target suffix must not be read after proposal receipt tampering")

    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_loader)
    monkeypatch.setattr(silverbox_final._MultisineFinalRecord, "_read_target_suffix", forbidden_target_reader)
    with pytest.raises(ValueError, match="proposal receipt bytes do not match"):
        silverbox_final.score_silverbox_final(fit, selection, archive_path)
    assert parser_calls == 0
    assert target_reader_calls == 0


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("returned_model", "different-model", "exact successful frozen Qwen request"),
        ("term_id", "u_cubed", "response content does not match"),
    ],
)
def test_hash_consistent_forged_proposal_content_is_rejected_before_archive(
    monkeypatch, tmp_path, synthetic_source, field, value, expected_error
):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, fit_body, selection_body = _complete_receipts(tmp_path, archive_bytes)
    run_dir = fit.receipt_path.parent.parent / "_orchestration" / fit.run_id
    stage_path = run_dir / "proposal_succeeded.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    proposal_path = Path(stage["proposal_receipt_path"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal[field] = value
    proposal_path.write_text(json.dumps(proposal, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    proposal_sha = hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    stage["proposal_receipt_sha256"] = proposal_sha
    if field == "term_id":
        stage["term_id"] = value
    stage_path.write_text(json.dumps(stage, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    # Re-freeze the fit and selection handles around the forged raw-receipt
    # digest so the receipt content checks, rather than stale links, deny it.
    fit_body["proposal_receipt_sha256"] = proposal_sha
    if field == "term_id":
        fit_body["term_id"] = value
    fit_path, fit_sha = _write_content_hashed(fit.receipt_path, fit_body)
    forged_fit = silverbox_first_fit.FrozenSilverboxFit(
        fit.run_id, fit_path, fit_sha, "complete", fit_body["term_id"], fit.source_sha256, proposal_sha
    )
    selection_body["proposal_receipt_sha256"] = proposal_sha
    selection_body["fit_receipt_sha256"] = fit_sha
    selection_path, selection_sha = _write_content_hashed(selection.receipt_path, selection_body)
    forged_selection = silverbox_first_fit.SilverboxSelection(
        "selected", selection.selected_hypothesis, selection_path, selection_sha, selection.reason
    )
    parser_calls = 0
    suffix_calls = 0

    def forbidden_archive_parser(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        raise AssertionError("archive parser must not run after forged proposal content")

    def forbidden_target_reader(self):
        nonlocal suffix_calls
        suffix_calls += 1
        raise AssertionError("target suffix must not be read after forged proposal content")

    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_archive_parser)
    monkeypatch.setattr(silverbox_final._MultisineFinalRecord, "_read_target_suffix", forbidden_target_reader)
    with pytest.raises(ValueError, match=expected_error):
        silverbox_final.score_silverbox_final(forged_fit, forged_selection, archive_path)
    assert parser_calls == 0
    assert suffix_calls == 0


def test_source_proposal_chain_mismatch_is_rejected_before_archive_parser_or_suffix(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    run_dir = fit.receipt_path.parent.parent / "_orchestration" / fit.run_id
    source_path = run_dir / "source_verified.json"
    source_stage = json.loads(source_path.read_text(encoding="utf-8"))
    source_stage["archive_sha256"] = "0" * 64
    source_path.write_text(json.dumps(source_stage), encoding="utf-8")
    calls = {"archive_parser": 0, "target_suffix": 0}

    def forbidden_archive_parser(*args, **kwargs):
        calls["archive_parser"] += 1
        raise AssertionError("archive parser must not run after source/proposal chain rejection")

    def forbidden_target_reader(self):
        calls["target_suffix"] += 1
        raise AssertionError("target suffix must not be read after source/proposal chain rejection")

    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_archive_parser)
    monkeypatch.setattr(silverbox_final._MultisineFinalRecord, "_read_target_suffix", forbidden_target_reader)
    with pytest.raises(ValueError, match="source receipt does not bind"):
        silverbox_final.score_silverbox_final(fit, selection, archive_path)
    assert calls == {"archive_parser": 0, "target_suffix": 0}


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("status", "unresolved", "terminal run receipt"),
        ("run_id", "different-run", "terminal run receipt"),
        ("source_sha256", "0" * 64, "terminal run receipt"),
        ("proposal_receipt_sha256", "1" * 64, "terminal run receipt"),
        ("fit_receipt_sha256", "2" * 64, "terminal run receipt"),
        ("selection_receipt_sha256", "3" * 64, "terminal run receipt"),
        ("reason", "tampered-selection-reason", "terminal run receipt"),
        ("finished_unix", "late", "outside the fixed wall-time budget"),
    ],
)
def test_terminal_run_finished_mismatch_is_denied_before_archive_access(
    monkeypatch, tmp_path, synthetic_source, field, value, expected_error
):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    stage_dir = _orchestration_dir(fit)
    finished_path = stage_dir / "run_finished.json"
    finished = json.loads(finished_path.read_text(encoding="utf-8"))
    if field == "finished_unix" and value == "late":
        run_started = json.loads((stage_dir / "run_started.json").read_text(encoding="utf-8"))
        finished[field] = run_started["started_unix"] + silverbox_first_fit.PILOT_WALL_SECONDS
    else:
        finished[field] = value
    finished_path.write_text(json.dumps(finished, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    _assert_denied_before_archive_access(
        monkeypatch, fit, selection, archive_path, expected_error
    )


def test_absent_terminal_run_finished_is_denied_before_archive_access(
    monkeypatch, tmp_path, synthetic_source
):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    (_orchestration_dir(fit) / "run_finished.json").unlink()

    _assert_denied_before_archive_access(
        monkeypatch, fit, selection, archive_path, "run_finished.json"
    )


@pytest.mark.parametrize(
    ("stage_filename", "field", "value", "expected_error"),
    [
        ("fit_completed.json", "fit_receipt_path", "/different/fit.json", "fit receipt path"),
        (
            "selection_completed.json",
            "selection_receipt_path",
            "/different/selection.json",
            "selection receipt path",
        ),
        ("selection_completed.json", "status", "unresolved", "selection stage"),
    ],
)
def test_terminal_stage_path_or_status_mismatch_is_denied_before_archive_access(
    monkeypatch, tmp_path, synthetic_source, stage_filename, field, value, expected_error
):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    stage_path = _orchestration_dir(fit) / stage_filename
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage[field] = value
    stage_path.write_text(json.dumps(stage, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    _assert_denied_before_archive_access(
        monkeypatch, fit, selection, archive_path, expected_error
    )


def test_proposer_code_hash_is_checked_before_archive_parser(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    calls = {"archive_parser": 0, "target_suffix": 0}

    def forbidden_archive_parser(*args, **kwargs):
        calls["archive_parser"] += 1
        raise AssertionError("archive parser must not run after proposer implementation mismatch")

    def forbidden_target_reader(self):
        calls["target_suffix"] += 1
        raise AssertionError("target suffix must not be read after proposer implementation mismatch")

    monkeypatch.setattr(silverbox_final, "SILVERBOX_PROPOSER_SHA256", "0" * 64)
    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_archive_parser)
    monkeypatch.setattr(silverbox_final._MultisineFinalRecord, "_read_target_suffix", forbidden_target_reader)
    with pytest.raises(ValueError, match="Qwen proposer source hash"):
        silverbox_final.score_silverbox_final(fit, selection, archive_path)
    assert calls == {"archive_parser": 0, "target_suffix": 0}


def test_source_adapter_hash_is_checked_before_suffix_load(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("source parser must not run after adapter hash mismatch")

    monkeypatch.setattr(silverbox_final, "SILVERBOX_ADAPTER_SHA256", "0" * 64)
    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", forbidden_loader)
    with pytest.raises(ValueError, match="source adapter hash"):
        silverbox_final.score_silverbox_final(fit, selection, tmp_path / "sealed-archive-not-opened.zip")


def test_source_hash_gate_rejects_before_suffix_reader(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    wrong_source = tmp_path / "different-index-coded-archive.zip"
    wrong_source.write_bytes(archive_bytes + b"changed-source")
    suffix_reader_calls = 0

    def counted_reader(*args, **kwargs):
        nonlocal suffix_reader_calls
        suffix_reader_calls += 1
        raise AssertionError("suffix parser must not run after a source hash mismatch")

    monkeypatch.setattr(silverbox_final, "_load_multisine_final_record", counted_reader)
    result = silverbox_final.score_silverbox_final(fit, selection, wrong_source)
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))

    assert result.status == "failed"
    assert result.failure_mode == "source_hash_mismatch"
    assert suffix_reader_calls == 0
    assert receipt["target_sha256"] is None
    assert receipt["source_archive_sha256"] == hashlib.sha256(wrong_source.read_bytes()).hexdigest()


def test_particle_failure_is_terminal_without_target_hash(monkeypatch, tmp_path, synthetic_source):
    archive_path, archive_bytes, _, _ = synthetic_source
    fit, selection, _, _ = _complete_receipts(tmp_path, archive_bytes)
    calls = 0

    def one_failure(input_u, initialization_y, parameters, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise silverbox_first_fit.SimulationFailure("divergence_bound_exceeded", "synthetic failure")
        return np.zeros(silverbox_final.TEST_PREDICTION_COUNT)

    monkeypatch.setattr(silverbox_first_fit, "simulate_controlled_ar2", one_failure)
    result = silverbox_final.score_silverbox_final(fit, selection, archive_path)
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))

    assert result.status == "failed"
    assert result.failure_mode == "particle_simulation_failure"
    assert calls == silverbox_first_fit.ABC_PARTICLES
    assert receipt["target_sha256"] is None
    assert receipt["rmse"] is None
    assert receipt["particle_failures"] == [
        {
            "particle_index": 6,
            "failure_mode": "divergence_bound_exceeded",
            "detail": "divergence_bound_exceeded: synthetic failure",
        }
    ]

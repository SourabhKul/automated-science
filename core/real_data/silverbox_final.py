"""Single-use, multisine-only Silverbox scorer for a frozen first-fit choice.

The public entry point verifies the frozen fit and development-selection
receipts before it opens the archive. These receipts are self-content-hashed,
not cryptographically signed; the caller-held hashes freeze their contents.
It then loads only the official
``test_multisine`` input, its first 50 outputs, and a private scorer-held
suffix. Simulations receive only the input and initializer. Targets are hashed
and scored only after every selected-particle forecast succeeds.

This module is intentionally not a command-line runner. The scorer is a narrow
post-lock interface, not a general Silverbox test-data loader.
"""

from __future__ import annotations

import base64
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from core import abc_smc_reference
from core.real_data import (
    silverbox,
    silverbox_controlled,
    silverbox_first_fit,
    silverbox_proposer,
    silverbox_v2_manifest,
)
from core.real_data.silverbox import (
    MULTISINE_STOP,
    MULTISINE_TRAIN_STOP,
    SAMPLE_COUNT,
    SAMPLE_TIME_SECONDS,
    SNLS_CSV_MEMBER,
    STATE_INITIALIZATION_WINDOW_LENGTH,
)

FINAL_RECEIPT_NAME = "final.json"
FINAL_STARTED_NAME = "final_started.json"
TEST_RECORD_ID = "test_multisine"
TEST_SOURCE_START = MULTISINE_TRAIN_STOP
TEST_SOURCE_STOP = MULTISINE_STOP
TEST_SAMPLE_COUNT = TEST_SOURCE_STOP - TEST_SOURCE_START
TEST_INITIALIZATION_LENGTH = STATE_INITIALIZATION_WINDOW_LENGTH
TEST_PREDICTION_COUNT = TEST_SAMPLE_COUNT - TEST_INITIALIZATION_LENGTH
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
# Frozen source parser and split contract reviewed for this scorer. Any adapter
# change requires an explicit review and a new expected digest before scoring.
SILVERBOX_ADAPTER_SHA256 = "d1d81a1c4145352de95f7f8965a7bf991512569031ea50ede055c7bc22cda356"
SILVERBOX_PROPOSER_SHA256 = "dbc3d62201841fc04f62b84d0a16d018e59d5121791e704cd96daa290a26ec95"


@dataclass(frozen=True)
class SilverboxFinalScore:
    """Identity and compact outcome of one locked final-score attempt."""

    status: Literal["complete", "failed"]
    run_id: str
    selected_hypothesis: Literal["linear", "nonlinear"]
    receipt_path: Path
    receipt_sha256: str
    rmse: float | None
    failure_mode: str | None


@dataclass(frozen=True)
class _MultisineFinalRecord:
    """Multisine-only scorer view; the target suffix stays private."""

    source_start: int
    source_stop: int
    input_u: np.ndarray
    initialization_y: np.ndarray
    _archive_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        input_u = np.asarray(self.input_u, dtype=float)
        initialization_y = np.asarray(self.initialization_y, dtype=float)
        if self.source_start != TEST_SOURCE_START or self.source_stop != TEST_SOURCE_STOP:
            raise ValueError("final scorer accepts only the fixed official multisine range")
        if input_u.shape != (TEST_SAMPLE_COUNT,):
            raise ValueError("multisine input has the wrong fixed length")
        if initialization_y.shape != (TEST_INITIALIZATION_LENGTH,):
            raise ValueError("multisine initializer must contain exactly 50 outputs")
        if not isinstance(self._archive_bytes, bytes):
            raise ValueError("scorer-held source archive must be immutable bytes")
        if any(not np.all(np.isfinite(values)) for values in (input_u, initialization_y)):
            raise ValueError("multisine record arrays must be finite")
        for name, values in (
            ("input_u", input_u),
            ("initialization_y", initialization_y),
        ):
            frozen = np.array(values, dtype=float, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    def _read_target_suffix(self) -> np.ndarray:
        """Materialize the one allowed target slice after all forecasts pass."""

        start = self.source_start + TEST_INITIALIZATION_LENGTH
        target = np.empty(TEST_PREDICTION_COUNT, dtype=float)
        for sample_index, row in _iter_snls_csv_rows(self._archive_bytes):
            if start <= sample_index < self.source_stop:
                target[sample_index - start] = _finite_csv_value(row[1], "multisine target")
        if target.shape != (TEST_PREDICTION_COUNT,) or not np.all(np.isfinite(target)):
            raise ValueError("multisine target suffix is incomplete or nonfinite")
        target.setflags(write=False)
        return target


@dataclass(frozen=True)
class _LockedModel:
    run_id: str
    protocol_id: str
    abc_attempts_per_population: int
    preflight_manifest_sha256: str | None
    preflight_verified_sha256: str | None
    reviewed_git_commit: str | None
    runtime_versions: dict[str, str] | None
    code_sha256: dict[str, str] | None
    expected_archive_bytes: int | None
    selected_hypothesis: Literal["linear", "nonlinear"]
    term_id: str
    source_sha256: str
    fit_receipt: dict[str, Any]
    fit_receipt_sha256: str
    selection_receipt: dict[str, Any]
    selection_receipt_sha256: str
    fit_receipt_path: Path
    selection_receipt_path: Path
    proposal_receipt_path: Path
    proposal_receipt_sha256: str
    proposal_request_payload_sha256: str
    proposal_response_sha256: str
    proposal_source_sha256: str
    silverbox_adapter_sha256: str
    silverbox_proposer_sha256: str
    parameters: np.ndarray = field(repr=False, compare=False)
    weights: np.ndarray = field(repr=False, compare=False)
    scalers: dict[str, float] = field(repr=False, compare=False)


def score_silverbox_final(
    fit: silverbox_first_fit.FrozenSilverboxFit,
    selection: silverbox_first_fit.SilverboxSelection,
    raw_archive: str | Path,
) -> SilverboxFinalScore:
    """Score one frozen selected model on exactly the multisine test suffix.

    Frozen-input errors raise before opening ``raw_archive``. Once a valid
    selection claims its single score slot, any archive or simulation failure
    is written as the terminal final receipt for that run. A started or
    completed run can never be scored a second time.
    """

    locked = _verify_locked_inputs(fit, selection)
    run_dir = locked.fit_receipt_path.parent
    final_path = run_dir / FINAL_RECEIPT_NAME
    started_path = run_dir / FINAL_STARTED_NAME

    with _claim_final_attempt(run_dir, started_path, locked):
        try:
            archive_bytes = Path(raw_archive).read_bytes()
        except Exception as error:
            return _write_failure(
                locked,
                final_path,
                "archive_read_failure",
                f"{type(error).__name__}: {error}",
            )

        observed_source_sha256 = _sha256_bytes(archive_bytes)
        if locked.expected_archive_bytes is not None and len(archive_bytes) != locked.expected_archive_bytes:
            return _write_failure(
                locked,
                final_path,
                "source_size_mismatch",
                "archive bytes do not match the source size frozen in the v2 preflight manifest",
                source_archive_sha256=observed_source_sha256,
            )
        if observed_source_sha256 != locked.source_sha256:
            return _write_failure(
                locked,
                final_path,
                "source_hash_mismatch",
                "archive bytes do not match the source hash frozen during development",
                source_archive_sha256=observed_source_sha256,
            )

        try:
            record = _load_multisine_final_record(archive_bytes)
        except Exception as error:
            return _write_failure(
                locked,
                final_path,
                "multisine_source_invalid",
                f"{type(error).__name__}: {error}",
                source_archive_sha256=observed_source_sha256,
            )

        predictions: list[np.ndarray] = []
        failures: list[dict[str, Any]] = []
        bound = locked.scalers["divergence_bound"]
        for particle_index, theta in enumerate(locked.parameters):
            try:
                prediction = silverbox_first_fit.simulate_controlled_ar2(
                    record.input_u,
                    record.initialization_y,
                    theta,
                    hypothesis=locked.selected_hypothesis,
                    term_id=locked.term_id,
                    s_y=locked.scalers["s_y"],
                    s_feature=locked.scalers["s_feature"],
                    divergence_bound=bound,
                )
                prediction = np.asarray(prediction, dtype=float)
                if prediction.shape != (TEST_PREDICTION_COUNT,):
                    raise silverbox_first_fit.SimulationFailure(
                        "wrong_output_shape",
                        f"expected {(TEST_PREDICTION_COUNT,)}, received {prediction.shape}",
                    )
                if not np.all(np.isfinite(prediction)):
                    raise silverbox_first_fit.SimulationFailure(
                        "nonfinite_trajectory", "simulator returned nonfinite predictions"
                    )
                predictions.append(prediction)
            except Exception as error:
                failure_mode = (
                    error.failure_mode
                    if isinstance(error, silverbox_first_fit.SimulationFailure)
                    else "simulator_exception"
                )
                failures.append(
                    {
                        "particle_index": particle_index,
                        "failure_mode": failure_mode,
                        "detail": str(error),
                    }
                )
                # Keep one entry per particle while forbidding drop/renormalize.
                predictions.append(np.full(TEST_PREDICTION_COUNT, np.nan, dtype=float))

        if failures:
            return _write_failure(
                locked,
                final_path,
                "particle_simulation_failure",
                "at least one selected particle failed; the ensemble was not reduced",
                source_archive_sha256=observed_source_sha256,
                input_sha256=_sha256_array(record.input_u),
                initializer_sha256=_sha256_array(record.initialization_y),
                particle_failures=failures,
            )

        particle_paths = np.asarray(predictions, dtype=float)
        try:
            forecast = silverbox_first_fit.pointwise_weighted_median(particle_paths, locked.weights)
        except Exception as error:
            return _write_failure(
                locked,
                final_path,
                "ensemble_median_failure",
                f"{type(error).__name__}: {error}",
                source_archive_sha256=observed_source_sha256,
                input_sha256=_sha256_array(record.input_u),
                initializer_sha256=_sha256_array(record.initialization_y),
            )
        if forecast.shape != (TEST_PREDICTION_COUNT,) or not np.all(np.isfinite(forecast)):
            return _write_failure(
                locked,
                final_path,
                "invalid_ensemble_forecast",
                "weighted-median forecast has the wrong shape or contains nonfinite values",
                source_archive_sha256=observed_source_sha256,
                input_sha256=_sha256_array(record.input_u),
                initializer_sha256=_sha256_array(record.initialization_y),
            )

        # This is the first materialized test-target slice: every particle passed.
        try:
            target = record._read_target_suffix()
        except Exception as error:
            return _write_failure(
                locked,
                final_path,
                "target_read_failure",
                f"{type(error).__name__}: {error}",
                source_archive_sha256=observed_source_sha256,
                input_sha256=_sha256_array(record.input_u),
                initializer_sha256=_sha256_array(record.initialization_y),
            )
        rmse = silverbox_first_fit._rmse(forecast, target)
        if not math.isfinite(rmse):
            return _write_failure(
                locked,
                final_path,
                "nonfinite_final_rmse",
                "forecast and target did not produce a finite RMSE",
                source_archive_sha256=observed_source_sha256,
                input_sha256=_sha256_array(record.input_u),
                initializer_sha256=_sha256_array(record.initialization_y),
                target_sha256=_sha256_array(target),
                forecast_sha256=_sha256_array(forecast),
            )

        receipt = _base_final_receipt(locked)
        receipt.update(
            {
                "status": "complete",
                "failure_mode": None,
                "failure_detail": None,
                "source_archive_sha256": observed_source_sha256,
                "source_record_id": TEST_RECORD_ID,
                "source_indices": [TEST_SOURCE_START, TEST_SOURCE_STOP],
                "sampling_time": SAMPLE_TIME_SECONDS,
                "input_sample_count": TEST_SAMPLE_COUNT,
                "target_sample_count": TEST_PREDICTION_COUNT,
                "input_sha256": _sha256_array(record.input_u),
                "initializer_sha256": _sha256_array(record.initialization_y),
                "target_sha256": _sha256_array(target),
                "forecast_sha256": _sha256_array(forecast),
                "selected_parameters_sha256": _sha256_array(locked.parameters),
                "selected_weights_sha256": _sha256_array(locked.weights),
                "rmse": rmse,
                "particle_count": int(locked.parameters.shape[0]),
                "particle_failures": [],
            }
        )
        path, digest = _write_content_hashed_json(final_path, receipt)
        return SilverboxFinalScore(
            "complete",
            locked.run_id,
            locked.selected_hypothesis,
            path,
            digest,
            rmse,
            None,
        )


def _verify_locked_inputs(
    fit: silverbox_first_fit.FrozenSilverboxFit,
    selection: silverbox_first_fit.SilverboxSelection,
) -> _LockedModel:
    """Verify all frozen material without touching the sealed data archive."""

    if type(fit) is not silverbox_first_fit.FrozenSilverboxFit:
        raise TypeError("final scoring requires a FrozenSilverboxFit handle")
    if type(selection) is not silverbox_first_fit.SilverboxSelection:
        raise TypeError("final scoring requires a SilverboxSelection handle")
    if fit.status != "complete" or selection.status != "selected":
        raise ValueError("final scoring requires a complete fit and selected development outcome")
    if selection.selected_hypothesis not in ("linear", "nonlinear"):
        raise ValueError("selected development hypothesis is missing or invalid")
    if not isinstance(fit.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", fit.run_id):
        raise ValueError("frozen fit run id is invalid")
    protocol_id = getattr(fit, "protocol_id", silverbox_first_fit.PROTOCOL_ID)
    attempt_cap = silverbox_first_fit.protocol_attempt_cap(protocol_id)
    manifest_sha256 = getattr(fit, "preflight_manifest_sha256", None)
    if getattr(selection, "protocol_id", silverbox_first_fit.PROTOCOL_ID) != protocol_id:
        raise ValueError("fit and selection handles name different Silverbox protocols")
    if protocol_id == silverbox_first_fit.V2_PROTOCOL_ID:
        _verify_hash(manifest_sha256, "v2 preflight manifest SHA-256")
        if getattr(selection, "preflight_manifest_sha256", None) != manifest_sha256:
            raise ValueError("fit and selection handles name different v2 manifests")
    elif manifest_sha256 is not None or getattr(selection, "preflight_manifest_sha256", None) is not None:
        raise ValueError("v1 fit or selection cannot carry a v2 preflight manifest")

    fit_path = Path(fit.receipt_path)
    selection_path = Path(selection.receipt_path)
    if fit_path.name != "fit.json" or selection_path.name != "selection.json":
        raise ValueError("frozen fit and selection must use their canonical receipt files")
    if fit_path.parent.resolve() != selection_path.parent.resolve():
        raise ValueError("frozen fit and selection receipts do not belong to the same run")
    if fit_path.parent.name != fit.run_id:
        raise ValueError("frozen run directory does not match its run id")

    fit_receipt = _load_content_hashed_json(fit_path, fit.receipt_sha256)
    _verify_hash(fit.source_sha256, "fit source_sha256")
    _verify_hash(fit.proposal_receipt_sha256, "fit proposal_receipt_sha256")
    if (
        fit_receipt.get("run_id") != fit.run_id
        or fit_receipt.get("source_sha256") != fit.source_sha256
        or fit_receipt.get("proposal_receipt_sha256") != fit.proposal_receipt_sha256
        or fit_receipt.get("term_id") != fit.term_id
        or fit_receipt.get("status") != fit.status
    ):
        raise ValueError("fit handle metadata does not match its frozen receipt")
    if fit_receipt.get("protocol_id") != protocol_id:
        raise ValueError("fit receipt protocol id does not match the frozen fit handle")
    if protocol_id == silverbox_first_fit.V2_PROTOCOL_ID:
        if (
            fit_receipt.get("abc_attempts_per_population") != attempt_cap
            or fit_receipt.get("preflight_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("v2 fit receipt does not bind its frozen attempt cap and manifest")
    elif fit_receipt.get("abc_attempts_per_population") is not None:
        raise ValueError("v1 fit receipt cannot be relabeled with a v2 attempt cap")
    if fit.term_id not in silverbox_first_fit.TERM_IDS:
        raise ValueError("fit receipt term id is invalid")
    proposer_sha256 = _sha256_file(Path(silverbox_proposer.__file__))
    if proposer_sha256 != SILVERBOX_PROPOSER_SHA256:
        raise ValueError("Silverbox Qwen proposer source hash differs from the reviewed scorer contract")
    adapter_sha256 = _sha256_file(Path(silverbox.__file__))
    if adapter_sha256 != SILVERBOX_ADAPTER_SHA256:
        raise ValueError("Silverbox source adapter hash differs from the reviewed scorer contract")

    selection_receipt = _load_content_hashed_json(selection_path, selection.receipt_sha256)
    if (
        selection_receipt.get("protocol_id") != protocol_id
        or selection_receipt.get("run_id") != fit.run_id
        or selection_receipt.get("source_sha256") != fit.source_sha256
        or selection_receipt.get("proposal_receipt_sha256") != fit.proposal_receipt_sha256
        or selection_receipt.get("fit_receipt_sha256") != fit.receipt_sha256
        or selection_receipt.get("status") != selection.status
        or selection_receipt.get("selected_hypothesis") != selection.selected_hypothesis
    ):
        raise ValueError("selection handle does not match a frozen selection for this fit")
    if protocol_id == silverbox_first_fit.V2_PROTOCOL_ID:
        if (
            selection_receipt.get("abc_attempts_per_population") != attempt_cap
            or selection_receipt.get("preflight_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("v2 selection receipt does not bind its frozen attempt cap and manifest")
    elif selection_receipt.get("abc_attempts_per_population") is not None:
        raise ValueError("v1 selection receipt cannot be relabeled with a v2 attempt cap")

    model_data = _verify_complete_fit(fit_receipt, protocol_id=protocol_id)
    _verify_selected_development_receipt(selection_receipt)
    proposal = _verify_orchestration_proposal(
        fit_path=fit_path,
        run_id=fit.run_id,
        source_sha256=fit.source_sha256,
        term_id=fit.term_id,
        proposal_receipt_sha256=fit.proposal_receipt_sha256,
    )
    provenance = _verify_terminal_run_receipts(
        fit=fit,
        selection=selection,
        fit_path=fit_path,
        selection_path=selection_path,
        proposal_receipt_path=proposal["receipt_path"],
    )
    expected_archive_bytes = provenance["archive_bytes"] if provenance is not None else None
    parameters, weights = model_data[selection.selected_hypothesis]
    scalers_raw = fit_receipt["training_scalers"]
    scalers = {
        name: float(scalers_raw[name]) for name in ("s_y", "s_feature", "divergence_bound")
    }
    parameters = np.array(parameters, dtype=float, copy=True)
    weights = np.array(weights, dtype=float, copy=True)
    parameters.setflags(write=False)
    weights.setflags(write=False)
    return _LockedModel(
        run_id=fit.run_id,
        protocol_id=protocol_id,
        abc_attempts_per_population=attempt_cap,
        preflight_manifest_sha256=manifest_sha256,
        preflight_verified_sha256=(provenance["preflight_verified_sha256"] if provenance else None),
        reviewed_git_commit=(provenance["reviewed_git_commit"] if provenance else None),
        runtime_versions=(provenance["runtime_versions"] if provenance else None),
        code_sha256=(provenance["code_sha256"] if provenance else None),
        expected_archive_bytes=expected_archive_bytes,
        selected_hypothesis=selection.selected_hypothesis,
        term_id=fit.term_id,
        source_sha256=fit.source_sha256,
        fit_receipt=fit_receipt,
        fit_receipt_sha256=fit.receipt_sha256,
        selection_receipt=selection_receipt,
        selection_receipt_sha256=selection.receipt_sha256,
        fit_receipt_path=fit_path,
        selection_receipt_path=selection_path,
        proposal_receipt_path=proposal["receipt_path"],
        proposal_receipt_sha256=proposal["receipt_sha256"],
        proposal_request_payload_sha256=proposal["request_payload_sha256"],
        proposal_response_sha256=proposal["response_sha256"],
        proposal_source_sha256=proposal["source_sha256"],
        silverbox_adapter_sha256=adapter_sha256,
        silverbox_proposer_sha256=proposer_sha256,
        parameters=parameters,
        weights=weights,
        scalers=scalers,
    )


def _verify_terminal_run_receipts(
    *,
    fit: silverbox_first_fit.FrozenSilverboxFit,
    selection: silverbox_first_fit.SilverboxSelection,
    fit_path: Path,
    selection_path: Path,
    proposal_receipt_path: Path,
) -> dict[str, Any] | None:
    """Require the orchestrator's terminal selected record to bind this lock.

    ``run_finished.json`` is a plain stage receipt, not a cryptographic
    signature. Its terminal status is accepted only when the adjacent source,
    proposal, fit, and selection stages agree with the caller-held,
    content-hashed fit/selection receipts and their canonical paths.
    """

    stage_dir = fit_path.parent.parent / "_orchestration" / fit.run_id
    if not stage_dir.is_dir() or stage_dir.is_symlink():
        raise ValueError("terminal run receipt directory is missing or invalid")

    def load_stage(filename: str) -> dict[str, Any]:
        try:
            return _load_json_object(stage_dir / filename)
        except OSError as error:
            raise ValueError(f"required orchestration stage receipt {filename} is missing or unreadable") from error

    run_started = load_stage("run_started.json")
    source_stage = load_stage("source_verified.json")
    proposal_stage = load_stage("proposal_succeeded.json")
    fit_stage = load_stage("fit_completed.json")
    selection_stage = load_stage("selection_completed.json")
    run_finished = load_stage("run_finished.json")

    protocol_id = getattr(fit, "protocol_id", silverbox_first_fit.PROTOCOL_ID)
    attempt_cap = silverbox_first_fit.protocol_attempt_cap(protocol_id)
    manifest_sha256 = getattr(fit, "preflight_manifest_sha256", None)
    if protocol_id == silverbox_first_fit.V2_PROTOCOL_ID:
        _verify_hash(manifest_sha256, "v2 preflight manifest SHA-256")
        for stage in (run_started, source_stage, proposal_stage, fit_stage, selection_stage, run_finished):
            if (
                stage.get("protocol_id") != protocol_id
                or stage.get("abc_attempts_per_population") != attempt_cap
                or stage.get("preflight_manifest_sha256") != manifest_sha256
            ):
                raise ValueError("v2 orchestration stage does not bind the frozen protocol, cap, and manifest")
        manifest_path = stage_dir / "preflight_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("v2 orchestration is missing its immutable preflight manifest copy")
        manifest_bytes = manifest_path.read_bytes()
        if _sha256_bytes(manifest_bytes) != manifest_sha256:
            raise ValueError("v2 preflight manifest bytes do not match the frozen handle digest")
        manifest_raw, manifest_document, manifest_digest = silverbox_v2_manifest.read_manifest(manifest_path)
        if manifest_raw != manifest_bytes or manifest_digest != manifest_sha256:
            raise ValueError("v2 preflight manifest changed while verifying its receipt")
        archive_bytes = source_stage.get("archive_bytes")
        if type(archive_bytes) is not int or archive_bytes <= 0:
            raise ValueError("v2 source stage has no valid archive byte count")
        preflight_verified = load_stage("preflight_verified.json")
        preflight_verified_path = stage_dir / "preflight_verified.json"
        if preflight_verified_path.is_symlink() or not preflight_verified_path.is_file():
            raise ValueError("v2 preflight verification receipt is not a regular run-local file")
        preflight_verified_sha256 = _sha256_file(preflight_verified_path)
        if (
            preflight_verified.get("run_id") != fit.run_id
            or preflight_verified.get("status") != "verified"
            or preflight_verified.get("protocol_id") != protocol_id
            or preflight_verified.get("abc_attempts_per_population") != attempt_cap
            or preflight_verified.get("preflight_manifest_sha256") != manifest_sha256
            or preflight_verified.get("reviewed_git_commit") != manifest_document.get("reviewed_git_commit")
            or preflight_verified.get("reviewed_tree_clean") is not True
            or preflight_verified.get("runtime_versions") != manifest_document.get("runtime")
            or preflight_verified.get("code_sha256") != manifest_document.get("code_sha256")
            or preflight_verified.get("archive_sha256") != fit.source_sha256
            or preflight_verified.get("archive_bytes") != archive_bytes
        ):
            raise ValueError("v2 preflight verification receipt does not match its frozen manifest")
        for stage in (proposal_stage, fit_stage, selection_stage, run_finished):
            if stage.get("preflight_verified_sha256") != preflight_verified_sha256:
                raise ValueError("v2 run stage does not bind its verified provenance receipt")
        actual_environment = silverbox_v2_manifest.current_environment(
            Path(__file__).resolve().parents[2]
        )
        silverbox_v2_manifest.verify_manifest(
            manifest_document,
            actual=actual_environment,
            archive_sha256=fit.source_sha256,
            archive_bytes=archive_bytes,
        )
        if (
            preflight_verified.get("reviewed_git_commit") != actual_environment.get("reviewed_git_commit")
            or preflight_verified.get("reviewed_tree_clean") is not actual_environment.get("reviewed_tree_clean")
            or preflight_verified.get("runtime_versions") != actual_environment.get("runtime")
            or preflight_verified.get("code_sha256") != actual_environment.get("code_sha256")
        ):
            raise ValueError("v2 actual provenance differs from the run's preflight verification receipt")
        expected_archive_bytes: dict[str, Any] | None = {
            "archive_bytes": archive_bytes,
            "preflight_verified_sha256": preflight_verified_sha256,
            "reviewed_git_commit": preflight_verified["reviewed_git_commit"],
            "runtime_versions": preflight_verified["runtime_versions"],
            "code_sha256": preflight_verified["code_sha256"],
        }
    else:
        expected_archive_bytes = None

    if (
        run_started.get("protocol_id") != protocol_id
        or run_started.get("run_id") != fit.run_id
        or run_started.get("status") != "started"
        or not _finite_number(run_started.get("started_monotonic"))
        or not _finite_number(run_started.get("started_unix"))
        or run_started.get("wall_limit_seconds") != silverbox_first_fit.PILOT_WALL_SECONDS
    ):
        raise ValueError("orchestration run-start receipt does not match the fixed pilot")
    if (
        source_stage.get("run_id") != fit.run_id
        or source_stage.get("status") != "verified"
        or source_stage.get("archive_sha256") != fit.source_sha256
    ):
        raise ValueError("orchestration source stage does not bind the terminal run")
    source_path_raw = source_stage.get("archive_path")
    if not isinstance(source_path_raw, str) or not Path(source_path_raw).is_absolute():
        raise ValueError("orchestration source stage has no canonical archive path")

    if (
        proposal_stage.get("run_id") != fit.run_id
        or proposal_stage.get("status") != "success"
        or proposal_stage.get("proposal_receipt_sha256") != fit.proposal_receipt_sha256
    ):
        raise ValueError("orchestration proposal stage does not bind the terminal run")
    _require_stage_path(
        proposal_stage.get("proposal_receipt_path"),
        proposal_receipt_path,
        "proposal receipt",
    )

    if (
        fit_stage.get("run_id") != fit.run_id
        or fit_stage.get("status") != "complete"
        or fit_stage.get("fit_receipt_sha256") != fit.receipt_sha256
        or fit_stage.get("proposal_receipt_sha256") != fit.proposal_receipt_sha256
        or fit_stage.get("validation_accessed") is not False
    ):
        raise ValueError("orchestration fit stage does not bind the terminal run")
    _require_stage_path(fit_stage.get("fit_receipt_path"), fit_path, "fit receipt")

    if (
        selection_stage.get("run_id") != fit.run_id
        or selection_stage.get("status") != "selected"
        or selection_stage.get("reason") != selection.reason
        or selection_stage.get("selected_hypothesis") != selection.selected_hypothesis
        or selection_stage.get("selection_receipt_sha256") != selection.receipt_sha256
        or selection_stage.get("fit_receipt_sha256") != fit.receipt_sha256
        or selection_stage.get("validation_accessed") is not True
    ):
        raise ValueError("orchestration selection stage does not bind the frozen selection")
    _require_stage_path(
        selection_stage.get("selection_receipt_path"),
        selection_path,
        "selection receipt",
    )

    started_unix = float(run_started["started_unix"])
    finished_unix = run_finished.get("finished_unix")
    if not _finite_number(finished_unix):
        raise ValueError("terminal run receipt has no finite completion time")
    elapsed_wall = float(finished_unix) - started_unix
    if elapsed_wall < 0 or elapsed_wall >= silverbox_first_fit.PILOT_WALL_SECONDS:
        raise ValueError("terminal run receipt completed outside the fixed wall-time budget")
    if (
        run_finished.get("run_id") != fit.run_id
        or run_finished.get("status") != "selected"
        or run_finished.get("reason") != selection.reason
        or run_finished.get("source_sha256") != fit.source_sha256
        or run_finished.get("proposal_receipt_sha256") != fit.proposal_receipt_sha256
        or run_finished.get("fit_receipt_sha256") != fit.receipt_sha256
        or run_finished.get("selection_receipt_sha256") != selection.receipt_sha256
    ):
        raise ValueError("terminal run receipt is not selected or does not bind the frozen receipt chain")
    return expected_archive_bytes


def _require_stage_path(value: Any, expected: Path, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"orchestration stage has no {label} path")
    supplied = Path(value)
    if supplied.is_symlink():
        raise ValueError(f"orchestration {label} path must not be a symlink")
    try:
        observed = supplied.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"orchestration {label} path is missing") from error
    if observed != canonical or not observed.is_file():
        raise ValueError(f"orchestration {label} path does not match the frozen receipt")


def _finite_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
    )


def _verify_complete_fit(
    receipt: dict[str, Any], *, protocol_id: str | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    protocol_id = protocol_id or receipt.get("protocol_id")
    try:
        attempt_cap = silverbox_first_fit.protocol_attempt_cap(protocol_id)
    except ValueError as error:
        raise ValueError("fit receipt names an unsupported Silverbox protocol") from error
    if receipt.get("status") != "complete":
        raise ValueError("fit receipt is not complete")
    if receipt.get("protocol_id") != protocol_id:
        raise ValueError("fit receipt protocol id does not match the selected protocol")
    if protocol_id == silverbox_first_fit.V2_PROTOCOL_ID:
        if receipt.get("abc_attempts_per_population") != attempt_cap:
            raise ValueError("v2 fit receipt does not record its frozen 2,048 attempt cap")
    elif receipt.get("abc_attempts_per_population") is not None:
        raise ValueError("v1 fit receipt cannot carry a v2 attempt cap")
    budget = receipt.get("budget")
    if not isinstance(budget, dict) or budget.get("stop_reason") is not None:
        raise ValueError("fit receipt reports a stopped or missing pilot budget")
    elapsed = budget.get("wall_elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed >= silverbox_first_fit.PILOT_WALL_SECONDS:
        raise ValueError("fit receipt exceeded or omitted the fixed pilot wall-time budget")
    if not isinstance(receipt.get("selection"), dict) or receipt["selection"].get("validation_accessed") is not False:
        raise ValueError("fit receipt must precede separate validation selection")
    if receipt.get("sampling_time") != SAMPLE_TIME_SECONDS:
        raise ValueError("fit receipt does not use the native Silverbox sample interval")
    expected_train_ranges = [
        [silverbox_controlled.TRAIN_SOURCE_START + start,
         silverbox_controlled.TRAIN_SOURCE_START + start + silverbox_controlled.TRAIN_WINDOW_LENGTH]
        for start in silverbox_controlled.TRAIN_WINDOW_RELATIVE_STARTS
    ]
    if receipt.get("source_indices") != expected_train_ranges:
        raise ValueError("fit receipt training ranges differ from the approved development contract")

    code_hashes = {
        "implementation_sha256": Path(silverbox_first_fit.__file__),
        "controlled_contract_sha256": Path(silverbox_controlled.__file__),
        "abc_reference_sha256": Path(abc_smc_reference.__file__),
    }
    for field_name, path in code_hashes.items():
        recorded = receipt.get(field_name)
        _verify_hash(recorded, field_name)
        if recorded != _sha256_file(path):
            raise ValueError(f"frozen fit code hash mismatch for {field_name}")

    constants = receipt.get("protocol_constants")
    if not isinstance(constants, dict) or constants.get("abc_particles") != silverbox_first_fit.ABC_PARTICLES:
        raise ValueError("fit protocol constants do not match the approved particle count")
    if constants.get("abc_attempts_per_population") != attempt_cap:
        raise ValueError("fit protocol constants do not match the frozen attempt cap")
    if constants.get("linear_parameter_names") != list(silverbox_first_fit.LINEAR_PARAMETER_NAMES):
        raise ValueError("linear parameter schema differs from the approved protocol")
    if constants.get("nonlinear_parameter_names") != list(silverbox_first_fit.NONLINEAR_PARAMETER_NAMES):
        raise ValueError("nonlinear parameter schema differs from the approved protocol")
    if constants.get("linear_bounds") != [list(row) for row in silverbox_first_fit.LINEAR_BOUNDS]:
        raise ValueError("linear parameter support differs from the approved protocol")
    if constants.get("nonlinear_bounds") != [list(row) for row in silverbox_first_fit.NONLINEAR_BOUNDS]:
        raise ValueError("nonlinear parameter support differs from the approved protocol")

    scalers = receipt.get("training_scalers")
    if not isinstance(scalers, dict):
        raise ValueError("complete fit receipt is missing its frozen training scalers")
    for key in ("s_y", "s_feature", "divergence_bound"):
        value = scalers.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"fit receipt has an invalid {key}")

    families = receipt.get("hypotheses")
    if not isinstance(families, dict) or set(families) != set(silverbox_first_fit.HYPOTHESES):
        raise ValueError("fit receipt does not contain both required hypotheses")
    model_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for hypothesis in silverbox_first_fit.HYPOTHESES:
        family = families[hypothesis]
        if not isinstance(family, dict) or family.get("status") != "complete":
            raise ValueError(f"{hypothesis} fit is incomplete")
        names, bounds = _hypothesis_schema(hypothesis)
        if family.get("parameter_names") != list(names) or family.get("bounds") != [list(row) for row in bounds]:
            raise ValueError(f"{hypothesis} parameter schema differs from the approved protocol")
        calibration = family.get("calibration")
        if (
            not isinstance(calibration, dict)
            or calibration.get("status") != "complete"
            or calibration.get("protocol_id") != protocol_id
            or int(calibration.get("finite_count", 0)) < silverbox_first_fit.CALIBRATION_MIN_FINITE
        ):
            raise ValueError(f"{hypothesis} calibration is incomplete")
        abc_result = family.get("abc_result")
        populations = abc_result.get("populations") if isinstance(abc_result, dict) else None
        if (
            not isinstance(abc_result, dict)
            or abc_result.get("status") != "complete"
            or abc_result.get("complete") is not True
            or abc_result.get("max_attempts_per_population") != [attempt_cap] * silverbox_first_fit.ABC_POPULATIONS
            or abc_result.get("max_attempts_per_population") != [attempt_cap] * silverbox_first_fit.ABC_POPULATIONS
            or not isinstance(populations, list)
            or len(populations) != silverbox_first_fit.ABC_POPULATIONS
        ):
            raise ValueError(f"{hypothesis} ABC-SMC fit is incomplete")
        last_parameters: np.ndarray | None = None
        last_weights: np.ndarray | None = None
        for generation, population in enumerate(populations):
            if not isinstance(population, dict) or population.get("generation") != generation:
                raise ValueError(f"{hypothesis} population sequence is malformed")
            diagnostics = population.get("diagnostics")
            if (
                not isinstance(diagnostics, dict)
                or diagnostics.get("complete") is not True
                or diagnostics.get("max_attempts") != attempt_cap
                or diagnostics.get("proposed", attempt_cap + 1) > attempt_cap
            ):
                raise ValueError(f"{hypothesis} population {generation} is incomplete")
            parameters = np.asarray(population.get("accepted_params"), dtype=float)
            weights = np.asarray(population.get("weights"), dtype=float)
            if parameters.shape != (silverbox_first_fit.ABC_PARTICLES, len(names)):
                raise ValueError(f"{hypothesis} population has an invalid parameter matrix")
            if weights.shape != (silverbox_first_fit.ABC_PARTICLES,):
                raise ValueError(f"{hypothesis} population has invalid particle weights")
            if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError(f"{hypothesis} population has nonfinite parameters or invalid weights")
            lower = np.asarray(bounds, dtype=float)[:, 0]
            upper = np.asarray(bounds, dtype=float)[:, 1]
            if np.any(parameters < lower) or np.any(parameters > upper):
                raise ValueError(f"{hypothesis} parameters fall outside the frozen prior support")
            weight_total = math.fsum(float(value) for value in weights)
            if weight_total <= 0 or not math.isclose(weight_total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{hypothesis} weights are not normalized")
            last_parameters = parameters
            last_weights = weights
        assert last_parameters is not None and last_weights is not None
        model_data[hypothesis] = (last_parameters, last_weights)
    return model_data


def _verify_selected_development_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("validation_accessed") is not True:
        raise ValueError("selection receipt did not complete the fixed validation decision")
    if receipt.get("validation_source_range") != [
        silverbox_controlled.VALIDATION_SOURCE_START,
        silverbox_controlled.VALIDATION_SOURCE_STOP,
    ]:
        raise ValueError("selection receipt uses an unexpected validation interval")
    if receipt.get("validation_target_count") != silverbox_controlled.VALIDATION_SOURCE_STOP - silverbox_controlled.VALIDATION_SOURCE_START - 50:
        raise ValueError("selection receipt has an unexpected validation target count")
    for key in ("validation_input_sha256", "validation_initializer_sha256", "validation_target_sha256"):
        _verify_hash(receipt.get(key), key)
    forecasts = receipt.get("forecasts")
    if not isinstance(forecasts, dict) or set(forecasts) != set(silverbox_first_fit.HYPOTHESES):
        raise ValueError("selection receipt must contain both frozen validation forecasts")
    for hypothesis in silverbox_first_fit.HYPOTHESES:
        forecast = forecasts[hypothesis]
        if not isinstance(forecast, dict) or forecast.get("status") != "success":
            raise ValueError("selection receipt contains a failed validation forecast")
        values = np.asarray(forecast.get("pointwise_weighted_median"), dtype=float)
        if values.shape != (1_742,) or not np.all(np.isfinite(values)):
            raise ValueError("selection receipt contains an invalid validation forecast")
        _verify_hash(forecast.get("forecast_sha256"), f"{hypothesis} validation forecast hash")
        if _sha256_array(values) != forecast["forecast_sha256"]:
            raise ValueError(f"{hypothesis} validation forecast hash does not match its frozen values")

    scores = receipt.get("scores")
    if not isinstance(scores, dict) or not isinstance(scores.get("rmse"), dict):
        raise ValueError("selection receipt has no validation score")
    rmse: dict[str, float] = {}
    for hypothesis in silverbox_first_fit.HYPOTHESES:
        value = scores["rmse"].get(hypothesis)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("selection receipt contains an invalid validation RMSE")
        rmse[hypothesis] = float(value)
    persistence = scores.get("persistence_rmse")
    if not isinstance(persistence, (int, float)) or not math.isfinite(float(persistence)) or float(persistence) < 0:
        raise ValueError("selection receipt contains an invalid persistence RMSE")
    if all(value >= float(persistence) for value in rmse.values()):
        raise ValueError("selection receipt should be unresolved when both models fail to beat persistence")
    selection_rule = receipt.get("selection_rule")
    if (
        not isinstance(selection_rule, dict)
        or selection_rule.get("minimum_margin") != silverbox_first_fit.NONLINEAR_PROMOTION_MARGIN
        or selection_rule.get("both_models_must_beat_persistence") is not True
    ):
        raise ValueError("selection receipt does not retain the approved selection rule")
    selected = receipt.get("selected_hypothesis")
    reason = receipt.get("reason")
    if selected == "nonlinear":
        if rmse["nonlinear"] > (1.0 - silverbox_first_fit.NONLINEAR_PROMOTION_MARGIN) * rmse["linear"]:
            raise ValueError("nonlinear selection does not satisfy the frozen 5% margin")
        if reason != "nonlinear_at_least_five_percent_below_linear":
            raise ValueError("nonlinear selection reason does not match its outcome")
    elif selected == "linear":
        if rmse["nonlinear"] <= (1.0 - silverbox_first_fit.NONLINEAR_PROMOTION_MARGIN) * rmse["linear"]:
            raise ValueError("linear selection conflicts with the frozen 5% margin")
        if reason != "linear_retained_by_five_percent_rule":
            raise ValueError("linear selection reason does not match its outcome")
    else:
        raise ValueError("selection receipt does not select a valid hypothesis")


def _verify_orchestration_proposal(
    *,
    fit_path: Path,
    run_id: str,
    source_sha256: str,
    term_id: str,
    proposal_receipt_sha256: str,
) -> dict[str, Any]:
    """Verify the exact proposal receipt bound by the frozen fit.

    The fit receipt stores the proposal digest but not its path. Resolve the
    companion orchestration record from the canonical run layout, then verify
    the on-disk receipt, exact request, preflight, raw response, and term.
    """

    root = fit_path.parent.parent
    stage_dir = root / "_orchestration" / run_id
    source_stage = _load_json_object(stage_dir / "source_verified.json")
    if (
        source_stage.get("run_id") != run_id
        or source_stage.get("status") != "verified"
        or source_stage.get("archive_sha256") != source_sha256
    ):
        raise ValueError("orchestration source receipt does not bind this fit's source hash")
    proposal_stage = _load_json_object(stage_dir / "proposal_succeeded.json")
    if (
        proposal_stage.get("run_id") != run_id
        or proposal_stage.get("status") != "success"
        or proposal_stage.get("model_id") != silverbox_proposer.MODEL_ID
        or proposal_stage.get("term_id") != term_id
        or proposal_stage.get("proposal_receipt_sha256") != proposal_receipt_sha256
        or proposal_stage.get("request_payload_sha256") != silverbox_proposer.FROZEN_REQUEST_PAYLOAD_SHA256
        or proposal_stage.get("validation_accessed") is not False
        or proposal_stage.get("falsifying_prediction_status") != "unscored_selection_independent"
    ):
        raise ValueError("orchestration proposal receipt link does not match the frozen fit")
    # The current orchestration schema places the source digest in
    # source_verified.json and the proposal digest in proposal_succeeded.json.
    # The content-hashed fit receipt freezes both values for this same run.
    # If a future stage schema repeats the source hash here, require equality.
    if "source_sha256" in proposal_stage and proposal_stage["source_sha256"] != source_sha256:
        raise ValueError("orchestration proposal stage is bound to a different source hash")
    receipt_path_raw = proposal_stage.get("proposal_receipt_path")
    if not isinstance(receipt_path_raw, str) or not receipt_path_raw:
        raise ValueError("orchestration proposal stage has no receipt path")
    receipt_path = Path(receipt_path_raw).resolve()
    expected_receipt_dir = (stage_dir / "proposal_receipts").resolve()
    if receipt_path.parent != expected_receipt_dir:
        raise ValueError("proposal receipt path escapes the run's fixed proposal directory")
    receipt_files = sorted(path.resolve() for path in expected_receipt_dir.glob("*.json"))
    if receipt_files != [receipt_path]:
        raise ValueError("proposal directory does not contain exactly the frozen successful receipt")
    receipt_bytes = receipt_path.read_bytes()
    observed_digest = _sha256_bytes(receipt_bytes)
    if observed_digest != proposal_receipt_sha256:
        raise ValueError("proposal receipt bytes do not match the fit's frozen proposal hash")
    receipt = _loads_json_object(receipt_bytes)

    expected_payload = silverbox_proposer._build_request_payload()
    request_payload = receipt.get("request_payload")
    payload_digest = (
        _sha256_bytes(_canonical_json(request_payload)) if isinstance(request_payload, dict) else None
    )
    if (
        receipt.get("status") != "success"
        or receipt.get("parser_outcome") != "success"
        or receipt.get("returned_model") != silverbox_proposer.MODEL_ID
        or receipt.get("finish_reason") != "stop"
        or receipt.get("endpoint") != silverbox_proposer.ENDPOINT
        or receipt.get("timeout_seconds") != silverbox_proposer.REQUEST_TIMEOUT_SECONDS
        or receipt.get("model_preflight_endpoint") != silverbox_proposer.MODELS_ENDPOINT
        or receipt.get("model_preflight_timeout_seconds") != silverbox_proposer.REQUEST_TIMEOUT_SECONDS
        or request_payload != expected_payload
        or receipt.get("request_payload_sha256") != silverbox_proposer.FROZEN_REQUEST_PAYLOAD_SHA256
        or payload_digest != silverbox_proposer.FROZEN_REQUEST_PAYLOAD_SHA256
        or receipt.get("chat_dispatched") is not True
        or receipt.get("term_id") != term_id
        or receipt.get("falsifying_prediction_status") != "unscored_selection_independent"
    ):
        raise ValueError("proposal receipt does not contain the exact successful frozen Qwen request")

    preflight_bytes = _decode_receipt_base64(receipt.get("model_preflight_raw_response_base64"))
    if (
        receipt.get("model_preflight_status") != "success"
        or receipt.get("model_preflight_http_status") != 200
        or receipt.get("model_preflight_response_sha256") != _sha256_bytes(preflight_bytes)
    ):
        raise ValueError("proposal receipt model preflight metadata is invalid")
    preflight = _loads_json_object(preflight_bytes)
    model_rows = preflight.get("data")
    listed_ids = [item.get("id") for item in model_rows if isinstance(item, dict)] if isinstance(model_rows, list) else []
    if (
        silverbox_proposer.MODEL_ID not in listed_ids
        or receipt.get("model_preflight_model_ids") != listed_ids
    ):
        raise ValueError("proposal receipt model preflight did not list the exact Qwen model")

    response_bytes = _decode_receipt_base64(receipt.get("raw_response_base64"))
    if (
        receipt.get("http_status") != 200
        or receipt.get("response_sha256") != _sha256_bytes(response_bytes)
        or proposal_stage.get("response_sha256") != receipt.get("response_sha256")
    ):
        raise ValueError("proposal response bytes do not match the frozen response hash")
    parsed_proposal, _ = silverbox_proposer._parse_chat_response(response_bytes, 200)
    if (
        parsed_proposal.get("term_id") != term_id
        or parsed_proposal.get("falsifying_prediction") != receipt.get("falsifying_prediction")
    ):
        raise ValueError("proposal response content does not match the frozen term and receipt")
    return {
        "receipt_path": receipt_path,
        "receipt_sha256": observed_digest,
        "request_payload_sha256": silverbox_proposer.FROZEN_REQUEST_PAYLOAD_SHA256,
        "response_sha256": receipt["response_sha256"],
        "source_sha256": source_sha256,
    }


def _load_multisine_final_record(archive_bytes: bytes) -> _MultisineFinalRecord:
    """Read only multisine inputs and its 50 initializer outputs."""

    test_input = np.empty(TEST_SAMPLE_COUNT, dtype=float)
    initializer = np.empty(TEST_INITIALIZATION_LENGTH, dtype=float)
    initializer_stop = TEST_SOURCE_START + TEST_INITIALIZATION_LENGTH
    for sample_index, row in _iter_snls_csv_rows(archive_bytes):
        if TEST_SOURCE_START <= sample_index < TEST_SOURCE_STOP:
            test_input[sample_index - TEST_SOURCE_START] = _finite_csv_value(row[0], "multisine input")
        if TEST_SOURCE_START <= sample_index < initializer_stop:
            initializer[sample_index - TEST_SOURCE_START] = _finite_csv_value(row[1], "multisine initializer")
    return _MultisineFinalRecord(
        source_start=TEST_SOURCE_START,
        source_stop=TEST_SOURCE_STOP,
        input_u=test_input,
        initialization_y=initializer,
        _archive_bytes=archive_bytes,
    )


def _iter_snls_csv_rows(archive_bytes: bytes) -> Iterator[tuple[int, list[str]]]:
    """Validate the canonical CSV framing without converting unrelated outputs."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        if SNLS_CSV_MEMBER not in names:
            raise ValueError(f"archive is missing official multisine source member {SNLS_CSV_MEMBER}")
        with archive.open(SNLS_CSV_MEMBER) as raw:
            import io as _io

            with _io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                reader = csv.reader(text)
                try:
                    header = next(reader)
                except StopIteration as error:
                    raise ValueError("Silverbox SNLS CSV is empty") from error
                if header[:2] != ["V1", "V2"] or any(cell.strip() for cell in header[2:]):
                    raise ValueError(f"unexpected Silverbox SNLS header: {header}")
                sample_index = 0
                seen_blank_row = False
                for row_number, row in enumerate(reader, start=2):
                    if not any(cell.strip() for cell in row):
                        seen_blank_row = True
                        continue
                    if seen_blank_row:
                        raise ValueError(f"non-empty Silverbox SNLS row follows trailing blank row {row_number}")
                    if len(row) < 2 or any(cell.strip() for cell in row[2:]):
                        raise ValueError(f"unexpected Silverbox SNLS row {row_number}: {row}")
                    if sample_index >= SAMPLE_COUNT:
                        raise ValueError(f"Silverbox SNLS CSV has more than {SAMPLE_COUNT} samples")
                    yield sample_index, row
                    sample_index += 1
                if sample_index != SAMPLE_COUNT:
                    raise ValueError(
                        f"Silverbox SNLS CSV must contain {SAMPLE_COUNT} samples, found {sample_index}"
                    )


def _finite_csv_value(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric {label} in Silverbox SNLS CSV") from error
    if not math.isfinite(parsed):
        raise ValueError(f"nonfinite {label} in Silverbox SNLS CSV")
    return parsed


@contextmanager
def _claim_final_attempt(run_dir: Path, started_path: Path, locked: _LockedModel) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".silverbox_final.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("a Silverbox final score is already in progress for this run") from error
        final_path = run_dir / FINAL_RECEIPT_NAME
        if started_path.exists() or final_path.exists():
            raise ValueError("this run already has a final scoring attempt")
        _write_content_hashed_json(
            started_path,
            {
                **_protocol_receipt_fields(locked),
                "run_id": locked.run_id,
                "status": "started",
                "fit_receipt_sha256": locked.fit_receipt_sha256,
                "selection_receipt_sha256": locked.selection_receipt_sha256,
                "started_unix": time.time(),
            },
        )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_failure(
    locked: _LockedModel,
    final_path: Path,
    failure_mode: str,
    detail: str,
    *,
    source_archive_sha256: str | None = None,
    input_sha256: str | None = None,
    initializer_sha256: str | None = None,
    target_sha256: str | None = None,
    forecast_sha256: str | None = None,
    particle_failures: list[dict[str, Any]] | None = None,
) -> SilverboxFinalScore:
    receipt = _base_final_receipt(locked)
    receipt.update(
        {
            "status": "failed",
            "failure_mode": failure_mode,
            "failure_detail": detail,
            "source_archive_sha256": source_archive_sha256,
            "source_record_id": TEST_RECORD_ID,
            "source_indices": [TEST_SOURCE_START, TEST_SOURCE_STOP],
            "sampling_time": SAMPLE_TIME_SECONDS,
            "input_sample_count": TEST_SAMPLE_COUNT,
            "target_sample_count": TEST_PREDICTION_COUNT,
            "input_sha256": input_sha256,
            "initializer_sha256": initializer_sha256,
            "target_sha256": target_sha256,
            "forecast_sha256": forecast_sha256,
            "selected_parameters_sha256": _sha256_array(locked.parameters),
            "selected_weights_sha256": _sha256_array(locked.weights),
            "rmse": None,
            "particle_count": int(locked.parameters.shape[0]),
            "particle_failures": particle_failures or [],
        }
    )
    path, digest = _write_content_hashed_json(final_path, receipt)
    return SilverboxFinalScore(
        "failed",
        locked.run_id,
        locked.selected_hypothesis,
        path,
        digest,
        None,
        failure_mode,
    )


def _base_final_receipt(locked: _LockedModel) -> dict[str, Any]:
    return {
        **_protocol_receipt_fields(locked),
        "run_id": locked.run_id,
        "created_unix": time.time(),
        "scorer_implementation_sha256": _sha256_file(Path(__file__)),
        "source_sha256": locked.source_sha256,
        "fit_receipt_path": str(locked.fit_receipt_path),
        "fit_receipt_sha256": locked.fit_receipt_sha256,
        "selection_receipt_path": str(locked.selection_receipt_path),
        "selection_receipt_sha256": locked.selection_receipt_sha256,
        "proposal_receipt_path": str(locked.proposal_receipt_path),
        "proposal_receipt_sha256": locked.proposal_receipt_sha256,
        "proposal_request_payload_sha256": locked.proposal_request_payload_sha256,
        "proposal_response_sha256": locked.proposal_response_sha256,
        "proposal_source_sha256": locked.proposal_source_sha256,
        "silverbox_adapter_sha256": locked.silverbox_adapter_sha256,
        "silverbox_proposer_sha256": locked.silverbox_proposer_sha256,
        "selected_hypothesis": locked.selected_hypothesis,
        "term_id": locked.term_id,
    }


def _protocol_receipt_fields(locked: _LockedModel) -> dict[str, Any]:
    fields: dict[str, Any] = {"protocol_id": locked.protocol_id}
    if locked.protocol_id == silverbox_first_fit.V2_PROTOCOL_ID:
        fields.update(
            {
                "abc_attempts_per_population": locked.abc_attempts_per_population,
                "preflight_manifest_sha256": locked.preflight_manifest_sha256,
                "preflight_verified_sha256": locked.preflight_verified_sha256,
                "reviewed_git_commit": locked.reviewed_git_commit,
                "runtime_versions": locked.runtime_versions,
                "code_sha256": locked.code_sha256,
                "archive_bytes": locked.expected_archive_bytes,
            }
        )
    return fields


def _hypothesis_schema(hypothesis: str) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]]:
    if hypothesis == "linear":
        return silverbox_first_fit.LINEAR_PARAMETER_NAMES, silverbox_first_fit.LINEAR_BOUNDS
    if hypothesis == "nonlinear":
        return silverbox_first_fit.NONLINEAR_PARAMETER_NAMES, silverbox_first_fit.NONLINEAR_BOUNDS
    raise ValueError(f"unsupported Silverbox hypothesis: {hypothesis}")


def _load_content_hashed_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    _verify_hash(expected_sha256, "receipt SHA-256")
    value = _loads_json_object(path.read_bytes())
    embedded_digest = value.pop("receipt_sha256", None)
    canonical = _canonical_json(value)
    computed_digest = hashlib.sha256(canonical).hexdigest()
    if embedded_digest != expected_sha256 or computed_digest != expected_sha256:
        raise ValueError("frozen receipt content hash mismatch")
    value["receipt_sha256"] = embedded_digest
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    return _loads_json_object(path.read_bytes())


def _loads_json_object(value: bytes) -> dict[str, Any]:
    parsed = json.loads(
        value.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_json_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("receipt must contain a JSON object")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _decode_receipt_base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ValueError("proposal receipt is missing raw response bytes")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError("proposal receipt contains invalid base64 bytes") from error


def _write_content_hashed_json(path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_json(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    serialized = _canonical_json({**payload, "receipt_sha256": digest}).decode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return path, digest


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _verify_hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")


__all__ = ["SilverboxFinalScore", "score_silverbox_final"]

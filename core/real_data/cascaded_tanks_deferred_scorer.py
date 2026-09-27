"""Private synthetic-only seam for deferred Cascaded Tanks development scores.

The synthetic entrypoint accepts paths and an explicit non-official archive
expectation. It snapshots and validates every JSON input before the one-use
marker, checks only the training and input stages before that marker, and reads
the 256 synthetic development targets only after the marker is durably synced.
The public production entrypoint is intentionally disabled until a separate
numeric/Qwen/baseline protocol and independent review authorize it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core.real_data import cascaded_tanks_controlled as _source
from core.real_data import cascaded_tanks_deferred_targets as _targets
from core.real_data import cascaded_tanks_development_forecast as _forecast
from core.real_data import cascaded_tanks_forecast_gate as _gate
from core.real_data import cascaded_tanks_training_fit as _fit
from core.real_data.cascaded_tanks_controlled import (
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_SHA256,
    ArchiveExpectation,
)

SCORER_SCHEMA = "cascaded-tanks-deferred-development-score-receipt-v1"
FORECAST_BUNDLE_SCHEMA = "cascaded-tanks-synthetic-forecast-bundle-v1"
BASELINE_BUNDLE_SCHEMA = "cascaded-tanks-synthetic-baseline-bundle-v1"
BASELINE_PROVENANCE_SCHEMA = "cascaded-tanks-synthetic-baseline-provenance-v1"
SCORING_DECLARATION_SCHEMA = "cascaded-tanks-synthetic-scoring-declaration-v1"
SCORE_CONDITION = "development-score-v1"
TRAIN_INDICES = [0, 768]
FORECAST_INDICES = [768, 1024]
_MAX_SYNTHETIC_ARCHIVE_BYTES = 16_000_000
_MAX_TOTAL_JSON_BYTES = 128_000_000
_MAX_JSON_BYTES = {
    "fit_manifest": 1_000_000,
    "fit_receipt": 64_000_000,
    "comparison_declaration": 1_000_000,
    "gate_receipt": 64_000_000,
    "candidate_bundle": 4_000_000,
    "baseline_bundle": 1_000_000,
    "baseline_provenance": 1_000_000,
    "scoring_declaration": 1_000_000,
}
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_ATTEMPT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
# The checkout's .gitignore excludes artifacts/*; callers cannot override this.
_MARKER_ROOT = _CHECKOUT_ROOT / "artifacts" / "cascaded_tanks_deferred_score_locks"
_GATE_RECEIPT_KEYS = {
    "schema",
    "status",
    "complete",
    "development_score_eligible",
    "target_access",
    "target_sha256",
    "scope",
    "protocol_id",
    "run_id",
    "comparison_declaration_sha256",
    "fit_manifest_sha256",
    "fit_receipt_sha256",
    "source",
    "archive_sha256",
    "source_stage_sha256",
    "forecast",
    "candidate_roster",
    "runtime",
    "code_sha256",
    "candidate_results",
    "failures",
    "receipt_sha256",
}


class CascadedTanksDeferredScorerError(ValueError):
    """A deferred-score artifact, source, or one-use condition failed closed."""


@dataclass(frozen=True, slots=True)
class SyntheticDeferredScoreOutcome:
    """Compact result; target arrays and raw source bytes are never returned."""

    status: Literal["complete", "terminal_failure"]
    receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Snapshot:
    raw: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _PinnedMarkerRoot:
    path: Path
    descriptor: int
    device: int
    inode: int


class _MarkerClaimFailure(RuntimeError):
    def __init__(self, code: str, *, consumed: bool, marker_sha256: str | None):
        super().__init__(code)
        self.code = code
        self.consumed = consumed
        self.marker_sha256 = marker_sha256


def run_cascaded_tanks_deferred_development_score(
    archive_path: str | os.PathLike[str],
    fit_manifest_path: str | os.PathLike[str],
    fit_receipt_path: str | os.PathLike[str],
    comparison_declaration_path: str | os.PathLike[str],
    gate_receipt_path: str | os.PathLike[str],
    candidate_bundle_path: str | os.PathLike[str],
    baseline_bundle_path: str | os.PathLike[str],
    baseline_provenance_path: str | os.PathLike[str],
    scoring_declaration_path: str | os.PathLike[str],
) -> SyntheticDeferredScoreOutcome:
    """Fail closed for real-source scoring until a separately reviewed protocol."""

    del (
        archive_path,
        fit_manifest_path,
        fit_receipt_path,
        comparison_declaration_path,
        gate_receipt_path,
        candidate_bundle_path,
        baseline_bundle_path,
        baseline_provenance_path,
        scoring_declaration_path,
    )
    raise CascadedTanksDeferredScorerError(
        "official-source deferred scoring is disabled pending a frozen numeric, "
        "Qwen, and baseline protocol plus independent review"
    )


def _score_synthetic_fixture_development(
    archive_path: str | os.PathLike[str],
    fit_manifest_path: str | os.PathLike[str],
    fit_receipt_path: str | os.PathLike[str],
    comparison_declaration_path: str | os.PathLike[str],
    gate_receipt_path: str | os.PathLike[str],
    candidate_bundle_path: str | os.PathLike[str],
    baseline_bundle_path: str | os.PathLike[str],
    baseline_provenance_path: str | os.PathLike[str],
    scoring_declaration_path: str | os.PathLike[str],
    *,
    expected_archive: ArchiveExpectation,
) -> SyntheticDeferredScoreOutcome:
    """Score one synthetic fixture attempt using only frozen filesystem inputs."""

    input_paths = {
        "fit_manifest": fit_manifest_path,
        "fit_receipt": fit_receipt_path,
        "comparison_declaration": comparison_declaration_path,
        "gate_receipt": gate_receipt_path,
        "candidate_bundle": candidate_bundle_path,
        "baseline_bundle": baseline_bundle_path,
        "baseline_provenance": baseline_provenance_path,
        "scoring_declaration": scoring_declaration_path,
    }
    input_hashes: dict[str, str | None] = {name: None for name in input_paths}
    context: dict[str, Any] = {
        "run_id": None,
        "protocol_id": None,
        "archive_sha256": None,
        "scoring_declaration_sha256": None,
        "attempt_id": None,
        "marker_sha256": None,
    }
    marker_root: _PinnedMarkerRoot | None = None

    try:
        _validate_synthetic_expectation(expected_archive)
        marker_root = _open_marker_root()
        _require_path(archive_path, "archive_path")
        snapshots = _read_all_json_snapshots(input_paths)
        for name, snapshot in snapshots.items():
            input_hashes[name] = snapshot.sha256

        documents = {
            name: _parse_json_snapshot(name, snapshot.raw)
            for name, snapshot in snapshots.items()
            if name != "fit_manifest"
        }
        manifest = _gate._parse_fit_manifest(snapshots["fit_manifest"].raw)
        context["run_id"] = manifest.run_id
        context["protocol_id"] = manifest.protocol_id
        context["archive_sha256"] = expected_archive.archive_sha256
        if manifest.scope != "synthetic-fixture-only":
            raise CascadedTanksDeferredScorerError("synthetic scope is required")
        _validate_run_id(manifest.run_id)
        if manifest.source["archive_sha256"] != expected_archive.archive_sha256:
            raise CascadedTanksDeferredScorerError(
                "fit manifest archive identity differs from the fixture expectation"
            )
        if manifest.source["archive_bytes"] != expected_archive.archive_bytes:
            raise CascadedTanksDeferredScorerError(
                "fit manifest archive size differs from the fixture expectation"
            )
        fit_receipt = documents["fit_receipt"]
        comparison = documents["comparison_declaration"]
        gate_receipt = documents["gate_receipt"]
        candidate_bundle = documents["candidate_bundle"]
        baseline_bundle = documents["baseline_bundle"]
        baseline_provenance = documents["baseline_provenance"]
        scoring_declaration = documents["scoring_declaration"]

        _gate._validate_fit_receipt_binding(
            fit_receipt, manifest, snapshots["fit_manifest"].sha256
        )
        roster = _gate._validate_comparison_declaration(
            comparison,
            manifest,
            snapshots["fit_manifest"].sha256,
            required_scope="synthetic-fixture-only",
        )
        comparison_sha256 = snapshots["comparison_declaration"].sha256
        context["scoring_declaration_sha256"] = snapshots[
            "scoring_declaration"
        ].sha256
        _validate_gate_receipt(
            gate_receipt,
            manifest=manifest,
            fit_manifest_sha256=snapshots["fit_manifest"].sha256,
            fit_receipt_sha256=snapshots["fit_receipt"].sha256,
            comparison_sha256=comparison_sha256,
            roster=roster,
            expected_archive=expected_archive,
        )

        # Parse only training yEst and the two uEst stages before the marker.
        # This loader never converts or returns the deferred suffix or uVal/yVal.
        data = _source._load_fixture_development_data(
            archive_path, expected_archive=expected_archive
        )
        archive_binding = {
            "archive_bytes": expected_archive.archive_bytes,
            "archive_sha256": expected_archive.archive_sha256,
            "csv_member": expected_archive.csv_member,
            "csv_member_bytes": _source.CSV_MEMBER_BYTES,
            "source_doi": "synthetic-fixture-only",
            "source_version": "synthetic-fixture-only",
        }
        _fit._validate_source_binding(
            manifest,
            source_kind="synthetic-fixture-only",
            data=data,
            archive=archive_binding,
        )
        _gate._validate_training_payload(fit_receipt, data)
        _gate._validate_fit_populations(fit_receipt, manifest)

        candidates = _validate_candidate_bundle(
            candidate_bundle,
            manifest=manifest,
            gate_receipt=gate_receipt,
            run_id=manifest.run_id,
            protocol_id=manifest.protocol_id,
            archive_sha256=expected_archive.archive_sha256,
            fit_manifest_sha256=snapshots["fit_manifest"].sha256,
            fit_receipt_sha256=snapshots["fit_receipt"].sha256,
            comparison_sha256=comparison_sha256,
            gate_receipt_sha256=snapshots["gate_receipt"].sha256,
        )
        baseline_id, baseline = _validate_baseline_bundle(
            baseline_bundle,
            run_id=manifest.run_id,
            protocol_id=manifest.protocol_id,
            archive_sha256=expected_archive.archive_sha256,
            gate_receipt_sha256=snapshots["gate_receipt"].sha256,
        )
        _validate_baseline_provenance(
            baseline_provenance,
            manifest=manifest,
            archive_sha256=expected_archive.archive_sha256,
            fit_manifest_sha256=snapshots["fit_manifest"].sha256,
            fit_receipt_sha256=snapshots["fit_receipt"].sha256,
            gate_receipt_sha256=snapshots["gate_receipt"].sha256,
            baseline_bundle_sha256=snapshots["baseline_bundle"].sha256,
            baseline_id=baseline_id,
        )
        attempt_id, decision_rule, structural_decision = (
            _validate_scoring_declaration(
                scoring_declaration,
                manifest=manifest,
                archive_sha256=expected_archive.archive_sha256,
                fit_manifest_sha256=snapshots["fit_manifest"].sha256,
                fit_receipt_sha256=snapshots["fit_receipt"].sha256,
                comparison_sha256=comparison_sha256,
                gate_receipt_sha256=snapshots["gate_receipt"].sha256,
                candidate_bundle_sha256=snapshots["candidate_bundle"].sha256,
                baseline_bundle_sha256=snapshots["baseline_bundle"].sha256,
                baseline_provenance_sha256=snapshots["baseline_provenance"].sha256,
                scoring_declaration_sha256=snapshots["scoring_declaration"].sha256,
            )
        )
        context["attempt_id"] = attempt_id
    except Exception as error:  # noqa: BLE001 - any preflight failure is terminal
        _close_marker_root(marker_root)
        return _outcome(
            status="terminal_failure",
            consumed=False,
            context=context,
            input_hashes=input_hashes,
            code="preflight_failure",
            detail=f"{type(error).__name__}: {error}"[:256],
        )

    try:
        _require_marker_root_identity(marker_root)
        marker_sha256, already_exists = _claim_one_use_marker(
            marker_root,
            run_id=manifest.run_id,
            archive_sha256=expected_archive.archive_sha256,
            declaration_sha256=snapshots["scoring_declaration"].sha256,
            attempt_id=attempt_id,
        )
        context["marker_sha256"] = marker_sha256
    except _MarkerClaimFailure as error:
        _close_marker_root(marker_root)
        context["marker_sha256"] = error.marker_sha256
        return _outcome(
            status="terminal_failure",
            consumed=error.consumed,
            context=context,
            input_hashes=input_hashes,
            code=error.code,
            detail="marker_claim_failed",
        )
    except Exception as error:  # noqa: BLE001 - uncertain claim identity fails closed
        _close_marker_root(marker_root)
        return _outcome(
            status="terminal_failure",
            consumed=False,
            context=context,
            input_hashes=input_hashes,
            code="marker_root_identity_failure",
            detail=f"{type(error).__name__}: {error}"[:256],
        )
    if already_exists:
        _close_marker_root(marker_root)
        return _outcome(
            status="terminal_failure",
            consumed=True,
            context=context,
            input_hashes=input_hashes,
            code="run_condition_already_consumed",
            detail="existing_marker_blocks_retry",
        )

    try:
        # Verify the fixed pathname still resolves to the directory whose fd
        # received the durable marker. The marker itself is claimed and synced
        # only relative to that pinned fd, so a concurrent path swap cannot
        # redirect the one-use slot.
        archive_snapshot = _read_archive_snapshot(archive_path, expected_archive)
        csv_wire = _csv_member_from_snapshot(archive_snapshot, expected_archive)
        _require_marker_root_identity(marker_root)
        targets = _targets._parse_synthetic_development_targets(csv_wire)
        target_values = _finite_vector(targets, 256, "synthetic development targets")
        target_array = np.asarray(target_values, dtype=np.float64)
        baseline_array = np.asarray(baseline, dtype=np.float64)
        candidate_scores: list[dict[str, Any]] = []
        for candidate_id, forecast_values in candidates:
            forecast_array = np.asarray(forecast_values, dtype=np.float64)
            candidate_scores.append(
                {
                    "candidate_id": candidate_id,
                    "rmse": _rmse_256(forecast_array, target_array),
                }
            )
        baseline_rmse = _rmse_256(baseline_array, target_array)
        selection_status, selected_candidate_id = _select_synthetic_candidate(
            candidate_scores,
            baseline_rmse,
            decision_rule,
            structural_decision,
        )
        return _outcome(
            status="complete",
            consumed=True,
            context={
                **context,
                "archive_sha256": archive_snapshot.sha256,
            },
            input_hashes=input_hashes,
            target_sha256=_fit._float_array_sha256(target_array),
            score={
                "metric": "root-mean-square-error",
                "forecast_indices": FORECAST_INDICES,
                "candidate_rmse": candidate_scores,
                "baseline_id": baseline_id,
                "baseline_rmse": baseline_rmse,
                "decision_rule": decision_rule,
                "structural_separation": structural_decision,
                "selection_status": selection_status,
                "selected_candidate_id": selected_candidate_id,
            },
        )
    except Exception as error:  # noqa: BLE001 - marker consumption is terminal
        return _outcome(
            status="terminal_failure",
            consumed=True,
            context=context,
            input_hashes=input_hashes,
            code="post_marker_failure",
            detail=f"{type(error).__name__}: {error}"[:256],
        )
    finally:
        _close_marker_root(marker_root)


def _validate_synthetic_expectation(expectation: ArchiveExpectation) -> None:
    if type(expectation) is not ArchiveExpectation:
        raise CascadedTanksDeferredScorerError(
            "an explicit synthetic ArchiveExpectation is required"
        )
    if (
        expectation.archive_bytes <= 0
        or expectation.archive_bytes > _MAX_SYNTHETIC_ARCHIVE_BYTES
        or expectation.archive_sha256 == SOURCE_ARCHIVE_SHA256
        or expectation.archive_bytes == SOURCE_ARCHIVE_BYTES
        or expectation.csv_member != _source.CSV_MEMBER
    ):
        raise CascadedTanksDeferredScorerError(
            "the official archive and non-synthetic source expectations are forbidden"
        )


def _open_marker_root() -> _PinnedMarkerRoot:
    root = _MARKER_ROOT
    descriptor: int | None = None
    try:
        if not root.is_absolute():
            raise OSError("fixed marker root must be absolute")
        root_info = root.lstat()
        parent_mode = root.parent.lstat().st_mode
        if not stat.S_ISDIR(root_info.st_mode) or not stat.S_ISDIR(parent_mode):
            raise OSError("fixed marker root and its parent must be real directories")
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            raise OSError("directory no-follow opens are unavailable")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(root, flags)
        opened_info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened_info.st_mode)
            or (opened_info.st_dev, opened_info.st_ino)
            != (root_info.st_dev, root_info.st_ino)
        ):
            raise OSError("fixed marker root changed while it was opened")
        return _PinnedMarkerRoot(
            path=root,
            descriptor=descriptor,
            device=opened_info.st_dev,
            inode=opened_info.st_ino,
        )
    except OSError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise CascadedTanksDeferredScorerError(
            "fixed marker root must already exist"
        ) from error


def _require_marker_root_identity(marker_root: _PinnedMarkerRoot) -> None:
    try:
        current = marker_root.path.lstat()
        parent_mode = marker_root.path.parent.lstat().st_mode
        pinned = os.fstat(marker_root.descriptor)
    except OSError as error:
        raise CascadedTanksDeferredScorerError(
            "fixed marker root identity is unavailable"
        ) from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or not stat.S_ISDIR(parent_mode)
        or not stat.S_ISDIR(pinned.st_mode)
        or (current.st_dev, current.st_ino) != (marker_root.device, marker_root.inode)
        or (pinned.st_dev, pinned.st_ino) != (marker_root.device, marker_root.inode)
    ):
        raise CascadedTanksDeferredScorerError(
            "fixed marker root no longer resolves to the pinned directory"
        )


def _close_marker_root(marker_root: _PinnedMarkerRoot | None) -> None:
    if marker_root is not None:
        try:
            os.close(marker_root.descriptor)
        except OSError:
            pass


def _require_path(value: object, label: str) -> None:
    if not isinstance(value, (str, os.PathLike)):
        raise CascadedTanksDeferredScorerError(f"{label} must be a filesystem path")


def _read_all_json_snapshots(
    paths: dict[str, str | os.PathLike[str]],
) -> dict[str, _Snapshot]:
    snapshots: dict[str, _Snapshot] = {}
    total = 0
    for name, path in paths.items():
        _require_path(path, f"{name}_path")
        snapshot = _read_bounded_snapshot(path, _MAX_JSON_BYTES[name], name)
        total += len(snapshot.raw)
        if total > _MAX_TOTAL_JSON_BYTES:
            raise CascadedTanksDeferredScorerError("JSON input set exceeds size limit")
        snapshots[name] = snapshot
    return snapshots


def _read_bounded_snapshot(
    path: str | os.PathLike[str], maximum: int, label: str
) -> _Snapshot:
    try:
        with Path(path).open("rb") as stream:
            initial_size = os.fstat(stream.fileno()).st_size
            if initial_size > maximum:
                raise CascadedTanksDeferredScorerError(
                    f"{label} exceeds its size limit"
                )
            raw = stream.read(maximum + 1)
    except CascadedTanksDeferredScorerError:
        raise
    except OSError as error:
        raise CascadedTanksDeferredScorerError(
            f"{label} could not be read"
        ) from error
    if len(raw) > maximum:
        raise CascadedTanksDeferredScorerError(f"{label} exceeds its size limit")
    return _Snapshot(raw=raw, sha256=hashlib.sha256(raw).hexdigest())


def _parse_json_snapshot(name: str, raw: bytes) -> Any:
    labels = {
        "fit_receipt": "fit receipt",
        "comparison_declaration": "comparison declaration",
        "gate_receipt": "gate receipt",
        "candidate_bundle": "candidate forecast bundle",
        "baseline_bundle": "baseline forecast bundle",
        "baseline_provenance": "baseline provenance",
        "scoring_declaration": "scoring declaration",
    }
    if name == "fit_manifest":
        return _gate._parse_fit_manifest(raw)
    value = _gate._parse_json_document(raw, labels[name])
    if not isinstance(value, dict):
        raise CascadedTanksDeferredScorerError(f"{labels[name]} must be an object")
    return value


def _validate_gate_receipt(
    receipt: object,
    *,
    manifest: _fit._FitManifest,
    fit_manifest_sha256: str,
    fit_receipt_sha256: str,
    comparison_sha256: str,
    roster: list[dict[str, Any]],
    expected_archive: ArchiveExpectation,
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != _GATE_RECEIPT_KEYS:
        raise CascadedTanksDeferredScorerError("gate receipt schema is invalid")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("receipt_sha256") != _fit._canonical_sha256(unsigned)
        or receipt.get("schema") != _gate.FORECAST_GATE_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("complete") is not True
        or receipt.get("development_score_eligible") is not False
        or receipt.get("target_access")
        != {
            "development_y_est_materialized": 0,
            "u_val_materialized": 0,
            "y_val_materialized": 0,
        }
        or receipt.get("target_sha256") is not None
        or receipt.get("scope") != "synthetic-fixture-only"
        or receipt.get("run_id") != manifest.run_id
        or receipt.get("protocol_id") != manifest.protocol_id
        or receipt.get("comparison_declaration_sha256") != comparison_sha256
        or receipt.get("fit_manifest_sha256") != fit_manifest_sha256
        or receipt.get("fit_receipt_sha256") != fit_receipt_sha256
        or receipt.get("source") != manifest.source
        or receipt.get("source_stage_sha256") != manifest.source["stage_sha256"]
        or receipt.get("runtime") != _fit._current_runtime()
        or receipt.get("code_sha256") != _gate._current_code_sha256()
        or receipt.get("archive_sha256") != expected_archive.archive_sha256
        or receipt.get("candidate_roster") != roster
        or receipt.get("failures") != []
    ):
        raise CascadedTanksDeferredScorerError(
            "gate receipt status, hash, source, or stage binding is invalid"
        )
    expected_forecast = {
        "training_indices": [0, 768],
        "boundary_state_index": 768,
        "forecast_indices": [768, 1024],
        "forecast_length": 256,
        "alignment": _gate._ALIGNMENT,
    }
    if receipt.get("forecast") != expected_forecast:
        raise CascadedTanksDeferredScorerError("gate forecast split is invalid")
    result_rows = receipt.get("candidate_results")
    if not isinstance(result_rows, list) or len(result_rows) != len(manifest.candidates):
        raise CascadedTanksDeferredScorerError("gate candidate roster is incomplete")
    for candidate, row in zip(manifest.candidates, result_rows, strict=True):
        if not isinstance(row, dict):
            raise CascadedTanksDeferredScorerError("gate candidate row is invalid")
        median_hash = row.get("weighted_median_forecast_sha256")
        particle_rows = row.get("particle_results")
        if (
            row.get("candidate_id") != candidate.candidate_id
            or row.get("model") != candidate.model.value
            or row.get("expected_particle_count") != candidate.target_samples
            or row.get("status") != "complete"
            or not _is_sha256(median_hash)
            or not isinstance(particle_rows, list)
            or len(particle_rows) != candidate.target_samples
            or row.get("failures") != []
        ):
            raise CascadedTanksDeferredScorerError(
                "gate candidate forecast is incomplete or reordered"
            )
        for index, particle in enumerate(particle_rows):
            if (
                not isinstance(particle, dict)
                or particle.get("identity")
                != {
                    "candidate_id": candidate.candidate_id,
                    "final_population_index": index,
                }
                or particle.get("status") != "complete"
                or particle.get("training_status") != "complete"
                or particle.get("development_status") != "complete"
                or not _is_sha256(particle.get("forecast_sha256"))
            ):
                raise CascadedTanksDeferredScorerError(
                    "gate particle forecast is incomplete or reordered"
                )


def _validate_candidate_bundle(
    bundle: object,
    *,
    manifest: _fit._FitManifest,
    gate_receipt: dict[str, Any],
    run_id: str,
    protocol_id: str,
    archive_sha256: str,
    fit_manifest_sha256: str,
    fit_receipt_sha256: str,
    comparison_sha256: str,
    gate_receipt_sha256: str,
) -> list[tuple[str, tuple[float, ...]]]:
    keys = {
        "schema",
        "scope",
        "run_id",
        "protocol_id",
        "archive_sha256",
        "fit_manifest_sha256",
        "fit_receipt_sha256",
        "comparison_declaration_sha256",
        "gate_receipt_sha256",
        "training_indices",
        "forecast_indices",
        "candidates",
        "bundle_sha256",
    }
    _require_self_hashed_object(bundle, keys, "bundle_sha256", "candidate bundle")
    assert isinstance(bundle, dict)
    if (
        bundle.get("schema") != FORECAST_BUNDLE_SCHEMA
        or bundle.get("scope") != "synthetic-fixture-only"
        or bundle.get("run_id") != run_id
        or bundle.get("protocol_id") != protocol_id
        or bundle.get("archive_sha256") != archive_sha256
        or bundle.get("fit_manifest_sha256") != fit_manifest_sha256
        or bundle.get("fit_receipt_sha256") != fit_receipt_sha256
        or bundle.get("comparison_declaration_sha256") != comparison_sha256
        or bundle.get("gate_receipt_sha256") != gate_receipt_sha256
        or bundle.get("training_indices") != TRAIN_INDICES
        or bundle.get("forecast_indices") != FORECAST_INDICES
    ):
        raise CascadedTanksDeferredScorerError(
            "candidate bundle source, stage, or receipt binding is invalid"
        )
    rows = bundle.get("candidates")
    gate_rows = gate_receipt["candidate_results"]
    if not isinstance(rows, list) or len(rows) != len(manifest.candidates):
        raise CascadedTanksDeferredScorerError("candidate bundle roster is incomplete")
    result: list[tuple[str, tuple[float, ...]]] = []
    for candidate, row, gate_row in zip(
        manifest.candidates, rows, gate_rows, strict=True
    ):
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "model",
            "forecast_sha256",
            "values",
        }:
            raise CascadedTanksDeferredScorerError("candidate bundle row is invalid")
        values = _finite_vector(row.get("values"), 256, "candidate forecast")
        forecast_sha256 = _forecast._sequence_sha256(values, start_index=768)
        if (
            row.get("candidate_id") != candidate.candidate_id
            or row.get("model") != candidate.model.value
            or row.get("forecast_sha256") != forecast_sha256
            or forecast_sha256 != gate_row["weighted_median_forecast_sha256"]
        ):
            raise CascadedTanksDeferredScorerError(
                "candidate bundle differs from the exact ordered gate forecasts"
            )
        result.append((candidate.candidate_id, values))
    return result


def _validate_baseline_bundle(
    bundle: object,
    *,
    run_id: str,
    protocol_id: str,
    archive_sha256: str,
    gate_receipt_sha256: str,
) -> tuple[str, tuple[float, ...]]:
    keys = {
        "schema",
        "scope",
        "run_id",
        "protocol_id",
        "archive_sha256",
        "gate_receipt_sha256",
        "forecast_indices",
        "baseline_id",
        "method",
        "forecast_sha256",
        "values",
        "bundle_sha256",
    }
    _require_self_hashed_object(bundle, keys, "bundle_sha256", "baseline bundle")
    assert isinstance(bundle, dict)
    if (
        bundle.get("schema") != BASELINE_BUNDLE_SCHEMA
        or bundle.get("scope") != "synthetic-fixture-only"
        or bundle.get("run_id") != run_id
        or bundle.get("protocol_id") != protocol_id
        or bundle.get("archive_sha256") != archive_sha256
        or bundle.get("gate_receipt_sha256") != gate_receipt_sha256
        or bundle.get("forecast_indices") != FORECAST_INDICES
        or not _is_nonempty_text(bundle.get("baseline_id"))
        or not _is_nonempty_text(bundle.get("method"))
    ):
        raise CascadedTanksDeferredScorerError(
            "baseline bundle source, provenance, or split binding is invalid"
        )
    values = _finite_vector(bundle.get("values"), 256, "baseline forecast")
    if bundle.get("forecast_sha256") != _forecast._sequence_sha256(
        values, start_index=768
    ):
        raise CascadedTanksDeferredScorerError("baseline forecast hash is invalid")
    return bundle["baseline_id"], values


def _validate_baseline_provenance(
    provenance: object,
    *,
    manifest: _fit._FitManifest,
    archive_sha256: str,
    fit_manifest_sha256: str,
    fit_receipt_sha256: str,
    gate_receipt_sha256: str,
    baseline_bundle_sha256: str,
    baseline_id: str,
) -> None:
    keys = {
        "schema",
        "scope",
        "run_id",
        "protocol_id",
        "archive_sha256",
        "fit_manifest_sha256",
        "fit_receipt_sha256",
        "gate_receipt_sha256",
        "baseline_bundle_sha256",
        "baseline_id",
        "training_indices",
        "forecast_indices",
        "training_stage_sha256",
        "forecast_inputs_stage_sha256",
        "development_y_est_materialized",
        "u_val_materialized",
        "y_val_materialized",
        "provenance_sha256",
    }
    _require_self_hashed_object(
        provenance, keys, "provenance_sha256", "baseline provenance"
    )
    assert isinstance(provenance, dict)
    if (
        provenance.get("schema") != BASELINE_PROVENANCE_SCHEMA
        or provenance.get("scope") != "synthetic-fixture-only"
        or provenance.get("run_id") != manifest.run_id
        or provenance.get("protocol_id") != manifest.protocol_id
        or provenance.get("archive_sha256") != archive_sha256
        or provenance.get("fit_manifest_sha256") != fit_manifest_sha256
        or provenance.get("fit_receipt_sha256") != fit_receipt_sha256
        or provenance.get("gate_receipt_sha256") != gate_receipt_sha256
        or provenance.get("baseline_bundle_sha256") != baseline_bundle_sha256
        or provenance.get("baseline_id") != baseline_id
        or provenance.get("training_indices") != TRAIN_INDICES
        or provenance.get("forecast_indices") != FORECAST_INDICES
        or provenance.get("training_stage_sha256")
        != manifest.source["stage_sha256"]["training_sha256"]
        or provenance.get("forecast_inputs_stage_sha256")
        != manifest.source["stage_sha256"]["forecast_inputs_sha256"]
        or provenance.get("development_y_est_materialized") != 0
        or provenance.get("u_val_materialized") != 0
        or provenance.get("y_val_materialized") != 0
        or not _is_nonempty_text(provenance.get("baseline_id"))
    ):
        raise CascadedTanksDeferredScorerError(
            "baseline provenance is not bound to target-free training stages"
        )


def _validate_scoring_declaration(
    declaration: object,
    *,
    manifest: _fit._FitManifest,
    archive_sha256: str,
    fit_manifest_sha256: str,
    fit_receipt_sha256: str,
    comparison_sha256: str,
    gate_receipt_sha256: str,
    candidate_bundle_sha256: str,
    baseline_bundle_sha256: str,
    baseline_provenance_sha256: str,
    scoring_declaration_sha256: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    keys = {
        "schema",
        "scope",
        "run_id",
        "protocol_id",
        "archive_sha256",
        "fit_manifest_sha256",
        "fit_receipt_sha256",
        "comparison_declaration_sha256",
        "gate_receipt_sha256",
        "candidate_bundle_sha256",
        "baseline_bundle_sha256",
        "baseline_provenance_sha256",
        "attempt_id",
        "metric",
        "decision_rule",
        "structural_separation",
        "qwen_disposition",
        "declaration_sha256",
    }
    _require_self_hashed_object(
        declaration, keys, "declaration_sha256", "scoring declaration"
    )
    assert isinstance(declaration, dict)
    rule = declaration.get("decision_rule")
    structural = declaration.get("structural_separation")
    if (
        declaration.get("schema") != SCORING_DECLARATION_SCHEMA
        or declaration.get("scope") != "synthetic-fixture-only"
        or declaration.get("run_id") != manifest.run_id
        or declaration.get("protocol_id") != manifest.protocol_id
        or declaration.get("archive_sha256") != archive_sha256
        or declaration.get("fit_manifest_sha256") != fit_manifest_sha256
        or declaration.get("fit_receipt_sha256") != fit_receipt_sha256
        or declaration.get("comparison_declaration_sha256") != comparison_sha256
        or declaration.get("gate_receipt_sha256") != gate_receipt_sha256
        or declaration.get("candidate_bundle_sha256") != candidate_bundle_sha256
        or declaration.get("baseline_bundle_sha256") != baseline_bundle_sha256
        or declaration.get("baseline_provenance_sha256")
        != baseline_provenance_sha256
        or declaration.get("metric") != "root-mean-square-error"
        or declaration.get("qwen_disposition")
        != "synthetic-abstain-no-proposal"
        or not _is_nonempty_text(declaration.get("attempt_id"))
        or not _SAFE_ATTEMPT.fullmatch(declaration["attempt_id"])
        or not isinstance(rule, dict)
        or set(rule)
        != {
            "rule",
            "minimum_absolute_improvement",
            "tie_policy",
        }
        or rule.get("rule")
        != "select-lowest-candidate-if-rmse-improves-baseline-by-more-than-minimum"
        or rule.get("tie_policy") != "retain-baseline"
        or not isinstance(structural, dict)
        or set(structural) != {"decision", "reason"}
        or structural.get("decision") not in {"eligible", "abstain"}
        or not _is_nonempty_text(structural.get("reason"))
    ):
        raise CascadedTanksDeferredScorerError(
            "scoring declaration bindings or explicit decision rules are invalid"
        )
    threshold = _finite_nonnegative(rule.get("minimum_absolute_improvement"))
    checked_rule = {
        "rule": rule["rule"],
        "minimum_absolute_improvement": threshold,
        "tie_policy": rule["tie_policy"],
    }
    return declaration["attempt_id"], checked_rule, structural


def _require_self_hashed_object(
    value: object, keys: set[str], hash_key: str, label: str
) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise CascadedTanksDeferredScorerError(f"{label} schema is invalid")
    unsigned = {key: item for key, item in value.items() if key != hash_key}
    expected = _canonical_sha256(unsigned)
    if value.get(hash_key) != expected:
        raise CascadedTanksDeferredScorerError(f"{label} content hash is invalid")


def _claim_one_use_marker(
    marker_root: _PinnedMarkerRoot,
    *,
    run_id: str,
    archive_sha256: str,
    declaration_sha256: str,
    attempt_id: str,
) -> tuple[str, bool]:
    _validate_run_id(run_id)
    marker_name = f"{run_id}.{SCORE_CONDITION}.started.json"
    marker = {
        "schema": "cascaded-tanks-deferred-score-one-use-marker-v1",
        "run_id": run_id,
        "condition": SCORE_CONDITION,
        "archive_sha256": archive_sha256,
        "scoring_declaration_sha256": declaration_sha256,
        "attempt_id": attempt_id,
    }
    marker_raw = _canonical_json(marker)
    marker_sha256 = hashlib.sha256(marker_raw).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(
            marker_name, flags, 0o600, dir_fd=marker_root.descriptor
        )
    except FileExistsError:
        try:
            existing = _read_marker_snapshot_at(
                marker_root.descriptor, marker_name, 16_384
            )
        except (OSError, CascadedTanksDeferredScorerError):
            existing_sha256 = None
        else:
            existing_sha256 = existing.sha256
        return existing_sha256, True
    except OSError as error:
        raise _MarkerClaimFailure(
            "marker_create_failure", consumed=False, marker_sha256=None
        ) from error

    try:
        _write_all(descriptor, marker_raw)
        os.fsync(descriptor)
    except OSError as error:
        raise _MarkerClaimFailure(
            "marker_file_sync_failure", consumed=True, marker_sha256=marker_sha256
        ) from error
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass

    try:
        os.fsync(marker_root.descriptor)
    except OSError as error:
        raise _MarkerClaimFailure(
            "marker_directory_sync_failure", consumed=True, marker_sha256=marker_sha256
        ) from error
    return marker_sha256, False


def _read_marker_snapshot_at(
    directory_descriptor: int, marker_name: str, maximum: int
) -> _Snapshot:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(marker_name, flags, dir_fd=directory_descriptor)
    try:
        initial_size = os.fstat(descriptor).st_size
        if initial_size > maximum:
            raise CascadedTanksDeferredScorerError("existing marker exceeds size limit")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) > maximum:
        raise CascadedTanksDeferredScorerError("existing marker exceeds size limit")
    return _Snapshot(raw=raw, sha256=hashlib.sha256(raw).hexdigest())


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("marker write made no progress")
        view = view[written:]


def _read_archive_snapshot(
    archive_path: str | os.PathLike[str], expectation: ArchiveExpectation
) -> _Snapshot:
    try:
        with Path(archive_path).open("rb") as stream:
            initial_size = os.fstat(stream.fileno()).st_size
            if initial_size != expectation.archive_bytes:
                raise CascadedTanksDeferredScorerError(
                    "archive byte count changed after marker claim"
                )
            raw = stream.read(expectation.archive_bytes + 1)
    except CascadedTanksDeferredScorerError:
        raise
    except OSError as error:
        raise CascadedTanksDeferredScorerError(
            "archive snapshot could not be read after marker claim"
        ) from error
    if len(raw) != expectation.archive_bytes:
        raise CascadedTanksDeferredScorerError(
            "archive snapshot size changed after marker claim"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expectation.archive_sha256:
        raise CascadedTanksDeferredScorerError(
            "archive snapshot hash changed after marker claim"
        )
    return _Snapshot(raw=raw, sha256=digest)


def _csv_member_from_snapshot(
    snapshot: _Snapshot, expectation: ArchiveExpectation
) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.raw), mode="r") as archive:
            csv_info = _source._validate_zip_members(archive.infolist(), expectation)
            if csv_info.filename != expectation.csv_member:
                raise CascadedTanksDeferredScorerError(
                    "CSV member path differs from the synthetic contract"
                )
            wire = archive.read(csv_info)
    except (OSError, RuntimeError, zipfile.BadZipFile, _source.CascadedTanksSourceError) as error:
        raise CascadedTanksDeferredScorerError(
            "archive snapshot does not satisfy the synthetic CSV contract"
        ) from error
    if len(wire) != _source.CSV_MEMBER_BYTES:
        raise CascadedTanksDeferredScorerError(
            "CSV member byte count differs from the synthetic contract"
        )
    return wire


def _finite_vector(value: object, expected_length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise CascadedTanksDeferredScorerError(
            f"{label} must contain exactly {expected_length} values"
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CascadedTanksDeferredScorerError(f"{label} contains a non-number")
        try:
            number = float(item)
        except (OverflowError, ValueError) as error:
            raise CascadedTanksDeferredScorerError(
                f"{label} contains an invalid number"
            ) from error
        if not math.isfinite(number):
            raise CascadedTanksDeferredScorerError(f"{label} contains a non-finite value")
        result.append(number)
    return tuple(result)


def _rmse_256(forecast: np.ndarray, target: np.ndarray) -> float:
    if forecast.shape != (256,) or target.shape != (256,):
        raise CascadedTanksDeferredScorerError("RMSE requires exact [768, 1024) vectors")
    value = float(np.sqrt(np.mean(np.square(forecast - target), dtype=np.float64)))
    if not math.isfinite(value):
        raise CascadedTanksDeferredScorerError("RMSE is non-finite")
    return value


def _select_synthetic_candidate(
    candidate_scores: list[dict[str, Any]],
    baseline_rmse: float,
    rule: dict[str, Any],
    structural: dict[str, Any],
) -> tuple[str, str | None]:
    if structural["decision"] == "abstain":
        return "structural_abstention", None
    best = min(candidate_scores, key=lambda row: row["rmse"])
    improvement = baseline_rmse - best["rmse"]
    if improvement > rule["minimum_absolute_improvement"]:
        return "candidate_selected", best["candidate_id"]
    return "baseline_retained", None


def _outcome(
    *,
    status: Literal["complete", "terminal_failure"],
    consumed: bool,
    context: dict[str, Any],
    input_hashes: dict[str, str | None],
    target_sha256: str | None = None,
    score: dict[str, Any] | None = None,
    code: str | None = None,
    detail: str | None = None,
) -> SyntheticDeferredScoreOutcome:
    code_hashes = _current_code_hashes()
    receipt: dict[str, Any] = {
        "schema": SCORER_SCHEMA,
        "status": status,
        "consumed": consumed,
        "run_id": context.get("run_id"),
        "protocol_id": context.get("protocol_id"),
        "attempt_id": context.get("attempt_id"),
        "scope": "synthetic-fixture-only",
        "archive_sha256": context.get("archive_sha256"),
        "scoring_declaration_sha256": context.get("scoring_declaration_sha256"),
        "marker_sha256": context.get("marker_sha256"),
        "input_sha256": input_hashes,
        "target_sha256": target_sha256,
        "score": score,
        "failure": None if code is None else {"code": code, "detail": detail},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "code_sha256": code_hashes,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return SyntheticDeferredScoreOutcome(status=status, receipt=receipt)


def _current_code_hashes() -> dict[str, str]:
    return {
        "deferred_scorer": _sha256_file(Path(__file__)),
        "synthetic_target_parser": _sha256_file(Path(_targets.__file__)),
        "source_adapter": _sha256_file(Path(_source.__file__)),
        "training_fit": _sha256_file(Path(_fit.__file__)),
        "forecast_gate": _sha256_file(Path(_gate.__file__)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _is_nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_nonnegative(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CascadedTanksDeferredScorerError(
            "minimum absolute improvement must be a finite nonnegative number"
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise CascadedTanksDeferredScorerError(
            "minimum absolute improvement is invalid"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise CascadedTanksDeferredScorerError(
            "minimum absolute improvement must be finite and nonnegative"
        )
    return result


def _validate_run_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not _SAFE_COMPONENT.fullmatch(value)
    ):
        raise CascadedTanksDeferredScorerError(
            "run_id must be a safe path component"
        )


__all__: tuple[str, ...] = ()

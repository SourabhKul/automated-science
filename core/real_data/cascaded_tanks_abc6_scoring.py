"""Synthetic-only deferred prospective scoring and diagnostics for ABC6.

The caller freezes the complete target-free forecast roster first.  Scoring
then verifies all durable fit/baseline receipts, identities, and forecast
hashes before creating one fixed project-root reveal marker.  The marker is
created with exclusive semantics and consumes the synthetic confirmation
condition even when any later operation fails.

This module has no real-data, Qwen, fitting, or model-selection interface.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

import numpy as np

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data.cascaded_tanks_abc6_cases import (
    CASE_COUNT,
    CASE_ROSTER,
    PARAMETER_ORDER,
    PROSPECTIVE_LENGTH,
    PROTOCOL_ID,
    RUN_ID,
    TRAINING_INPUT_L,
    TRAINING_INPUT_S,
    TRAINING_LENGTH,
    ABC6ProspectiveTargets,
    ABC6TrainingBundle,
    ABC6TrainingCaseData,
    open_deferred_abc6_target_gate,
)
from core.real_data.cascaded_tanks_abc6_forecast import (
    ABC6PosteriorBaselineForecast,
    forecast_abc6_posterior_and_baseline,
    weighted_left_inverse_quantile,
)
from core.real_data.cascaded_tanks_abc6_training import PARTICLE_COUNT
from core.real_data.cascaded_tanks_models import (
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

_PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
_SOURCE_ROOT: Final = Path(__file__).resolve().parents[2]
_TRAINING_SUMMARY_FILENAME: Final = "campaign.training-summary.json"
_FORECAST_ARTIFACT_FILENAME: Final = "campaign.target-free-forecasts.json"
TARGET_ARRAYS_ARTIFACT_FILENAME: Final = "campaign.prospective-target-arrays.json"
SCORE_RECEIPT_FILENAME: Final = "campaign.deferred-score-receipt.json"
_INTEGRATED_SOURCE_PATHS: Final = (
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
)
_REVEAL_MARKER_PATH: Final = (
    _PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "cascaded_tanks_abc6_scoring"
    / "claims"
    / f"{RUN_ID}.claim"
)
_WEIGHT_SUM_ABS_TOL: Final = 1.0e-12
_MAX_HANDOFF_ARTIFACT_BYTES: Final = 128 * 1024 * 1024
_MAX_SCORE_RECEIPT_BYTES: Final = 16 * 1024 * 1024

_TRUTH_PARAMETERS: Final = {
    "A": ("O2", (0.50, 0.40, 0.50, 0.50, 0.50, 3.0)),
    "B": ("C2", (0.54, 0.36, 0.46, 0.25, 0.75, 3.2)),
}
_NO_CROSSING_PRIOR_CEILING_QUANTILES: Final = (910.0, 1000.0, 1090.0)
_M_RESIDUAL_GROUPS: Final = ((0, 24, "u8_first_24"), (24, 204, "u0_last_180"))


class ABC6ScoringError(ValueError):
    """A frozen roster, receipt, target, or score failed closed validation."""


class ABC6RevealAlreadyConsumedError(ABC6ScoringError):
    """The fixed one-use marker already exists; this run cannot be retried."""


class ABC6DeferredScoreConsumedError(RuntimeError):
    """A post-marker error; the synthetic confirmation condition stays consumed."""

    def __init__(
        self,
        message: str,
        *,
        marker_sha256: str,
        score_receipt_path: Path | None = None,
        score_receipt_sha256: str | None = None,
        failure_stage: str | None = None,
    ):
        super().__init__(
            f"{message}; reveal condition consumed, retry forbidden "
            f"(marker sha256 {marker_sha256})"
        )
        self.condition_consumed = True
        self.retry_forbidden = True
        self.marker_sha256 = marker_sha256
        self.score_receipt_path = score_receipt_path
        self.score_receipt_sha256 = score_receipt_sha256
        self.failure_stage = failure_stage


@dataclass(frozen=True, slots=True)
class ABC6FrozenForecastRoster:
    """Ordered target-free forecasts and hashes fixed before target reveal."""

    forecasts: tuple[ABC6PosteriorBaselineForecast, ...]
    case_sha256: tuple[str, ...]
    roster_sha256: str


@dataclass(frozen=True, slots=True)
class ABC6ParameterInterval:
    parameter: str
    q05: float
    q95: float
    truth_in_interval: bool


@dataclass(frozen=True, slots=True)
class ABC6CaseScore:
    case_index: int
    case_id: str
    truth_id: str
    fit_model: str
    input_window: str
    replicate: int | None
    abc_weighted_mean_rmse: float | None
    abc_weighted_median_rmse: float | None
    baseline_coherent_rmse: float | None
    abc_q05_q95_envelope_inclusion_fraction: float | None
    terminal_effective_sample_size: float | None
    parameter_intervals: tuple[ABC6ParameterInterval, ...] | None


@dataclass(frozen=True, slots=True)
class ABC6PairedHorizonContrasts:
    truth_id: Literal["A", "B"]
    replicate: int
    abc_weighted_mean_rmse_s_minus_l: float | None
    abc_weighted_median_rmse_s_minus_l: float | None
    baseline_coherent_rmse_s_minus_l: float | None


@dataclass(frozen=True, slots=True)
class ABC6ParameterInclusionCount:
    truth_id: Literal["A", "B"]
    input_window: Literal["S", "L"]
    parameter: str
    included_count: int
    eligible_fit_count: int
    planned_fit_count: Literal[4] = 4
    interpretation: Literal["descriptive_not_calibrated_coverage"] = (
        "descriptive_not_calibrated_coverage"
    )


@dataclass(frozen=True, slots=True)
class ABC6NoCrossingDiagnostic:
    case_index: int
    fit_model: str
    ceiling_q05: float | None
    ceiling_q50: float | None
    ceiling_q95: float | None
    prior_reference_q05: Literal[910.0] = 910.0
    prior_reference_q50: Literal[1000.0] = 1000.0
    prior_reference_q95: Literal[1090.0] = 1090.0
    mechanism_abstention: Literal[True] = True
    prospective_score: None = None


@dataclass(frozen=True, slots=True)
class ABC6MTrainingResidualMeans:
    case_index: int
    fit_model: str
    baseline_u8_first_24_mean: float | None
    baseline_u0_last_180_mean: float | None
    abc_weighted_mean_u8_first_24_mean: float | None
    abc_weighted_mean_u0_last_180_mean: float | None
    residual_definition: Literal["observed_minus_predicted"] = (
        "observed_minus_predicted"
    )
    truth_inclusion_claim: Literal[False] = False


@dataclass(frozen=True, slots=True)
class ABC6DeferredScoreResult:
    protocol_id: str
    run_id: str
    reveal_marker_sha256: str
    target_sha256_by_truth: tuple[tuple[str, str], ...]
    case_scores: tuple[ABC6CaseScore, ...]
    paired_horizon_contrasts: tuple[ABC6PairedHorizonContrasts, ...]
    parameter_inclusion_counts: tuple[ABC6ParameterInclusionCount, ...]
    no_crossing_diagnostics: tuple[ABC6NoCrossingDiagnostic, ...]
    m_training_residual_means: tuple[ABC6MTrainingResidualMeans, ...]
    model_choice: None = None
    bayes_factors: None = None
    target_arrays_artifact_path: Path | None = None
    target_arrays_artifact_sha256: str | None = None
    forecast_array_sha256: str | None = None
    score_result_sha256: str | None = None
    score_receipt_path: Path | None = None
    score_receipt_sha256: str | None = None
    score_event_monotonic_ns: int | None = None
    score_event_utc: str | None = None


@dataclass(slots=True)
class _ABC6RevealMarkerClaim:
    """Durable claim digest plus the exact parent directory used to create it."""

    marker_sha256: str
    marker_parent_anchor: cases._DirectoryAnchor | None
    marker_relative_path: str
    marker_parent_device: int
    marker_parent_inode: int
    marker_device: int
    marker_inode: int
    marker_size: int
    marker_mtime_ns: int
    marker_ctime_ns: int

    def take_parent_anchor(self) -> cases._DirectoryAnchor:
        anchor = self.marker_parent_anchor
        if anchor is None:
            raise ABC6ScoringError("reveal marker directory anchor was already transferred")
        self.marker_parent_anchor = None
        return anchor

    def duplicate_parent_anchor(self) -> cases._DirectoryAnchor:
        anchor = self.marker_parent_anchor
        if anchor is None:
            raise ABC6ScoringError(
                "reveal marker parent was already transferred to its scoring handoff"
            )
        try:
            return cases._DirectoryAnchor(anchor.path, os.dup(anchor.descriptor))
        except Exception as error:
            raise ABC6ScoringError(
                "cannot retain the pinned reveal-marker parent for score evidence"
            ) from error

    def receipt_identity(self) -> dict[str, object]:
        return {
            "relative_path": self.marker_relative_path,
            "sha256": self.marker_sha256,
            "parent_device": self.marker_parent_device,
            "parent_inode": self.marker_parent_inode,
            "device": self.marker_device,
            "inode": self.marker_inode,
            "size_bytes": self.marker_size,
            "mtime_ns": self.marker_mtime_ns,
            "ctime_ns": self.marker_ctime_ns,
        }

    def close(self) -> None:
        if self.marker_parent_anchor is not None:
            self.marker_parent_anchor.close()
            self.marker_parent_anchor = None

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _root_anchor_for_identity(
    receipt_root_identity: object,
) -> cases._DirectoryAnchor:
    if not isinstance(receipt_root_identity, campaign_fit.ABC6ReceiptRootIdentity):
        raise ABC6ScoringError("typed campaign receipt-root identity is required")
    identity = receipt_root_identity
    try:
        identity.verify()
    except Exception as error:
        raise ABC6ScoringError(
            "campaign checkout or receipt-root identity changed"
        ) from error
    if identity.receipt_root_relative != campaign_fit.RECEIPT_ROOT_RELATIVE:
        raise ABC6ScoringError("campaign receipt root differs from the fixed path")
    root_path = Path(identity.repository_root_realpath)
    if not root_path.is_absolute() or str(root_path) != identity.repository_root_realpath:
        raise ABC6ScoringError("campaign checkout root path is not canonical")
    for configured_root, label in (
        (campaign_fit._REPO_ROOT, "campaign source root"),
        (_SOURCE_ROOT, "scorer source root"),
        (_PROJECT_ROOT, "scorer marker root"),
        (cases._PROJECT_ROOT, "case source root"),
    ):
        if cases._absolute_lexical_path(configured_root) != root_path:
            raise ABC6ScoringError(f"{label} differs from the typed campaign root")
    try:
        descriptor = identity.duplicate_repository_root_fd()
        anchor = cases._DirectoryAnchor(root_path, descriptor)
    except Exception as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ABC6ScoringError("cannot retain the typed checkout-root descriptor") from error
    runtime = identity.runtime_identity
    if (anchor.device, anchor.inode) != (
        runtime["repository_root_device"],
        runtime["repository_root_inode"],
    ):
        anchor.close()
        raise ABC6ScoringError("typed checkout-root descriptor identity changed")
    return anchor


def _read_relative_project_file(
    root_anchor: cases._DirectoryAnchor,
    relative_path: str,
    *,
    label: str,
) -> bytes:
    parts = cases._validate_relative_directory_path(
        relative_path, label=f"{label} path"
    )
    parent_path = "/".join(parts[:-1])
    parent_anchor = cases._open_relative_directory_anchor(
        root_anchor,
        parent_path,
        label=f"{label} parent",
    )
    try:
        return cases._read_regular_file_at(
            parent_anchor.descriptor,
            parts[-1],
            maximum_bytes=128 * 1024 * 1024,
            label=label,
        )
    finally:
        parent_anchor.close()


def _verified_integrated_source_hashes(
    receipt_root_identity: object,
) -> tuple[tuple[str, str], ...]:
    """Require the runner, forecast, and scorer in the frozen source contract."""

    required_paths = getattr(campaign_fit, "_REQUIRED_SOURCE_PATHS", None)
    reviewed = getattr(campaign_fit, "_REVIEWED_SOURCE_SHA256", None)
    if (
        type(required_paths) is not tuple
        or any(type(path) is not str for path in required_paths)
        or len(set(required_paths)) != len(required_paths)
        or not set(_INTEGRATED_SOURCE_PATHS).issubset(required_paths)
        or not isinstance(reviewed, Mapping)
    ):
        raise ABC6ScoringError(
            "campaign source contract must allowlist and review runner, forecast, and scorer"
        )
    root_anchor = _root_anchor_for_identity(receipt_root_identity)
    try:
        current = {
            relative_path: _sha256(
                _read_relative_project_file(
                    root_anchor,
                    relative_path,
                    label=f"campaign source {relative_path}",
                )
            )
            for relative_path in required_paths
        }
    except Exception as error:
        if isinstance(error, ABC6ScoringError):
            raise
        raise ABC6ScoringError(
            "cannot verify source files through the pinned checkout descriptor"
        ) from error
    finally:
        root_anchor.close()

    checked: list[tuple[str, str]] = []
    for relative_path in _INTEGRATED_SOURCE_PATHS:
        current_digest = current.get(relative_path)
        reviewed_digest = reviewed.get(relative_path)
        if (
            not _is_sha256(current_digest)
            or not _is_sha256(reviewed_digest)
            or current_digest != reviewed_digest
        ):
            raise ABC6ScoringError(
                f"integrated source is not independently pinned: {relative_path}"
            )
        checked.append((relative_path, current_digest))
    return tuple(checked)


def _validate_campaign_execution_identity(
    execution,
) -> tuple[campaign_fit.ABC6ReceiptRootIdentity, Path, tuple[str, ...]]:
    """Check the typed campaign identity and the exact 48 status links."""

    campaign_result = execution.campaign_result
    if not isinstance(campaign_result, campaign_fit.ABC6CampaignResult):
        raise ABC6ScoringError("verified execution has no typed campaign result")
    if (
        campaign_result.protocol_id != PROTOCOL_ID
        or campaign_result.run_id != RUN_ID
        or campaign_result.prospective_targets_generated is not False
        or campaign_result.forecasts_run is not False
        or not _is_sha256(campaign_result.manifest_sha256)
        or not _is_sha256(campaign_result.claim_sha256)
        or not _is_sha256(campaign_result.summary_sha256)
        or not _is_sha256(execution.evidence_manifest_sha256)
    ):
        raise ABC6ScoringError("verified campaign execution identity is invalid")
    identity = getattr(execution, "receipt_root_identity", None)
    if not isinstance(identity, campaign_fit.ABC6ReceiptRootIdentity):
        raise ABC6ScoringError("campaign execution has no typed receipt-root identity")
    try:
        identity.verify()
    except Exception as error:
        raise ABC6ScoringError("campaign receipt-root identity is not live") from error
    if identity.receipt_root_relative != campaign_fit.RECEIPT_ROOT_RELATIVE:
        raise ABC6ScoringError("campaign receipt path is not the fixed schema-v2 path")
    receipt_directory = Path(identity.repository_root_realpath) / identity.receipt_root_relative
    summary_path = campaign_result.summary_path
    if (
        not isinstance(summary_path, Path)
        or summary_path != receipt_directory / _TRAINING_SUMMARY_FILENAME
        or execution.evidence_manifest_path
        != receipt_directory / campaign_fit.EVIDENCE_MANIFEST_FILENAME
        or len(campaign_result.case_statuses) != CASE_COUNT
    ):
        raise ABC6ScoringError("verified campaign summary/status roster is incomplete")

    expected_receipt_names: list[str] = []
    statuses_complete = True
    for index, (case, status) in enumerate(
        zip(CASE_ROSTER, campaign_result.case_statuses, strict=True)
    ):
        if (
            type(status.case_index) is not int
            or status.case_index != index
            or status.case_id != case.case_id
            or type(status.fit_status) is not str
            or type(status.baseline_status) is not str
            or not _is_sha256(status.fit_receipt_sha256)
            or not _is_sha256(status.baseline_receipt_sha256)
        ):
            raise ABC6ScoringError(
                f"campaign status identity or receipt digest is invalid at case {index}"
            )
        if status.fit_status not in {"complete", "incomplete", "unresolved", "failed"}:
            raise ABC6ScoringError(f"campaign fit status is invalid at case {index}")
        if status.baseline_status not in {"complete", "incomplete", "failed"}:
            raise ABC6ScoringError(
                f"campaign baseline status is invalid at case {index}"
            )
        statuses_complete = statuses_complete and (
            status.fit_status == "complete" and status.baseline_status == "complete"
        )
        expected_receipt_names.extend(
            (
                f"case-{index:02d}.fit-status.json",
                f"case-{index:02d}.baseline-status.json",
            )
        )
    expected_campaign_status = "complete" if statuses_complete else "incomplete"
    if campaign_result.status != expected_campaign_status:
        raise ABC6ScoringError("campaign overall status disagrees with its 48 cases")
    return identity, receipt_directory, tuple(expected_receipt_names)


def _expected_forecast_artifact_payload(
    *,
    campaign_result,
    evidence_manifest_sha256: str,
    source_hashes: tuple[tuple[str, str], ...],
    receipt_hashes: tuple[tuple[str, str], ...],
    summary_sha256: str,
    frozen: ABC6FrozenForecastRoster,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "training_manifest_sha256": campaign_result.manifest_sha256,
        "training_claim_sha256": campaign_result.claim_sha256,
        "training_summary_filename": _TRAINING_SUMMARY_FILENAME,
        "training_summary_sha256": summary_sha256,
        "training_evidence_manifest_sha256": evidence_manifest_sha256,
        "integrated_source_hashes": [
            {"path": path, "sha256": digest} for path, digest in source_hashes
        ],
        "ordered_case_identities": [
            {
                "case_index": case.case_index,
                "case_id": case.case_id,
                "truth_id": case.truth_id,
                "input_window": case.input_window,
                "replicate": case.replicate,
                "fit_model": case.fit_model.value,
            }
            for case in CASE_ROSTER
        ],
        "status_receipts": [
            {"filename": filename, "sha256": digest}
            for filename, digest in receipt_hashes
        ],
        "forecast_case_sha256": list(frozen.case_sha256),
        "forecast_roster_sha256": frozen.roster_sha256,
        "forecasts": [_json_ready(item) for item in frozen.forecasts],
        "target_free": True,
        "prospective_targets_generated_by_runner": False,
        "retry_allowed": False,
    }
    body["payload_sha256"] = _sha256(_canonical_json(body))
    return _canonical_json(body)


def _verify_frozen_forecast_artifact(
    path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    receipt_anchor: cases._DirectoryAnchor,
    campaign_result,
    evidence_manifest_sha256: str,
    source_hashes: tuple[tuple[str, str], ...],
    receipt_hashes: tuple[tuple[str, str], ...],
    summary_sha256: str,
    frozen: ABC6FrozenForecastRoster,
) -> str:
    if not _is_sha256(expected_sha256):
        raise ABC6ScoringError("forecast artifact digest must be a lowercase SHA-256")
    artifact_path = Path(path)
    expected_path = receipt_anchor.path / _FORECAST_ARTIFACT_FILENAME
    if artifact_path != expected_path:
        raise ABC6ScoringError("forecast artifact must be the fixed campaign artifact")
    try:
        cases._verify_directory_anchor_path(
            receipt_anchor, label="status receipt directory"
        )
        raw = cases._read_regular_file_at(
            receipt_anchor.descriptor,
            _FORECAST_ARTIFACT_FILENAME,
            maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
            label="target-free forecast artifact",
        )
    except cases.ABC6StatusReceiptError as error:
        raise ABC6ScoringError(
            "durable target-free forecast artifact is unreadable"
        ) from error
    expected_payload = _expected_forecast_artifact_payload(
        campaign_result=campaign_result,
        evidence_manifest_sha256=evidence_manifest_sha256,
        source_hashes=source_hashes,
        receipt_hashes=receipt_hashes,
        summary_sha256=summary_sha256,
        frozen=frozen,
    )
    if _sha256(raw) != expected_sha256 or raw != expected_payload:
        raise ABC6ScoringError(
            "durable forecast artifact does not bind the verified evidence/source/forecast roster"
        )
    try:
        cases._verify_directory_anchor_path(
            receipt_anchor, label="status receipt directory"
        )
    except cases.ABC6StatusReceiptError as error:
        raise ABC6ScoringError(
            "campaign receipt path changed while verifying the forecast artifact"
        ) from error
    return expected_sha256


def _json_ready(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ABC6ScoringError("forecast evidence contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ABC6ScoringError(
        f"forecast evidence contains unsupported value type {type(value).__name__}"
    )


def _forecast_case_hash(
    case_index: int, forecast: ABC6PosteriorBaselineForecast
) -> str:
    case = CASE_ROSTER[case_index]
    if not isinstance(forecast, ABC6PosteriorBaselineForecast):
        raise ABC6ScoringError("every case requires a target-free ABC6 forecast")
    if type(forecast.case_index) is not int:
        raise ABC6ScoringError("forecast case_index must be an exact integer")
    if forecast.case_index != case_index or forecast.case_id != case.case_id:
        raise ABC6ScoringError(
            "forecast identity does not match the frozen case roster"
        )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "case_index": case_index,
        "case_id": case.case_id,
        "truth_id": case.truth_id,
        "input_window": case.input_window,
        "replicate": case.replicate,
        "fit_model": case.fit_model.value,
        "forecast": _json_ready(forecast),
    }
    return _sha256(_canonical_json(payload))


def freeze_abc6_forecasts(
    forecasts: Sequence[ABC6PosteriorBaselineForecast],
) -> ABC6FrozenForecastRoster:
    """Hash the ordered 24-case target-free roster before any reveal attempt."""

    if isinstance(forecasts, (str, bytes, bytearray)):
        raise TypeError("forecasts must be an ordered sequence of 24 forecast records")
    try:
        forecast_tuple = tuple(forecasts)
    except TypeError as error:
        raise TypeError(
            "forecasts must be an ordered sequence of 24 records"
        ) from error
    if len(forecast_tuple) != CASE_COUNT:
        raise ABC6ScoringError("forecast roster must contain exactly 24 ordered cases")
    case_hashes = tuple(
        _forecast_case_hash(index, forecast)
        for index, forecast in enumerate(forecast_tuple)
    )
    roster_hash = _sha256(
        _canonical_json(
            {
                "protocol_id": PROTOCOL_ID,
                "run_id": RUN_ID,
                "case_sha256": case_hashes,
            }
        )
    )
    return ABC6FrozenForecastRoster(forecast_tuple, case_hashes, roster_hash)


def _finite_vector(values: object, *, label: str, length: int) -> tuple[float, ...]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ABC6ScoringError(f"{label} must be a numeric vector") from error
    if array.ndim != 1 or array.dtype.kind not in "fiu" or array.size != length:
        raise ABC6ScoringError(f"{label} must be a numeric vector of length {length}")
    result = tuple(float(value) for value in array)
    if not all(math.isfinite(value) for value in result):
        raise ABC6ScoringError(f"{label} must contain only finite values")
    return result


def _validated_posterior(
    result: object, case_index: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]] | None:
    abc_status = getattr(result, "abc_status", None)
    abc_result = getattr(result, "abc_result", None)
    if abc_status != "complete":
        return None
    if not isinstance(abc_result, Mapping) or abc_result.get("complete") is not True:
        raise ABC6ScoringError("complete fit is missing a complete ABC posterior")
    posterior = abc_result.get("posterior")
    if not isinstance(posterior, Mapping):
        raise ABC6ScoringError("complete fit is missing its terminal posterior")
    if tuple(posterior.get("parameter_order", ())) != tuple(PARAMETER_ORDER):
        raise ABC6ScoringError("posterior parameter order does not match the protocol")
    free_values = posterior.get("free_parameter_values")
    if not isinstance(free_values, Mapping) or set(free_values) != set(PARAMETER_ORDER):
        raise ABC6ScoringError("posterior parameter columns do not match the protocol")
    try:
        columns = [np.asarray(free_values[name]) for name in PARAMETER_ORDER]
        weight_array = np.asarray(posterior.get("weights"))
    except (TypeError, ValueError) as error:
        raise ABC6ScoringError("posterior arrays are malformed") from error
    if any(column.ndim != 1 or column.dtype.kind not in "fiu" for column in columns):
        raise ABC6ScoringError("posterior parameter columns must be numeric vectors")
    particle_count = columns[0].size
    if particle_count < 1 or any(column.size != particle_count for column in columns):
        raise ABC6ScoringError("posterior parameter columns have inconsistent lengths")
    if particle_count != PARTICLE_COUNT:
        raise ABC6ScoringError(
            f"terminal posterior must contain exactly {PARTICLE_COUNT} particles"
        )
    if (
        weight_array.ndim != 1
        or weight_array.dtype.kind not in "fiu"
        or weight_array.size != particle_count
    ):
        raise ABC6ScoringError("posterior weights do not match the particle count")
    matrix = np.column_stack(columns).astype(float, copy=False)
    weights = weight_array.astype(float, copy=False)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(weights)):
        raise ABC6ScoringError("posterior values and weights must be finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ABC6ScoringError("posterior weights must be nonnegative and nonzero")
    if not math.isclose(
        float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=_WEIGHT_SUM_ABS_TOL
    ):
        raise ABC6ScoringError("posterior weights must sum to one")
    bounds = np.asarray(CASE_ROSTER[case_index].prior_bounds, dtype=float)
    if np.any(matrix < bounds[:, 0]) or np.any(matrix > bounds[:, 1]):
        raise ABC6ScoringError("posterior parameter lies outside its frozen prior box")
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    weight_tuple = tuple(float(value) for value in weights)
    return rows, weight_tuple


def _validate_result_identity(result: object, case_index: int) -> None:
    case = CASE_ROSTER[case_index]
    if type(getattr(result, "case_index", None)) is not int:
        raise ABC6ScoringError("training result case_index must be an exact integer")
    identity = (
        getattr(result, "case_index", None),
        getattr(result, "case_id", None),
        getattr(result, "model", None),
        getattr(result, "fit_length", None),
    )
    expected = (case.case_index, case.case_id, case.fit_model.value, case.input_length)
    if identity != expected:
        raise ABC6ScoringError(
            "training result identity does not match the case roster"
        )
    abc_status = getattr(result, "abc_status", None)
    baseline_status = getattr(result, "baseline_status", None)
    status = getattr(result, "status", None)
    if abc_status not in {"complete", "incomplete", "unresolved"}:
        raise ABC6ScoringError("training result has an unsupported ABC status")
    if baseline_status not in {"complete", "incomplete"}:
        raise ABC6ScoringError("training result has an unsupported baseline status")
    expected_status = (
        "unresolved"
        if abc_status == "unresolved"
        else "complete"
        if abc_status == "complete" and baseline_status == "complete"
        else "incomplete"
    )
    if status != expected_status:
        raise ABC6ScoringError(
            "overall training status conflicts with component statuses"
        )
    if abc_status == "incomplete":
        abc_result = getattr(result, "abc_result", None)
        if (
            not isinstance(abc_result, Mapping)
            or abc_result.get("complete") is not False
        ):
            raise ABC6ScoringError(
                "incomplete ABC fit must not claim a terminal posterior"
            )
        if "posterior" in abc_result:
            raise ABC6ScoringError(
                "incomplete ABC fit must not expose a terminal posterior"
            )
    if abc_status == "unresolved" and getattr(result, "abc_result", None) is not None:
        raise ABC6ScoringError("unresolved calibration must not expose an ABC result")
    _validated_posterior(result, case_index)


def _validate_training_bundle(
    training: ABC6TrainingBundle,
) -> tuple[ABC6TrainingCaseData, ...]:
    if not isinstance(training, ABC6TrainingBundle):
        raise TypeError("training must be the synthetic ABC6 training bundle")
    data_by_case: list[ABC6TrainingCaseData] = []
    for case in CASE_ROSTER:
        data = training.data_for_case(case.case_index)
        expected_inputs = (
            TRAINING_INPUT_S if case.input_window == "S" else TRAINING_INPUT_L
        )
        if (
            not isinstance(data, ABC6TrainingCaseData)
            or type(data.case.case_index) is not int
            or data.case != case
        ):
            raise ABC6ScoringError(
                "training data identity does not match the frozen roster"
            )
        if data.inputs != expected_inputs:
            raise ABC6ScoringError(
                "training inputs do not match the frozen input window"
            )
        _finite_vector(
            data.observed_outputs,
            label="training observations",
            length=case.input_length,
        )
        data_by_case.append(data)
    return tuple(data_by_case)


def _strict_json_object(pairs):
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_summary_snapshot(raw: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ABC6ScoringError("training summary is malformed JSON") from error
    if not isinstance(decoded, dict):
        raise ABC6ScoringError("training summary must be a JSON object")
    return decoded


def _read_verified_training_summary(
    receipt_anchor: cases._DirectoryAnchor,
    receipt_statuses: tuple[tuple[str, str], ...],
    receipt_hashes: tuple[tuple[str, str], ...],
) -> str:
    try:
        raw = cases._read_regular_file_at(
            receipt_anchor.descriptor,
            _TRAINING_SUMMARY_FILENAME,
            maximum_bytes=1_000_000,
            label="training summary",
        )
    except cases.ABC6StatusReceiptError as error:
        raise ABC6ScoringError(
            "durable training summary is missing or unreadable"
        ) from error
    decoded = _parse_summary_snapshot(raw)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "claim_sha256",
        "written_at_utc",
        "status",
        "case_count",
        "fit_status_counts",
        "baseline_status_counts",
        "case_statuses",
        "training_only",
        "prospective_targets_generated",
        "forecasts_run",
        "postfit_target_gate_opened",
        "external_watchdog_enforced_here",
        "independent_manifest_approval_enforced_here",
        "resume_allowed",
        "payload_sha256",
    }
    if set(decoded) != expected_keys:
        raise ABC6ScoringError("training summary fields do not match the frozen schema")
    if type(decoded["schema_version"]) is not int or decoded["schema_version"] != 1:
        raise ABC6ScoringError("training summary schema_version must be integer 1")
    if type(decoded["case_count"]) is not int or decoded["case_count"] != CASE_COUNT:
        raise ABC6ScoringError("training summary case_count must be integer 24")
    if decoded["protocol_id"] != PROTOCOL_ID or decoded["run_id"] != RUN_ID:
        raise ABC6ScoringError("training summary protocol or run identity mismatch")
    for name in ("manifest_sha256", "claim_sha256", "payload_sha256"):
        value = decoded[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ABC6ScoringError(
                f"training summary {name} must be a lowercase SHA-256"
            )
    if not isinstance(decoded["written_at_utc"], str):
        raise ABC6ScoringError("training summary timestamp must be text")
    try:
        datetime.strptime(decoded["written_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ABC6ScoringError(
            "training summary timestamp must be UTC seconds"
        ) from error

    complete = all(
        fit == "complete" and baseline == "complete"
        for fit, baseline in receipt_statuses
    )
    expected_overall = "complete" if complete else "incomplete"
    if decoded["status"] != expected_overall:
        raise ABC6ScoringError("training summary status disagrees with case receipts")
    expected_flags = {
        "training_only": True,
        "prospective_targets_generated": False,
        "forecasts_run": False,
        "postfit_target_gate_opened": False,
        "external_watchdog_enforced_here": False,
        "independent_manifest_approval_enforced_here": False,
        "resume_allowed": False,
    }
    for key, value in expected_flags.items():
        if type(decoded[key]) is not bool or decoded[key] is not value:
            raise ABC6ScoringError(f"training summary {key} must be {value}")

    expected_rows: list[dict[str, object]] = []
    for case, (fit_status, baseline_status) in zip(
        CASE_ROSTER, receipt_statuses, strict=True
    ):
        fit_hash = dict(receipt_hashes)[f"case-{case.case_index:02d}.fit-status.json"]
        baseline_hash = dict(receipt_hashes)[
            f"case-{case.case_index:02d}.baseline-status.json"
        ]
        expected_rows.append(
            {
                "case_index": case.case_index,
                "case_id": case.case_id,
                "fit_status": fit_status,
                "baseline_status": baseline_status,
                "fit_receipt_sha256": fit_hash,
                "baseline_receipt_sha256": baseline_hash,
            }
        )
    summary_rows = decoded["case_statuses"]
    if not isinstance(summary_rows, list) or len(summary_rows) != CASE_COUNT:
        raise ABC6ScoringError("training summary must list exactly 24 ordered cases")
    for row_index, (actual, expected) in enumerate(
        zip(summary_rows, expected_rows, strict=True)
    ):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ABC6ScoringError(
                f"training summary case row {row_index} has wrong schema"
            )
        if type(actual.get("case_index")) is not int:
            raise ABC6ScoringError(
                f"training summary case row {row_index} index must be integer"
            )
        if any(
            not isinstance(actual.get(key), str)
            for key in (
                "case_id",
                "fit_status",
                "baseline_status",
                "fit_receipt_sha256",
                "baseline_receipt_sha256",
            )
        ):
            raise ABC6ScoringError(
                f"training summary case row {row_index} text fields are invalid"
            )
        if actual != expected:
            raise ABC6ScoringError(
                f"training summary case row {row_index} identity/status/hash mismatch"
            )

    fit_counts: dict[str, int] = {}
    baseline_counts: dict[str, int] = {}
    for fit_status, baseline_status in receipt_statuses:
        fit_counts[fit_status] = fit_counts.get(fit_status, 0) + 1
        baseline_counts[baseline_status] = baseline_counts.get(baseline_status, 0) + 1
    for key, expected in (
        ("fit_status_counts", fit_counts),
        ("baseline_status_counts", baseline_counts),
    ):
        actual = decoded[key]
        if not isinstance(actual, dict) or any(
            not isinstance(name, str) or type(count) is not int
            for name, count in actual.items()
        ):
            raise ABC6ScoringError(f"training summary {key} has invalid types")
        if actual != expected:
            raise ABC6ScoringError(
                f"training summary {key} disagrees with case receipts"
            )

    body = {key: value for key, value in decoded.items() if key != "payload_sha256"}
    if decoded["payload_sha256"] != _sha256(_canonical_json(body)):
        raise ABC6ScoringError("training summary payload hash mismatch")
    if raw != _canonical_json(decoded):
        raise ABC6ScoringError("training summary is not canonical JSON")
    return _sha256(raw)


def _read_verified_statuses(receipt_source: object):
    """Read status and summary bytes through one no-follow directory anchor.

    The scoring path passes the typed campaign identity and receives a target
    gate retaining both checkout and receipt descriptors. The runner's earlier
    target-free status check may pass only a path; that read-only mode opens a
    no-follow descriptor and never returns a target capability.
    """

    gate = None
    temporary_anchor: cases._DirectoryAnchor | None = None
    if isinstance(receipt_source, campaign_fit.ABC6ReceiptRootIdentity):
        gate = open_deferred_abc6_target_gate(receipt_source)
        receipt_anchor = gate._receipt_anchor
    else:
        receipt_path = cases._absolute_lexical_path(receipt_source)
        if not receipt_path.as_posix().endswith("/" + cases.RECEIPT_ROOT_RELATIVE):
            raise ABC6ScoringError("status receipt path differs from the fixed run root")
        temporary_anchor = cases._open_directory_anchor(
            receipt_path,
            label="status receipt directory",
        )
        receipt_anchor = temporary_anchor
    statuses: list[tuple[str, str]] = []
    hashes: list[tuple[str, str]] = []
    try:
        if gate is not None:
            _verify_gate_anchors(gate)
        else:
            cases._verify_directory_anchor_path(
                receipt_anchor, label="status receipt directory"
            )
        verified = cases._verify_all_status_receipts_at(receipt_anchor.descriptor)
        for case in CASE_ROSTER:
            components: list[str] = []
            for component in ("fit", "baseline"):
                filename = f"case-{case.case_index:02d}.{component}-status.json"
                raw = cases._read_regular_file_at(
                    receipt_anchor.descriptor,
                    filename,
                    maximum_bytes=cases._MAX_STATUS_RECEIPT_BYTES,
                    label="status receipt",
                )
                status, digest = cases._validate_status_receipt_bytes(
                    raw,
                    filename,
                    case_index=case.case_index,
                    component=component,
                )
                components.append(status)
                hashes.append((filename, digest))
            statuses.append((components[0], components[1]))
        if tuple(hashes) != verified:
            raise ABC6ScoringError("status receipts changed during scoring preflight")
        if gate is not None and tuple(hashes) != gate._verified_receipts:
            raise ABC6ScoringError("status receipt roster changed while opening the gate")
        status_tuple = tuple(statuses)
        hash_tuple = tuple(hashes)
        summary_hash = _read_verified_training_summary(
            receipt_anchor, status_tuple, hash_tuple
        )
        if gate is not None:
            _verify_gate_anchors(gate)
        else:
            cases._verify_directory_anchor_path(
                receipt_anchor, label="status receipt directory"
            )
        return gate, status_tuple, hash_tuple, summary_hash
    finally:
        if temporary_anchor is not None:
            temporary_anchor.close()


def _verify_gate_anchors(gate: cases.DeferredABC6TargetGate) -> None:
    if not isinstance(gate, cases.DeferredABC6TargetGate):
        raise ABC6ScoringError("scoring requires the typed campaign status gate")
    try:
        cases._verify_directory_anchor_path(
            gate._root_anchor, label="campaign checkout root"
        )
        cases._verify_directory_anchor_path(
            gate._receipt_anchor, label="status receipt directory"
        )
    except cases.ABC6StatusReceiptError as error:
        raise ABC6ScoringError("campaign directory identity changed") from error
    if gate._receipt_anchor.path != (
        gate._root_anchor.path / cases.RECEIPT_ROOT_RELATIVE
    ):
        raise ABC6ScoringError("campaign receipt gate is bound to another checkout root")


def _validated_forecast(
    forecast: ABC6PosteriorBaselineForecast,
    result: object,
    case_index: int,
    posterior: tuple[tuple[tuple[float, ...], ...], tuple[float, ...]] | None,
) -> None:
    case = CASE_ROSTER[case_index]
    if forecast.parameter_order != tuple(PARAMETER_ORDER):
        raise ABC6ScoringError("forecast parameter order does not match the protocol")
    if forecast.fit_window != case.input_window:
        raise ABC6ScoringError("forecast window does not match the frozen case")
    if case.truth_id == "N":
        if (
            forecast.status != "abstained_n"
            or forecast.prospective_inputs is not None
            or forecast.common_state_index is not None
            or forecast.particles
            or forecast.weights
            or forecast.particle_trajectories
            or forecast.aggregate_status != "unavailable"
            or forecast.pointwise_weighted_mean is not None
            or forecast.pointwise_weighted_median is not None
            or forecast.pointwise_q05 is not None
            or forecast.pointwise_q95 is not None
            or forecast.baseline_trajectory is not None
        ):
            raise ABC6ScoringError("N must abstain without any prospective forecast")
        return

    abc_status = result.abc_status
    if abc_status != "complete":
        if (
            forecast.status != "incomplete_abc_fit"
            or forecast.prospective_inputs is not None
            or forecast.common_state_index is not None
            or forecast.particles
            or forecast.weights
            or forecast.particle_trajectories
            or forecast.aggregate_status != "unavailable"
            or forecast.pointwise_weighted_mean is not None
            or forecast.pointwise_weighted_median is not None
            or forecast.pointwise_q05 is not None
            or forecast.pointwise_q95 is not None
            or forecast.baseline_trajectory is not None
        ):
            raise ABC6ScoringError("incomplete fit must have a null forecast record")
        return
    if posterior is None:
        raise ABC6ScoringError("complete fit has no validated terminal posterior")
    parameter_rows, weights = posterior
    if (
        forecast.status != "complete"
        or forecast.prospective_inputs != tuple(cases.PROSPECTIVE_INPUT)
        or forecast.common_state_index != TRAINING_LENGTH
        or forecast.pointwise_summaries_are_coherent_trajectories is not False
    ):
        raise ABC6ScoringError("complete forecast metadata does not match the protocol")
    if len(forecast.particles) != len(parameter_rows):
        raise ABC6ScoringError("forecast particle count differs from the posterior")
    if tuple(forecast.weights) != weights:
        raise ABC6ScoringError("forecast weights differ from the terminal posterior")
    if len(forecast.particle_trajectories) != len(parameter_rows):
        raise ABC6ScoringError("forecast trajectory count differs from the posterior")
    if not math.isclose(
        float(forecast.effective_sample_size),
        1.0 / sum(weight * weight for weight in weights),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ABC6ScoringError("forecast ESS does not match the terminal weights")

    for particle_index, (particle, parameters, weight, trajectory) in enumerate(
        zip(
            forecast.particles,
            parameter_rows,
            weights,
            forecast.particle_trajectories,
            strict=True,
        )
    ):
        if type(particle.particle_index) is not int:
            raise ABC6ScoringError("particle_index must be an exact integer")
        if (
            particle.particle_index != particle_index
            or tuple(particle.parameter_values) != parameters
            or particle.weight != weight
            or particle.trajectory != trajectory
            or (trajectory is None) != (particle.failure is not None)
        ):
            raise ABC6ScoringError(
                "forecast particle identity or weights are inconsistent"
            )
        if trajectory is not None:
            _finite_vector(
                trajectory,
                label="particle prospective trajectory",
                length=PROSPECTIVE_LENGTH,
            )
    all_complete = all(
        trajectory is not None for trajectory in forecast.particle_trajectories
    )
    if all_complete:
        if forecast.aggregate_status != "complete":
            raise ABC6ScoringError(
                "all-particle forecast must have a complete aggregate"
            )
        matrix = np.asarray(forecast.particle_trajectories, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        expected_mean = tuple(
            float(v) for v in np.sum(matrix * weight_array[:, None], axis=0)
        )
        expected_median = tuple(
            weighted_left_inverse_quantile(matrix[:, time], weights, 0.50)
            for time in range(PROSPECTIVE_LENGTH)
        )
        expected_q05 = tuple(
            weighted_left_inverse_quantile(matrix[:, time], weights, 0.05)
            for time in range(PROSPECTIVE_LENGTH)
        )
        expected_q95 = tuple(
            weighted_left_inverse_quantile(matrix[:, time], weights, 0.95)
            for time in range(PROSPECTIVE_LENGTH)
        )
        if (
            not np.allclose(
                forecast.pointwise_weighted_mean,
                expected_mean,
                rtol=0.0,
                atol=1.0e-12,
            )
            or forecast.pointwise_weighted_median != expected_median
            or forecast.pointwise_q05 != expected_q05
            or forecast.pointwise_q95 != expected_q95
        ):
            raise ABC6ScoringError("forecast pointwise summaries fail recomputation")
    else:
        if forecast.aggregate_status != "suppressed_particle_failure":
            raise ABC6ScoringError(
                "a particle failure must suppress every ABC aggregate"
            )
        if any(
            summary is not None
            for summary in (
                forecast.pointwise_weighted_mean,
                forecast.pointwise_weighted_median,
                forecast.pointwise_q05,
                forecast.pointwise_q95,
            )
        ):
            raise ABC6ScoringError(
                "failed all-particle aggregate must retain null summaries"
            )
    if forecast.baseline_trajectory is not None:
        _finite_vector(
            forecast.baseline_trajectory,
            label="baseline prospective trajectory",
            length=PROSPECTIVE_LENGTH,
        )
        baseline_parameters = _finite_vector(
            forecast.baseline_parameter_values,
            label="baseline parameter values",
            length=len(PARAMETER_ORDER),
        )
        baseline_bounds = np.asarray(case.prior_bounds, dtype=float)
        if np.any(np.asarray(baseline_parameters) < baseline_bounds[:, 0]) or np.any(
            np.asarray(baseline_parameters) > baseline_bounds[:, 1]
        ):
            raise ABC6ScoringError(
                "baseline parameters lie outside the frozen prior box"
            )


def _prepare_marker_parent(
    marker_path: Path,
    root_anchor: cases._DirectoryAnchor,
) -> cases._DirectoryAnchor:
    """Create the fixed claims parent relative to the campaign root FD."""

    marker_path = cases._absolute_lexical_path(marker_path)
    expected_marker = cases._fixed_reveal_marker_path(root_anchor)
    if marker_path != expected_marker:
        raise ABC6ScoringError(
            "reveal marker path differs from the typed campaign checkout root"
        )
    try:
        cases._verify_directory_anchor_path(
            root_anchor, label="campaign checkout root"
        )
        parent_anchor = cases._open_relative_directory_anchor(
            root_anchor,
            "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims",
            label="reveal marker parent",
            create=True,
        )
        cases._verify_directory_anchor_path(
            parent_anchor, label="reveal marker parent"
        )
        return parent_anchor
    except cases.ABC6StatusReceiptError as error:
        raise ABC6ScoringError(
            "reveal marker ancestry is missing or contains a symlink"
        ) from error


def _claim_reveal_marker(
    marker_path: Path,
    marker_payload: Mapping[str, object],
    *,
    root_anchor: cases._DirectoryAnchor,
) -> _ABC6RevealMarkerClaim:
    marker_path = cases._absolute_lexical_path(marker_path)
    parent_anchor = _prepare_marker_parent(marker_path, root_anchor)
    parent_fd = parent_anchor.descriptor
    payload = _canonical_json(dict(marker_payload))
    transferred = False
    marker_info: os.stat_result | None = None
    try:
        try:
            descriptor = os.open(
                marker_path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise ABC6RevealAlreadyConsumedError(
                "the fixed ABC6 reveal marker already exists; this run cannot be retried"
            ) from error
        except OSError as error:
            raise ABC6ScoringError(
                "could not create the fixed reveal marker"
            ) from error
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("reveal marker write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            marker_info = os.fstat(descriptor)
            if not stat.S_ISREG(marker_info.st_mode) or marker_info.st_size != len(payload):
                raise OSError("reveal marker is not a complete regular file")
        except OSError as error:
            # Leave any partial marker in place: uncertain creation consumes the ID.
            raise ABC6ScoringError(
                "reveal marker durability is uncertain; do not retry"
            ) from error
        finally:
            os.close(descriptor)
        try:
            os.fsync(parent_fd)
        except OSError as error:
            # The exclusive marker remains; it is never removed or reused.
            raise ABC6ScoringError(
                "reveal marker directory durability is uncertain"
            ) from error
        try:
            recorded_bytes = cases._read_regular_file_at(
                parent_fd,
                marker_path.name,
                maximum_bytes=cases._MAX_REVEAL_MARKER_BYTES,
                label="fixed scoring reveal marker",
            )
        except cases.ABC6StatusReceiptError as error:
            raise ABC6ScoringError(
                "could not read back the durable reveal marker"
            ) from error
        if recorded_bytes != payload:
            raise ABC6ScoringError(
                "durable reveal marker bytes differ from the claim payload"
            )
        marker_anchor: cases._DirectoryAnchor | None = None
        try:
            marker_anchor = parent_anchor
            cases._verify_directory_anchor_path(
                marker_anchor, label="reveal marker parent"
            )
            cases._verify_directory_anchor_path(
                root_anchor, label="campaign checkout root"
            )
        except Exception as error:
            if marker_anchor is not None:
                marker_anchor.close()
            raise ABC6ScoringError(
                "durable reveal marker path identity changed; retry forbidden"
            ) from error
        if marker_info is None:  # pragma: no cover - guarded by successful write.
            raise ABC6ScoringError("reveal marker has no captured file identity")
        relative_path = marker_path.relative_to(root_anchor.path).as_posix()
        claim = _ABC6RevealMarkerClaim(
            marker_sha256=_sha256(recorded_bytes),
            marker_parent_anchor=marker_anchor,
            marker_relative_path=relative_path,
            marker_parent_device=marker_anchor.device,
            marker_parent_inode=marker_anchor.inode,
            marker_device=marker_info.st_dev,
            marker_inode=marker_info.st_ino,
            marker_size=marker_info.st_size,
            marker_mtime_ns=marker_info.st_mtime_ns,
            marker_ctime_ns=marker_info.st_ctime_ns,
        )
        transferred = True
        return claim
    finally:
        if not transferred:
            parent_anchor.close()


def _rmse(prediction: Sequence[float], target: Sequence[float]) -> float:
    predicted = _finite_vector(
        prediction, label="forecast score input", length=PROSPECTIVE_LENGTH
    )
    observed = _finite_vector(
        target, label="prospective target", length=PROSPECTIVE_LENGTH
    )
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sqrt(np.mean(np.square(np.asarray(predicted) - observed))))
    if not math.isfinite(value):
        raise ABC6ScoringError("prospective RMSE is non-finite")
    return value


def _parameter_intervals(
    result: object, case_index: int
) -> tuple[ABC6ParameterInterval, ...] | None:
    case = CASE_ROSTER[case_index]
    if case.truth_id not in _TRUTH_PARAMETERS:
        return None
    true_model, truth = _TRUTH_PARAMETERS[case.truth_id]
    if case.fit_model.value != true_model:
        return None
    posterior = _validated_posterior(result, case_index)
    if posterior is None:
        return None
    rows, weights = posterior
    matrix = np.asarray(rows, dtype=float)
    intervals: list[ABC6ParameterInterval] = []
    for column, parameter in enumerate(PARAMETER_ORDER):
        q05 = weighted_left_inverse_quantile(matrix[:, column], weights, 0.05)
        q95 = weighted_left_inverse_quantile(matrix[:, column], weights, 0.95)
        intervals.append(
            ABC6ParameterInterval(
                parameter=parameter,
                q05=q05,
                q95=q95,
                truth_in_interval=q05 <= truth[column] <= q95,
            )
        )
    return tuple(intervals)


def _score_case(
    case_index: int,
    forecast: ABC6PosteriorBaselineForecast,
    result: object,
    target: tuple[float, ...] | None,
) -> ABC6CaseScore:
    case = CASE_ROSTER[case_index]
    if case.truth_id == "N":
        return ABC6CaseScore(
            case_index,
            case.case_id,
            case.truth_id,
            case.fit_model.value,
            case.input_window,
            case.replicate,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    if target is None:
        raise ABC6ScoringError("A/B/M scores require their deferred target")
    mean_rmse = median_rmse = envelope_fraction = None
    if forecast.aggregate_status == "complete":
        if any(
            item is None
            for item in (
                forecast.pointwise_weighted_mean,
                forecast.pointwise_weighted_median,
                forecast.pointwise_q05,
                forecast.pointwise_q95,
            )
        ):
            raise ABC6ScoringError("complete aggregate is missing pointwise summaries")
        mean_rmse = _rmse(forecast.pointwise_weighted_mean, target)
        median_rmse = _rmse(forecast.pointwise_weighted_median, target)
        lower = np.asarray(forecast.pointwise_q05, dtype=float)
        upper = np.asarray(forecast.pointwise_q95, dtype=float)
        observed = np.asarray(target, dtype=float)
        envelope_fraction = float(np.mean((lower <= observed) & (observed <= upper)))
    baseline_rmse = (
        None
        if forecast.baseline_trajectory is None
        else _rmse(forecast.baseline_trajectory, target)
    )
    return ABC6CaseScore(
        case_index=case_index,
        case_id=case.case_id,
        truth_id=case.truth_id,
        fit_model=case.fit_model.value,
        input_window=case.input_window,
        replicate=case.replicate,
        abc_weighted_mean_rmse=mean_rmse,
        abc_weighted_median_rmse=median_rmse,
        baseline_coherent_rmse=baseline_rmse,
        abc_q05_q95_envelope_inclusion_fraction=envelope_fraction,
        terminal_effective_sample_size=forecast.effective_sample_size,
        parameter_intervals=_parameter_intervals(result, case_index),
    )


def _paired_contrasts(
    scores: tuple[ABC6CaseScore, ...],
) -> tuple[ABC6PairedHorizonContrasts, ...]:
    score_by_index = {score.case_index: score for score in scores}
    output: list[ABC6PairedHorizonContrasts] = []
    for truth in ("A", "B"):
        same_family = _TRUTH_PARAMETERS[truth][0]
        for replicate in range(4):
            pair: dict[str, ABC6CaseScore] = {}
            for case in CASE_ROSTER:
                if (
                    case.truth_id == truth
                    and case.replicate == replicate
                    and case.fit_model.value == same_family
                ):
                    pair[case.input_window] = score_by_index[case.case_index]
            if set(pair) != {"S", "L"}:
                raise ABC6ScoringError("frozen same-family S/L pair is incomplete")

            def contrast(
                field: str, pair: Mapping[str, ABC6CaseScore] = pair
            ) -> float | None:
                short_score = getattr(pair["S"], field)
                long_score = getattr(pair["L"], field)
                if short_score is None or long_score is None:
                    return None
                return float(short_score - long_score)

            output.append(
                ABC6PairedHorizonContrasts(
                    truth_id=truth,  # type: ignore[arg-type]
                    replicate=replicate,
                    abc_weighted_mean_rmse_s_minus_l=contrast("abc_weighted_mean_rmse"),
                    abc_weighted_median_rmse_s_minus_l=contrast(
                        "abc_weighted_median_rmse"
                    ),
                    baseline_coherent_rmse_s_minus_l=contrast("baseline_coherent_rmse"),
                )
            )
    return tuple(output)


def _parameter_inclusion_counts(
    scores: tuple[ABC6CaseScore, ...],
) -> tuple[ABC6ParameterInclusionCount, ...]:
    score_by_index = {score.case_index: score for score in scores}
    output: list[ABC6ParameterInclusionCount] = []
    for truth in ("A", "B"):
        same_family = _TRUTH_PARAMETERS[truth][0]
        for window in ("S", "L"):
            eligible = [
                case
                for case in CASE_ROSTER
                if case.truth_id == truth
                and case.input_window == window
                and case.fit_model.value == same_family
            ]
            if len(eligible) != 4:
                raise ABC6ScoringError(
                    "frozen inclusion roster must have four fits per window"
                )
            for parameter in PARAMETER_ORDER:
                included_count = 0
                eligible_count = 0
                for case in eligible:
                    intervals = score_by_index[case.case_index].parameter_intervals
                    if intervals is None:
                        continue
                    matched = next(
                        (item for item in intervals if item.parameter == parameter),
                        None,
                    )
                    if matched is None:
                        raise ABC6ScoringError("parameter interval set is incomplete")
                    eligible_count += 1
                    included_count += int(matched.truth_in_interval)
                output.append(
                    ABC6ParameterInclusionCount(
                        truth_id=truth,  # type: ignore[arg-type]
                        input_window=window,  # type: ignore[arg-type]
                        parameter=parameter,
                        included_count=included_count,
                        eligible_fit_count=eligible_count,
                    )
                )
    return tuple(output)


def _no_crossing_diagnostics(
    results: Sequence[object],
) -> tuple[ABC6NoCrossingDiagnostic, ...]:
    diagnostics: list[ABC6NoCrossingDiagnostic] = []
    for case_index in (8, 9):
        result = results[case_index]
        posterior = _validated_posterior(result, case_index)
        if posterior is None:
            quantiles: tuple[float | None, float | None, float | None] = (
                None,
                None,
                None,
            )
        else:
            rows, weights = posterior
            ceiling = tuple(row[PARAMETER_ORDER.index("ceiling")] for row in rows)
            quantiles = tuple(
                weighted_left_inverse_quantile(ceiling, weights, probability)
                for probability in (0.05, 0.50, 0.95)
            )  # type: ignore[assignment]
        diagnostics.append(
            ABC6NoCrossingDiagnostic(
                case_index=case_index,
                fit_model=CASE_ROSTER[case_index].fit_model.value,
                ceiling_q05=quantiles[0],
                ceiling_q50=quantiles[1],
                ceiling_q95=quantiles[2],
            )
        )
    return tuple(diagnostics)


def _simulate_training_prediction(
    values: Sequence[float],
    data: ABC6TrainingCaseData,
    *,
    simulator,
) -> tuple[float, ...] | None:
    parameters = _finite_vector(
        values, label="training prediction parameters", length=6
    )
    named = dict(zip(PARAMETER_ORDER, parameters, strict=True))
    try:
        outcome = simulator(
            data.inputs,
            TankParameters(named["a"], named["c"], named["p"]),
            TankState(named["x1_0"], named["x2_0"]),
            model=data.case.fit_model,
            ceiling=named["ceiling"],
            limits=TankSimulationLimits(max_steps=len(data.inputs)),
        )
    except Exception:  # noqa: BLE001 - simulator failure leaves residual diagnostics null.
        return None
    if isinstance(outcome, TankSimulationFailure) or not isinstance(
        outcome, TankSimulationSuccess
    ):
        return None
    try:
        return _finite_vector(
            outcome.observations,
            label="M training prediction",
            length=TRAINING_LENGTH,
        )
    except ABC6ScoringError:
        return None


def _group_means(
    observed: Sequence[float], predicted: Sequence[float] | None
) -> tuple[float | None, float | None]:
    if predicted is None:
        return None, None
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    if residual.shape != (TRAINING_LENGTH,) or not np.all(np.isfinite(residual)):
        return None, None
    values: list[float] = []
    for start, stop, _label in _M_RESIDUAL_GROUPS:
        mean = float(np.mean(residual[start:stop]))
        if not math.isfinite(mean):
            return None, None
        values.append(mean)
    return values[0], values[1]


def _m_training_residuals(
    results: Sequence[object],
    forecasts: Sequence[ABC6PosteriorBaselineForecast],
    data: Sequence[ABC6TrainingCaseData],
    *,
    simulator,
) -> tuple[ABC6MTrainingResidualMeans, ...]:
    output: list[ABC6MTrainingResidualMeans] = []
    for case_index in (10, 11):
        result = results[case_index]
        forecast = forecasts[case_index]
        case_data = data[case_index]
        observed = _finite_vector(
            case_data.observed_outputs,
            label="M training observations",
            length=TRAINING_LENGTH,
        )
        baseline_prediction = None
        if forecast.baseline_parameter_values is not None:
            baseline_prediction = _simulate_training_prediction(
                forecast.baseline_parameter_values, case_data, simulator=simulator
            )
        baseline_groups = _group_means(observed, baseline_prediction)

        abc_prediction: tuple[float, ...] | None = None
        posterior = _validated_posterior(result, case_index)
        if posterior is not None:
            parameter_rows, weights = posterior
            predictions = [
                _simulate_training_prediction(row, case_data, simulator=simulator)
                for row in parameter_rows
            ]
            if all(prediction is not None for prediction in predictions):
                matrix = np.asarray(predictions, dtype=float)
                mean = np.sum(matrix * np.asarray(weights)[:, None], axis=0)
                if np.all(np.isfinite(mean)):
                    abc_prediction = tuple(float(value) for value in mean)
        abc_groups = _group_means(observed, abc_prediction)
        output.append(
            ABC6MTrainingResidualMeans(
                case_index=case_index,
                fit_model=CASE_ROSTER[case_index].fit_model.value,
                baseline_u8_first_24_mean=baseline_groups[0],
                baseline_u0_last_180_mean=baseline_groups[1],
                abc_weighted_mean_u8_first_24_mean=abc_groups[0],
                abc_weighted_mean_u0_last_180_mean=abc_groups[1],
            )
        )
    return tuple(output)


def _target_vectors_and_hashes(
    targets: ABC6ProspectiveTargets,
) -> tuple[dict[str, tuple[float, ...]], tuple[tuple[str, str], ...]]:
    if not isinstance(targets, ABC6ProspectiveTargets):
        raise ABC6ScoringError("case gate returned an unexpected target object")
    if targets.truth_ids != ("A", "B", "M"):
        raise ABC6ScoringError(
            "prospective target roster must contain A, B, and M only"
        )
    declared_hashes = targets.sha256_by_truth
    if tuple(name for name, _digest in declared_hashes) != ("A", "B", "M"):
        raise ABC6ScoringError("target hash roster must match A/B/M order exactly")
    hash_map = dict(declared_hashes)
    vectors: dict[str, tuple[float, ...]] = {}
    computed: list[tuple[str, str]] = []
    for truth in ("A", "B", "M"):
        values = _finite_vector(
            targets.for_truth(truth),
            label=f"{truth} prospective target",
            length=PROSPECTIVE_LENGTH,
        )
        digest = _sha256(np.asarray(values, dtype="<f8").tobytes(order="C"))
        if hash_map.get(truth) != digest:
            raise ABC6ScoringError(f"{truth} prospective target hash mismatch")
        vectors[truth] = values
        computed.append((truth, digest))
    return vectors, tuple(computed)


def _forecast_array_sha256(frozen: ABC6FrozenForecastRoster) -> str:
    """Hash every retained particle and baseline array in roster order."""

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "forecast_roster_sha256": frozen.roster_sha256,
        "ordered_case_sha256": list(frozen.case_sha256),
        "forecasts": [_json_ready(item) for item in frozen.forecasts],
    }
    return _sha256(_canonical_json(payload))


def _score_result_payload(result: ABC6DeferredScoreResult) -> dict[str, object]:
    """Return the stable score body, excluding receipt metadata to avoid recursion."""

    return {
        "protocol_id": result.protocol_id,
        "run_id": result.run_id,
        "reveal_marker_sha256": result.reveal_marker_sha256,
        "target_sha256_by_truth": [list(row) for row in result.target_sha256_by_truth],
        "case_scores": _json_ready(result.case_scores),
        "paired_horizon_contrasts": _json_ready(result.paired_horizon_contrasts),
        "parameter_inclusion_counts": _json_ready(result.parameter_inclusion_counts),
        "no_crossing_diagnostics": _json_ready(result.no_crossing_diagnostics),
        "m_training_residual_means": _json_ready(result.m_training_residual_means),
        "model_choice": None,
        "bayes_factors": None,
    }


def _target_arrays_artifact(
    target_vectors: Mapping[str, tuple[float, ...]],
    target_hashes: tuple[tuple[str, str], ...],
) -> bytes:
    if tuple(target_vectors) != ("A", "B", "M") or tuple(
        name for name, _digest in target_hashes
    ) != ("A", "B", "M"):
        raise ABC6ScoringError("target-array artifact requires the fixed A/B/M order")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "cascaded_tanks_abc6_prospective_target_arrays",
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "truth_ids": ["A", "B", "M"],
        "target_sha256_encoding": "float64-little-endian-c-order",
        "target_sha256_by_truth": [list(row) for row in target_hashes],
        "targets": [
            {"truth_id": truth, "values": list(target_vectors[truth])}
            for truth in ("A", "B", "M")
        ],
        "N": {
            "prospective_target_generated": False,
            "prospective_score_computed": False,
            "status": "mechanism_abstention_no_target_no_score",
        },
    }
    body["payload_sha256"] = _sha256(_canonical_json(body))
    return _canonical_json(body)


def _training_evidence_links(
    receipt_anchor: cases._DirectoryAnchor,
    execution: campaign_fit.ABC6TrainingCampaignExecution,
) -> dict[str, object]:
    """Link every durable fit/baseline cost and failure artifact into the result."""

    try:
        raw = cases._read_regular_file_at(
            receipt_anchor.descriptor,
            campaign_fit.EVIDENCE_MANIFEST_FILENAME,
            maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
            label="training evidence manifest for score receipt",
        )
        manifest = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {value}")
            ),
        )
    except Exception as error:
        raise ABC6ScoringError(
            "verified training evidence manifest could not be linked to score receipt"
        ) from error
    if (
        not isinstance(manifest, dict)
        or _sha256(raw) != execution.evidence_manifest_sha256
        or raw != _canonical_json(manifest)
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("run_id") != RUN_ID
        or manifest.get("manifest_sha256")
        != execution.campaign_result.manifest_sha256
        or manifest.get("claim_sha256") != execution.campaign_result.claim_sha256
        or manifest.get("training_summary_filename") != _TRAINING_SUMMARY_FILENAME
        or manifest.get("training_summary_sha256")
        != execution.campaign_result.summary_sha256
    ):
        raise ABC6ScoringError(
            "training evidence manifest changed before score receipt publication"
        )
    case_artifacts = manifest.get("case_artifacts")
    bundle_artifact = manifest.get("bundle_artifact")
    if (
        not isinstance(case_artifacts, list)
        or len(case_artifacts) != CASE_COUNT
        or not isinstance(bundle_artifact, dict)
        or set(bundle_artifact) != {"filename", "sha256"}
        or bundle_artifact.get("filename") != "training-bundle.evidence.json"
        or not _is_sha256(bundle_artifact.get("sha256"))
    ):
        raise ABC6ScoringError("training evidence artifact roster is malformed")
    expected_case_links: list[dict[str, object]] = []
    expected_receipts = [
        {
            "filename": f"case-{status.case_index:02d}.{component}-status.json",
            "sha256": getattr(status, f"{component}_receipt_sha256"),
            "case_index": status.case_index,
            "case_id": status.case_id,
            "component": component,
            "status": getattr(status, f"{component}_status"),
        }
        for status in execution.campaign_result.case_statuses
        for component in ("fit", "baseline")
    ]
    if manifest.get("status_receipts") != expected_receipts:
        raise ABC6ScoringError(
            "training evidence manifest status chain differs from score execution"
        )
    evidence_anchor: cases._DirectoryAnchor | None = None
    try:
        evidence_anchor = cases._open_relative_directory_anchor(
            receipt_anchor,
            campaign_fit.EVIDENCE_DIRECTORY_NAME,
            label="training evidence artifacts",
        )
        bundle_raw = cases._read_regular_file_at(
            evidence_anchor.descriptor,
            str(bundle_artifact["filename"]),
            maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
            label="training bundle evidence",
        )
        if _sha256(bundle_raw) != bundle_artifact["sha256"]:
            raise ABC6ScoringError("training bundle evidence digest changed")
        for index, (row, case, status) in enumerate(
            zip(
                case_artifacts,
                CASE_ROSTER,
                execution.campaign_result.case_statuses,
                strict=True,
            )
        ):
            expected_filename = f"case-{index:02d}.training-evidence.json"
            if (
                not isinstance(row, dict)
                or row.get("filename") != expected_filename
                or row.get("roster_index") != index
                or row.get("case_identity") != campaign_fit._case_identity(case)
                or row.get("fit_status") != status.fit_status
                or row.get("baseline_status") != status.baseline_status
                or not _is_sha256(row.get("sha256"))
            ):
                raise ABC6ScoringError(
                    f"training evidence artifact link is malformed at case {index}"
                )
            case_raw = cases._read_regular_file_at(
                evidence_anchor.descriptor,
                expected_filename,
                maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
                label=f"training evidence for case {index}",
            )
            if _sha256(case_raw) != row["sha256"]:
                raise ABC6ScoringError(
                    f"training evidence digest changed at case {index}"
                )
            expected_case_links.append(
                {
                    "roster_index": index,
                    "filename": expected_filename,
                    "sha256": row["sha256"],
                    "fit_status": row["fit_status"],
                    "baseline_status": row["baseline_status"],
                }
            )
        cases._verify_directory_anchor_path(
            evidence_anchor, label="training evidence artifacts"
        )
    except Exception as error:
        if isinstance(error, ABC6ScoringError):
            raise
        raise ABC6ScoringError(
            "training evidence artifact links could not be reverified"
        ) from error
    finally:
        if evidence_anchor is not None:
            evidence_anchor.close()
    return {
        "manifest_filename": campaign_fit.EVIDENCE_MANIFEST_FILENAME,
        "manifest_sha256": execution.evidence_manifest_sha256,
        "artifact_directory": campaign_fit.EVIDENCE_DIRECTORY_NAME,
        "bundle_artifact": {
            "filename": bundle_artifact["filename"],
            "sha256": bundle_artifact["sha256"],
        },
        "case_artifacts": expected_case_links,
        "diagnostics_location": "case_artifacts contain complete fit/baseline costs and failure evidence",
    }


def _score_receipt_payload(
    *,
    outcome: Literal["complete", "failed"],
    execution: campaign_fit.ABC6TrainingCampaignExecution,
    receipt_hashes: tuple[tuple[str, str], ...],
    summary_sha256: str,
    forecast_artifact_sha256: str,
    frozen: ABC6FrozenForecastRoster,
    forecast_array_sha256: str,
    marker_identity: Mapping[str, object],
    evidence_links: Mapping[str, object],
    target_hashes: tuple[tuple[str, str], ...] | None,
    target_arrays_sha256: str | None,
    score_result: Mapping[str, object] | None,
    score_result_sha256: str | None,
    score_event: Mapping[str, object],
    failure_checkpoint: Mapping[str, object] | None,
) -> bytes:
    campaign_result = execution.campaign_result
    target_hash_map = dict(target_hashes or ())
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "cascaded_tanks_abc6_deferred_score_receipt",
        "outcome": outcome,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "training": {
            "manifest_sha256": campaign_result.manifest_sha256,
            "claim_sha256": campaign_result.claim_sha256,
            "status": campaign_result.status,
            "summary_filename": campaign_result.summary_path.name,
            "summary_sha256": summary_sha256,
            "status_receipts": [
                {"filename": filename, "sha256": digest}
                for filename, digest in receipt_hashes
            ],
            "case_statuses": [
                {
                    "case_index": status.case_index,
                    "case_id": status.case_id,
                    "fit_status": status.fit_status,
                    "baseline_status": status.baseline_status,
                    "fit_receipt_sha256": status.fit_receipt_sha256,
                    "baseline_receipt_sha256": status.baseline_receipt_sha256,
                }
                for status in campaign_result.case_statuses
            ],
            "evidence": dict(evidence_links),
        },
        "target_free_forecast": {
            "artifact_filename": _FORECAST_ARTIFACT_FILENAME,
            "artifact_sha256": forecast_artifact_sha256,
            "forecast_roster_sha256": frozen.roster_sha256,
            "ordered_case_sha256": list(frozen.case_sha256),
            "forecast_array_sha256": forecast_array_sha256,
            "forecast_array_digest_encoding": (
                "canonical-json-ordered-full-forecast-records-v1"
            ),
        },
        "reveal_marker": dict(marker_identity),
        "prospective_targets": {
            "target_array_artifact_filename": TARGET_ARRAYS_ARTIFACT_FILENAME
            if target_arrays_sha256 is not None
            else None,
            "target_array_artifact_sha256": target_arrays_sha256,
            "target_sha256_by_truth": [
                [truth, target_hash_map.get(truth)] for truth in ("A", "B", "M")
            ],
            "target_sha256_encoding": "float64-little-endian-c-order",
            "truth_ids": ["A", "B", "M"] if target_hashes is not None else [],
            "N": {
                "prospective_target_generated": False,
                "prospective_score_computed": False,
                "mechanism_abstention": True,
                "status": "mechanism_abstention_no_target_no_score",
            },
        },
        "score_result": None if score_result is None else dict(score_result),
        "score_result_sha256": score_result_sha256,
        "score_result_digest_encoding": "canonical-json-score-result-body-v1",
        "score_event": dict(score_event),
        "failure_checkpoint": (
            None if failure_checkpoint is None else dict(failure_checkpoint)
        ),
        "retry_allowed": False,
    }
    body["payload_sha256"] = _sha256(_canonical_json(body))
    return _canonical_json(body)


def _score_event_snapshot(stage: str) -> dict[str, object]:
    """Capture a bounded same-host ordering point and an audit timestamp."""

    monotonic_ns = time.monotonic_ns()
    if type(monotonic_ns) is not int or not 0 <= monotonic_ns <= (2**63 - 1):
        raise ABC6ScoringError("system monotonic clock returned an invalid value")
    occurred_at_utc = (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    if len(occurred_at_utc) > 32:
        raise ABC6ScoringError("UTC audit timestamp exceeds its fixed bound")
    return {
        "stage": stage,
        "monotonic_ns": monotonic_ns,
        "occurred_at_utc": occurred_at_utc,
        "clock": "host-local-monotonic-ns",
    }


def _require_score_output_absent(
    receipt_anchor: cases._DirectoryAnchor,
) -> None:
    for filename in (TARGET_ARRAYS_ARTIFACT_FILENAME, SCORE_RECEIPT_FILENAME):
        try:
            os.stat(filename, dir_fd=receipt_anchor.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ABC6ScoringError(
                f"could not check for an existing score artifact: {filename}"
            ) from error
        raise ABC6ScoringError(
            f"score output already exists and cannot be replaced: {filename}"
        )


def _duplicate_postscore_root_anchors(
    gate: cases.DeferredABC6TargetGate,
    identity: campaign_fit.ABC6ReceiptRootIdentity,
) -> tuple[cases._DirectoryAnchor, cases._DirectoryAnchor]:
    """Retain the exact preclaim root and receipt descriptors for terminal output."""

    root_anchor: cases._DirectoryAnchor | None = None
    receipt_anchor: cases._DirectoryAnchor | None = None
    try:
        root_anchor = cases._DirectoryAnchor(
            gate._root_anchor.path,
            os.dup(gate._root_anchor.descriptor),
        )
        receipt_anchor = cases._DirectoryAnchor(
            gate._receipt_anchor.path,
            os.dup(gate._receipt_anchor.descriptor),
        )
        runtime = identity.runtime_identity
        if root_anchor.path != Path(identity.repository_root_realpath) or (
            receipt_anchor.path
            != Path(identity.repository_root_realpath)
            / campaign_fit.RECEIPT_ROOT_RELATIVE
        ):
            raise ABC6ScoringError("gate anchors differ from the typed receipt root")
        if (root_anchor.device, root_anchor.inode) != (
            runtime["repository_root_device"],
            runtime["repository_root_inode"],
        ) or (receipt_anchor.device, receipt_anchor.inode) != (
            runtime["receipt_root_device"],
            runtime["receipt_root_inode"],
        ):
            raise ABC6ScoringError("gate anchor descriptors differ from runtime identity")
        return root_anchor, receipt_anchor
    except Exception:
        if receipt_anchor is not None:
            receipt_anchor.close()
        if root_anchor is not None:
            root_anchor.close()
        raise


def _reject_existing_reveal_marker(
    marker_path: Path,
    root_anchor: cases._DirectoryAnchor,
) -> None:
    parent_anchor = _prepare_marker_parent(marker_path, root_anchor)
    try:
        try:
            os.stat(
                marker_path.name,
                dir_fd=parent_anchor.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as error:
            raise ABC6ScoringError("cannot inspect the fixed reveal marker") from error
        raise ABC6RevealAlreadyConsumedError(
            "the fixed ABC6 reveal marker already exists; this run cannot be retried"
        )
    finally:
        parent_anchor.close()


def _read_marker_snapshot(
    marker_anchor: cases._DirectoryAnchor,
    marker_identity: Mapping[str, object],
) -> bytes:
    filename = str(marker_identity["relative_path"]).rsplit("/", 1)[-1]
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=marker_anchor.descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > cases._MAX_REVEAL_MARKER_BYTES
        ):
            raise ABC6ScoringError("pinned reveal marker is not a bounded regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(
                descriptor,
                min(65536, cases._MAX_REVEAL_MARKER_BYTES + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > cases._MAX_REVEAL_MARKER_BYTES:
                raise ABC6ScoringError("pinned reveal marker exceeds its byte limit")
        after = os.fstat(descriptor)
        named = os.stat(
            filename,
            dir_fd=marker_anchor.descriptor,
            follow_symlinks=False,
        )
    except ABC6ScoringError:
        raise
    except OSError as error:
        raise ABC6ScoringError("pinned reveal marker could not be revalidated") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if stable != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or stable != (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    ):
        raise ABC6ScoringError("pinned reveal marker changed while reading")
    expected = (
        marker_identity["device"],
        marker_identity["inode"],
        marker_identity["size_bytes"],
        marker_identity["mtime_ns"],
        marker_identity["ctime_ns"],
    )
    if stable != expected:
        raise ABC6ScoringError("pinned reveal marker identity changed")
    raw = b"".join(chunks)
    if _sha256(raw) != marker_identity["sha256"]:
        raise ABC6ScoringError("pinned reveal marker digest changed")
    return raw


def _verify_postscore_anchors(
    *,
    identity: campaign_fit.ABC6ReceiptRootIdentity,
    root_anchor: cases._DirectoryAnchor,
    receipt_anchor: cases._DirectoryAnchor,
    marker_anchor: cases._DirectoryAnchor,
    marker_identity: Mapping[str, object],
    strict_paths: bool,
) -> None:
    root_path = Path(identity.repository_root_realpath)
    if (
        identity.receipt_root_relative != campaign_fit.RECEIPT_ROOT_RELATIVE
        or root_anchor.path != root_path
        or receipt_anchor.path != root_path / campaign_fit.RECEIPT_ROOT_RELATIVE
        or marker_anchor.path != root_path / str(marker_identity["relative_path"]).rsplit("/", 1)[0]
    ):
        raise ABC6ScoringError("post-score receipt or marker path identity is invalid")
    runtime = identity.runtime_identity
    for anchor, device_key, inode_key, label in (
        (root_anchor, "repository_root_device", "repository_root_inode", "checkout"),
        (receipt_anchor, "receipt_root_device", "receipt_root_inode", "receipt root"),
    ):
        info = os.fstat(anchor.descriptor)
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (
            runtime[device_key],
            runtime[inode_key],
        ):
            raise ABC6ScoringError(f"pinned {label} descriptor identity changed")
    marker_info = os.fstat(marker_anchor.descriptor)
    if not stat.S_ISDIR(marker_info.st_mode) or (
        marker_info.st_dev,
        marker_info.st_ino,
    ) != (
        marker_identity["parent_device"],
        marker_identity["parent_inode"],
    ):
        raise ABC6ScoringError("pinned reveal-marker parent identity changed")
    if strict_paths:
        try:
            identity.verify()
            cases._verify_directory_anchor_path(root_anchor, label="pinned checkout")
            cases._verify_directory_anchor_path(
                receipt_anchor, label="pinned receipt root"
            )
            cases._verify_directory_anchor_path(
                marker_anchor, label="pinned reveal-marker parent"
            )
        except Exception as error:
            raise ABC6ScoringError(
                "post-score receipt-root or marker path changed"
            ) from error
    _read_marker_snapshot(marker_anchor, marker_identity)


def _publish_postscore_artifact(
    *,
    filename: str,
    payload: bytes,
    execution: campaign_fit.ABC6TrainingCampaignExecution,
    root_anchor: cases._DirectoryAnchor,
    receipt_anchor: cases._DirectoryAnchor,
    marker_anchor: cases._DirectoryAnchor,
    marker_identity: Mapping[str, object],
    strict_paths: bool = True,
) -> str:
    _verify_postscore_anchors(
        identity=execution.receipt_root_identity,
        root_anchor=root_anchor,
        receipt_anchor=receipt_anchor,
        marker_anchor=marker_anchor,
        marker_identity=marker_identity,
        strict_paths=strict_paths,
    )
    try:
        campaign_fit._write_exclusive_durable_at(
            receipt_anchor.descriptor, filename, payload
        )
        raw = cases._read_regular_file_at(
            receipt_anchor.descriptor,
            filename,
            maximum_bytes=(
                _MAX_HANDOFF_ARTIFACT_BYTES
                if filename == TARGET_ARRAYS_ARTIFACT_FILENAME
                else _MAX_SCORE_RECEIPT_BYTES
            ),
            label=filename,
        )
    except Exception as error:
        raise ABC6ScoringError(
            f"durable post-score artifact publication failed: {filename}"
        ) from error
    if raw != payload:
        raise ABC6ScoringError(
            f"durable post-score artifact readback changed: {filename}"
        )
    _verify_postscore_anchors(
        identity=execution.receipt_root_identity,
        root_anchor=root_anchor,
        receipt_anchor=receipt_anchor,
        marker_anchor=marker_anchor,
        marker_identity=marker_identity,
        strict_paths=strict_paths,
    )
    return _sha256(raw)


def score_deferred_abc6_synthetic(
    execution: campaign_fit.ABC6TrainingCampaignExecution,
    frozen_forecasts: ABC6FrozenForecastRoster,
    forecast_artifact_path: str | os.PathLike[str],
    forecast_artifact_sha256: str,
    *,
    simulator=simulate_cascaded_tanks,
) -> ABC6DeferredScoreResult:
    """Reveal, hash, then score the fixed synthetic A/B/M prospective targets.

    A durable campaign execution is the only accepted training input. Its
    verified accessor supplies the exact bundle and results. The scorer then
    rechecks all 48 durable receipts, recomputes the complete target-free
    forecast roster from that evidence, and verifies the durable forecast
    artifact and integrated source pins before the fixed project-root O_EXCL
    marker. After it exists, target arrays and a terminal score-result receipt
    are published under anchored descriptors. A later error records a failure
    checkpoint when the pinned paths permit; the marker is never removed.
    """

    if not callable(simulator):
        raise TypeError("simulator must be callable")
    if not isinstance(execution, campaign_fit.ABC6TrainingCampaignExecution):
        raise TypeError(
            "execution must be an ABC6TrainingCampaignExecution; bare training data is rejected"
        )
    if not isinstance(frozen_forecasts, ABC6FrozenForecastRoster):
        raise TypeError("frozen_forecasts must come from freeze_abc6_forecasts")
    if (
        len(frozen_forecasts.forecasts) != CASE_COUNT
        or len(frozen_forecasts.case_sha256) != CASE_COUNT
    ):
        raise ABC6ScoringError("frozen forecast roster must contain exactly 24 cases")

    # The campaign manifest must freeze all three integrated sources before a
    # reveal attempt. The current campaign allowlist omits them, so production
    # scoring remains closed until its independent manifest review is updated.
    source_hashes = _verified_integrated_source_hashes(
        execution.receipt_root_identity
    )
    try:
        verified = execution.load_verified_training_evidence()
    except Exception as error:
        raise ABC6ScoringError(
            "campaign execution failed durable training-evidence verification"
        ) from error
    if not isinstance(verified, campaign_fit.ABC6VerifiedTrainingEvidence):
        raise ABC6ScoringError("execution returned no typed verified training evidence")
    training = verified.training_bundle
    results = verified.training_results
    if not isinstance(results, tuple) or len(results) != CASE_COUNT:
        raise ABC6ScoringError(
            "verified training results must contain exactly 24 cases"
        )
    data = _validate_training_bundle(training)
    receipt_root_identity, receipt_directory, expected_receipt_names = (
        _validate_campaign_execution_identity(execution)
    )

    # The case gate is opened only after every status receipt is durable and
    # verified.  No prospective simulator is called by this preflight.
    gate, receipt_statuses, receipt_hashes, summary_sha256 = _read_verified_statuses(
        receipt_root_identity
    )
    if gate is None:
        raise ABC6ScoringError("typed scoring call did not open its target gate")
    campaign_result = execution.campaign_result
    expected_receipt_hashes: list[tuple[str, str]] = []
    for status in campaign_result.case_statuses:
        expected_receipt_hashes.extend(
            (
                (
                    f"case-{status.case_index:02d}.fit-status.json",
                    status.fit_receipt_sha256,
                ),
                (
                    f"case-{status.case_index:02d}.baseline-status.json",
                    status.baseline_receipt_sha256,
                ),
            )
        )
    if tuple(name for name, _digest in receipt_hashes) != expected_receipt_names:
        raise ABC6ScoringError(
            "verified status receipt order differs from campaign execution"
        )
    if tuple(receipt_hashes) != tuple(expected_receipt_hashes):
        raise ABC6ScoringError(
            "verified receipt digests differ from campaign execution"
        )
    if summary_sha256 != campaign_result.summary_sha256:
        raise ABC6ScoringError(
            "verified training summary digest differs from execution"
        )
    for case_index, (result, forecast, (fit_status, baseline_status)) in enumerate(
        zip(results, frozen_forecasts.forecasts, receipt_statuses, strict=True)
    ):
        _validate_result_identity(result, case_index)
        if (fit_status, baseline_status) != (
            result.abc_status,
            result.baseline_status,
        ):
            raise ABC6ScoringError(
                "durable status receipts disagree with training results"
            )
        computed_hash = _forecast_case_hash(case_index, forecast)
        if computed_hash != frozen_forecasts.case_sha256[case_index]:
            raise ABC6ScoringError("target-free forecast hash mismatch")
        posterior = _validated_posterior(result, case_index)
        _validated_forecast(forecast, result, case_index, posterior)

    try:
        recomputed_forecasts = tuple(
            forecast_abc6_posterior_and_baseline(
                training.data_for_case(case_index),
                results[case_index],
                simulator=simulator,
            )
            for case_index in range(CASE_COUNT)
        )
        recomputed_roster = freeze_abc6_forecasts(recomputed_forecasts)
    except Exception as error:
        raise ABC6ScoringError(
            "verified training evidence could not reproduce target-free forecasts"
        ) from error
    if (
        recomputed_roster.case_sha256 != frozen_forecasts.case_sha256
        or recomputed_roster.roster_sha256 != frozen_forecasts.roster_sha256
    ):
        raise ABC6ScoringError(
            "submitted forecast roster differs from verified training evidence"
        )
    _verify_frozen_forecast_artifact(
        forecast_artifact_path,
        forecast_artifact_sha256,
        receipt_anchor=gate._receipt_anchor,
        campaign_result=campaign_result,
        evidence_manifest_sha256=execution.evidence_manifest_sha256,
        source_hashes=source_hashes,
        receipt_hashes=receipt_hashes,
        summary_sha256=summary_sha256,
        frozen=frozen_forecasts,
    )
    roster_hash = _sha256(
        _canonical_json(
            {
                "protocol_id": PROTOCOL_ID,
                "run_id": RUN_ID,
                "case_sha256": frozen_forecasts.case_sha256,
            }
        )
    )
    if roster_hash != frozen_forecasts.roster_sha256:
        raise ABC6ScoringError("frozen forecast roster hash mismatch")
    forecast_array_sha256 = _forecast_array_sha256(frozen_forecasts)
    evidence_links = _training_evidence_links(gate._receipt_anchor, execution)
    _reject_existing_reveal_marker(_REVEAL_MARKER_PATH, gate._root_anchor)
    _require_score_output_absent(gate._receipt_anchor)

    marker_payload = {
        "schema": "cascaded-tanks-abc6-synthetic-reveal-v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "condition": "synthetic-prospective-target-confirmation-v1",
        "semantics": "consumed-on-create; success-or-failure; no-retry",
        "forecast_roster_sha256": roster_hash,
        "forecast_artifact_sha256": forecast_artifact_sha256,
        "training_manifest_sha256": campaign_result.manifest_sha256,
        "training_evidence_manifest_sha256": execution.evidence_manifest_sha256,
        "integrated_source_hashes": source_hashes,
        "status_receipt_sha256": receipt_hashes,
        "training_summary_sha256": summary_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    postscore_root_anchor, postscore_receipt_anchor = (
        _duplicate_postscore_root_anchors(gate, receipt_root_identity)
    )
    try:
        reveal_claim = _claim_reveal_marker(
            _REVEAL_MARKER_PATH,
            marker_payload,
            root_anchor=gate._root_anchor,
        )
    except BaseException:
        postscore_receipt_anchor.close()
        postscore_root_anchor.close()
        raise
    marker_sha256 = reveal_claim.marker_sha256
    marker_identity = reveal_claim.receipt_identity()
    postscore_marker_anchor: cases._DirectoryAnchor | None = None
    target_vectors: dict[str, tuple[float, ...]] | None = None
    target_hashes: tuple[tuple[str, str], ...] | None = None
    target_arrays_sha256: str | None = None
    score_payload: dict[str, object] | None = None
    score_result_sha256: str | None = None
    score_receipt_sha256: str | None = None
    stage = "post_marker_anchor_retention"
    try:
        postscore_marker_anchor = reveal_claim.duplicate_parent_anchor()
        _verify_postscore_anchors(
            identity=receipt_root_identity,
            root_anchor=postscore_root_anchor,
            receipt_anchor=postscore_receipt_anchor,
            marker_anchor=postscore_marker_anchor,
            marker_identity=marker_identity,
            strict_paths=True,
        )
        # Only the scoring owner can issue the gate capability, and it does so
        # after the fixed marker is durably claimed and read back.
        stage = "target_materialization"
        scoring_handoff = cases._issue_scoring_target_handoff(
            gate,
            marker_sha256,
            marker_anchor=reveal_claim.take_parent_anchor(),
        )
        targets = gate.generate_targets(
            training,
            simulator=simulator,
            _scoring_handoff=scoring_handoff,
        )
        stage = "target_hash_validation"
        target_vectors, target_hashes = _target_vectors_and_hashes(targets)
        stage = "target_array_artifact_publication"
        target_array_bytes = _target_arrays_artifact(target_vectors, target_hashes)
        target_arrays_sha256 = _publish_postscore_artifact(
            filename=TARGET_ARRAYS_ARTIFACT_FILENAME,
            payload=target_array_bytes,
            execution=execution,
            root_anchor=postscore_root_anchor,
            receipt_anchor=postscore_receipt_anchor,
            marker_anchor=postscore_marker_anchor,
            marker_identity=marker_identity,
        )
        # Target hashes are fixed above before any metric is evaluated.
        stage = "case_scoring"
        scores = tuple(
            _score_case(
                case_index,
                frozen_forecasts.forecasts[case_index],
                results[case_index],
                None
                if CASE_ROSTER[case_index].truth_id == "N"
                else target_vectors[CASE_ROSTER[case_index].truth_id],
            )
            for case_index in range(CASE_COUNT)
        )
        stage = "paired_contrast_diagnostics"
        paired = _paired_contrasts(scores)
        stage = "parameter_inclusion_diagnostics"
        inclusion_counts = _parameter_inclusion_counts(scores)
        stage = "N_ceiling_diagnostics"
        n_diagnostics = _no_crossing_diagnostics(results)
        stage = "M_training_residual_diagnostics"
        m_residuals = _m_training_residuals(
            results,
            frozen_forecasts.forecasts,
            data,
            simulator=simulator,
        )
        score_result = ABC6DeferredScoreResult(
            protocol_id=PROTOCOL_ID,
            run_id=RUN_ID,
            reveal_marker_sha256=marker_sha256,
            target_sha256_by_truth=target_hashes,
            case_scores=scores,
            paired_horizon_contrasts=paired,
            parameter_inclusion_counts=inclusion_counts,
            no_crossing_diagnostics=n_diagnostics,
            m_training_residual_means=m_residuals,
        )
        score_payload = _score_result_payload(score_result)
        score_result_sha256 = _sha256(_canonical_json(score_payload))
        stage = "score_receipt_publication"
        score_event = _score_event_snapshot("score_complete")
        score_receipt_bytes = _score_receipt_payload(
            outcome="complete",
            execution=execution,
            receipt_hashes=receipt_hashes,
            summary_sha256=summary_sha256,
            forecast_artifact_sha256=forecast_artifact_sha256,
            frozen=frozen_forecasts,
            forecast_array_sha256=forecast_array_sha256,
            marker_identity=marker_identity,
            evidence_links=evidence_links,
            target_hashes=target_hashes,
            target_arrays_sha256=target_arrays_sha256,
            score_result=score_payload,
            score_result_sha256=score_result_sha256,
            score_event=score_event,
            failure_checkpoint=None,
        )
        score_receipt_sha256 = _publish_postscore_artifact(
            filename=SCORE_RECEIPT_FILENAME,
            payload=score_receipt_bytes,
            execution=execution,
            root_anchor=postscore_root_anchor,
            receipt_anchor=postscore_receipt_anchor,
            marker_anchor=postscore_marker_anchor,
            marker_identity=marker_identity,
        )
        return dataclasses.replace(
            score_result,
            target_arrays_artifact_path=(
                postscore_receipt_anchor.path / TARGET_ARRAYS_ARTIFACT_FILENAME
            ),
            target_arrays_artifact_sha256=target_arrays_sha256,
            forecast_array_sha256=forecast_array_sha256,
            score_result_sha256=score_result_sha256,
            score_receipt_path=postscore_receipt_anchor.path / SCORE_RECEIPT_FILENAME,
            score_receipt_sha256=score_receipt_sha256,
            score_event_monotonic_ns=int(score_event["monotonic_ns"]),
            score_event_utc=str(score_event["occurred_at_utc"]),
        )
    except Exception as error:
        failure_receipt_error: Exception | None = None
        failure_receipt_sha256: str | None = None
        # If duplicating the marker-parent descriptor failed immediately after
        # the marker claim, the claim still owns its original pinned descriptor.
        # Transfer that descriptor to the failure writer so the consumed run can
        # leave a durable checkpoint without reopening a path.
        if (
            postscore_marker_anchor is None
            and reveal_claim.marker_parent_anchor is not None
        ):
            try:
                postscore_marker_anchor = reveal_claim.take_parent_anchor()
            except Exception as anchor_error:
                failure_receipt_error = anchor_error
        failure_event: dict[str, object] | None = None
        try:
            failure_event = _score_event_snapshot(stage)
        except Exception as clock_error:
            failure_receipt_error = failure_receipt_error or clock_error
        if (
            postscore_root_anchor is not None
            and postscore_receipt_anchor is not None
            and postscore_marker_anchor is not None
            and failure_event is not None
        ):
            checkpoint = {
                "stage": stage,
                "occurred_at_monotonic_ns": failure_event["monotonic_ns"],
                "occurred_at_utc": failure_event["occurred_at_utc"],
                "exception_type": type(error).__name__,
                "message": str(error)[:512],
                "retry_forbidden": True,
                "condition_consumed": True,
            }
            try:
                failure_receipt = _score_receipt_payload(
                    outcome="failed",
                    execution=execution,
                    receipt_hashes=receipt_hashes,
                    summary_sha256=summary_sha256,
                    forecast_artifact_sha256=forecast_artifact_sha256,
                    frozen=frozen_forecasts,
                    forecast_array_sha256=forecast_array_sha256,
                    marker_identity=marker_identity,
                    evidence_links=evidence_links,
                    target_hashes=target_hashes,
                    target_arrays_sha256=target_arrays_sha256,
                    score_result=score_payload,
                    score_result_sha256=score_result_sha256,
                    score_event=failure_event,
                    failure_checkpoint=checkpoint,
                )
                failure_receipt_sha256 = _publish_postscore_artifact(
                    filename=SCORE_RECEIPT_FILENAME,
                    payload=failure_receipt,
                    execution=execution,
                    root_anchor=postscore_root_anchor,
                    receipt_anchor=postscore_receipt_anchor,
                    marker_anchor=postscore_marker_anchor,
                    marker_identity=marker_identity,
                    strict_paths=False,
                )
            except Exception as checkpoint_error:
                failure_receipt_error = checkpoint_error
        raise ABC6DeferredScoreConsumedError(
            "deferred synthetic scoring failed after target reveal "
            f"at {stage}: {type(error).__name__}"
            + (
                "; durable failure checkpoint could not be published"
                if failure_receipt_error is not None
                else "; durable failure checkpoint recorded when possible"
            ),
            marker_sha256=marker_sha256,
            score_receipt_path=(
                None
                if failure_receipt_sha256 is None
                or postscore_receipt_anchor is None
                else postscore_receipt_anchor.path / SCORE_RECEIPT_FILENAME
            ),
            score_receipt_sha256=failure_receipt_sha256,
            failure_stage=stage,
        ) from error
    finally:
        if postscore_marker_anchor is not None:
            postscore_marker_anchor.close()
        if postscore_receipt_anchor is not None:
            postscore_receipt_anchor.close()
        if postscore_root_anchor is not None:
            postscore_root_anchor.close()
        reveal_claim.close()


__all__ = (
    "ABC6CaseScore",
    "ABC6DeferredScoreConsumedError",
    "ABC6DeferredScoreResult",
    "ABC6FrozenForecastRoster",
    "ABC6MTrainingResidualMeans",
    "ABC6NoCrossingDiagnostic",
    "ABC6PairedHorizonContrasts",
    "ABC6ParameterInclusionCount",
    "ABC6ParameterInterval",
    "ABC6RevealAlreadyConsumedError",
    "ABC6ScoringError",
    "freeze_abc6_forecasts",
    "score_deferred_abc6_synthetic",
)

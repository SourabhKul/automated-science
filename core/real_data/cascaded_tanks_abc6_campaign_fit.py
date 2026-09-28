"""One-use, training-only orchestration for the frozen ABC6 case roster.

This module connects the reviewed 24-case roster to the reviewed one-case fit
adapter.  It creates only synthetic training data and writes immutable
fit/baseline status receipts.  It never opens the prospective-target gate,
materializes future outputs, forecasts, or scores targets.

The manifest hash/source/runtime checks below are preflight checks, not an
independent approval system.  The 900-second/2-GiB sampled watchdog and final
manifest approval remain external launch gates; this module does not enforce
either.  The caller must not invoke the production entry point until those
separate gates have passed.  A fixed project-root O_EXCL registry claim, not
the caller-selected receipt directory, consumes the run ID permanently:
partial runs cannot be resumed or retried under it.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import platform
import stat
import struct
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

import numpy as np

from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_training as training
from core.real_data.cascaded_tanks_abc6_cases import (
    CASE_COUNT,
    CASE_ROSTER,
    PROTOCOL_ID,
    RUN_ID,
    ABC6Case,
    ABC6TrainingBundle,
    ABC6TrainingCaseData,
    build_synthetic_training_bundle,
    case_by_index,
)

MANIFEST_SCHEMA_VERSION: Final = 1
CLAIM_FILENAME: Final = "campaign.claim"
SUMMARY_FILENAME: Final = "campaign.training-summary.json"
EVIDENCE_DIRECTORY_NAME: Final = "training-evidence"
EVIDENCE_MANIFEST_FILENAME: Final = "campaign.training-evidence-manifest.json"
FAILURE_FILENAME: Final = "campaign.failure.json"
MAX_MANIFEST_BYTES: Final = 2 * 1024 * 1024
GLOBAL_WATCHDOG_SECONDS: Final = 900
RUNNER_TREE_RSS_LIMIT_BYTES: Final = 2 * 1024**3
PROJECTED_ARTIFACT_LIMIT_BYTES: Final = 1024**3
MAX_WORKERS: Final = 1

REVIEWED_PROPOSAL_PATH: Final = (
    "reports/cascaded-tanks-six-parameter-synthetic-abc-proposal-2026-09-28.md"
)
REVIEWED_PROPOSAL_SHA256: Final = (
    "8f406a1e3ceb12cc7d835b9ed686e5dd2ce181725c97f9c40047ba84a1e92b0f"
)

# These are checks against the already-reviewed stable local runtime named by
# the proposal.  The final manifest must independently pin and hash them too.
PINNED_INTERPRETER_PATH: Final = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/"
    "Versions/3.14/bin/python3.14"
)
PINNED_INTERPRETER_SHA256: Final = (
    "cbf84109626aa1013bbe408fbb9590bd0f1c1548f038b2221c6b8b87de26ca43"
)
PINNED_NUMPY_PATH: Final = (
    "/opt/homebrew/lib/python3.14/site-packages/numpy/__init__.py"
)
PINNED_NUMPY_SHA256: Final = (
    "2e8da3e4385e79c4885b3f7324a8b957e6f01732b239e99e266a12c62a008b8d"
)
PINNED_PSUTIL_PATH: Final = (
    "/Users/sourabh/Library/Python/3.14/lib/python/site-packages/psutil/__init__.py"
)
PINNED_PSUTIL_SHA256: Final = (
    "d138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a"
)

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_PROJECT_RUN_CLAIM_PATH: Final = (
    _REPO_ROOT
    / "artifacts"
    / "evaluations"
    / "cascaded_tanks_abc6_campaign_fit"
    / "claims"
    / f"{RUN_ID}.claim"
)
_REQUIRED_SOURCE_PATHS: Final = (
    "core/real_data/cascaded_tanks_abc6_campaign_fit.py",
    "core/real_data/cascaded_tanks_abc6_cases.py",
    "core/real_data/cascaded_tanks_abc6_training.py",
    "core/real_data/cascaded_tanks_pattern_search.py",
    "core/real_data/cascaded_tanks_synthetic_abc.py",
    "core/real_data/cascaded_tanks_models.py",
    REVIEWED_PROPOSAL_PATH,
)
_REVIEWED_SOURCE_SHA256: Final = {
    "core/real_data/cascaded_tanks_abc6_cases.py": (
        "3e20935ab4324dd4ffc8b352281ce02fae05f8f57516f5ac8c9cd8c606cd119b"
    ),
    "core/real_data/cascaded_tanks_abc6_training.py": (
        "a12e0f1e2c4fb40e632b0785fdd47f8467ecffbdc36e1396e62d147b2981c59e"
    ),
    "core/real_data/cascaded_tanks_pattern_search.py": (
        "c7800c76c73b5d6063a6e15ea45cd756e7a542b7ec584e99a18108e3ea914458"
    ),
    REVIEWED_PROPOSAL_PATH: REVIEWED_PROPOSAL_SHA256,
}
_FINAL_STATUSES: Final = frozenset({"complete", "incomplete", "unresolved", "failed"})
_COMPONENTS: Final = ("fit", "baseline")


class ABC6CampaignError(RuntimeError):
    """Base error for fail-closed campaign preflight or orchestration."""


class ABC6CampaignPreflightError(ABC6CampaignError):
    """The immutable run manifest or output inventory failed preflight."""


class ABC6CampaignAlreadyClaimedError(ABC6CampaignError):
    """The one-use run ID has already been claimed and cannot be resumed."""


class ABC6CampaignExecutionError(ABC6CampaignError):
    """Training stopped after the run claim; the run ID remains consumed."""


@dataclass(frozen=True, slots=True)
class ABC6CaseStatus:
    case_index: int
    case_id: str
    fit_status: str
    baseline_status: str
    fit_receipt_sha256: str
    baseline_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ABC6CampaignResult:
    """Compact terminal status only; training evidence stays caller-owned."""

    protocol_id: str
    run_id: str
    manifest_sha256: str
    claim_sha256: str
    status: Literal["complete", "incomplete"]
    case_statuses: tuple[ABC6CaseStatus, ...]
    summary_path: Path
    summary_sha256: str
    prospective_targets_generated: Literal[False] = False
    forecasts_run: Literal[False] = False


@dataclass(frozen=True, slots=True)
class ABC6VerifiedTrainingEvidence:
    """Fresh immutable copies validated against the durable evidence chain."""

    training_bundle: ABC6TrainingBundle = field(repr=False, compare=False)
    training_results: tuple[training.ABC6TrainingResult, ...] = field(
        repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class ABC6TrainingCampaignExecution:
    """Private in-memory evidence plus an accessor that verifies it before use.

    The mutable numerical graph is deliberately private. Call
    :meth:`load_verified_training_evidence` immediately before forecasting; it
    rehashes the durable manifest, summary, status receipts, bundle, and all
    case evidence, then compares the private in-memory graph to those digests.
    """

    campaign_result: ABC6CampaignResult
    evidence_manifest_path: Path
    evidence_manifest_sha256: str
    _training_bundle: ABC6TrainingBundle = field(repr=False, compare=False)
    _training_results: tuple[training.ABC6TrainingResult, ...] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_execution_shape(
            self.campaign_result, self._training_bundle, self._training_results
        )
        if not isinstance(self.evidence_manifest_path, Path):
            raise TypeError("evidence_manifest_path must be a Path")
        if not _is_sha256(self.evidence_manifest_sha256):
            raise ValueError("evidence_manifest_sha256 must be a lowercase SHA-256")

    def load_verified_training_evidence(self) -> ABC6VerifiedTrainingEvidence:
        """Revalidate durable and in-memory evidence, then return frozen copies."""

        _verify_training_execution_evidence(self)
        bundle_copy = _freeze_evidence_value(copy.deepcopy(self._training_bundle))
        result_copies = tuple(
            _freeze_evidence_value(copy.deepcopy(result))
            for result in self._training_results
        )
        return ABC6VerifiedTrainingEvidence(
            training_bundle=cast(ABC6TrainingBundle, bundle_copy),
            training_results=cast(
                tuple[training.ABC6TrainingResult, ...], result_copies
            ),
        )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _float_classification(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "positive_infinity" if value > 0.0 else "negative_infinity"
    return "finite"


def _numpy_float_classification(value: np.generic) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "negative_infinity" if np.signbit(value) else "positive_infinity"
    return "finite"


def _canonical_evidence_value(value: object) -> object:
    """Encode the reviewed result graph without pickle or JSON non-finite tokens.

    Numeric arrays preserve dtype, shape, and exact C-order bytes.  Every
    non-finite floating array element is separately labelled, and scalar
    floating values carry their exact IEEE-754 bit pattern and classification.
    Unsupported values fail closed rather than being stringified.
    """

    if value is None or type(value) in (bool, str):
        return value
    if type(value) is int:
        return {"$integer": str(value)}
    if type(value) is float:
        return {
            "$float64_bits": struct.pack(">d", value).hex(),
            "classification": _float_classification(value),
        }
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if (
            array.dtype.hasobject
            or array.dtype.kind not in "biuf"
            or array.dtype.itemsize > 8
        ):
            raise TypeError(
                f"unsupported evidence ndarray dtype: {array.dtype!s}"
            )
        contiguous = np.ascontiguousarray(array)
        non_finite: list[dict[str, object]] = []
        if contiguous.dtype.kind == "f":
            flat = contiguous.reshape(-1)
            for index in np.flatnonzero(~np.isfinite(flat)):
                number = float(flat[int(index)])
                non_finite.append(
                    {
                        "flat_index": int(index),
                        "classification": _numpy_float_classification(
                            flat[int(index)]
                        ),
                        "bits": np.asarray(flat[int(index)]).tobytes().hex(),
                    }
                )
        return {
            "$ndarray": {
                "dtype": contiguous.dtype.str,
                "shape": [int(length) for length in contiguous.shape],
                "data_base64": base64.b64encode(
                    contiguous.tobytes(order="C")
                ).decode("ascii"),
                "non_finite": non_finite,
            }
        }
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        if (
            scalar.dtype.hasobject
            or scalar.dtype.kind not in "biuf"
            or scalar.dtype.itemsize > 8
        ):
            raise TypeError(f"unsupported evidence NumPy scalar: {scalar.dtype!s}")
        encoded: dict[str, object] = {
            "$numpy_scalar": {
                "dtype": scalar.dtype.str,
                "bits": scalar.tobytes().hex(),
            }
        }
        if scalar.dtype.kind == "f":
            encoded["classification"] = _numpy_float_classification(scalar[()])
        return encoded
    if isinstance(value, Enum):
        return {
            "$enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_evidence_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _canonical_evidence_value(getattr(value, field.name))]
                for field in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("evidence mappings must use string keys")
        items: list[list[object]] = []
        for key in sorted(value):
            items.append([key, _canonical_evidence_value(value[key])])
        return {"$mapping": items}
    if isinstance(value, tuple):
        return {"$tuple": [_canonical_evidence_value(item) for item in value]}
    if isinstance(value, list):
        return {"$list": [_canonical_evidence_value(item) for item in value]}
    raise TypeError(f"unsupported value in durable ABC6 evidence: {type(value)!r}")


def _freeze_evidence_value(value: object) -> object:
    """Copy an evidence graph and remove mutable mappings/array aliases."""

    if isinstance(value, np.ndarray):
        frozen = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype)
        frozen = frozen.reshape(value.shape)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_evidence_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_evidence_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_evidence_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        frozen = copy.copy(value)
        for field in fields(value):
            object.__setattr__(
                frozen,
                field.name,
                _freeze_evidence_value(getattr(value, field.name)),
            )
        return frozen
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in durable evidence JSON: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _read_canonical_evidence_json(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"durable evidence path is not a regular file: {path}")
        raw = path.read_bytes()
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read durable evidence JSON: {path}") from error
    if not isinstance(decoded, dict) or _canonical_json(decoded) != raw:
        raise ValueError(f"durable evidence JSON is not canonical: {path}")
    return raw, decoded


def _canonical_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _roster_identities() -> list[dict[str, object]]:
    return [_case_identity(case) for case in CASE_ROSTER]


def _roster_sha256() -> str:
    return _canonical_sha256(_canonical_json({"ordered_cases": _roster_identities()}))


def _validate_execution_shape(
    campaign_result: ABC6CampaignResult,
    bundle: ABC6TrainingBundle,
    results: tuple[training.ABC6TrainingResult, ...],
) -> None:
    if not isinstance(campaign_result, ABC6CampaignResult):
        raise TypeError("campaign_result must be an ABC6CampaignResult")
    if not isinstance(bundle, ABC6TrainingBundle):
        raise TypeError("training bundle must be an ABC6TrainingBundle")
    if not isinstance(results, tuple) or len(results) != CASE_COUNT:
        raise ValueError("training results must contain exactly 24 ordered cases")
    if len(campaign_result.case_statuses) != CASE_COUNT:
        raise ValueError("campaign result must contain exactly 24 case statuses")
    for index, (case, status, result) in enumerate(
        zip(CASE_ROSTER, campaign_result.case_statuses, results, strict=True)
    ):
        if (
            status.case_index != index
            or status.case_index != case.case_index
            or status.case_id != case.case_id
        ):
            raise ValueError("campaign statuses are not in frozen roster order")
        if bundle.data_for_case(index).case != case:
            raise ValueError("training bundle does not match the frozen roster")
        result_statuses = _result_statuses(result, case)
        if result_statuses != (status.fit_status, status.baseline_status):
            raise ValueError(
                "training result statuses do not match durable campaign receipts"
            )


def _bundle_evidence_payload(bundle: ABC6TrainingBundle) -> bytes:
    return _canonical_json(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "kind": "abc6_training_bundle",
            "ordered_case_identities": _roster_identities(),
            "bundle": _canonical_evidence_value(bundle),
        }
    )


def _case_evidence_payload(
    case: ABC6Case,
    result: training.ABC6TrainingResult,
) -> bytes:
    fit_status, baseline_status = _result_statuses(result, case)
    return _canonical_json(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "kind": "abc6_training_case_result",
            "roster_index": case.case_index,
            "case_identity": _case_identity(case),
            "fit_status": fit_status,
            "baseline_status": baseline_status,
            "result": _canonical_evidence_value(result),
        }
    )


def _verify_summary_link(campaign_result: ABC6CampaignResult) -> None:
    summary_path = campaign_result.summary_path
    raw, decoded = _read_canonical_evidence_json(summary_path)
    if _canonical_sha256(raw) != campaign_result.summary_sha256:
        raise ValueError("training summary digest does not match campaign result")
    summary_body = {
        key: value for key, value in decoded.items() if key != "payload_sha256"
    }
    if decoded.get("payload_sha256") != _canonical_sha256(
        _canonical_json(summary_body)
    ):
        raise ValueError("training summary payload digest is invalid")
    if (
        decoded.get("protocol_id") != campaign_result.protocol_id
        or decoded.get("run_id") != campaign_result.run_id
        or decoded.get("manifest_sha256") != campaign_result.manifest_sha256
        or decoded.get("claim_sha256") != campaign_result.claim_sha256
        or decoded.get("status") != campaign_result.status
        or decoded.get("case_count") != CASE_COUNT
        or decoded.get("training_only") is not True
        or decoded.get("prospective_targets_generated") is not False
        or decoded.get("forecasts_run") is not False
        or decoded.get("postfit_target_gate_opened") is not False
    ):
        raise ValueError("training summary identity differs from campaign result")
    expected_statuses = [
        {
            "case_index": status.case_index,
            "case_id": status.case_id,
            "fit_status": status.fit_status,
            "baseline_status": status.baseline_status,
            "fit_receipt_sha256": status.fit_receipt_sha256,
            "baseline_receipt_sha256": status.baseline_receipt_sha256,
        }
        for status in campaign_result.case_statuses
    ]
    if decoded.get("case_statuses") != expected_statuses:
        raise ValueError("training summary case statuses differ from campaign result")


def _verify_training_execution_evidence(
    execution: ABC6TrainingCampaignExecution,
) -> None:
    result = execution.campaign_result
    _validate_execution_shape(
        result, execution._training_bundle, execution._training_results
    )
    manifest_path = execution.evidence_manifest_path
    expected_manifest_path = result.summary_path.parent / EVIDENCE_MANIFEST_FILENAME
    if manifest_path != expected_manifest_path:
        raise ValueError("evidence manifest is outside the campaign receipt directory")
    manifest_raw, manifest = _read_canonical_evidence_json(manifest_path)
    expected_manifest_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "claim_sha256",
        "training_summary_filename",
        "training_summary_sha256",
        "roster_sha256",
        "ordered_case_identities",
        "status_receipts",
        "bundle_artifact",
        "case_artifacts",
        "payload_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("evidence manifest has an unexpected schema")
    if _canonical_sha256(manifest_raw) != execution.evidence_manifest_sha256:
        raise ValueError("evidence manifest digest does not match execution")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    if manifest.get("payload_sha256") != _canonical_sha256(
        _canonical_json(manifest_body)
    ):
        raise ValueError("evidence manifest payload digest is invalid")
    expected_manifest_fields = {
        "schema_version": 1,
        "protocol_id": result.protocol_id,
        "run_id": result.run_id,
        "manifest_sha256": result.manifest_sha256,
        "claim_sha256": result.claim_sha256,
        "training_summary_filename": SUMMARY_FILENAME,
        "training_summary_sha256": result.summary_sha256,
        "roster_sha256": _roster_sha256(),
        "ordered_case_identities": _roster_identities(),
    }
    if any(
        manifest.get(key) != value
        for key, value in expected_manifest_fields.items()
    ):
        raise ValueError("evidence manifest identity differs from campaign execution")
    if manifest_path.parent.is_symlink() or not manifest_path.parent.is_dir():
        raise ValueError("campaign receipt directory must be a real directory")
    if result.summary_path.parent != manifest_path.parent:
        raise ValueError("summary and evidence manifest directories differ")

    _verify_summary_link(result)
    local_claim_path = manifest_path.parent / CLAIM_FILENAME
    claim_raw, claim = _read_canonical_evidence_json(local_claim_path)
    if _canonical_sha256(claim_raw) != result.claim_sha256:
        raise ValueError("local claim receipt digest does not match campaign result")
    if (
        claim.get("protocol_id") != result.protocol_id
        or claim.get("run_id") != result.run_id
        or claim.get("manifest_sha256") != result.manifest_sha256
        or claim.get("claim_semantics") != "consumed_once_no_resume"
    ):
        raise ValueError("local claim receipt identity differs from campaign result")

    verified_receipts = cases._verify_all_status_receipts(manifest_path.parent)
    receipt_entries = manifest.get("status_receipts")
    expected_receipts: list[dict[str, object]] = []
    for case, status in zip(CASE_ROSTER, result.case_statuses, strict=True):
        for component in _COMPONENTS:
            expected_receipts.append(
                {
                    "filename": f"case-{case.case_index:02d}.{component}-status.json",
                    "sha256": getattr(status, f"{component}_receipt_sha256"),
                    "case_index": case.case_index,
                    "case_id": case.case_id,
                    "component": component,
                    "status": getattr(status, f"{component}_status"),
                }
            )
    if receipt_entries != expected_receipts:
        raise ValueError("evidence manifest does not bind the exact 48 status receipts")
    if verified_receipts != tuple(
        (entry["filename"], entry["sha256"]) for entry in expected_receipts
    ):
        raise ValueError("durable status receipt chain differs from evidence manifest")

    evidence_directory = manifest_path.parent / EVIDENCE_DIRECTORY_NAME
    if evidence_directory.is_symlink() or not evidence_directory.is_dir():
        raise ValueError("durable training evidence directory is missing or unsafe")
    bundle_entry = manifest.get("bundle_artifact")
    if not isinstance(bundle_entry, dict) or set(bundle_entry) != {"filename", "sha256"}:
        raise ValueError("evidence manifest bundle entry is invalid")
    if (
        bundle_entry.get("filename") != "training-bundle.evidence.json"
        or not _is_sha256(bundle_entry.get("sha256"))
    ):
        raise ValueError("evidence manifest bundle filename/digest is invalid")
    bundle_path = evidence_directory / str(bundle_entry["filename"])
    bundle_raw, bundle_decoded = _read_canonical_evidence_json(bundle_path)
    bundle_digest = _canonical_sha256(bundle_raw)
    if bundle_digest != bundle_entry["sha256"]:
        raise ValueError("durable training bundle artifact digest is invalid")
    if (
        bundle_decoded.get("schema_version") != 1
        or bundle_decoded.get("kind") != "abc6_training_bundle"
        or bundle_decoded.get("protocol_id") != result.protocol_id
        or bundle_decoded.get("run_id") != result.run_id
        or bundle_decoded.get("ordered_case_identities") != _roster_identities()
    ):
        raise ValueError("durable training bundle identity is invalid")
    if _canonical_sha256(_bundle_evidence_payload(execution._training_bundle)) != bundle_digest:
        raise ValueError("in-memory training bundle differs from durable evidence")

    case_entries = manifest.get("case_artifacts")
    if not isinstance(case_entries, list) or len(case_entries) != CASE_COUNT:
        raise ValueError("evidence manifest must contain 24 ordered case artifacts")
    for index, (case, status, training_result, entry) in enumerate(
        zip(
            CASE_ROSTER,
            result.case_statuses,
            execution._training_results,
            case_entries,
            strict=True,
        )
    ):
        if not isinstance(entry, dict) or set(entry) != {
            "filename", "sha256", "roster_index", "case_identity",
            "fit_status", "baseline_status",
        }:
            raise ValueError("case evidence manifest entry is invalid")
        expected_identity = _case_identity(case)
        fit_status, baseline_status = _result_statuses(training_result, case)
        if (
            entry.get("roster_index") != index
            or entry.get("case_identity") != expected_identity
            or entry.get("fit_status") != status.fit_status
            or entry.get("baseline_status") != status.baseline_status
            or (fit_status, baseline_status)
            != (status.fit_status, status.baseline_status)
        ):
            raise ValueError("case evidence identity/status differs from roster receipt")
        filename = f"case-{index:02d}.training-evidence.json"
        if entry.get("filename") != filename or not _is_sha256(entry.get("sha256")):
            raise ValueError("case evidence filename or digest is invalid")
        case_path = evidence_directory / filename
        case_raw, case_decoded = _read_canonical_evidence_json(case_path)
        case_digest = _canonical_sha256(case_raw)
        if case_digest != entry["sha256"]:
            raise ValueError("durable case evidence artifact digest is invalid")
        if (
            case_decoded.get("schema_version") != 1
            or case_decoded.get("kind") != "abc6_training_case_result"
            or case_decoded.get("protocol_id") != result.protocol_id
            or case_decoded.get("run_id") != result.run_id
            or case_decoded.get("roster_index") != index
            or case_decoded.get("case_identity") != expected_identity
            or case_decoded.get("fit_status") != fit_status
            or case_decoded.get("baseline_status") != baseline_status
        ):
            raise ValueError("durable case evidence identity/status is invalid")
        if _canonical_sha256(_case_evidence_payload(case, training_result)) != case_digest:
            raise ValueError("in-memory case result differs from durable evidence")


def _sha256_file(path: Path) -> str:
    try:
        file_stat = path.stat()
    except OSError as error:
        raise ABC6CampaignPreflightError(f"cannot stat pinned file: {path}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ABC6CampaignPreflightError(f"pinned path is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ABC6CampaignPreflightError(f"cannot read pinned file: {path}") from error
    return digest.hexdigest()


def _case_identity(case: ABC6Case) -> dict[str, object]:
    return {
        "case_index": case.case_index,
        "case_id": case.case_id,
        "truth_id": case.truth_id,
        "input_window": case.input_window,
        "replicate": case.replicate,
        "fit_model": case.fit_model.value,
        "calibration_seed": case.calibration_seed,
        "abc_seed": case.abc_seed,
        "input_length": case.input_length,
        "prior_bounds": [list(bounds) for bounds in case.prior_bounds],
    }


def _manifest_identity() -> dict[str, object]:
    """Stable identity fields; source/runtime digests remain externally pinned."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "reviewed_proposal": {
            "path": REVIEWED_PROPOSAL_PATH,
            "sha256": REVIEWED_PROPOSAL_SHA256,
        },
        "training_control_manifest_sha256": training.TRAINING_CONTROL_MANIFEST_SHA256,
        "ordered_cases": [_case_identity(case) for case in CASE_ROSTER],
    }


def _current_source_hashes() -> dict[str, str]:
    return {
        relative_path: _sha256_file(_REPO_ROOT / relative_path)
        for relative_path in _REQUIRED_SOURCE_PATHS
    }


def _current_runtime_fingerprint() -> dict[str, object]:
    """Return actual in-process runtime evidence for manifest comparison."""

    import numpy
    import psutil

    try:
        resolved_interpreter = Path(sys.executable).resolve(strict=True)
        numpy_path = Path(numpy.__file__).absolute()
        psutil_path = Path(psutil.__file__).absolute()
    except OSError as error:
        raise ABC6CampaignPreflightError("runtime path resolution failed") from error
    return {
        "declared_launch_interpreter_path": PINNED_INTERPRETER_PATH,
        "resolved_running_interpreter_path": str(resolved_interpreter),
        "python_version": platform.python_version(),
        "interpreter_sha256": _sha256_file(Path(PINNED_INTERPRETER_PATH)),
        "numpy_version": numpy.__version__,
        "numpy_module_path": str(numpy_path),
        "numpy_module_sha256": _sha256_file(numpy_path),
        "psutil_version": psutil.__version__,
        "psutil_module_path": str(psutil_path),
        "psutil_module_sha256": _sha256_file(psutil_path),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _require_reviewed_runtime(fingerprint: Mapping[str, object]) -> None:
    expected = {
        "declared_launch_interpreter_path": PINNED_INTERPRETER_PATH,
        "resolved_running_interpreter_path": str(
            Path(PINNED_INTERPRETER_PATH).resolve(strict=True)
        ),
        "python_version": "3.14.3",
        "interpreter_sha256": PINNED_INTERPRETER_SHA256,
        "numpy_version": "2.4.2",
        "numpy_module_path": PINNED_NUMPY_PATH,
        "numpy_module_sha256": PINNED_NUMPY_SHA256,
        "psutil_version": "7.2.2",
        "psutil_module_path": PINNED_PSUTIL_PATH,
        "psutil_module_sha256": PINNED_PSUTIL_SHA256,
    }
    if any(fingerprint.get(key) != value for key, value in expected.items()):
        raise ABC6CampaignPreflightError(
            "process runtime does not match the reviewed Python/NumPy/psutil pins"
        )


def _execution_contract_is_valid(value: object) -> bool:
    expected_keys = {
        "wall_clock_limit_seconds",
        "runner_tree_rss_limit_bytes",
        "projected_artifact_bytes",
        "worker_count",
        "watchdog_enforcement",
        "independent_manifest_approval_required",
        "prospective_targets_before_receipts",
    }
    if type(value) is not dict or set(value) != expected_keys:
        return False
    projected = value.get("projected_artifact_bytes")
    return (
        type(value.get("wall_clock_limit_seconds")) is int
        and value["wall_clock_limit_seconds"] == GLOBAL_WATCHDOG_SECONDS
        and type(value.get("runner_tree_rss_limit_bytes")) is int
        and value["runner_tree_rss_limit_bytes"] == RUNNER_TREE_RSS_LIMIT_BYTES
        and type(projected) is int
        and 0 <= projected < PROJECTED_ARTIFACT_LIMIT_BYTES
        and type(value.get("worker_count")) is int
        and value["worker_count"] == MAX_WORKERS
        and value.get("watchdog_enforcement") == "external"
        and value.get("independent_manifest_approval_required") is True
        and value.get("prospective_targets_before_receipts") is False
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_case_entry(entry: object) -> bool:
    expected_keys = {
        "case_index",
        "case_id",
        "truth_id",
        "input_window",
        "replicate",
        "fit_model",
        "calibration_seed",
        "abc_seed",
        "input_length",
        "prior_bounds",
    }
    if type(entry) is not dict or set(entry) != expected_keys:
        return False
    for key in ("case_index", "calibration_seed", "abc_seed", "input_length"):
        if type(entry[key]) is not int:
            return False
    if entry["replicate"] is not None and type(entry["replicate"]) is not int:
        return False
    for key in ("case_id", "truth_id", "input_window", "fit_model"):
        if type(entry[key]) is not str:
            return False
    bounds = entry["prior_bounds"]
    if type(bounds) is not list or len(bounds) != 6:
        return False
    for pair in bounds:
        if type(pair) is not list or len(pair) != 2:
            return False
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in pair
        ):
            return False
    return True


def _strict_manifest_schema(decoded: object) -> bool:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "reviewed_proposal",
        "training_control_manifest_sha256",
        "ordered_cases",
        "source_hashes",
        "runtime_fingerprint",
        "execution_contract",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys:
        return False
    if type(decoded["schema_version"]) is not int:
        return False
    if type(decoded["protocol_id"]) is not str or type(decoded["run_id"]) is not str:
        return False
    if not _is_sha256(decoded["training_control_manifest_sha256"]):
        return False

    proposal = decoded["reviewed_proposal"]
    if (
        type(proposal) is not dict
        or set(proposal) != {"path", "sha256"}
        or type(proposal["path"]) is not str
        or not _is_sha256(proposal["sha256"])
    ):
        return False

    roster = decoded["ordered_cases"]
    if (
        type(roster) is not list
        or len(roster) != CASE_COUNT
        or any(not _strict_case_entry(entry) for entry in roster)
    ):
        return False

    source_hashes = decoded["source_hashes"]
    if (
        type(source_hashes) is not dict
        or set(source_hashes) != set(_REQUIRED_SOURCE_PATHS)
        or any(
            type(path) is not str or not _is_sha256(digest)
            for path, digest in source_hashes.items()
        )
    ):
        return False

    runtime = decoded["runtime_fingerprint"]
    runtime_keys = {
        "declared_launch_interpreter_path",
        "resolved_running_interpreter_path",
        "python_version",
        "interpreter_sha256",
        "numpy_version",
        "numpy_module_path",
        "numpy_module_sha256",
        "psutil_version",
        "psutil_module_path",
        "psutil_module_sha256",
        "platform",
        "machine",
    }
    if (
        type(runtime) is not dict
        or set(runtime) != runtime_keys
        or any(type(value) is not str for value in runtime.values())
        or any(
            not _is_sha256(runtime[key])
            for key in (
                "interpreter_sha256",
                "numpy_module_sha256",
                "psutil_module_sha256",
            )
        )
    ):
        return False
    return _execution_contract_is_valid(decoded["execution_contract"])


def preflight_abc6_campaign_manifest(
    manifest_path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    require_reviewed_runtime: bool = True,
) -> str:
    """Validate an external immutable manifest and return its exact file hash.

    The expected hash is supplied by the caller after the manifest has been
    frozen.  This function deliberately does not approve the manifest, attest
    an independent reviewer, or verify/enforce an external watchdog receipt.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ABC6CampaignPreflightError(
            "expected_sha256 must be 64 lowercase hex chars"
        )
    path = Path(manifest_path)
    try:
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_size > MAX_MANIFEST_BYTES
        ):
            raise ABC6CampaignPreflightError(
                "manifest must be a small regular file, not a symlink"
            )
        raw = path.read_bytes()
    except ABC6CampaignPreflightError:
        raise
    except OSError as error:
        raise ABC6CampaignPreflightError("manifest is missing or unreadable") from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ABC6CampaignPreflightError("manifest SHA-256 does not match caller pin")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ABC6CampaignPreflightError(
            "manifest must be canonical ASCII JSON"
        ) from error
    try:
        canonical_decoded = _canonical_json(decoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise ABC6CampaignPreflightError(
            "manifest JSON contains values outside the strict schema"
        ) from error
    if type(decoded) is not dict or canonical_decoded != raw:
        raise ABC6CampaignPreflightError("manifest JSON is not a canonical object")
    if not _strict_manifest_schema(decoded):
        raise ABC6CampaignPreflightError("manifest schema/type validation failed")

    identity = _manifest_identity()
    if any(decoded.get(key) != value for key, value in identity.items()):
        raise ABC6CampaignPreflightError(
            "manifest protocol, report, training controls, or ordered roster mismatch"
        )
    source_hashes = decoded.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ABC6CampaignPreflightError("manifest lacks source_hashes mapping")
    current_sources = _current_source_hashes()
    if any(
        current_sources.get(name) != digest
        for name, digest in _REVIEWED_SOURCE_SHA256.items()
    ):
        raise ABC6CampaignPreflightError(
            "reviewed proposal, case roster, training seam, or baseline source changed"
        )
    if any(
        source_hashes.get(name) != digest for name, digest in current_sources.items()
    ):
        raise ABC6CampaignPreflightError("manifest source hash mismatch")
    runtime = decoded.get("runtime_fingerprint")
    if not isinstance(runtime, dict):
        raise ABC6CampaignPreflightError("manifest lacks runtime_fingerprint object")
    current_runtime = _current_runtime_fingerprint()
    if runtime != current_runtime:
        raise ABC6CampaignPreflightError("manifest runtime fingerprint mismatch")
    if require_reviewed_runtime:
        _require_reviewed_runtime(current_runtime)
    if not _execution_contract_is_valid(decoded.get("execution_contract")):
        raise ABC6CampaignPreflightError(
            "manifest execution contract must declare the external 900s/2-GiB "
            "single-worker limits and a projected artifact size below 1 GiB"
        )
    return actual_sha256


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:  # pragma: no cover - guarded by the OS write contract.
            raise OSError("durable file write made no progress")
        remaining = remaining[count:]


def _write_exclusive_durable(path: Path, payload: bytes) -> None:
    """Atomically publish immutable bytes with no replacement on collision."""

    directory = path.parent
    temporary_path = directory / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # Same-directory hard-link publication is atomic and fails if the final
        # immutable receipt already exists; os.replace would permit rewriting.
        os.link(temporary_path, path)
        _fsync_directory(directory)
    finally:
        try:
            temporary_path.unlink()
            _fsync_directory(directory)
        except FileNotFoundError:
            pass


def _status_receipt_payload(case_index: int, component: str, status: str) -> bytes:
    case = case_by_index(case_index)
    body: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "case_index": case_index,
        "case_id": case.case_id,
        "component": component,
        "status": status,
    }
    body["payload_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return _canonical_json(body)


def _write_case_status(
    receipt_directory: Path,
    case_index: int,
    component: str,
    status: str,
) -> tuple[Path, str]:
    if component not in _COMPONENTS:
        raise ValueError("component must be 'fit' or 'baseline'")
    if status not in _FINAL_STATUSES:
        raise ValueError("status must be a final receipt status")
    path = receipt_directory / f"case-{case_index:02d}.{component}-status.json"
    payload = _status_receipt_payload(case_index, component, status)
    _write_exclusive_durable(path, payload)
    return path, hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _claim_run(
    receipt_directory: Path,
    manifest_sha256: str,
    claim_registry_path: Path,
) -> str:
    local_claim_path = receipt_directory / CLAIM_FILENAME
    if local_claim_path.exists() or local_claim_path.is_symlink():
        raise ABC6CampaignAlreadyClaimedError(
            "the one-use run ID is already claimed; resume and retry are forbidden"
        )
    try:
        entries = tuple(receipt_directory.iterdir())
    except OSError as error:
        raise ABC6CampaignPreflightError(
            "cannot inventory run output directory"
        ) from error
    if entries:
        raise ABC6CampaignPreflightError(
            "run output directory must be empty before the one-use claim"
        )
    if not receipt_directory.is_dir() or receipt_directory.is_symlink():
        raise ABC6CampaignPreflightError(
            "run output directory must be a real directory"
        )
    try:
        claim_registry_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ABC6CampaignPreflightError(
            "cannot create the fixed project-root run-ID registry"
        ) from error
    if (
        claim_registry_path.parent.is_symlink()
        or not claim_registry_path.parent.is_dir()
    ):
        raise ABC6CampaignPreflightError(
            "run-ID registry parent must be a real directory"
        )
    claim = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "receipt_directory": str(receipt_directory.resolve()),
        "claimed_at_utc": _utc_now(),
        "claim_semantics": "consumed_once_no_resume",
    }
    payload = _canonical_json(claim)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            claim_registry_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _fsync_directory(claim_registry_path.parent)
    except FileExistsError as error:
        raise ABC6CampaignAlreadyClaimedError(
            "the fixed project-root registry already consumed this one-use run ID"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    # Retain a local linkable receipt in the per-run output folder.  The
    # registry claim above is authoritative and remains consumed if this
    # secondary publication fails.
    _write_exclusive_durable(local_claim_path, payload)
    return hashlib.sha256(payload).hexdigest()


def _write_failure_receipt(
    receipt_directory: Path,
    *,
    manifest_sha256: str,
    claim_sha256: str,
    completed_cases: int,
    case_index: int | None,
    stop_reason: str,
    exception_type: str,
) -> None:
    failure: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "claim_sha256": claim_sha256,
        "failed_at_utc": _utc_now(),
        "completed_case_count": completed_cases,
        "case_index": case_index,
        "case_id": None if case_index is None else case_by_index(case_index).case_id,
        "stop_reason": stop_reason,
        "exception_type": exception_type,
        "resume_allowed": False,
    }
    failure["payload_sha256"] = hashlib.sha256(_canonical_json(failure)).hexdigest()
    _write_exclusive_durable(
        receipt_directory / FAILURE_FILENAME,
        _canonical_json(failure),
    )


def _result_statuses(
    result: object,
    case: ABC6Case,
) -> tuple[str, str]:
    # Attribute validation keeps the private injected seam lightweight for
    # control-flow tests.  The production callable is the reviewed typed
    # one-case adapter above.
    required = ("case_index", "case_id", "model", "abc_status", "baseline_status")
    if any(not hasattr(result, name) for name in required):
        raise TypeError("fit callable returned an object without case status fields")
    if (
        type(result.case_index) is not int
        or result.case_index != case.case_index
        or result.case_id != case.case_id
        or result.model != case.fit_model.value
    ):
        raise ValueError("one-case training result identity does not match roster")
    fit_status = result.abc_status
    baseline_status = result.baseline_status
    if fit_status not in {"complete", "incomplete", "unresolved"}:
        raise ValueError("one-case training result has an invalid ABC status")
    if baseline_status not in {"complete", "incomplete"}:
        raise ValueError("one-case training result has an invalid baseline status")
    return fit_status, baseline_status


def _record_failed_case(
    receipt_directory: Path,
    *,
    case: ABC6Case,
    manifest_sha256: str,
    claim_sha256: str,
    completed_cases: int,
    stop_reason: str,
    error: BaseException,
) -> None:
    # The case-level schema intentionally accepts "failed" for both components.
    # Existing immutable receipts are never rewritten if storage failed midway.
    for component in _COMPONENTS:
        path = receipt_directory / f"case-{case.case_index:02d}.{component}-status.json"
        if not path.exists():
            _write_case_status(receipt_directory, case.case_index, component, "failed")
    _write_failure_receipt(
        receipt_directory,
        manifest_sha256=manifest_sha256,
        claim_sha256=claim_sha256,
        completed_cases=completed_cases,
        case_index=case.case_index,
        stop_reason=stop_reason,
        exception_type=type(error).__name__,
    )


def _summary_payload(
    manifest_sha256: str,
    claim_sha256: str,
    case_statuses: tuple[ABC6CaseStatus, ...],
) -> bytes:
    overall_status = (
        "complete"
        if len(case_statuses) == CASE_COUNT
        and all(
            item.fit_status == "complete" and item.baseline_status == "complete"
            for item in case_statuses
        )
        else "incomplete"
    )
    fit_counts = Counter(item.fit_status for item in case_statuses)
    baseline_counts = Counter(item.baseline_status for item in case_statuses)
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        "claim_sha256": claim_sha256,
        "written_at_utc": _utc_now(),
        "status": overall_status,
        "case_count": len(case_statuses),
        "fit_status_counts": dict(sorted(fit_counts.items())),
        "baseline_status_counts": dict(sorted(baseline_counts.items())),
        "case_statuses": [
            {
                "case_index": item.case_index,
                "case_id": item.case_id,
                "fit_status": item.fit_status,
                "baseline_status": item.baseline_status,
                "fit_receipt_sha256": item.fit_receipt_sha256,
                "baseline_receipt_sha256": item.baseline_receipt_sha256,
            }
            for item in case_statuses
        ],
        "training_only": True,
        "prospective_targets_generated": False,
        "forecasts_run": False,
        "postfit_target_gate_opened": False,
        "external_watchdog_enforced_here": False,
        "independent_manifest_approval_enforced_here": False,
        "resume_allowed": False,
    }
    summary["payload_sha256"] = hashlib.sha256(_canonical_json(summary)).hexdigest()
    return _canonical_json(summary)


def _publish_training_evidence(
    receipt_directory: Path,
    *,
    campaign_result: ABC6CampaignResult,
    bundle: ABC6TrainingBundle,
    results: tuple[training.ABC6TrainingResult, ...],
) -> tuple[Path, str]:
    """Durably publish canonical bundle/case evidence and its terminal index."""

    _validate_execution_shape(campaign_result, bundle, results)
    evidence_directory = receipt_directory / EVIDENCE_DIRECTORY_NAME
    evidence_directory.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(receipt_directory)

    bundle_filename = "training-bundle.evidence.json"
    bundle_payload = _bundle_evidence_payload(bundle)
    _write_exclusive_durable(evidence_directory / bundle_filename, bundle_payload)
    bundle_digest = _canonical_sha256(bundle_payload)

    case_entries: list[dict[str, object]] = []
    for case, status, result in zip(
        CASE_ROSTER, campaign_result.case_statuses, results, strict=True
    ):
        payload = _case_evidence_payload(case, result)
        filename = f"case-{case.case_index:02d}.training-evidence.json"
        _write_exclusive_durable(evidence_directory / filename, payload)
        case_entries.append(
            {
                "filename": filename,
                "sha256": _canonical_sha256(payload),
                "roster_index": case.case_index,
                "case_identity": _case_identity(case),
                "fit_status": status.fit_status,
                "baseline_status": status.baseline_status,
            }
        )

    status_receipts: list[dict[str, object]] = []
    for case, status in zip(CASE_ROSTER, campaign_result.case_statuses, strict=True):
        for component in _COMPONENTS:
            status_receipts.append(
                {
                    "filename": f"case-{case.case_index:02d}.{component}-status.json",
                    "sha256": getattr(status, f"{component}_receipt_sha256"),
                    "case_index": case.case_index,
                    "case_id": case.case_id,
                    "component": component,
                    "status": getattr(status, f"{component}_status"),
                }
            )
    terminal: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": campaign_result.protocol_id,
        "run_id": campaign_result.run_id,
        "manifest_sha256": campaign_result.manifest_sha256,
        "claim_sha256": campaign_result.claim_sha256,
        "training_summary_filename": campaign_result.summary_path.name,
        "training_summary_sha256": campaign_result.summary_sha256,
        "roster_sha256": _roster_sha256(),
        "ordered_case_identities": _roster_identities(),
        "status_receipts": status_receipts,
        "bundle_artifact": {
            "filename": bundle_filename,
            "sha256": bundle_digest,
        },
        "case_artifacts": case_entries,
    }
    terminal["payload_sha256"] = _canonical_sha256(_canonical_json(terminal))
    terminal_bytes = _canonical_json(terminal)
    manifest_path = receipt_directory / EVIDENCE_MANIFEST_FILENAME
    _write_exclusive_durable(manifest_path, terminal_bytes)
    return manifest_path, _canonical_sha256(terminal_bytes)


def _execute_campaign(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    require_reviewed_runtime: bool,
    claim_registry_path: Path,
    capture_training_evidence: bool = False,
) -> ABC6CampaignResult | ABC6TrainingCampaignExecution:
    # Hash and source/runtime checks happen before the irreversible O_EXCL claim.
    verified_manifest_sha256 = preflight_abc6_campaign_manifest(
        manifest_path,
        manifest_sha256,
        require_reviewed_runtime=require_reviewed_runtime,
    )
    directory = Path(receipt_directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ABC6CampaignPreflightError(
            "run output directory must already exist as a real directory"
        )
    claim_sha256 = _claim_run(
        directory,
        verified_manifest_sha256,
        claim_registry_path,
    )

    try:
        bundle = build_synthetic_training_bundle()
    except BaseException as error:
        _write_failure_receipt(
            directory,
            manifest_sha256=verified_manifest_sha256,
            claim_sha256=claim_sha256,
            completed_cases=0,
            case_index=None,
            stop_reason="synthetic_training_bundle_construction_failed",
            exception_type=type(error).__name__,
        )
        raise ABC6CampaignExecutionError(
            "training-data construction failed after the one-use claim; run is terminal"
        ) from error

    case_statuses: list[ABC6CaseStatus] = []
    training_results: list[training.ABC6TrainingResult] | None = (
        [] if capture_training_evidence else None
    )
    for expected_index, expected_case in enumerate(CASE_ROSTER):
        # Never trust a consumer-supplied order: verify the frozen view and then
        # retrieve it again by its exact roster index.
        if expected_case != case_by_index(expected_index):
            raise ABC6CampaignPreflightError(
                "runtime case roster changed after preflight"
            )
        data = bundle.data_for_case(expected_index)
        if data.case != expected_case:
            error = ValueError("training bundle view identity differs from roster")
            _record_failed_case(
                directory,
                case=expected_case,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(case_statuses),
                stop_reason="training_case_view_identity_mismatch",
                error=error,
            )
            raise ABC6CampaignExecutionError(
                f"case {expected_index} identity mismatch; run is terminal"
            ) from error

        try:
            result = fit_callable(data)
        except BaseException as error:
            try:
                _record_failed_case(
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="one_case_training_callable_raised",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "one-case training stopped and failure receipt could not be sealed; "
                    "run remains claimed and non-resumable"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} training failed; run is terminal and non-resumable"
            ) from error
        try:
            fit_status, baseline_status = _result_statuses(result, expected_case)
        except BaseException as error:
            try:
                _record_failed_case(
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="one_case_result_validation_failed",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "invalid one-case result and failure receipt could not be sealed; "
                    "run remains claimed and non-resumable"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} result failed validation; run is terminal"
            ) from error

        try:
            fit_path, fit_receipt_sha256 = _write_case_status(
                directory, expected_index, "fit", fit_status
            )
            baseline_path, baseline_receipt_sha256 = _write_case_status(
                directory, expected_index, "baseline", baseline_status
            )
        except BaseException as error:
            try:
                _write_failure_receipt(
                    directory,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    case_index=expected_index,
                    stop_reason="case_status_receipt_publication_failed",
                    exception_type=type(error).__name__,
                )
            except OSError as receipt_error:
                raise ABC6CampaignExecutionError(
                    "case status publication and failure receipt both failed; "
                    "run is terminal"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} status publication failed; run is terminal"
            ) from error
        del fit_path, baseline_path
        case_statuses.append(
            ABC6CaseStatus(
                case_index=expected_index,
                case_id=expected_case.case_id,
                fit_status=fit_status,
                baseline_status=baseline_status,
                fit_receipt_sha256=fit_receipt_sha256,
                baseline_receipt_sha256=baseline_receipt_sha256,
            )
        )
        if training_results is not None:
            # Production execution uses the typed one-case training adapter.
            # The private test seam may inject lightweight result doubles.
            training_results.append(cast(training.ABC6TrainingResult, result))

    statuses = tuple(case_statuses)
    summary_bytes = _summary_payload(
        verified_manifest_sha256,
        claim_sha256,
        statuses,
    )
    summary_path = directory / SUMMARY_FILENAME
    try:
        _write_exclusive_durable(summary_path, summary_bytes)
    except OSError as error:
        try:
            _write_failure_receipt(
                directory,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(statuses),
                case_index=None,
                stop_reason="campaign_summary_publication_failed",
                exception_type=type(error).__name__,
            )
        except OSError as receipt_error:
            raise ABC6CampaignExecutionError(
                "campaign summary and failure receipt both failed; run is terminal"
            ) from receipt_error
        raise ABC6CampaignExecutionError(
            "all case receipts exist but summary publication failed; run is terminal"
        ) from error
    overall_status: Literal["complete", "incomplete"] = (
        "complete"
        if all(
            item.fit_status == "complete" and item.baseline_status == "complete"
            for item in statuses
        )
        else "incomplete"
    )
    campaign_result = ABC6CampaignResult(
        protocol_id=PROTOCOL_ID,
        run_id=RUN_ID,
        manifest_sha256=verified_manifest_sha256,
        claim_sha256=claim_sha256,
        status=overall_status,
        case_statuses=statuses,
        summary_path=summary_path,
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
    )
    if training_results is None:
        return campaign_result
    detailed_results = tuple(training_results)
    try:
        evidence_manifest_path, evidence_manifest_sha256 = _publish_training_evidence(
            directory,
            campaign_result=campaign_result,
            bundle=bundle,
            results=detailed_results,
        )
    except BaseException as error:
        try:
            _write_failure_receipt(
                directory,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(statuses),
                case_index=None,
                stop_reason="training_evidence_publication_failed",
                exception_type=type(error).__name__,
            )
        except BaseException as receipt_error:
            raise ABC6CampaignExecutionError(
                "training evidence publication and failure receipt both failed; "
                "run is terminal"
            ) from receipt_error
        raise ABC6CampaignExecutionError(
            "all status receipts exist but durable training evidence publication "
            "failed; run is terminal"
        ) from error
    execution = ABC6TrainingCampaignExecution(
        campaign_result=campaign_result,
        evidence_manifest_path=evidence_manifest_path,
        evidence_manifest_sha256=evidence_manifest_sha256,
        _training_bundle=bundle,
        _training_results=detailed_results,
    )
    try:
        _verify_training_execution_evidence(execution)
    except BaseException as error:
        try:
            _write_failure_receipt(
                directory,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(statuses),
                case_index=None,
                stop_reason="training_evidence_terminal_verification_failed",
                exception_type=type(error).__name__,
            )
        except BaseException as receipt_error:
            raise ABC6CampaignExecutionError(
                "training evidence verification and failure receipt both failed; "
                "run is terminal"
            ) from receipt_error
        raise ABC6CampaignExecutionError(
            "durable training evidence failed terminal verification; run is terminal"
        ) from error
    return execution


def run_abc6_training_campaign(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
) -> ABC6CampaignResult:
    """Run the 24 production training fits once, after separate external gates.

    This is not a launch approval.  Before calling it, an independent reviewer
    must approve the immutable manifest and a separate watchdog must enforce
    and record the full runner-tree budget.  The function itself verifies the
    manifest hash, exact roster/source/runtime identity, and declared limits;
    it cannot prove reviewer independence or monitor global time/RSS.
    """

    result = _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=training.run_abc6_training_case,
        require_reviewed_runtime=True,
        claim_registry_path=_PROJECT_RUN_CLAIM_PATH,
        capture_training_evidence=False,
    )
    if not isinstance(result, ABC6CampaignResult):  # pragma: no cover - invariant.
        raise AssertionError("status-only campaign unexpectedly returned evidence")
    return result


def run_abc6_training_campaign_with_evidence(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
) -> ABC6TrainingCampaignExecution:
    """Run the claimed 24-case campaign and return its exact in-memory evidence.

    This follows the same manifest preflight, one-use claim, bundle build,
    ordered fits, durable status receipts, and terminal-summary path as
    :func:`run_abc6_training_campaign`.  It returns only after every receipt
    and the summary have been durably published.  A raised case or publication
    failure remains terminal and returns no execution object.
    """

    result = _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=training.run_abc6_training_case,
        require_reviewed_runtime=True,
        claim_registry_path=_PROJECT_RUN_CLAIM_PATH,
        capture_training_evidence=True,
    )
    if not isinstance(result, ABC6TrainingCampaignExecution):  # pragma: no cover.
        raise AssertionError("evidence campaign did not return its training evidence")
    return result


def _run_campaign_with_fit_callable_for_test(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    claim_registry_path: str | os.PathLike[str],
) -> ABC6CampaignResult:
    """Test-only bounded seam; permits fake fits and a non-production runtime."""

    if not callable(fit_callable):
        raise TypeError("fit_callable must be callable")
    return _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=fit_callable,
        require_reviewed_runtime=False,
        claim_registry_path=Path(claim_registry_path),
        capture_training_evidence=False,
    )


def _run_campaign_with_fit_callable_and_evidence_for_test(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    claim_registry_path: str | os.PathLike[str],
) -> ABC6TrainingCampaignExecution:
    """Test-only evidence seam with fake fits and a private claim registry."""

    if not callable(fit_callable):
        raise TypeError("fit_callable must be callable")
    result = _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=fit_callable,
        require_reviewed_runtime=False,
        claim_registry_path=Path(claim_registry_path),
        capture_training_evidence=True,
    )
    if not isinstance(result, ABC6TrainingCampaignExecution):
        raise AssertionError("test evidence campaign did not return its evidence")
    return result


__all__ = [
    "ABC6CampaignAlreadyClaimedError",
    "ABC6CampaignError",
    "ABC6CampaignExecutionError",
    "ABC6CampaignPreflightError",
    "ABC6CampaignResult",
    "ABC6CaseStatus",
    "ABC6VerifiedTrainingEvidence",
    "ABC6TrainingCampaignExecution",
    "preflight_abc6_campaign_manifest",
    "run_abc6_training_campaign",
    "run_abc6_training_campaign_with_evidence",
]

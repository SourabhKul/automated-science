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
import subprocess
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

from core.real_data import cascaded_tanks_abc6_authority as authority_module
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_receipt_io as receipt_io
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

MANIFEST_SCHEMA_VERSION: Final = 2
RECEIPT_ROOT_RELATIVE: Final = (
    "artifacts/cascaded_tanks_abc6_synthetic/"
    "ct-abc6-20260928-v1/receipts"
)
CAMPAIGN_CLAIM_PARENT_RELATIVE: Final = (
    "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims"
)
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
    *authority_module.ABC6_MANIFEST_SOURCE_PATHS,
)
# The source-key shape is frozen while final bytes/runtime are not reviewed.
# Empty slots make the current production pin intentionally unusable; fake
# tests may replace this mapping with digests in their isolated fixtures.
_REVIEWED_SOURCE_SHA256: Final = MappingProxyType(
    {path: None for path in authority_module.ABC6_STATIC_REVIEWED_SOURCE_PATHS}
)
_FINAL_STATUSES: Final = frozenset({"complete", "incomplete", "unresolved", "failed"})
_COMPONENTS: Final = ("fit", "baseline")
_TRAINING_EXECUTION_SEAL: Final = object()


class ABC6CampaignError(RuntimeError):
    """Base error for fail-closed campaign preflight or orchestration."""


class ABC6CampaignPreflightError(ABC6CampaignError):
    """The immutable run manifest or output inventory failed preflight."""


class ABC6CampaignAlreadyClaimedError(ABC6CampaignError):
    """The one-use run ID has already been claimed and cannot be resumed."""


class ABC6CampaignExecutionError(ABC6CampaignError):
    """Training stopped after the run claim; the run ID remains consumed."""


class ABC6ReceiptRootIdentity:
    """Live checkout and receipt directory anchors for one local campaign.

    Device/inode pairs are runtime path identities, not portable manifest
    values. The owned descriptors keep the opened directories alive until the
    execution is closed or collected.
    """

    __slots__ = (
        "repository_root_realpath",
        "receipt_root_relative",
        "repository_root_device",
        "repository_root_inode",
        "receipt_root_device",
        "receipt_root_inode",
        "source_hashes",
        "_repository_root_fd",
        "_receipt_root_fd",
    )

    def __init__(
        self,
        *,
        repository_root_realpath: str,
        receipt_root_relative: str,
        repository_root_fd: int,
        receipt_root_fd: int,
        source_hashes: tuple[tuple[str, str], ...] = (),
    ) -> None:
        root_stat = os.fstat(repository_root_fd)
        receipt_stat = os.fstat(receipt_root_fd)
        if not stat.S_ISDIR(root_stat.st_mode) or not stat.S_ISDIR(receipt_stat.st_mode):
            raise ABC6CampaignPreflightError("campaign path anchors must be directories")
        self.repository_root_realpath = repository_root_realpath
        self.receipt_root_relative = receipt_root_relative
        self.repository_root_device = root_stat.st_dev
        self.repository_root_inode = root_stat.st_ino
        self.receipt_root_device = receipt_stat.st_dev
        self.receipt_root_inode = receipt_stat.st_ino
        self.source_hashes = tuple(source_hashes)
        self._repository_root_fd = repository_root_fd
        self._receipt_root_fd = receipt_root_fd

    @property
    def receipt_root_path(self) -> Path:
        return Path(self.repository_root_realpath) / self.receipt_root_relative

    def verify(self) -> None:
        """Fail if either retained descriptor or its canonical path changed."""

        _validate_receipt_root_relative(self.receipt_root_relative)
        try:
            root_stat = os.fstat(self._repository_root_fd)
            receipt_stat = os.fstat(self._receipt_root_fd)
        except (OSError, TypeError) as error:
            raise ABC6CampaignPreflightError("campaign path anchor is closed") from error
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or (root_stat.st_dev, root_stat.st_ino)
            != (self.repository_root_device, self.repository_root_inode)
            or not stat.S_ISDIR(receipt_stat.st_mode)
            or (receipt_stat.st_dev, receipt_stat.st_ino)
            != (self.receipt_root_device, self.receipt_root_inode)
        ):
            raise ABC6CampaignPreflightError("campaign path anchor identity changed")

        root_reopened = _open_directory_nofollow(
            Path(self.repository_root_realpath), label="frozen checkout root"
        )
        try:
            current = os.fstat(root_reopened)
            if (current.st_dev, current.st_ino) != (
                self.repository_root_device,
                self.repository_root_inode,
            ):
                raise ABC6CampaignPreflightError(
                    "frozen checkout root identity changed"
                )
            receipt_reopened = _open_relative_directory_nofollow(
                root_reopened,
                self.receipt_root_relative,
                label="campaign receipt root",
            )
            try:
                receipt_current = os.fstat(receipt_reopened)
                if (receipt_current.st_dev, receipt_current.st_ino) != (
                    self.receipt_root_device,
                    self.receipt_root_inode,
                ):
                    raise ABC6CampaignPreflightError(
                        "campaign receipt root identity changed"
                    )
            finally:
                os.close(receipt_reopened)
        finally:
            os.close(root_reopened)
        if self.source_hashes:
            try:
                actual_sources = _current_source_hashes(
                    root_fd=self._repository_root_fd
                )
            except ABC6CampaignPreflightError:
                raise
            except OSError as error:
                raise ABC6CampaignPreflightError(
                    "pinned campaign sources could not be revalidated"
                ) from error
            if actual_sources != dict(self.source_hashes):
                raise ABC6CampaignPreflightError(
                    "pinned campaign source identity changed"
                )

    def duplicate_repository_root_fd(self) -> int:
        self.verify()
        return os.dup(self._repository_root_fd)

    def duplicate_receipt_root_fd(self) -> int:
        self.verify()
        return os.dup(self._receipt_root_fd)

    def duplicate_receipt_root_fd_for_durable_write(self) -> int:
        """Duplicate the already-open receipt directory without path lookup.

        This is intentionally usable after an ancestor path swap so terminal
        failure evidence remains directed at the directory opened pre-claim.
        """

        try:
            receipt_stat = os.fstat(self._receipt_root_fd)
        except (OSError, TypeError) as error:
            raise ABC6CampaignPreflightError(
                "campaign receipt directory anchor is closed"
            ) from error
        if (
            not stat.S_ISDIR(receipt_stat.st_mode)
            or (receipt_stat.st_dev, receipt_stat.st_ino)
            != (self.receipt_root_device, self.receipt_root_inode)
        ):
            raise ABC6CampaignPreflightError(
                "campaign receipt directory anchor identity changed"
            )
        return os.dup(self._receipt_root_fd)

    @property
    def runtime_identity(self) -> dict[str, int]:
        return {
            "repository_root_device": self.repository_root_device,
            "repository_root_inode": self.repository_root_inode,
            "receipt_root_device": self.receipt_root_device,
            "receipt_root_inode": self.receipt_root_inode,
        }

    def close(self) -> None:
        for name in ("_receipt_root_fd", "_repository_root_fd"):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, name, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
    receipt_root_identity: ABC6ReceiptRootIdentity = field(
        repr=False, compare=False
    )
    _authority: authority_module.ABC6LaunchAuthority | None = field(
        repr=False, compare=False
    )
    _grant_sha256: str | None = field(repr=False, compare=False)
    _run_id: str = field(repr=False, compare=False)
    _manifest_sha256: str = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _TRAINING_EXECUTION_SEAL:
            raise TypeError("training executions are issued only by the private campaign path")
        _validate_execution_shape(
            self.campaign_result, self._training_bundle, self._training_results
        )
        if not isinstance(self.evidence_manifest_path, Path):
            raise TypeError("evidence_manifest_path must be a Path")
        if not _is_sha256(self.evidence_manifest_sha256):
            raise ValueError("evidence_manifest_sha256 must be a lowercase SHA-256")
        if not isinstance(self.receipt_root_identity, ABC6ReceiptRootIdentity):
            raise TypeError("receipt_root_identity must be an anchored identity")
        _validate_execution_receipt_paths(
            self.campaign_result, self.evidence_manifest_path, self.receipt_root_identity
        )
        if (
            self._run_id != self.campaign_result.run_id
            or self._manifest_sha256 != self.campaign_result.manifest_sha256
        ):
            raise ValueError("training execution binding differs from its campaign result")
        if self._authority is None:
            if self._grant_sha256 is not None:
                raise ValueError("offline training execution cannot carry a grant digest")
        elif (
            type(self._authority) is not authority_module.ABC6LaunchAuthority
            or not _is_sha256(self._grant_sha256)
        ):
            raise TypeError("supervised training execution requires its sealed launch grant")

    def load_verified_training_evidence(self) -> ABC6VerifiedTrainingEvidence:
        """Revalidate durable and in-memory evidence, then return frozen copies."""

        if self._authority is not None:
            self._authority._assert_registered_training_execution(self)
        if not isinstance(self.receipt_root_identity, ABC6ReceiptRootIdentity):
            raise ValueError("training execution has no anchored receipt identity")
        self.receipt_root_identity.verify()
        _verify_training_execution_evidence(self)
        self.receipt_root_identity.verify()
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

    def close(self) -> None:
        """Release the local directory descriptors retained by this execution."""

        if (
            self._authority is not None
            and not self._authority._owns_training_execution_identity(self)
        ):
            return
        identity = self.receipt_root_identity
        if isinstance(identity, ABC6ReceiptRootIdentity):
            identity.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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


def _decode_canonical_evidence_json(
    raw: bytes, *, label: str
) -> tuple[bytes, dict[str, object]]:
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read durable evidence JSON: {label}") from error
    if not isinstance(decoded, dict) or _canonical_json(decoded) != raw:
        raise ValueError(f"durable evidence JSON is not canonical: {label}")
    return raw, decoded


def _read_canonical_evidence_json_at(
    directory_fd: int, filename: str, *, label: str
) -> tuple[bytes, dict[str, object]]:
    try:
        raw = receipt_io.read_regular_file_at(
            directory_fd,
            filename,
            maximum_bytes=PROJECTED_ARTIFACT_LIMIT_BYTES,
            label=label,
        )
    except (OSError, receipt_io.ABC6ReceiptIOError) as error:
        raise ValueError(f"cannot read durable evidence JSON: {label}") from error
    return _decode_canonical_evidence_json(raw, label=label)


def _open_evidence_directory_at(receipt_directory_fd: int) -> int:
    try:
        descriptor = os.open(
            EVIDENCE_DIRECTORY_NAME,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=receipt_directory_fd,
        )
    except OSError as error:
        raise ValueError("durable training evidence directory is missing or unsafe") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("durable training evidence path is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


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


def _validate_execution_receipt_paths(
    campaign_result: ABC6CampaignResult,
    evidence_manifest_path: Path,
    receipt_root_identity: ABC6ReceiptRootIdentity,
) -> None:
    receipt_root = receipt_root_identity.receipt_root_path
    if not isinstance(campaign_result.summary_path, Path) or (
        campaign_result.summary_path != receipt_root / SUMMARY_FILENAME
    ):
        raise ValueError("campaign summary path is outside the anchored receipt root")
    if not isinstance(evidence_manifest_path, Path) or (
        evidence_manifest_path != receipt_root / EVIDENCE_MANIFEST_FILENAME
    ):
        raise ValueError("evidence manifest path is outside the anchored receipt root")


def _verify_summary_link(
    campaign_result: ABC6CampaignResult,
    raw: bytes,
    decoded: Mapping[str, object],
) -> None:
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
    identity = execution.receipt_root_identity
    if not isinstance(identity, ABC6ReceiptRootIdentity):
        raise ValueError("training execution has no anchored receipt identity")
    _validate_execution_receipt_paths(
        execution.campaign_result, execution.evidence_manifest_path, identity
    )
    identity.verify()
    receipt_directory_fd = identity.duplicate_receipt_root_fd()
    try:
        _verify_training_execution_evidence_at(execution, receipt_directory_fd)
    finally:
        os.close(receipt_directory_fd)
    identity.verify()


def _verify_training_execution_evidence_at(
    execution: ABC6TrainingCampaignExecution,
    receipt_directory_fd: int,
) -> None:
    result = execution.campaign_result
    _validate_execution_shape(
        result, execution._training_bundle, execution._training_results
    )
    manifest_raw, manifest = _read_canonical_evidence_json_at(
        receipt_directory_fd,
        EVIDENCE_MANIFEST_FILENAME,
        label="training evidence manifest",
    )
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

    summary_raw, summary = _read_canonical_evidence_json_at(
        receipt_directory_fd,
        SUMMARY_FILENAME,
        label="campaign training summary",
    )
    _verify_summary_link(result, summary_raw, summary)
    claim_raw, claim = _read_canonical_evidence_json_at(
        receipt_directory_fd,
        CLAIM_FILENAME,
        label="campaign claim receipt",
    )
    if _canonical_sha256(claim_raw) != result.claim_sha256:
        raise ValueError("local claim receipt digest does not match campaign result")
    if (
        claim.get("protocol_id") != result.protocol_id
        or claim.get("run_id") != result.run_id
        or claim.get("manifest_sha256") != result.manifest_sha256
        or claim.get("claim_semantics") != "consumed_once_no_resume"
    ):
        raise ValueError("local claim receipt identity differs from campaign result")

    verified_receipts = receipt_io.verify_status_receipts_at(
        receipt_directory_fd,
        protocol_id=result.protocol_id,
        run_id=result.run_id,
        case_ids=tuple(case.case_id for case in CASE_ROSTER),
    )
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

    evidence_directory_fd = _open_evidence_directory_at(receipt_directory_fd)
    try:
        _verify_training_evidence_artifacts(
            execution, result, manifest, evidence_directory_fd
        )
    finally:
        os.close(evidence_directory_fd)


def _verify_training_evidence_artifacts(
    execution: ABC6TrainingCampaignExecution,
    result: ABC6CampaignResult,
    manifest: Mapping[str, object],
    evidence_directory_fd: int,
) -> None:
    bundle_entry = manifest.get("bundle_artifact")
    if not isinstance(bundle_entry, dict) or set(bundle_entry) != {"filename", "sha256"}:
        raise ValueError("evidence manifest bundle entry is invalid")
    if (
        bundle_entry.get("filename") != "training-bundle.evidence.json"
        or not _is_sha256(bundle_entry.get("sha256"))
    ):
        raise ValueError("evidence manifest bundle filename/digest is invalid")
    bundle_raw, bundle_decoded = _read_canonical_evidence_json_at(
        evidence_directory_fd,
        str(bundle_entry["filename"]),
        label="durable training bundle",
    )
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
        case_raw, case_decoded = _read_canonical_evidence_json_at(
            evidence_directory_fd,
            filename,
            label=f"durable training case {index:02d}",
        )
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


def _active_checkout_root_path() -> Path:
    """Return the lexical source checkout root without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(__file__))).parents[2]


def _current_git_head() -> str:
    """Return the active source checkout's exact Git HEAD."""

    root = _active_checkout_root_path()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ABC6CampaignPreflightError(
            "cannot read the active checkout Git HEAD"
        ) from error
    head = result.stdout.strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ABC6CampaignPreflightError("active checkout Git HEAD is invalid")
    return head


def _canonical_absolute_directory_text(value: object, *, label: str) -> Path:
    if type(value) is not str:
        raise ABC6CampaignPreflightError(f"{label} must be a string")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ABC6CampaignPreflightError(f"{label} must be a canonical absolute path")
    return path


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    """Open an absolute directory by walking every component from `/`."""

    if not path.is_absolute() or str(path) != os.fspath(path):
        raise ABC6CampaignPreflightError(f"{label} path is not canonical and absolute")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ABC6CampaignPreflightError(
            "platform cannot enforce no-follow campaign path traversal"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        current_fd = os.open(os.sep, flags)
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise ABC6CampaignPreflightError(
                    f"{label} path contains a noncanonical component"
                )
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise ABC6CampaignPreflightError(f"{label} is not a directory")
        return current_fd
    except ABC6CampaignPreflightError:
        if "current_fd" in locals():
            os.close(current_fd)
        raise
    except OSError as error:
        if "current_fd" in locals():
            os.close(current_fd)
        raise ABC6CampaignPreflightError(
            f"{label} is missing or has a symlinked path component"
        ) from error


def _validate_receipt_root_relative(value: object) -> str:
    if type(value) is not str or value != RECEIPT_ROOT_RELATIVE:
        raise ABC6CampaignPreflightError(
            "receipt_root_relative must equal the fixed canonical campaign path"
        )
    return _validate_canonical_relative_path(
        value, label="receipt_root_relative"
    )


def _validate_canonical_relative_path(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or Path(value).is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(Path(value)) != value
    ):
        raise ABC6CampaignPreflightError(
            f"{label} must be a canonical traversal-free relative path"
        )
    return value


def _open_relative_directory_nofollow(
    root_fd: int, relative_path: str, *, label: str, create: bool = False
) -> int:
    """Walk a canonical relative directory path from an already-open root."""

    _validate_canonical_relative_path(relative_path, label=label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.dup(root_fd)
    try:
        for component in relative_path.split("/"):
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
                # Persist each newly created directory entry before descending.
                os.fsync(current_fd)
                next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise ABC6CampaignPreflightError(f"{label} is not a directory")
        if create:
            os.fsync(current_fd)
        return current_fd
    except ABC6CampaignPreflightError:
        os.close(current_fd)
        raise
    except OSError as error:
        os.close(current_fd)
        raise ABC6CampaignPreflightError(
            f"{label} is missing or has a symlinked path component"
        ) from error


def _open_matching_checkout_root(repository_root_realpath: object) -> int:
    frozen_path = _canonical_absolute_directory_text(
        repository_root_realpath, label="repository_root_realpath"
    )
    active_path = _canonical_absolute_directory_text(
        str(_active_checkout_root_path()), label="active source checkout root"
    )
    if active_path != frozen_path:
        raise ABC6CampaignPreflightError(
            "manifest checkout root differs from the active source checkout root"
        )
    frozen_fd = _open_directory_nofollow(frozen_path, label="frozen checkout root")
    try:
        active_fd = _open_directory_nofollow(
            active_path, label="active source checkout root"
        )
    except BaseException:
        os.close(frozen_fd)
        raise
    try:
        frozen_stat = os.fstat(frozen_fd)
        active_stat = os.fstat(active_fd)
        if (frozen_stat.st_dev, frozen_stat.st_ino) != (
            active_stat.st_dev,
            active_stat.st_ino,
        ):
            raise ABC6CampaignPreflightError(
                "manifest and active checkout root identities differ"
            )
    except BaseException:
        os.close(frozen_fd)
        raise
    finally:
        os.close(active_fd)
    return frozen_fd


def _open_receipt_root_identity(
    repository_root_fd: int,
    repository_root_realpath: str,
    receipt_root_relative: str,
    requested_receipt_directory: str | os.PathLike[str],
    *,
    training_session: authority_module.ABC6TrainingSession | None = None,
    manifest_sha256: str | None = None,
    source_hashes: tuple[tuple[str, str], ...] = (),
) -> ABC6ReceiptRootIdentity:
    relative = _validate_receipt_root_relative(receipt_root_relative)
    expected_path = Path(repository_root_realpath) / relative
    try:
        requested_text = os.fspath(requested_receipt_directory)
    except TypeError as error:
        raise ABC6CampaignPreflightError(
            "receipt directory must be an absolute path"
        ) from error
    if type(requested_text) is not str or requested_text != str(expected_path):
        raise ABC6CampaignPreflightError(
            "receipt directory must equal the manifest checkout/root join"
        )

    if source_hashes and _current_source_hashes(root_fd=repository_root_fd) != dict(
        source_hashes
    ):
        raise ABC6CampaignPreflightError(
            "pinned campaign source identity changed before receipt setup"
        )

    receipt_fd = _open_relative_directory_nofollow(
        repository_root_fd,
        relative,
        label="campaign receipt root",
        create=True,
    )
    repository_root_duplicate = os.dup(repository_root_fd)
    try:
        identity = ABC6ReceiptRootIdentity(
            repository_root_realpath=repository_root_realpath,
            receipt_root_relative=relative,
            repository_root_fd=repository_root_duplicate,
            receipt_root_fd=receipt_fd,
            source_hashes=source_hashes,
        )
    except BaseException:
        os.close(repository_root_duplicate)
        os.close(receipt_fd)
        raise
    try:
        identity.verify()
        _check_preclaim_receipt_root(
            identity,
            identity._receipt_root_fd,
            training_session,
            scan="root-open",
            manifest_sha256=manifest_sha256,
        )
        os.fsync(identity._receipt_root_fd)
    except BaseException:
        identity.close()
        raise
    return identity


def _check_preclaim_receipt_root(
    receipt_root_identity: ABC6ReceiptRootIdentity,
    receipt_directory_fd: int,
    training_session: authority_module.ABC6TrainingSession | None,
    *,
    scan: str,
    manifest_sha256: str | None = None,
) -> None:
    """Apply direct empty-only or supervised pinned-grant preclaim policy."""

    # The directory listing and later O_EXCL claim are separate operations.
    # This methods-control gate assumes the trusted local run owns its receipt
    # root; it does not serialize or defeat an untrusted concurrent writer.
    try:
        receipt_root_identity.verify()
        if training_session is None:
            if os.listdir(receipt_directory_fd):
                raise ABC6CampaignPreflightError(
                    "run output directory must be empty before the one-use claim"
                )
            return
        if type(training_session) is not authority_module.ABC6TrainingSession:
            raise ABC6CampaignPreflightError(
                "campaign preclaim requires a sealed training session"
            )
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise ABC6CampaignPreflightError(
                "supervised campaign preclaim has no verified manifest digest"
            )
        training_session.verify_preclaim_root(
            receipt_root_identity._repository_root_fd,
            receipt_directory_fd,
            scan=scan,
            repository_root_realpath=receipt_root_identity.repository_root_realpath,
            receipt_root_relative=receipt_root_identity.receipt_root_relative,
            protocol_id=PROTOCOL_ID,
            run_id=RUN_ID,
            manifest_sha256=manifest_sha256,
        )
    except authority_module.ABC6AuthorityError as error:
        raise ABC6CampaignPreflightError(
            f"supervised campaign preclaim failed at {scan}: {error}"
        ) from error
    except OSError as error:
        raise ABC6CampaignPreflightError(
            f"campaign receipt-root scan failed at {scan}"
        ) from error


def _manifest_identity() -> dict[str, object]:
    """Stable identity fields; source/runtime digests remain externally pinned."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "repository_root_realpath": str(_active_checkout_root_path()),
        "receipt_root_relative": RECEIPT_ROOT_RELATIVE,
        "reviewed_git_head": _current_git_head(),
        "reviewed_proposal": {
            "path": REVIEWED_PROPOSAL_PATH,
            "sha256": REVIEWED_PROPOSAL_SHA256,
        },
        "training_control_manifest_sha256": training.TRAINING_CONTROL_MANIFEST_SHA256,
        "ordered_cases": [_case_identity(case) for case in CASE_ROSTER],
    }


def _source_stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _assert_source_root_path_matches(root_fd: int) -> None:
    """Keep source reads tied to the canonical checkout-root path identity."""

    try:
        descriptor_info = os.fstat(root_fd)
    except OSError as error:
        raise ABC6CampaignPreflightError("pinned source root descriptor is unavailable") from error
    if not stat.S_ISDIR(descriptor_info.st_mode):
        raise ABC6CampaignPreflightError("pinned source root is not a directory")
    path_fd = _open_directory_nofollow(Path(_REPO_ROOT), label="pinned source root")
    try:
        path_info = os.fstat(path_fd)
        if (descriptor_info.st_dev, descriptor_info.st_ino) != (
            path_info.st_dev,
            path_info.st_ino,
        ):
            raise ABC6CampaignPreflightError("pinned source root identity changed")
    finally:
        os.close(path_fd)


def _hash_source_beneath(root_fd: int, relative_path: str) -> str:
    """Hash one source through no-follow directory descriptors and verify identity."""

    _validate_canonical_relative_path(relative_path, label="pinned source path")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ABC6CampaignPreflightError(
            "platform cannot enforce no-follow pinned source reads"
        )
    parts = relative_path.split("/")
    leaf = parts[-1]
    parent_relative = "/".join(parts[:-1])
    parent_fd = (
        _open_relative_directory_nofollow(
            root_fd, parent_relative, label=f"source parent for {relative_path}"
        )
        if parent_relative
        else os.dup(root_fd)
    )
    source_fd = -1
    try:
        parent_before = os.fstat(parent_fd)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            source_fd = os.open(leaf, flags, dir_fd=parent_fd)
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ABC6CampaignPreflightError(
                    f"pinned source is not a regular file: {relative_path}"
                )
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > 64 * 1024 * 1024:
                    raise ABC6CampaignPreflightError(
                        f"pinned source exceeds the 64 MiB read limit: {relative_path}"
                    )
                digest.update(chunk)
            after = os.fstat(source_fd)
            named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except ABC6CampaignPreflightError:
            raise
        except OSError as error:
            raise ABC6CampaignPreflightError(
                f"pinned source is missing, symlinked, or unreadable: {relative_path}"
            ) from error

        if (
            byte_count != before.st_size
            or _source_stat_identity(before) != _source_stat_identity(after)
            or _source_stat_identity(before) != _source_stat_identity(named)
            or not stat.S_ISREG(named.st_mode)
        ):
            raise ABC6CampaignPreflightError(
                f"pinned source identity changed while hashing: {relative_path}"
            )

        if parent_relative:
            current_parent_fd = _open_relative_directory_nofollow(
                root_fd,
                parent_relative,
                label=f"source parent for {relative_path}",
            )
        else:
            current_parent_fd = os.dup(root_fd)
        try:
            parent_after = os.fstat(parent_fd)
            current_parent = os.fstat(current_parent_fd)
            if (
                (parent_before.st_dev, parent_before.st_ino)
                != (parent_after.st_dev, parent_after.st_ino)
                or (parent_before.st_dev, parent_before.st_ino)
                != (current_parent.st_dev, current_parent.st_ino)
            ):
                raise ABC6CampaignPreflightError(
                    f"pinned source ancestor identity changed while hashing: {relative_path}"
                )
        finally:
            os.close(current_parent_fd)
        return digest.hexdigest()
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        os.close(parent_fd)


def _current_source_hashes(*, root_fd: int | None = None) -> dict[str, str]:
    """Hash the exact candidate source roster through the pinned checkout root."""

    owned_root_fd = root_fd is None
    if root_fd is None:
        root_fd = _open_directory_nofollow(
            Path(_REPO_ROOT), label="pinned source root"
        )
    try:
        _assert_source_root_path_matches(root_fd)
        hashes = {
            relative_path: _hash_source_beneath(root_fd, relative_path)
            for relative_path in _REQUIRED_SOURCE_PATHS
        }
        _assert_source_root_path_matches(root_fd)
        return hashes
    finally:
        if owned_root_fd:
            os.close(root_fd)


def _reviewed_source_map_is_complete() -> bool:
    reviewed = _REVIEWED_SOURCE_SHA256
    return (
        isinstance(reviewed, Mapping)
        and set(reviewed) == set(authority_module.ABC6_STATIC_REVIEWED_SOURCE_PATHS)
        and all(_is_sha256(digest) for digest in reviewed.values())
    )


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
        "repository_root_realpath",
        "receipt_root_relative",
        "reviewed_git_head",
        "reviewed_proposal",
        "training_control_manifest_sha256",
        "ordered_cases",
        "source_hashes",
        "runtime_fingerprint",
        "execution_contract",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys:
        return False
    if type(decoded["schema_version"]) is not int or decoded["schema_version"] != 2:
        return False
    if type(decoded["protocol_id"]) is not str or type(decoded["run_id"]) is not str:
        return False
    try:
        _canonical_absolute_directory_text(
            decoded["repository_root_realpath"], label="repository_root_realpath"
        )
        _validate_receipt_root_relative(decoded["receipt_root_relative"])
    except ABC6CampaignPreflightError:
        return False
    if (
        type(decoded["reviewed_git_head"]) is not str
        or len(decoded["reviewed_git_head"]) != 40
        or any(char not in "0123456789abcdef" for char in decoded["reviewed_git_head"])
    ):
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
    pinned_manifest_bytes: bytes | None = None,
) -> str:
    """Validate an external immutable manifest and return its exact file hash.

    The expected hash is supplied by the caller after the manifest has been
    frozen. This verifies the declared physical checkout and active HEAD, but
    does not validate the separate approval record or enforce the watchdog.
    The approval owner must require its reviewed_git_head to match the manifest.
    A watchdog that has already read the manifest through its pinned no-follow
    descriptor may pass those exact bytes here, avoiding a second path lookup.
    """

    actual_sha256, decoded, root_fd = _preflight_manifest_and_open_checkout_root(
        manifest_path,
        expected_sha256,
        require_reviewed_runtime=require_reviewed_runtime,
        pinned_manifest_bytes=pinned_manifest_bytes,
    )
    del decoded
    os.close(root_fd)
    return actual_sha256


def _preflight_manifest_and_open_checkout_root(
    manifest_path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    require_reviewed_runtime: bool,
    pinned_manifest_bytes: bytes | None = None,
) -> tuple[str, dict[str, object], int]:
    """Validate the v2 manifest and return its verified checkout-root FD."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ABC6CampaignPreflightError(
            "expected_sha256 must be 64 lowercase hex chars"
        )
    if pinned_manifest_bytes is not None:
        if type(pinned_manifest_bytes) is not bytes:
            raise ABC6CampaignPreflightError(
                "pinned manifest bytes must be an exact bytes value"
            )
        raw = pinned_manifest_bytes
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ABC6CampaignPreflightError(
                "manifest must be a small regular file, not a symlink"
            )
    else:
        raw = _read_manifest_bytes_nofollow(manifest_path)
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

    root_fd = _open_matching_checkout_root(decoded["repository_root_realpath"])
    try:
        active_head = _current_git_head()
        if decoded.get("reviewed_git_head") != active_head:
            raise ABC6CampaignPreflightError(
                "active checkout Git HEAD differs from the manifest reviewed_git_head"
            )
        identity = _manifest_identity()
        if any(decoded.get(key) != value for key, value in identity.items()):
            raise ABC6CampaignPreflightError(
                "manifest protocol, report, training controls, or ordered roster mismatch"
            )
    except BaseException:
        os.close(root_fd)
        raise

    try:
        source_hashes = decoded.get("source_hashes")
        if not isinstance(source_hashes, dict):
            raise ABC6CampaignPreflightError("manifest lacks source_hashes mapping")
        current_sources = _current_source_hashes(root_fd=root_fd)
        if not _reviewed_source_map_is_complete():
            raise ABC6CampaignPreflightError(
                "static reviewed source pins remain incomplete"
            )
        if any(
            current_sources.get(name) != digest
            for name, digest in _REVIEWED_SOURCE_SHA256.items()
        ):
            raise ABC6CampaignPreflightError(
                "reviewed proposal, case roster, training seam, or baseline source changed"
            )
        if any(
            source_hashes.get(name) != digest
            for name, digest in current_sources.items()
        ):
            raise ABC6CampaignPreflightError("manifest source hash mismatch")
        runtime = decoded.get("runtime_fingerprint")
        if not isinstance(runtime, dict):
            raise ABC6CampaignPreflightError(
                "manifest lacks runtime_fingerprint object"
            )
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
        return actual_sha256, decoded, root_fd
    except BaseException:
        os.close(root_fd)
        raise


def _read_manifest_bytes_nofollow(
    manifest_path: str | os.PathLike[str],
) -> bytes:
    """Read one bounded regular manifest without a FIFO or symlink race."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise ABC6CampaignPreflightError(
            "platform cannot enforce no-follow nonblocking manifest reads"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            manifest_path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_MANIFEST_BYTES:
            raise ABC6CampaignPreflightError(
                "manifest must be a small regular file, not a symlink"
            )
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_MANIFEST_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_MANIFEST_BYTES:
                raise ABC6CampaignPreflightError(
                    "manifest exceeds its byte limit"
                )
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != after.st_size:
            raise ABC6CampaignPreflightError(
                "manifest identity changed while being read"
            )
        return raw
    except ABC6CampaignPreflightError:
        raise
    except OSError as error:
        raise ABC6CampaignPreflightError(
            "manifest is missing or unreadable"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory_at(directory_fd: int) -> None:
    info = os.fstat(directory_fd)
    if not stat.S_ISDIR(info.st_mode):
        raise OSError("durable publication parent is not a directory")
    os.fsync(directory_fd)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:  # pragma: no cover - guarded by the OS write contract.
            raise OSError("durable file write made no progress")
        remaining = remaining[count:]


def _write_exclusive_durable_at(
    directory_fd: int, filename: str, payload: bytes
) -> None:
    """Atomically publish immutable bytes through an anchored directory FD."""

    if (
        not isinstance(filename, str)
        or filename in {"", ".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("durable publication filename must be a leaf")
    temporary_name = (
        f".{filename}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    temporary_created = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        # Same-directory hard-link publication is atomic, descriptor-relative,
        # and fails if the final immutable receipt already exists.
        os.link(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _fsync_directory_at(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            else:
                _fsync_directory_at(directory_fd)


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


def _relative_file_parts_under_checkout(
    repository_root_realpath: str,
    file_path: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[str, str, str]:
    try:
        file_text = os.fspath(file_path)
    except TypeError as error:
        raise ABC6CampaignPreflightError(f"{label} must be an absolute path") from error
    if type(file_text) is not str:
        raise ABC6CampaignPreflightError(f"{label} must use a text path")
    path = Path(file_text)
    if not path.is_absolute() or str(path) != file_text:
        raise ABC6CampaignPreflightError(f"{label} must be canonical and absolute")
    root = Path(repository_root_realpath)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ABC6CampaignPreflightError(
            f"{label} must be inside the anchored checkout root"
        ) from error
    relative = _validate_canonical_relative_path(relative, label=label)
    parts = relative.split("/")
    if len(parts) < 2:
        raise ABC6CampaignPreflightError(f"{label} must include a parent directory")
    return "/".join(parts[:-1]), parts[-1], relative


def _write_case_status(
    receipt_directory_fd: int,
    receipt_directory_path: Path,
    case_index: int,
    component: str,
    status: str,
) -> tuple[Path, str]:
    if component not in _COMPONENTS:
        raise ValueError("component must be 'fit' or 'baseline'")
    if status not in _FINAL_STATUSES:
        raise ValueError("status must be a final receipt status")
    filename = f"case-{case_index:02d}.{component}-status.json"
    path = receipt_directory_path / filename
    payload = _status_receipt_payload(case_index, component, status)
    _write_exclusive_durable_at(receipt_directory_fd, filename, payload)
    return path, hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _claim_run(
    receipt_directory: Path,
    manifest_sha256: str,
    claim_registry_path: Path,
    *,
    receipt_directory_fd: int,
    receipt_root_identity: ABC6ReceiptRootIdentity,
    require_fixed_claim_path: bool,
    training_session: authority_module.ABC6TrainingSession | None = None,
) -> str:
    if receipt_directory != receipt_root_identity.receipt_root_path:
        raise ABC6CampaignPreflightError(
            "campaign receipt path differs from its anchored receipt root"
        )
    _check_preclaim_receipt_root(
        receipt_root_identity,
        receipt_directory_fd,
        training_session,
        scan="claim-entry",
        manifest_sha256=manifest_sha256,
    )
    claim_parent_relative, claim_filename, claim_relative = (
        _relative_file_parts_under_checkout(
            receipt_root_identity.repository_root_realpath,
            claim_registry_path,
            label="campaign global claim path",
        )
    )
    expected_claim_relative = (
        f"{CAMPAIGN_CLAIM_PARENT_RELATIVE}/{RUN_ID}.claim"
    )
    if require_fixed_claim_path and claim_relative != expected_claim_relative:
        raise ABC6CampaignPreflightError(
            "campaign global claim path differs from its fixed source-derived path"
        )

    receipt_root_identity.verify()
    repository_root_fd = receipt_root_identity.duplicate_repository_root_fd()
    try:
        claim_parent_fd = _open_relative_directory_nofollow(
            repository_root_fd,
            claim_parent_relative,
            label="campaign global claim parent",
            create=True,
        )
    finally:
        os.close(repository_root_fd)

    claim = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_sha256,
        # This path was checked against the frozen manifest join and opened
        # from the checkout-root descriptor; never resolve caller input here.
        "receipt_directory": str(receipt_directory),
        "claimed_at_utc": _utc_now(),
        "claim_semantics": "consumed_once_no_resume",
    }
    payload = _canonical_json(claim)
    claim_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        # Recheck the opened tree and receipt before the irreversible global
        # claim. Every subsequent write still uses the retained directory FDs.
        _check_preclaim_receipt_root(
            receipt_root_identity,
            receipt_directory_fd,
            training_session,
            scan="pre-durable-claim",
            manifest_sha256=manifest_sha256,
        )
        verification_root_fd = receipt_root_identity.duplicate_repository_root_fd()
        try:
            current_claim_parent_fd = _open_relative_directory_nofollow(
                verification_root_fd,
                claim_parent_relative,
                label="campaign global claim parent",
            )
            try:
                opened_info = os.fstat(claim_parent_fd)
                current_info = os.fstat(current_claim_parent_fd)
                if (opened_info.st_dev, opened_info.st_ino) != (
                    current_info.st_dev,
                    current_info.st_ino,
                ):
                    raise ABC6CampaignPreflightError(
                        "campaign global claim parent identity changed"
                    )
            finally:
                os.close(current_claim_parent_fd)
        finally:
            os.close(verification_root_fd)
        try:
            _write_exclusive_durable_at(
                claim_parent_fd,
                claim_filename,
                payload,
            )
        except FileExistsError as error:
            raise ABC6CampaignAlreadyClaimedError(
                "the fixed project-root registry already consumed this one-use run ID"
            ) from error
        except BaseException as error:
            try:
                _write_failure_receipt(
                    receipt_directory_fd,
                    manifest_sha256=manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=0,
                    case_index=None,
                    stop_reason="global_campaign_claim_publication_uncertain",
                    exception_type=type(error).__name__,
                )
            except BaseException as failure_error:
                raise ABC6CampaignExecutionError(
                    "global campaign claim publication is uncertain and no terminal "
                    "failure receipt could be durably published"
                ) from failure_error
            raise ABC6CampaignExecutionError(
                "global campaign claim publication is uncertain; run is terminal"
            ) from error
        # The registry claim above is authoritative. The local copy is written
        # through the already-open receipt directory, never through its path.
        try:
            _write_exclusive_durable_at(
                receipt_directory_fd, CLAIM_FILENAME, payload
            )
        except BaseException as error:
            try:
                _write_failure_receipt(
                    receipt_directory_fd,
                    manifest_sha256=manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=0,
                    case_index=None,
                    stop_reason="local_campaign_claim_publication_failed",
                    exception_type=type(error).__name__,
                )
            except BaseException as failure_error:
                raise ABC6CampaignExecutionError(
                    "global campaign claim is committed but local claim and terminal "
                    "failure receipt could not be durably published"
                ) from failure_error
            raise ABC6CampaignExecutionError(
                "global campaign claim is committed but local claim publication failed"
            ) from error
        return claim_sha256
    finally:
        os.close(claim_parent_fd)


def _write_failure_receipt(
    receipt_directory_fd: int,
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
    _write_exclusive_durable_at(
        receipt_directory_fd,
        FAILURE_FILENAME,
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
    receipt_directory_fd: int,
    receipt_directory_path: Path,
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
        filename = f"case-{case.case_index:02d}.{component}-status.json"
        try:
            os.stat(filename, dir_fd=receipt_directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            _write_case_status(
                receipt_directory_fd,
                receipt_directory_path,
                case.case_index,
                component,
                "failed",
            )
    _write_failure_receipt(
        receipt_directory_fd,
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
    receipt_directory_fd: int,
    receipt_directory_path: Path,
    *,
    campaign_result: ABC6CampaignResult,
    bundle: ABC6TrainingBundle,
    results: tuple[training.ABC6TrainingResult, ...],
) -> tuple[Path, str]:
    """Durably publish canonical bundle/case evidence and its terminal index."""

    _validate_execution_shape(campaign_result, bundle, results)
    try:
        os.mkdir(EVIDENCE_DIRECTORY_NAME, mode=0o700, dir_fd=receipt_directory_fd)
    except OSError as error:
        raise ABC6CampaignExecutionError(
            "durable training evidence directory could not be created safely"
        ) from error
    _fsync_directory_at(receipt_directory_fd)
    evidence_directory_fd = _open_evidence_directory_at(receipt_directory_fd)
    try:
        bundle_filename = "training-bundle.evidence.json"
        bundle_payload = _bundle_evidence_payload(bundle)
        _write_exclusive_durable_at(
            evidence_directory_fd, bundle_filename, bundle_payload
        )
        bundle_digest = _canonical_sha256(bundle_payload)

        case_entries: list[dict[str, object]] = []
        for case, status, result in zip(
            CASE_ROSTER, campaign_result.case_statuses, results, strict=True
        ):
            payload = _case_evidence_payload(case, result)
            filename = f"case-{case.case_index:02d}.training-evidence.json"
            _write_exclusive_durable_at(evidence_directory_fd, filename, payload)
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
        _write_exclusive_durable_at(
            receipt_directory_fd,
            EVIDENCE_MANIFEST_FILENAME,
            terminal_bytes,
        )
        manifest_path = receipt_directory_path / EVIDENCE_MANIFEST_FILENAME
        return manifest_path, _canonical_sha256(terminal_bytes)
    finally:
        os.close(evidence_directory_fd)


def _execute_campaign(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    require_reviewed_runtime: bool,
    claim_registry_path: Path,
    capture_training_evidence: bool = False,
    training_session: authority_module.ABC6TrainingSession | None = None,
    case_permits: tuple[authority_module.ABC6TrainingPermit, ...] | None = None,
) -> ABC6CampaignResult | ABC6TrainingCampaignExecution:
    """Preflight the manifest/path, then execute against its anchored root."""

    if require_reviewed_runtime and (
        training_session is None
        or case_permits is None
        or not capture_training_evidence
    ):
        raise ABC6CampaignPreflightError(
            "production campaign execution requires one supervised evidence session"
        )
    if training_session is not None and not capture_training_evidence:
        raise ABC6CampaignPreflightError(
            "supervised training authority cannot enter the status-only private path"
        )
    if training_session is None:
        if case_permits is not None:
            raise ABC6CampaignPreflightError(
                "case permits cannot be used without the supervised training session"
            )
    elif (
        type(case_permits) is not tuple
        or len(case_permits) != CASE_COUNT
        or any(
            type(permit) is not authority_module.ABC6TrainingPermit
            or permit.case_index != index
            or permit._authority is not training_session._authority
            for index, permit in enumerate(case_permits)
        )
    ):
        raise ABC6CampaignPreflightError(
            "supervised campaign requires the exact ordered 24 case permits"
        )

    verified_manifest_sha256, manifest, repository_root_fd = (
        _preflight_manifest_and_open_checkout_root(
            manifest_path,
            manifest_sha256,
            require_reviewed_runtime=require_reviewed_runtime,
        )
    )
    try:
        receipt_identity = _open_receipt_root_identity(
            repository_root_fd,
            str(manifest["repository_root_realpath"]),
            str(manifest["receipt_root_relative"]),
            receipt_directory,
            training_session=training_session,
            manifest_sha256=verified_manifest_sha256,
            source_hashes=tuple(sorted(dict(manifest["source_hashes"]).items())),
        )
    finally:
        os.close(repository_root_fd)

    transferred = False
    try:
        result = _execute_campaign_after_preflight(
            verified_manifest_sha256,
            receipt_identity,
            fit_callable=fit_callable,
            claim_registry_path=claim_registry_path,
            capture_training_evidence=capture_training_evidence,
            require_fixed_claim_path=require_reviewed_runtime,
            training_session=training_session,
            case_permits=case_permits,
        )
        if isinstance(result, ABC6TrainingCampaignExecution):
            transferred = True
        return result
    finally:
        if not transferred:
            receipt_identity.close()


def _execute_campaign_after_preflight(
    verified_manifest_sha256: str,
    receipt_identity: ABC6ReceiptRootIdentity,
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    claim_registry_path: Path,
    capture_training_evidence: bool,
    require_fixed_claim_path: bool,
    training_session: authority_module.ABC6TrainingSession | None = None,
    case_permits: tuple[authority_module.ABC6TrainingPermit, ...] | None = None,
) -> ABC6CampaignResult | ABC6TrainingCampaignExecution:
    """Claim once, then keep writes anchored to the opened receipt directory."""

    receipt_identity.verify()
    directory = receipt_identity.receipt_root_path
    receipt_directory_fd = receipt_identity.duplicate_receipt_root_fd_for_durable_write()
    try:
        _check_preclaim_receipt_root(
            receipt_identity,
            receipt_directory_fd,
            training_session,
            scan="execution-preclaim",
            manifest_sha256=verified_manifest_sha256,
        )
        claim_sha256 = _claim_run(
            directory,
            verified_manifest_sha256,
            claim_registry_path,
            receipt_directory_fd=receipt_directory_fd,
            receipt_root_identity=receipt_identity,
            require_fixed_claim_path=require_fixed_claim_path,
            training_session=training_session,
        )
        try:
            receipt_identity.verify()
        except BaseException as error:
            try:
                _write_failure_receipt(
                    receipt_directory_fd,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=0,
                    case_index=None,
                    stop_reason="campaign_path_identity_changed_after_claim",
                    exception_type=type(error).__name__,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "campaign path changed after the one-use claim and terminal "
                    "failure receipt could not be sealed"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                "campaign path changed after the one-use claim; run is terminal"
            ) from error
        return _execute_campaign_after_claim(
            verified_manifest_sha256,
            claim_sha256,
            receipt_identity,
            receipt_directory_fd,
            directory,
            fit_callable=fit_callable,
            capture_training_evidence=capture_training_evidence,
            training_session=training_session,
            case_permits=case_permits,
        )
    finally:
        os.close(receipt_directory_fd)


def _execute_campaign_after_claim(
    verified_manifest_sha256: str,
    claim_sha256: str,
    receipt_root_identity: ABC6ReceiptRootIdentity,
    receipt_directory_fd: int,
    directory: Path,
    *,
    fit_callable: Callable[[ABC6TrainingCaseData], object],
    capture_training_evidence: bool,
    training_session: authority_module.ABC6TrainingSession | None = None,
    case_permits: tuple[authority_module.ABC6TrainingPermit, ...] | None = None,
) -> ABC6CampaignResult | ABC6TrainingCampaignExecution:

    try:
        bundle = build_synthetic_training_bundle()
        if training_session is not None:
            if case_permits is None:
                raise ABC6CampaignPreflightError(
                    "supervised campaign has no preissued case permits"
                )
            cases._bind_training_bundle_to_authority(
                bundle, training_session._authority
            )
    except BaseException as error:
        _write_failure_receipt(
            receipt_directory_fd,
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
    try:
        receipt_root_identity.verify()
    except BaseException as error:
        try:
            _write_failure_receipt(
                receipt_directory_fd,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=0,
                case_index=None,
                stop_reason="campaign_path_identity_changed_during_training_data_build",
                exception_type=type(error).__name__,
            )
        except BaseException as receipt_error:
            raise ABC6CampaignExecutionError(
                "campaign path changed after claim and terminal failure receipt "
                "could not be sealed"
            ) from receipt_error
        raise ABC6CampaignExecutionError(
            "campaign path changed after claim during training-data construction"
        ) from error

    case_statuses: list[ABC6CaseStatus] = []
    training_results: list[training.ABC6TrainingResult] | None = (
        [] if capture_training_evidence else None
    )
    for expected_index, expected_case in enumerate(CASE_ROSTER):
        try:
            receipt_root_identity.verify()
        except BaseException as error:
            try:
                _record_failed_case(
                    receipt_directory_fd,
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="campaign_path_identity_changed_before_case",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "campaign path changed after claim and current case failure "
                    "could not be sealed"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"campaign path changed before case {expected_index}; run is terminal"
            ) from error
        # Never trust a consumer-supplied order: verify the frozen view and then
        # retrieve it again by its exact roster index.
        try:
            if expected_case != case_by_index(expected_index):
                raise ValueError("runtime case roster changed after preflight")
            if training_session is None:
                data = bundle.data_for_case(expected_index)
                if data.case != expected_case:
                    raise ValueError("training bundle view identity differs from roster")
            else:
                if case_permits is None:
                    raise ValueError("supervised campaign has no case permit roster")
                data = cases.get_training_case_data(
                    bundle, expected_index, case_permits[expected_index]
                )
                if (
                    data._authority is not training_session._authority
                    or data._case_index != expected_index
                    or data._bundle is not bundle
                    or data._case_data is not bundle.case_data[expected_index]
                    or data._case_data.case is not expected_case
                ):
                    raise ValueError(
                        "authorized training case differs from its exact grant, bundle, or roster"
                    )
        except BaseException as error:
            try:
                _record_failed_case(
                    receipt_directory_fd,
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="training_case_view_identity_mismatch",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "case view identity failed and terminal failure receipt could not "
                    "be sealed"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"case {expected_index} identity mismatch; run is terminal"
            ) from error
        try:
            result = fit_callable(data)
        except BaseException as error:
            try:
                _record_failed_case(
                    receipt_directory_fd,
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
            receipt_root_identity.verify()
        except BaseException as error:
            try:
                _record_failed_case(
                    receipt_directory_fd,
                    directory,
                    case=expected_case,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(case_statuses),
                    stop_reason="campaign_path_identity_changed_during_case",
                    error=error,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "campaign path changed during a case and terminal failure "
                    "receipt could not be sealed"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                f"campaign path changed during case {expected_index}; run is terminal"
            ) from error
        try:
            fit_status, baseline_status = _result_statuses(result, expected_case)
        except BaseException as error:
            try:
                _record_failed_case(
                    receipt_directory_fd,
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
                receipt_directory_fd,
                directory,
                expected_index,
                "fit",
                fit_status,
            )
            baseline_path, baseline_receipt_sha256 = _write_case_status(
                receipt_directory_fd,
                directory,
                expected_index,
                "baseline",
                baseline_status,
            )
        except BaseException as error:
            try:
                _write_failure_receipt(
                    receipt_directory_fd,
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
            training_results.append(cast(training.ABC6TrainingResult, result))

    statuses = tuple(case_statuses)
    summary_bytes = _summary_payload(
        verified_manifest_sha256,
        claim_sha256,
        statuses,
    )
    summary_path = directory / SUMMARY_FILENAME
    try:
        receipt_root_identity.verify()
        _write_exclusive_durable_at(
            receipt_directory_fd,
            SUMMARY_FILENAME,
            summary_bytes,
        )
    except OSError as error:
        try:
            _write_failure_receipt(
                receipt_directory_fd,
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
    except BaseException as error:
        try:
            _write_failure_receipt(
                receipt_directory_fd,
                manifest_sha256=verified_manifest_sha256,
                claim_sha256=claim_sha256,
                completed_cases=len(statuses),
                case_index=None,
                stop_reason="campaign_path_identity_changed_before_summary",
                exception_type=type(error).__name__,
            )
        except BaseException as receipt_error:
            raise ABC6CampaignExecutionError(
                "campaign path changed before summary and terminal failure receipt "
                "could not be sealed"
            ) from receipt_error
        raise ABC6CampaignExecutionError(
            "campaign path changed before summary; run is terminal"
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
        try:
            receipt_root_identity.verify()
        except BaseException as error:
            try:
                _write_failure_receipt(
                    receipt_directory_fd,
                    manifest_sha256=verified_manifest_sha256,
                    claim_sha256=claim_sha256,
                    completed_cases=len(statuses),
                    case_index=None,
                    stop_reason="campaign_path_identity_changed_after_summary",
                    exception_type=type(error).__name__,
                )
            except BaseException as receipt_error:
                raise ABC6CampaignExecutionError(
                    "campaign path changed after summary and terminal failure "
                    "receipt could not be sealed"
                ) from receipt_error
            raise ABC6CampaignExecutionError(
                "campaign path changed after summary; run is terminal"
            ) from error
        return campaign_result
    detailed_results = tuple(training_results)
    try:
        evidence_manifest_path, evidence_manifest_sha256 = _publish_training_evidence(
            receipt_directory_fd,
            directory,
            campaign_result=campaign_result,
            bundle=bundle,
            results=detailed_results,
        )
    except BaseException as error:
        try:
            _write_failure_receipt(
                receipt_directory_fd,
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
        receipt_root_identity=receipt_root_identity,
        _training_bundle=bundle,
        _training_results=detailed_results,
        _authority=(
            None if training_session is None else training_session._authority
        ),
        _grant_sha256=(
            None
            if training_session is None
            else training_session._authority.grant_digest
        ),
        _run_id=campaign_result.run_id,
        _manifest_sha256=campaign_result.manifest_sha256,
        _seal=_TRAINING_EXECUTION_SEAL,
    )
    try:
        _verify_training_execution_evidence(execution)
    except BaseException as error:
        try:
            _write_failure_receipt(
                receipt_directory_fd,
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
    *,
    launch_authority: authority_module.ABC6LaunchAuthority,
) -> ABC6CampaignResult:
    """Return compact status from the same supervised evidence entrypoint."""

    execution = run_abc6_training_campaign_with_evidence(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        launch_authority=launch_authority,
    )
    return execution.campaign_result


def run_abc6_training_campaign_with_evidence(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    launch_authority: authority_module.ABC6LaunchAuthority,
) -> ABC6TrainingCampaignExecution:
    """Run the claimed 24-case campaign and return its exact in-memory evidence.

    This follows the same manifest preflight, one-use claim, bundle build,
    ordered fits, durable status receipts, and terminal-summary path as
    :func:`run_abc6_training_campaign`.  It returns only after every receipt
    and the summary have been durably published.  A raised case or publication
    failure remains terminal and returns no execution object.
    """

    if type(launch_authority) is not authority_module.ABC6LaunchAuthority:
        raise ABC6CampaignPreflightError(
            "supervised campaign requires the received launch authority"
        )
    try:
        # This is deliberately the first supervised action in the public
        # entrypoint: begin_training() checks this exact role/source/function.
        training_session = launch_authority.begin_training()
        # Issuance is deliberately in this exact frozen role frame and happens
        # before preflight or claim, making every failed attempt terminal.
        permits_list: list[authority_module.ABC6TrainingPermit] = []
        for index in range(CASE_COUNT):
            permits_list.append(training_session.issue_case_permit(index))
        case_permits = tuple(permits_list)
    except authority_module.ABC6AuthorityError as error:
        raise ABC6CampaignPreflightError(
            f"supervised campaign training authority was rejected: {error}"
        ) from error

    result = _execute_campaign(
        manifest_path,
        manifest_sha256,
        receipt_directory,
        fit_callable=training.run_abc6_training_case,
        require_reviewed_runtime=True,
        claim_registry_path=_PROJECT_RUN_CLAIM_PATH,
        capture_training_evidence=True,
        training_session=training_session,
        case_permits=case_permits,
    )
    if not isinstance(result, ABC6TrainingCampaignExecution):  # pragma: no cover.
        raise AssertionError("evidence campaign did not return its training evidence")
    try:
        launch_authority.register_training_execution(result)
    except authority_module.ABC6AuthorityError as error:
        result.close()
        raise ABC6CampaignExecutionError(
            f"completed training execution could not be registered; run is terminal: {error}"
        ) from error
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
    "ABC6ReceiptRootIdentity",
    "ABC6VerifiedTrainingEvidence",
    "ABC6TrainingCampaignExecution",
    "preflight_abc6_campaign_manifest",
    "run_abc6_training_campaign",
    "run_abc6_training_campaign_with_evidence",
]

"""Read-only verifier for already-persisted ABC6 deferred-score evidence.

This module never opens the campaign target gate and never calls a simulator.
It accepts only an explicitly supplied frozen identity and its live
campaign-fit receipt-root descriptors.  The current production manifest does
not yet freeze the integrated runner/forecast/scorer source hashes, so callers
must supply those hashes through the trusted identity object; the module is
not wired into a production entrypoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from core.real_data import (
    cascaded_tanks_abc6_authority as grant_authority,
    cascaded_tanks_abc6_campaign_fit as campaign_fit,
    cascaded_tanks_abc6_cases as cases,
    cascaded_tanks_abc6_receipt_io as receipt_io,
    cascaded_tanks_abc6_scoring as scoring,
)

MAX_JSON_BYTES: Final = 128 * 1024 * 1024
MAX_STATUS_BYTES: Final = receipt_io.MAX_STATUS_RECEIPT_BYTES
MAX_CLAIM_BYTES: Final = 64 * 1024
MAX_TRAINING_SUMMARY_BYTES: Final = 4 * 1024 * 1024
MAX_EVIDENCE_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_EVIDENCE_FILE_BYTES: Final = 32 * 1024 * 1024
MAX_EVIDENCE_TOTAL_BYTES: Final = 1024 * 1024 * 1024
MAX_MARKER_BYTES: Final = cases._MAX_REVEAL_MARKER_BYTES
TERMINAL_FILENAME: Final = "watchdog-terminal-receipt.json"
TERMINAL_ACK_FILENAME: Final = "terminal-readback.ok"
GRANT_RECORD_FILENAME: Final = "watchdog-child-grant.json"
_FORECAST_FILENAME: Final = "campaign.target-free-forecasts.json"
_BOOTSTRAP_GRANT_FIELDS: Final = frozenset(
    {
        "transport",
        "environment_locator",
        "descriptor_fd",
        "grant_sha256",
        "grant_record_sha256",
        "delivery_status",
        "digest_receipt_leaf",
    }
)
_INTEGRATED_SOURCE_PATHS: Final = (
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
)
_SOURCE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_WATCHDOG_INTERVENTION_REASONS: Final = {
    # The reviewed watchdog records the observed stop_reason as stage.
    "budget_cap": frozenset(
        {"sampled_rss_limit_exceeded", "wall_clock_limit_exceeded"}
    ),
    "operator_stop": frozenset({"watchdog_interrupted"}),
    "watchdog_stop": frozenset(
        {
            "child_exited_with_live_process_group_members",
            "process_group_membership_unknown",
            "unexpected_descendant_process",
            "watchdog_identity_or_monitoring_error",
            "watchdog_monitoring_error",
            "child_exited_process_group_reap_unverified",
            "child_runtime_attestation_timeout",
        }
    ),
}

ReplayClassification = Literal[
    "premarker_absent",
    "failed",
    "incomplete",
    "complete",
    "unreplayable",
]

_STAGE_B1_SCORER_CLAIM_PARENT_RELATIVE: Final = (
    "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims"
)
_STAGE_B1_MAX_LAUNCH_ATTESTATION_BYTES: Final = 64 * 1024
_STAGE_B1_WATCHDOG_CLAIM_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "approval_record_sha256",
        "reviewed_git_head",
        "repository_root_realpath",
        "receipt_root_relative",
        "repository_root_device",
        "repository_root_inode",
        "receipt_root_device",
        "receipt_root_inode",
        "watchdog_claim_parent_device",
        "watchdog_claim_parent_inode",
        "scorer_claim_parent_device",
        "scorer_claim_parent_inode",
        "claimed_at_utc",
        "declared_child_launch_vector",
        "declared_child_executable_path",
        "declared_child_executable_resolved_path",
        "declared_child_executable_sha256",
        "accepted_observed_child_argv0",
        "expected_observed_child_argv_tail",
        "expected_observed_child_image_path",
        "expected_observed_child_image_sha256",
        "watchdog_attestation",
        "claim_semantics",
    }
)
_STAGE_B1_WATCHDOG_ATTESTATION_KEYS: Final = frozenset(
    {
        "declared_launch_vector",
        "declared_python_bin",
        "declared_python_bin_resolved_path",
        "declared_python_bin_sha256",
        "observed_live_argv",
        "observed_python_app_image_path",
        "observed_python_app_image_sha256",
        "python_version",
        "psutil_version",
        "psutil_module_path",
        "psutil_module_sha256",
        "original_exec_alias_attestable_from_process_apis",
    }
)
_STAGE_B1_LAUNCH_ATTESTATION_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "approval_sha256",
        "review_sha256",
        "reviewed_git_head",
        "watchdog_claim_sha256",
        "physical_root",
        "root_device",
        "root_inode",
        "receipt_root_relative",
        "receipt_device",
        "receipt_inode",
        "claim_parent_device",
        "claim_parent_inode",
        "observed_monotonic_ns",
        "watchdog_process",
        "child_process",
    }
)
_STAGE_B1_PROCESS_KEYS: Final = frozenset(
    {"pid", "start_identity", "argv", "image_path", "image_sha256"}
)
_STAGE_B1_CHILD_PROCESS_KEYS: Final = _STAGE_B1_PROCESS_KEYS | frozenset(
    {
        "declared_launch_vector",
        "declared_launch_image_path",
        "declared_launch_image_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class ABC6StageB1WatchdogRuntimeExpectation:
    """Independently declared fake-fixture values for the watchdog runtime."""

    declared_launch_vector: tuple[str, ...]
    declared_python_bin: str
    declared_python_bin_resolved_path: str
    declared_python_bin_sha256: str
    observed_live_argv: tuple[str, ...]
    observed_python_app_image_path: str
    observed_python_app_image_sha256: str
    python_version: str
    psutil_version: str
    psutil_module_path: str
    psutil_module_sha256: str


@dataclass(frozen=True, slots=True)
class ABC6StageB1ProcessExpectation:
    """Fixture process identity, supplied only when captured independently."""

    pid: int | None = None
    start_identity: str | None = None


@dataclass(frozen=True, slots=True)
class ABC6StageB1Expectation:
    """Typed, independent fixture bindings for the isolated Stage B1 helper.

    This deliberately has no production builder and is not consumed by the
    public replay entrypoint. Tests must construct it from fixture manifest,
    approval, root anchors and separately declared launch/runtime facts.
    """

    protocol_id: str
    run_id: str
    manifest_sha256: str
    approval_record_sha256: str
    review_sha256: str
    reviewed_git_head: str
    watchdog_claim_sha256: str
    repository_root_realpath: str
    repository_root_device: int
    repository_root_inode: int
    receipt_root_relative: str
    receipt_root_device: int
    receipt_root_inode: int
    watchdog_claim_parent_device: int
    watchdog_claim_parent_inode: int
    scorer_claim_parent_device: int
    scorer_claim_parent_inode: int
    child_declared_launch_vector: tuple[str, ...]
    child_declared_executable_path: str
    child_declared_executable_resolved_path: str
    child_declared_executable_sha256: str
    child_accepted_observed_argv0: tuple[str, ...]
    child_expected_observed_argv_tail: tuple[str, ...]
    child_expected_observed_image_path: str
    child_expected_observed_image_sha256: str
    watchdog_runtime: ABC6StageB1WatchdogRuntimeExpectation
    watchdog_process: ABC6StageB1ProcessExpectation | None = None
    child_process: ABC6StageB1ProcessExpectation | None = None


@dataclass(frozen=True, slots=True)
class ABC6StageB1Verification:
    """Structural link result; it never classifies a campaign outcome."""

    status: Literal["evidence_linked", "unreplayable"]
    detail: str
    claim_sha256: str | None = None
    launch_attestation_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ABC6FrozenReplayIdentity:
    """Trusted frozen IDs and the already-open physical checkout/receipt root.

    ``integrated_source_hashes`` must come from an independently frozen input,
    not from the score receipt being checked.  The current production campaign
    manifest does not yet contain this three-file allowlist, which keeps a
    production caller from constructing a valid identity until that separate
    review and source-pin migration are complete. When a child-grant record is
    present, ``expected_child_grant_bindings`` must also be independently
    constructed from the frozen manifest/review/approval, pinned root, verified
    watchdog claim and separately captured child-process attestation. It is
    never inferred from the terminal, grant record or score receipt.
    """

    receipt_root_identity: campaign_fit.ABC6ReceiptRootIdentity = field(
        repr=False, compare=False
    )
    protocol_id: str
    run_id: str
    manifest_sha256: str
    approval_record_sha256: str
    reviewed_git_head: str
    watchdog_claim_sha256: str
    integrated_source_hashes: tuple[tuple[str, str], ...]
    expected_child_grant_bindings: grant_authority.ABC6AuthorityBindings | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.receipt_root_identity, campaign_fit.ABC6ReceiptRootIdentity
        ):
            raise TypeError("receipt_root_identity must be an anchored campaign identity")
        for name, value in (
            ("protocol_id", self.protocol_id),
            ("run_id", self.run_id),
        ):
            if type(value) is not str or not value or value in {".", ".."}:
                raise ValueError(f"{name} must be nonempty text")
            if Path(value).name != value or "/" in value or "\\" in value:
                raise ValueError(f"{name} must be one safe path component")
        for name, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("approval_record_sha256", self.approval_record_sha256),
            ("watchdog_claim_sha256", self.watchdog_claim_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if (
            type(self.reviewed_git_head) is not str
            or len(self.reviewed_git_head) != 40
            or any(char not in "0123456789abcdef" for char in self.reviewed_git_head)
        ):
            raise ValueError("reviewed_git_head must be a lowercase Git commit ID")
        if (
            not isinstance(self.integrated_source_hashes, tuple)
            or tuple(path for path, _digest in self.integrated_source_hashes)
            != _INTEGRATED_SOURCE_PATHS
            or any(not _is_sha256(digest) for _path, digest in self.integrated_source_hashes)
        ):
            raise ValueError(
                "integrated source hashes must cover the frozen runner/forecast/scorer order"
            )
        if self.expected_child_grant_bindings is not None and not isinstance(
            self.expected_child_grant_bindings,
            grant_authority.ABC6AuthorityBindings,
        ):
            raise TypeError(
                "expected_child_grant_bindings must be a separately supplied typed authority binding"
            )


@dataclass(frozen=True, slots=True)
class ABC6ReplayGate:
    classification: ReplayClassification
    marker_present: bool
    score_receipt_sha256: str | None = None
    score_outcome: Literal["complete", "failed"] | None = None
    score_event_stage: str | None = None
    score_event_monotonic_ns: int | None = None
    score_event_utc: str | None = None
    score_event_clock: str | None = None
    failure_stage: str | None = None
    detail: str | None = None
    terminal_status: str | None = None
    watchdog_intervention_type: str | None = None
    watchdog_intervention_stage: str | None = None
    watchdog_intervention_monotonic_ns: int | None = None
    watchdog_intervention_utc: str | None = None
    watchdog_intervention_clock: str | None = None
    watchdog_intervention_capture_status: str | None = None
    watchdog_intervention_attempted_type: str | None = None
    watchdog_intervention_attempted_stage: str | None = None
    watchdog_intervention_failure_kind: str | None = None
    status_receipt_count: int = 0
    all_statuses_complete: bool | None = None
    scientifically_ready: bool | None = None
    verified_target_hashes: tuple[tuple[str, str], ...] = ()


class _Reject(ValueError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, info: os.stat_result) -> "_FileIdentity":
        return cls(
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class _LeafPin:
    parent: cases._DirectoryAnchor = field(repr=False, compare=False)
    filename: str
    maximum_bytes: int
    identity: _FileIdentity
    sha256: str


@dataclass(frozen=True, slots=True)
class _AbsencePin:
    parent: cases._DirectoryAnchor = field(repr=False, compare=False)
    filename: str


class _ReadSession:
    """Own anchored directories and revalidate every referenced leaf at exit."""

    def __init__(self, frozen: ABC6FrozenReplayIdentity):
        self.frozen = frozen
        self.anchors: list[cases._DirectoryAnchor] = []
        self.pins: list[_LeafPin] = []
        self.absence_pins: list[_AbsencePin] = []
        self.directory_entry_pins: list[
            tuple[cases._DirectoryAnchor, tuple[str, ...]]
        ] = []
        identity = frozen.receipt_root_identity
        try:
            identity.verify()
        except Exception as error:
            raise _Reject(
                "receipt_root_identity", "pinned campaign receipt root is no longer valid"
            ) from error
        if (
            identity.receipt_root_relative != campaign_fit.RECEIPT_ROOT_RELATIVE
            or identity.repository_root_realpath != str(Path(identity.repository_root_realpath))
            or not Path(identity.repository_root_realpath).is_absolute()
            or frozen.protocol_id != campaign_fit.PROTOCOL_ID
            or frozen.protocol_id != cases.PROTOCOL_ID
            or frozen.protocol_id != scoring.PROTOCOL_ID
            or frozen.run_id != campaign_fit.RUN_ID
            or frozen.run_id != cases.RUN_ID
            or frozen.run_id != scoring.RUN_ID
        ):
            raise _Reject("frozen_identity", "frozen campaign identity differs from source contract")
        if (
            frozen.receipt_root_identity.repository_root_device
            != os.fstat(identity._repository_root_fd).st_dev
            or frozen.receipt_root_identity.repository_root_inode
            != os.fstat(identity._repository_root_fd).st_ino
            or frozen.receipt_root_identity.receipt_root_device
            != os.fstat(identity._receipt_root_fd).st_dev
            or frozen.receipt_root_identity.receipt_root_inode
            != os.fstat(identity._receipt_root_fd).st_ino
        ):
            raise _Reject("frozen_identity", "frozen directory identity differs from retained descriptors")
        self.root, self.receipt = cases._open_execution_directory_anchors(identity)
        self.anchors.extend((self.root, self.receipt))
        self.campaign_claim_parent = self.open_directory(
            campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE,
            label="campaign claim parent",
        )
        self.scorer_claim_parent = self.open_optional_directory(
            "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims",
            label="scorer claim parent",
        )

    def open_directory(
        self,
        relative: str,
        *,
        label: str,
        base: cases._DirectoryAnchor | None = None,
    ) -> cases._DirectoryAnchor:
        try:
            anchor = cases._open_relative_directory_anchor(
                self.root if base is None else base, relative, label=label
            )
        except Exception as error:
            raise _Reject(label.replace(" ", "_"), f"{label} is missing or unsafe") from error
        self.anchors.append(anchor)
        return anchor

    def open_optional_directory(
        self, relative: str, *, label: str
    ) -> cases._DirectoryAnchor | None:
        """Open an optional directory below the pinned root without following links."""

        try:
            parts = cases._validate_relative_directory_path(relative, label=label)
            current_fd = os.dup(self.root.descriptor)
            for component in parts:
                try:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                except FileNotFoundError:
                    os.close(current_fd)
                    return None
                os.close(current_fd)
                current_fd = next_fd
            anchor = cases._DirectoryAnchor(
                self.root.path.joinpath(*parts), current_fd
            )
            self.anchors.append(anchor)
            return anchor
        except _Reject:
            raise
        except Exception as error:
            if "current_fd" in locals():
                try:
                    os.close(current_fd)
                except OSError:
                    pass
            raise _Reject(
                label.replace(" ", "_"), f"{label} is unsafe"
            ) from error

    def read(
        self,
        parent: cases._DirectoryAnchor,
        filename: str,
        *,
        maximum_bytes: int,
        label: str,
        stage: str,
        pin: bool = True,
    ) -> tuple[bytes, dict[str, object], _FileIdentity]:
        if filename in {"", ".", ".."} or Path(filename).name != filename:
            raise _Reject(stage, f"{label} is not a leaf file")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                filename,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent.descriptor,
            )
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise _Reject(stage, f"{label} is not a bounded regular file")
            parts: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
                if not chunk:
                    break
                parts.append(chunk)
                size += len(chunk)
                if size > maximum_bytes:
                    raise _Reject(stage, f"{label} exceeds its byte limit")
            after = os.fstat(descriptor)
            named = os.stat(filename, dir_fd=parent.descriptor, follow_symlinks=False)
        except _Reject:
            raise
        except FileNotFoundError as error:
            raise _Reject(stage, f"{label} is missing") from error
        except OSError as error:
            raise _Reject(stage, f"{label} is missing or unsafe") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        before_identity = _FileIdentity.from_stat(before)
        if (
            before_identity != _FileIdentity.from_stat(after)
            or before_identity != _FileIdentity.from_stat(named)
            or size != before.st_size
        ):
            raise _Reject(stage, f"{label} changed while being read")
        raw = b"".join(parts)
        digest = hashlib.sha256(raw).hexdigest()
        if pin:
            self.pins.append(
                _LeafPin(parent, filename, maximum_bytes, before_identity, digest)
            )
        decoded = _decode_canonical(raw, label=label, stage=stage)
        return raw, decoded, before_identity

    def maybe_present(self, parent: cases._DirectoryAnchor, filename: str, *, stage: str) -> bool:
        try:
            os.stat(filename, dir_fd=parent.descriptor, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False
        except OSError as error:
            raise _Reject(stage, "cannot inspect the fixed evidence leaf") from error

    def require_absent(
        self,
        parent: cases._DirectoryAnchor,
        filename: str,
        *,
        label: str,
        stage: str,
    ) -> None:
        """Pin an anchored no-follow absence for final namespace revalidation."""

        if filename in {"", ".", ".."} or Path(filename).name != filename:
            raise _Reject(stage, f"{label} is not a leaf file")
        try:
            os.stat(filename, dir_fd=parent.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            self.absence_pins.append(_AbsencePin(parent, filename))
            return
        except OSError as error:
            raise _Reject(stage, f"cannot verify absence of {label}") from error
        raise _Reject(stage, f"{label} unexpectedly exists")

    def require_directory_entries(
        self,
        parent: cases._DirectoryAnchor,
        *,
        allowed: set[str],
        required: set[str],
        label: str,
        stage: str,
    ) -> None:
        """Reject unrecognized directory entries and recheck the exact listing."""

        try:
            entries = tuple(sorted(os.listdir(parent.descriptor)))
        except OSError as error:
            raise _Reject(stage, f"cannot enumerate {label}") from error
        if not required.issubset(entries) or not set(entries).issubset(allowed):
            raise _Reject(stage, f"{label} contains missing or unexpected artifacts")
        self.directory_entry_pins.append((parent, entries))

    def verify_stable(self) -> None:
        try:
            self.frozen.receipt_root_identity.verify()
            # A directory sync is part of the frozen replay contract.  It does
            # not publish or modify evidence; it makes the pinned namespace
            # state observable before accepting the readback chain.
            os.fsync(self.receipt.descriptor)
            for anchor in self.anchors:
                cases._verify_directory_anchor_path(
                    anchor, label="ABC6 replay evidence directory"
                )
            for pin in self.pins:
                raw, _decoded, identity = self.read(
                    pin.parent,
                    pin.filename,
                    maximum_bytes=pin.maximum_bytes,
                    label="replay evidence leaf revalidation",
                    stage="final_identity_revalidation",
                    pin=False,
                )
                if (
                    hashlib.sha256(raw).hexdigest() != pin.sha256
                    or identity != pin.identity
                ):
                    raise _Reject(
                        "final_identity_revalidation",
                        f"{pin.filename} changed after its initial verification",
                    )
            for pin in self.absence_pins:
                try:
                    os.stat(
                        pin.filename,
                        dir_fd=pin.parent.descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise _Reject(
                        "final_identity_revalidation",
                        f"absence of {pin.filename} could not be revalidated",
                    ) from error
                raise _Reject(
                    "final_identity_revalidation",
                    f"{pin.filename} appeared during replay",
                )
            for parent, expected_entries in self.directory_entry_pins:
                try:
                    entries = tuple(sorted(os.listdir(parent.descriptor)))
                except OSError as error:
                    raise _Reject(
                        "final_identity_revalidation",
                        "anchored evidence directory changed during replay",
                    ) from error
                if entries != expected_entries:
                    raise _Reject(
                        "final_identity_revalidation",
                        "anchored evidence directory entries changed during replay",
                    )
            self.frozen.receipt_root_identity.verify()
        except _Reject:
            raise
        except Exception as error:
            raise _Reject(
                "final_identity_revalidation", "anchored evidence changed during replay"
            ) from error

    def close(self) -> None:
        for anchor in reversed(self.anchors):
            try:
                anchor.close()
            except OSError:
                pass
        self.anchors.clear()


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA_RE.fullmatch(value) is not None


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("value cannot be encoded as canonical finite JSON") from error


def _reject_duplicate_keys(pairs):
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"duplicate JSON key: {key}")
        decoded[key] = value
    return decoded


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _check_finite(value: object, *, label: str, stage: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _Reject(stage, f"{label} contains a non-finite JSON number")
    if isinstance(value, dict):
        for nested in value.values():
            _check_finite(nested, label=label, stage=stage)
    elif isinstance(value, list):
        for nested in value:
            _check_finite(nested, label=label, stage=stage)


def _decode_canonical(raw: bytes, *, label: str, stage: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if type(decoded) is not dict:
            raise ValueError("JSON value is not an object")
        _check_finite(decoded, label=label, stage=stage)
        if _canonical_json(decoded) != raw:
            raise ValueError("JSON object is not canonical")
        return decoded
    except _Reject:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as error:
        raise _Reject(stage, f"{label} is not canonical finite ASCII JSON") from error


def _payload_hash_valid(value: dict[str, object], *, stage: str) -> None:
    payload_digest = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not _is_sha256(payload_digest) or hashlib.sha256(_canonical_json(body)).hexdigest() != payload_digest:
        raise _Reject(stage, "canonical payload digest is invalid")


def _open_presence(ctx: _ReadSession, parent: cases._DirectoryAnchor, name: str) -> bool:
    return ctx.maybe_present(parent, name, stage="evidence_presence")


def _verify_claims(
    ctx: _ReadSession,
) -> tuple[str, dict[str, object], dict[str, object]]:
    frozen = ctx.frozen
    receipt_raw, receipt_claim, _ = ctx.read(
        ctx.receipt, campaign_fit.CLAIM_FILENAME,
        maximum_bytes=MAX_CLAIM_BYTES, label="campaign receipt claim", stage="campaign_claim",
    )
    expected_campaign_claim = {
        "schema_version", "protocol_id", "run_id", "manifest_sha256",
        "receipt_directory", "claimed_at_utc", "claim_semantics",
    }
    if (
        set(receipt_claim) != expected_campaign_claim
        or receipt_claim.get("schema_version") != 1
        or receipt_claim.get("protocol_id") != frozen.protocol_id
        or receipt_claim.get("run_id") != frozen.run_id
        or receipt_claim.get("manifest_sha256") != frozen.manifest_sha256
        or receipt_claim.get("receipt_directory") != str(ctx.receipt.path)
        or receipt_claim.get("claim_semantics") != "consumed_once_no_resume"
        or type(receipt_claim.get("claimed_at_utc")) is not str
    ):
        raise _Reject("campaign_claim", "campaign receipt claim identity is invalid")
    campaign_claim_sha = hashlib.sha256(receipt_raw).hexdigest()

    global_raw, global_claim, _ = ctx.read(
        ctx.campaign_claim_parent,
        f"{frozen.run_id}.claim",
        maximum_bytes=MAX_CLAIM_BYTES,
        label="global campaign claim",
        stage="global_campaign_claim",
    )
    if global_raw != receipt_raw or global_claim != receipt_claim:
        raise _Reject("global_campaign_claim", "global and receipt campaign claims differ")

    _watchdog_sha, watchdog_claim = _verify_watchdog_claim(ctx)
    return campaign_claim_sha, receipt_claim, watchdog_claim


def _stage_b1_expected_attestation(
    runtime: ABC6StageB1WatchdogRuntimeExpectation,
) -> dict[str, object]:
    if type(runtime) is not ABC6StageB1WatchdogRuntimeExpectation:
        raise _Reject("stage_b1_expectation", "watchdog runtime expectation is unavailable")
    scalar_text = (
        runtime.declared_python_bin,
        runtime.declared_python_bin_resolved_path,
        runtime.observed_python_app_image_path,
        runtime.python_version,
        runtime.psutil_version,
        runtime.psutil_module_path,
    )
    if any(type(value) is not str or not value for value in scalar_text):
        raise _Reject("stage_b1_expectation", "watchdog runtime text is incomplete")
    for value in (runtime.declared_python_bin, runtime.declared_python_bin_resolved_path,
                  runtime.observed_python_app_image_path, runtime.psutil_module_path):
        if not _stage_b1_absolute_path(value):
            raise _Reject("stage_b1_expectation", "watchdog runtime path is not canonical")
    if not _is_sha256(runtime.declared_python_bin_sha256) or not _is_sha256(
        runtime.observed_python_app_image_sha256
    ) or not _is_sha256(runtime.psutil_module_sha256):
        raise _Reject("stage_b1_expectation", "watchdog runtime digest is invalid")
    for vector in (runtime.declared_launch_vector, runtime.observed_live_argv):
        if not _stage_b1_expected_vector(vector):
            raise _Reject("stage_b1_expectation", "watchdog runtime argv is incomplete")
    return {
        "declared_launch_vector": list(runtime.declared_launch_vector),
        "declared_python_bin": runtime.declared_python_bin,
        "declared_python_bin_resolved_path": runtime.declared_python_bin_resolved_path,
        "declared_python_bin_sha256": runtime.declared_python_bin_sha256,
        "observed_live_argv": list(runtime.observed_live_argv),
        "observed_python_app_image_path": runtime.observed_python_app_image_path,
        "observed_python_app_image_sha256": runtime.observed_python_app_image_sha256,
        "python_version": runtime.python_version,
        "psutil_version": runtime.psutil_version,
        "psutil_module_path": runtime.psutil_module_path,
        "psutil_module_sha256": runtime.psutil_module_sha256,
        "original_exec_alias_attestable_from_process_apis": False,
    }


def _stage_b1_expected_vector(value: object) -> bool:
    return (
        type(value) is tuple
        and bool(value)
        and len(value) <= 256
        and all(
            type(entry) is str and bool(entry) and len(entry) <= 4096 and "\x00" not in entry
            for entry in value
        )
    )


def _stage_b1_json_vector(value: object) -> bool:
    return (
        type(value) is list
        and bool(value)
        and len(value) <= 256
        and all(
            type(entry) is str and bool(entry) and len(entry) <= 4096 and "\x00" not in entry
            for entry in value
        )
    )


def _stage_b1_absolute_path(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and len(value) <= 4096
        and "\x00" not in value
        and Path(value).is_absolute()
        and str(Path(value)) == value
        and all(part not in {".", ".."} for part in value.split("/")[1:])
    )


def _stage_b1_safe_text(value: object, *, maximum: int = 256) -> bool:
    return (
        type(value) is str
        and bool(value)
        and len(value) <= maximum
        and all(
            character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-_"
            for character in value
        )
    )


def _stage_b1_validate_expectation(
    expected: ABC6StageB1Expectation | None,
) -> dict[str, object]:
    if type(expected) is not ABC6StageB1Expectation:
        raise _Reject(
            "stage_b1_expectation",
            "independent typed expectation is unavailable; terminal-v4 evidence is unreplayable",
        )
    if (
        not _stage_b1_safe_text(expected.protocol_id)
        or not _stage_b1_safe_text(expected.run_id)
        or expected.run_id in {".", ".."}
        or not _is_sha256(expected.manifest_sha256)
        or not _is_sha256(expected.approval_record_sha256)
        or not _is_sha256(expected.review_sha256)
        or not _is_sha256(expected.watchdog_claim_sha256)
        or type(expected.reviewed_git_head) is not str
        or len(expected.reviewed_git_head) != 40
        or any(char not in "0123456789abcdef" for char in expected.reviewed_git_head)
        or not _stage_b1_absolute_path(expected.repository_root_realpath)
        or type(expected.receipt_root_relative) is not str
        or not expected.receipt_root_relative
        or expected.receipt_root_relative != campaign_fit.RECEIPT_ROOT_RELATIVE
        or Path(expected.receipt_root_relative).is_absolute()
        or str(Path(expected.receipt_root_relative)) != expected.receipt_root_relative
        or "\\" in expected.receipt_root_relative
        or any(part in {"", ".", ".."} for part in expected.receipt_root_relative.split("/"))
    ):
        raise _Reject("stage_b1_expectation", "independent fixture identity is incomplete")
    for value in (
        expected.repository_root_device,
        expected.repository_root_inode,
        expected.receipt_root_device,
        expected.receipt_root_inode,
        expected.watchdog_claim_parent_device,
        expected.watchdog_claim_parent_inode,
        expected.scorer_claim_parent_device,
        expected.scorer_claim_parent_inode,
    ):
        if type(value) is not int or value < 0:
            raise _Reject("stage_b1_expectation", "independent directory identity is invalid")
    if any(
        value <= 0
        for value in (
            expected.repository_root_inode,
            expected.receipt_root_inode,
            expected.watchdog_claim_parent_inode,
            expected.scorer_claim_parent_inode,
        )
    ):
        raise _Reject("stage_b1_expectation", "independent directory inode is invalid")
    if (
        not _stage_b1_expected_vector(expected.child_declared_launch_vector)
        or not _stage_b1_absolute_path(expected.child_declared_executable_path)
        or not _stage_b1_absolute_path(expected.child_declared_executable_resolved_path)
        or not _is_sha256(expected.child_declared_executable_sha256)
        or type(expected.child_accepted_observed_argv0) is not tuple
        or not expected.child_accepted_observed_argv0
        or any(not _stage_b1_absolute_path(value) for value in expected.child_accepted_observed_argv0)
        or type(expected.child_expected_observed_argv_tail) is not tuple
        or any(
            type(value) is not str or not value or len(value) > 4096 or "\x00" in value
            for value in expected.child_expected_observed_argv_tail
        )
        or not _stage_b1_absolute_path(expected.child_expected_observed_image_path)
        or not _is_sha256(expected.child_expected_observed_image_sha256)
        or expected.child_declared_launch_vector[0] != expected.child_declared_executable_path
        or tuple(expected.child_declared_launch_vector[1:])
        != expected.child_expected_observed_argv_tail
    ):
        raise _Reject("stage_b1_expectation", "independent child launch facts are incomplete")
    for process in (expected.watchdog_process, expected.child_process):
        if process is not None:
            if type(process) is not ABC6StageB1ProcessExpectation:
                raise _Reject("stage_b1_expectation", "process expectation has the wrong type")
            if process.pid is not None and (type(process.pid) is not int or process.pid <= 0):
                raise _Reject("stage_b1_expectation", "independent process PID is invalid")
            if process.start_identity is not None and not _stage_b1_safe_text(
                process.start_identity
            ):
                raise _Reject("stage_b1_expectation", "independent process start identity is invalid")
    attestation = _stage_b1_expected_attestation(expected.watchdog_runtime)
    return attestation


def _stage_b1_expected_claim_fields(
    expected: ABC6StageB1Expectation,
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": expected.protocol_id,
        "run_id": expected.run_id,
        "manifest_sha256": expected.manifest_sha256,
        "approval_record_sha256": expected.approval_record_sha256,
        "reviewed_git_head": expected.reviewed_git_head,
        "repository_root_realpath": expected.repository_root_realpath,
        "receipt_root_relative": expected.receipt_root_relative,
        "repository_root_device": expected.repository_root_device,
        "repository_root_inode": expected.repository_root_inode,
        "receipt_root_device": expected.receipt_root_device,
        "receipt_root_inode": expected.receipt_root_inode,
        "watchdog_claim_parent_device": expected.watchdog_claim_parent_device,
        "watchdog_claim_parent_inode": expected.watchdog_claim_parent_inode,
        "scorer_claim_parent_device": expected.scorer_claim_parent_device,
        "scorer_claim_parent_inode": expected.scorer_claim_parent_inode,
        "declared_child_launch_vector": list(expected.child_declared_launch_vector),
        "declared_child_executable_path": expected.child_declared_executable_path,
        "declared_child_executable_resolved_path": expected.child_declared_executable_resolved_path,
        "declared_child_executable_sha256": expected.child_declared_executable_sha256,
        "accepted_observed_child_argv0": list(expected.child_accepted_observed_argv0),
        "expected_observed_child_argv_tail": list(expected.child_expected_observed_argv_tail),
        "expected_observed_child_image_path": expected.child_expected_observed_image_path,
        "expected_observed_child_image_sha256": expected.child_expected_observed_image_sha256,
        "watchdog_attestation": runtime,
        "claim_semantics": "consumed_once_no_resume",
    }


def _stage_b1_validate_claim(
    raw: bytes,
    claim: dict[str, object],
    expected: ABC6StageB1Expectation,
    runtime: dict[str, object],
) -> None:
    if set(claim) != _STAGE_B1_WATCHDOG_CLAIM_KEYS:
        raise _Reject("stage_b1_claim", "watchdog claim has missing or extra fields")
    if type(claim.get("schema_version")) is not int or claim["schema_version"] != 1:
        raise _Reject("stage_b1_claim", "watchdog claim schema version is invalid")
    for name in (
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "approval_record_sha256",
        "reviewed_git_head",
        "repository_root_realpath",
        "receipt_root_relative",
        "declared_child_executable_path",
        "declared_child_executable_resolved_path",
        "claim_semantics",
    ):
        if type(claim.get(name)) is not str:
            raise _Reject("stage_b1_claim", f"watchdog claim field {name} has the wrong type")
    for name in (
        "repository_root_device",
        "repository_root_inode",
        "receipt_root_device",
        "receipt_root_inode",
        "watchdog_claim_parent_device",
        "watchdog_claim_parent_inode",
        "scorer_claim_parent_device",
        "scorer_claim_parent_inode",
    ):
        if type(claim.get(name)) is not int:
            raise _Reject("stage_b1_claim", f"watchdog claim field {name} is not an exact integer")
    for name in (
        "declared_child_launch_vector",
        "accepted_observed_child_argv0",
        "expected_observed_child_argv_tail",
    ):
        if not _stage_b1_json_vector(claim.get(name)):
            raise _Reject("stage_b1_claim", f"watchdog claim field {name} is not a bounded vector")
    for name in (
        "declared_child_executable_sha256",
        "expected_observed_child_image_sha256",
    ):
        if not _is_sha256(claim.get(name)):
            raise _Reject("stage_b1_claim", f"watchdog claim field {name} is not lowercase SHA-256")
    for name in (
        "declared_child_executable_path",
        "declared_child_executable_resolved_path",
        "expected_observed_child_image_path",
    ):
        if not _stage_b1_absolute_path(claim.get(name)):
            raise _Reject("stage_b1_claim", f"watchdog claim field {name} is not a canonical path")
    if (
        type(claim.get("claimed_at_utc")) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", claim["claimed_at_utc"])
        is None
    ):
        raise _Reject("stage_b1_claim", "watchdog claim timestamp has an invalid exact shape")
    try:
        claimed = datetime.fromisoformat(str(claim["claimed_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise _Reject("stage_b1_claim", "watchdog claim timestamp is invalid") from error
    if claimed.tzinfo != timezone.utc:
        raise _Reject("stage_b1_claim", "watchdog claim timestamp is not UTC")
    nested = claim.get("watchdog_attestation")
    if type(nested) is not dict or set(nested) != _STAGE_B1_WATCHDOG_ATTESTATION_KEYS:
        raise _Reject("stage_b1_claim", "nested watchdog attestation has missing or extra fields")
    if (
        type(nested.get("declared_launch_vector")) is not list
        or not _stage_b1_json_vector(nested.get("declared_launch_vector"))
        or type(nested.get("observed_live_argv")) is not list
        or not _stage_b1_json_vector(nested.get("observed_live_argv"))
    ):
        raise _Reject("stage_b1_claim", "nested watchdog attestation argv is invalid")
    for name in (
        "declared_python_bin",
        "declared_python_bin_resolved_path",
        "observed_python_app_image_path",
        "python_version",
        "psutil_version",
        "psutil_module_path",
    ):
        if type(nested.get(name)) is not str or not nested.get(name):
            raise _Reject("stage_b1_claim", f"nested watchdog attestation field {name} is invalid")
    for name in (
        "declared_python_bin",
        "declared_python_bin_resolved_path",
        "observed_python_app_image_path",
        "psutil_module_path",
    ):
        if not _stage_b1_absolute_path(nested.get(name)):
            raise _Reject("stage_b1_claim", f"nested watchdog attestation path {name} is invalid")
    for name in (
        "declared_python_bin_sha256",
        "observed_python_app_image_sha256",
        "psutil_module_sha256",
    ):
        if not _is_sha256(nested.get(name)):
            raise _Reject("stage_b1_claim", f"nested watchdog attestation field {name} is invalid")
    if type(nested.get("original_exec_alias_attestable_from_process_apis")) is not bool:
        raise _Reject("stage_b1_claim", "nested watchdog attestation boolean has the wrong type")
    if _canonical_json(claim) != raw:
        raise _Reject("stage_b1_claim", "watchdog claim is not exact canonical JSON")
    claim_identity = {key: value for key, value in claim.items() if key != "claimed_at_utc"}
    if claim_identity != _stage_b1_expected_claim_fields(expected, runtime):
        raise _Reject(
            "stage_b1_claim",
            "watchdog claim differs from independent manifest, approval, root, launch or runtime facts",
        )


def _stage_b1_file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stage_b1_read_leaf(
    parent: cases._DirectoryAnchor,
    leaf: str,
    *,
    maximum_bytes: int,
    label: str,
    stage: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    if leaf in {"", ".", ".."} or Path(leaf).name != leaf:
        raise _Reject(stage, f"{label} is not a fixed leaf")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent.descriptor,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise _Reject(stage, f"{label} is not a bounded regular file")
        pieces: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
            if not chunk:
                break
            pieces.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise _Reject(stage, f"{label} exceeds its byte bound")
        after = os.fstat(descriptor)
        named = os.stat(leaf, dir_fd=parent.descriptor, follow_symlinks=False)
    except _Reject:
        raise
    except FileNotFoundError as error:
        raise _Reject(stage, f"{label} is missing") from error
    except OSError as error:
        raise _Reject(stage, f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = _stage_b1_file_identity(before)
    if (
        before_identity != _stage_b1_file_identity(after)
        or before_identity != _stage_b1_file_identity(named)
        or size != before.st_size
    ):
        raise _Reject(stage, f"{label} changed while being read")
    return b"".join(pieces), before_identity


def _stage_b1_verify_anchor(
    root: cases._DirectoryAnchor,
    supplied: cases._DirectoryAnchor,
    relative: str,
    *,
    expected_device: int,
    expected_inode: int,
    label: str,
) -> None:
    if type(root) is not cases._DirectoryAnchor or type(supplied) is not cases._DirectoryAnchor:
        raise _Reject("stage_b1_anchor", f"{label} is not a typed directory anchor")
    try:
        cases._verify_directory_anchor_path(root, label="Stage B1 physical root")
        cases._verify_directory_anchor_path(supplied, label=f"Stage B1 {label}")
        reopened = cases._open_relative_directory_anchor(root, relative, label=f"Stage B1 {label}")
    except Exception as error:
        raise _Reject("stage_b1_anchor", f"{label} ancestry is missing or unsafe") from error
    try:
        info = os.fstat(supplied.descriptor)
        reopened_info = os.fstat(reopened.descriptor)
        if (
            supplied.path != root.path.joinpath(*relative.split("/"))
            or (info.st_dev, info.st_ino) != (expected_device, expected_inode)
            or (reopened_info.st_dev, reopened_info.st_ino) != (expected_device, expected_inode)
            or (supplied.device, supplied.inode) != (expected_device, expected_inode)
        ):
            raise _Reject("stage_b1_anchor", f"{label} differs from independent directory identity")
    finally:
        reopened.close()


def _stage_b1_validate_sidecar(
    raw: bytes,
    sidecar: dict[str, object],
    expected: ABC6StageB1Expectation,
) -> None:
    if type(sidecar) is not dict or set(sidecar) != _STAGE_B1_LAUNCH_ATTESTATION_KEYS:
        raise _Reject("stage_b1_sidecar", "launch attestation has missing or extra fields")
    if type(sidecar.get("schema_version")) is not int or sidecar["schema_version"] != 1:
        raise _Reject("stage_b1_sidecar", "launch attestation schema version is invalid")
    for name in (
        "protocol_id",
        "run_id",
        "manifest_sha256",
        "approval_sha256",
        "review_sha256",
        "reviewed_git_head",
        "watchdog_claim_sha256",
        "physical_root",
        "receipt_root_relative",
    ):
        if type(sidecar.get(name)) is not str:
            raise _Reject("stage_b1_sidecar", f"launch attestation field {name} has the wrong type")
    for name in (
        "root_device",
        "root_inode",
        "receipt_device",
        "receipt_inode",
        "claim_parent_device",
        "claim_parent_inode",
        "observed_monotonic_ns",
    ):
        if type(sidecar.get(name)) is not int or sidecar[name] <= 0:
            raise _Reject("stage_b1_sidecar", f"launch attestation field {name} is not a positive exact integer")
    for name in ("manifest_sha256", "approval_sha256", "review_sha256", "watchdog_claim_sha256"):
        if not _is_sha256(sidecar.get(name)):
            raise _Reject("stage_b1_sidecar", f"launch attestation field {name} is not lowercase SHA-256")
    if (
        not _stage_b1_absolute_path(sidecar.get("physical_root"))
        or not _stage_b1_absolute_path(expected.repository_root_realpath)
        or type(sidecar.get("receipt_root_relative")) is not str
    ):
        raise _Reject("stage_b1_sidecar", "launch attestation root path is invalid")
    watchdog = sidecar.get("watchdog_process")
    child = sidecar.get("child_process")
    if type(watchdog) is not dict or set(watchdog) != _STAGE_B1_PROCESS_KEYS:
        raise _Reject("stage_b1_sidecar", "watchdog process attestation has missing or extra fields")
    if type(child) is not dict or set(child) != _STAGE_B1_CHILD_PROCESS_KEYS:
        raise _Reject("stage_b1_sidecar", "child process attestation has missing or extra fields")
    for label, process in (("watchdog", watchdog), ("child", child)):
        if type(process.get("pid")) is not int or process["pid"] <= 0:
            raise _Reject("stage_b1_sidecar", f"{label} process PID is not a positive exact integer")
        if not _stage_b1_safe_text(process.get("start_identity")):
            raise _Reject("stage_b1_sidecar", f"{label} process start identity is invalid")
        if not _stage_b1_json_vector(process.get("argv")):
            raise _Reject("stage_b1_sidecar", f"{label} process argv is invalid")
        if not _stage_b1_absolute_path(process.get("image_path")):
            raise _Reject("stage_b1_sidecar", f"{label} process image path is invalid")
        if not _is_sha256(process.get("image_sha256")):
            raise _Reject("stage_b1_sidecar", f"{label} process image digest is invalid")
    if (
        not _stage_b1_json_vector(child.get("declared_launch_vector"))
        or not _stage_b1_absolute_path(child.get("declared_launch_image_path"))
        or not _is_sha256(child.get("declared_launch_image_sha256"))
    ):
        raise _Reject("stage_b1_sidecar", "child declared launch binding is invalid")
    expected_processes = (
        (watchdog, expected.watchdog_process, "watchdog"),
        (child, expected.child_process, "child"),
    )
    for observed, process_expectation, label in expected_processes:
        if process_expectation is not None:
            if process_expectation.pid is not None and observed["pid"] != process_expectation.pid:
                raise _Reject("stage_b1_sidecar", f"{label} process PID differs from independent fixture data")
            if (
                process_expectation.start_identity is not None
                and observed["start_identity"] != process_expectation.start_identity
            ):
                raise _Reject("stage_b1_sidecar", f"{label} start identity differs from independent fixture data")
    runtime = expected.watchdog_runtime
    if (
        watchdog["argv"] != list(runtime.observed_live_argv)
        or watchdog["image_path"] != runtime.observed_python_app_image_path
        or watchdog["image_sha256"] != runtime.observed_python_app_image_sha256
        or child["argv"][0] not in expected.child_accepted_observed_argv0
        or child["argv"][1:] != list(expected.child_expected_observed_argv_tail)
        or child["image_path"] != expected.child_expected_observed_image_path
        or child["image_sha256"] != expected.child_expected_observed_image_sha256
        or child["declared_launch_vector"] != list(expected.child_declared_launch_vector)
        or child["declared_launch_image_path"] != expected.child_declared_executable_resolved_path
        or child["declared_launch_image_sha256"] != expected.child_declared_executable_sha256
    ):
        raise _Reject("stage_b1_sidecar", "process vectors or images differ from independent fixture data")
    common = {
        "protocol_id": expected.protocol_id,
        "run_id": expected.run_id,
        "manifest_sha256": expected.manifest_sha256,
        "approval_sha256": expected.approval_record_sha256,
        "review_sha256": expected.review_sha256,
        "reviewed_git_head": expected.reviewed_git_head,
        "watchdog_claim_sha256": expected.watchdog_claim_sha256,
        "physical_root": expected.repository_root_realpath,
        "root_device": expected.repository_root_device,
        "root_inode": expected.repository_root_inode,
        "receipt_root_relative": expected.receipt_root_relative,
        "receipt_device": expected.receipt_root_device,
        "receipt_inode": expected.receipt_root_inode,
        "claim_parent_device": expected.watchdog_claim_parent_device,
        "claim_parent_inode": expected.watchdog_claim_parent_inode,
    }
    if any(sidecar.get(name) != value for name, value in common.items()):
        raise _Reject("stage_b1_sidecar", "launch attestation differs from independent identity bindings")
    if _canonical_json(sidecar) != raw:
        raise _Reject("stage_b1_sidecar", "launch attestation is not exact canonical JSON")


def _stage_b1_verify(
    root: cases._DirectoryAnchor,
    receipt: cases._DirectoryAnchor,
    claim_parent: cases._DirectoryAnchor,
    scorer_claim_parent: cases._DirectoryAnchor,
    terminal: object,
    expected: ABC6StageB1Expectation,
) -> tuple[str, str]:
    runtime = _stage_b1_validate_expectation(expected)
    if (
        type(terminal) is not dict
        or type(terminal.get("schema_version")) is not int
        or terminal["schema_version"] != 4
    ):
        raise _Reject("stage_b1_terminal", "terminal is not exact schema-v4 evidence")
    reference = terminal.get("launch_attestation")
    reference_keys = {"leaf", "sha256", "device", "inode", "mode", "size_bytes"}
    if type(reference) is not dict or set(reference) != reference_keys:
        raise _Reject(
            "stage_b1_reference",
            "terminal launch_attestation reference has missing or extra fields",
        )
    if (
        type(reference.get("leaf")) is not str
        or reference["leaf"] != f"{expected.run_id}.launch-attestation.json"
        or not _is_sha256(reference.get("sha256"))
        or any(type(reference.get(name)) is not int for name in ("device", "inode", "mode", "size_bytes"))
        or reference["device"] < 0
        or reference["inode"] <= 0
        or reference["mode"] <= 0
        or reference["size_bytes"] <= 0
        or reference["size_bytes"] > _STAGE_B1_MAX_LAUNCH_ATTESTATION_BYTES
        or not stat.S_ISREG(reference["mode"])
    ):
        raise _Reject("stage_b1_reference", "terminal launch_attestation reference has invalid exact types or values")
    if terminal.get("watchdog_claim_sha256") != expected.watchdog_claim_sha256:
        raise _Reject("stage_b1_reference", "terminal watchdog claim digest differs from independent input")
    if (
        root.path != Path(expected.repository_root_realpath)
        or (root.device, root.inode)
        != (expected.repository_root_device, expected.repository_root_inode)
    ):
        raise _Reject("stage_b1_anchor", "physical root differs from independent fixture identity")
    if (
        receipt.path != root.path.joinpath(*expected.receipt_root_relative.split("/"))
        or (receipt.device, receipt.inode)
        != (expected.receipt_root_device, expected.receipt_root_inode)
    ):
        raise _Reject("stage_b1_anchor", "receipt root differs from independent fixture identity")
    _stage_b1_verify_anchor(
        root,
        receipt,
        expected.receipt_root_relative,
        expected_device=expected.receipt_root_device,
        expected_inode=expected.receipt_root_inode,
        label="receipt root",
    )
    _stage_b1_verify_anchor(
        root,
        claim_parent,
        campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE,
        expected_device=expected.watchdog_claim_parent_device,
        expected_inode=expected.watchdog_claim_parent_inode,
        label="watchdog claim parent",
    )
    _stage_b1_verify_anchor(
        root,
        scorer_claim_parent,
        _STAGE_B1_SCORER_CLAIM_PARENT_RELATIVE,
        expected_device=expected.scorer_claim_parent_device,
        expected_inode=expected.scorer_claim_parent_inode,
        label="scorer claim parent",
    )
    claim_leaf = f"{expected.run_id}.watchdog.claim"
    claim_raw, claim_identity = _stage_b1_read_leaf(
        claim_parent,
        claim_leaf,
        maximum_bytes=MAX_CLAIM_BYTES,
        label="watchdog one-use claim",
        stage="stage_b1_claim",
    )
    claim_sha256 = hashlib.sha256(claim_raw).hexdigest()
    if claim_sha256 != expected.watchdog_claim_sha256:
        raise _Reject("stage_b1_claim", "watchdog claim raw digest differs from independent input")
    claim = _decode_canonical(claim_raw, label="watchdog one-use claim", stage="stage_b1_claim")
    _stage_b1_validate_claim(claim_raw, claim, expected, runtime)
    sidecar_leaf = str(reference["leaf"])
    sidecar_raw, sidecar_identity = _stage_b1_read_leaf(
        claim_parent,
        sidecar_leaf,
        maximum_bytes=_STAGE_B1_MAX_LAUNCH_ATTESTATION_BYTES,
        label="launch attestation sidecar",
        stage="stage_b1_sidecar",
    )
    if (
        len(sidecar_raw) != reference["size_bytes"]
        or sidecar_identity[:4]
        != (reference["device"], reference["inode"], reference["mode"], reference["size_bytes"])
        or hashlib.sha256(sidecar_raw).hexdigest() != reference["sha256"]
    ):
        raise _Reject(
            "stage_b1_sidecar",
            "launch attestation bytes or (device,inode,mode,size) differ from terminal reference",
        )
    sidecar = _decode_canonical(
        sidecar_raw, label="launch attestation sidecar", stage="stage_b1_sidecar"
    )
    _stage_b1_validate_sidecar(sidecar_raw, sidecar, expected)
    claim_raw_after, claim_identity_after = _stage_b1_read_leaf(
        claim_parent,
        claim_leaf,
        maximum_bytes=MAX_CLAIM_BYTES,
        label="watchdog one-use claim revalidation",
        stage="stage_b1_revalidation",
    )
    sidecar_raw_after, sidecar_identity_after = _stage_b1_read_leaf(
        claim_parent,
        sidecar_leaf,
        maximum_bytes=_STAGE_B1_MAX_LAUNCH_ATTESTATION_BYTES,
        label="launch attestation revalidation",
        stage="stage_b1_revalidation",
    )
    if claim_raw_after != claim_raw or claim_identity_after != claim_identity:
        raise _Reject("stage_b1_revalidation", "watchdog claim changed during B1 verification")
    if sidecar_raw_after != sidecar_raw or sidecar_identity_after != sidecar_identity:
        raise _Reject("stage_b1_revalidation", "launch attestation changed during B1 verification")
    return claim_sha256, str(reference["sha256"])


def verify_abc6_stage_b1_launch_evidence(
    root: cases._DirectoryAnchor,
    receipt: cases._DirectoryAnchor,
    claim_parent: cases._DirectoryAnchor,
    scorer_claim_parent: cases._DirectoryAnchor,
    terminal: object,
    expected: ABC6StageB1Expectation | None,
) -> ABC6StageB1Verification:
    """Structurally link fake-fixture claim/sidecar evidence under pinned anchors.

    A successful result proves only that supplied bytes, fixed paths, exact
    structures, file identities and independently supplied fixture values
    agree. It does not classify a run or authorize production replay. Missing
    independent expectations and every malformed or unstable input return an
    explicit ``unreplayable`` result.
    """

    try:
        if expected is None:
            raise _Reject(
                "stage_b1_expectation",
                "independent typed expectation is unavailable; terminal-v4 evidence is unreplayable",
            )
        claim_sha256, sidecar_sha256 = _stage_b1_verify(
            root, receipt, claim_parent, scorer_claim_parent, terminal, expected
        )
        return ABC6StageB1Verification(
            status="evidence_linked",
            detail="Exact fixture claim and launch attestation links match independent inputs.",
            claim_sha256=claim_sha256,
            launch_attestation_sha256=sidecar_sha256,
        )
    except Exception as error:
        detail = error.args[0] if isinstance(error, _Reject) and error.args else "B1 evidence could not be verified"
        return ABC6StageB1Verification(status="unreplayable", detail=str(detail))


def _verify_watchdog_claim(
    ctx: _ReadSession,
) -> tuple[str, dict[str, object]]:
    """Verify the frozen one-use watchdog claim under its anchored parent."""

    if ctx.scorer_claim_parent is None:
        raise _Reject("watchdog_claim", "pinned scorer claim parent is unavailable")
    frozen = ctx.frozen
    watchdog_raw, watchdog_claim, _ = ctx.read(
        ctx.campaign_claim_parent,
        f"{frozen.run_id}.watchdog.claim",
        maximum_bytes=MAX_CLAIM_BYTES,
        label="watchdog one-use claim",
        stage="watchdog_claim",
    )
    watchdog_sha = hashlib.sha256(watchdog_raw).hexdigest()
    if watchdog_sha != frozen.watchdog_claim_sha256:
        raise _Reject("watchdog_claim", "watchdog claim digest differs from frozen identity")
    expected_fields = {
        "protocol_id": frozen.protocol_id,
        "run_id": frozen.run_id,
        "manifest_sha256": frozen.manifest_sha256,
        "approval_record_sha256": frozen.approval_record_sha256,
        "reviewed_git_head": frozen.reviewed_git_head,
        "repository_root_realpath": str(ctx.root.path),
        "receipt_root_relative": ctx.receipt.path.relative_to(ctx.root.path).as_posix(),
        "repository_root_device": ctx.root.device,
        "repository_root_inode": ctx.root.inode,
        "receipt_root_device": ctx.receipt.device,
        "receipt_root_inode": ctx.receipt.inode,
        "watchdog_claim_parent_device": ctx.campaign_claim_parent.device,
        "watchdog_claim_parent_inode": ctx.campaign_claim_parent.inode,
        "scorer_claim_parent_device": ctx.scorer_claim_parent.device,
        "scorer_claim_parent_inode": ctx.scorer_claim_parent.inode,
        "claim_semantics": "consumed_once_no_resume",
    }
    if any(watchdog_claim.get(key) != value for key, value in expected_fields.items()):
        raise _Reject("watchdog_claim", "watchdog claim identity differs from frozen root")
    return watchdog_sha, watchdog_claim


def _verify_no_campaign_chain(ctx: _ReadSession) -> None:
    """Pin no-follow absence of all fixed campaign and post-score artifacts."""

    if ctx.scorer_claim_parent is None:
        raise _Reject("reveal_marker", "pinned scorer claim parent is unavailable")
    ctx.require_absent(
        ctx.receipt,
        campaign_fit.CLAIM_FILENAME,
        label="receipt campaign claim",
        stage="campaign_claim_absence",
    )
    ctx.require_absent(
        ctx.campaign_claim_parent,
        f"{ctx.frozen.run_id}.claim",
        label="global campaign claim",
        stage="campaign_claim_absence",
    )
    ctx.require_absent(
        ctx.scorer_claim_parent,
        f"{ctx.frozen.run_id}.claim",
        label="reveal marker",
        stage="reveal_marker_absence",
    )
    for case in cases.CASE_ROSTER:
        for component in ("fit", "baseline"):
            filename = f"case-{case.case_index:02d}.{component}-status.json"
            ctx.require_absent(
                ctx.receipt,
                filename,
                label=f"{component} status receipt {case.case_index}",
                stage="campaign_status_absence",
            )
    for filename, label in (
        (campaign_fit.SUMMARY_FILENAME, "training summary"),
        (campaign_fit.EVIDENCE_MANIFEST_FILENAME, "training evidence manifest"),
        (campaign_fit.FAILURE_FILENAME, "campaign failure receipt"),
        (_FORECAST_FILENAME, "target-free forecast artifact"),
        (scoring.TARGET_ARRAYS_ARTIFACT_FILENAME, "prospective target artifact"),
        (scoring.SCORE_RECEIPT_FILENAME, "deferred score receipt"),
    ):
        ctx.require_absent(
            ctx.receipt,
            filename,
            label=label,
            stage="campaign_artifact_absence",
        )


def _verify_pre_campaign_child_state(
    ctx: _ReadSession,
    terminal: dict[str, object],
) -> Literal["prechild", "child"]:
    """Separate a genuine prechild terminal from an attested child stop."""

    fit_gate = terminal.get("fit_phase_gate")
    if (
        type(fit_gate) is not dict
        or fit_gate.get("status_receipt_count") != 0
        or fit_gate.get("status_receipts") != []
        or fit_gate.get("summary_sha256") is not None
        or fit_gate.get("evidence_manifest_sha256") is not None
        or fit_gate.get("target_free_forecast_sha256") is not None
        or fit_gate.get("pre_score_artifact_chain_valid") is not False
        or fit_gate.get("all_48_status_receipts_present_and_linked") is not False
        or fit_gate.get("training_evidence_chain_valid") is not False
        or fit_gate.get("target_free_forecast_chain_valid") is not False
    ):
        raise _Reject(
            "terminal_fit_gate",
            "terminal does not retain an empty pre-campaign fit-stage snapshot",
        )
    launch = terminal.get("child_launch")
    if type(launch) is not dict or type(launch.get("image_observations")) is not list:
        raise _Reject("terminal_child_attestation", "terminal child launch snapshot is malformed")
    observed = launch.get("observed_process")
    if observed is None:
        if ctx.frozen.expected_child_grant_bindings is not None:
            raise _Reject(
                "child_grant", "prechild replay cannot carry expected child grant bindings"
            )
        if (
            launch.get("image_observations") != []
            or terminal.get("process_return_code") is not None
            or terminal.get("kill_and_reap") is not None
        ):
            raise _Reject(
                "terminal_child_attestation",
                "prechild terminal contains child execution or cleanup evidence",
            )
        bootstrap = terminal.get("bootstrap_grant")
        if bootstrap is not None:
            if (
                type(bootstrap) is not dict
                or set(bootstrap) != _BOOTSTRAP_GRANT_FIELDS
                or bootstrap.get("transport") != "inherited-anonymous-unix-stream-socket"
                or bootstrap.get("environment_locator") != "ABC6_GRANT_FD"
                or type(bootstrap.get("descriptor_fd")) is not int
                or bootstrap["descriptor_fd"] < 0
                or bootstrap.get("grant_sha256") is not None
                or bootstrap.get("grant_record_sha256") is not None
                or bootstrap.get("delivery_status") != "not_attempted"
                or bootstrap.get("digest_receipt_leaf") != GRANT_RECORD_FILENAME
            ):
                raise _Reject(
                    "terminal_child_attestation",
                    "prechild bootstrap grant fields are inconsistent",
                )
        ctx.require_absent(
            ctx.receipt,
            GRANT_RECORD_FILENAME,
            label="durable child grant record",
            stage="terminal_child_attestation",
        )
        ctx.require_directory_entries(
            ctx.receipt,
            allowed={TERMINAL_FILENAME, TERMINAL_ACK_FILENAME},
            required={TERMINAL_FILENAME, TERMINAL_ACK_FILENAME},
            label="prechild receipt root",
            stage="campaign_artifact_absence",
        )
        return "prechild"

    _require_verified_child_reap(terminal)
    if type(terminal.get("process_return_code")) is not int:
        raise _Reject(
            "terminal_child_reap", "observed child has no retained process return code"
        )
    ctx.require_directory_entries(
        ctx.receipt,
        allowed={TERMINAL_FILENAME, TERMINAL_ACK_FILENAME, GRANT_RECORD_FILENAME},
        required={TERMINAL_FILENAME, TERMINAL_ACK_FILENAME},
        label="pre-campaign child receipt root",
        stage="campaign_artifact_absence",
    )
    return "child"


def _verify_child_grant_record(
    ctx: _ReadSession,
    terminal: dict[str, object],
) -> None:
    """If a child grant was durably issued, verify its anchored receipt link."""

    present = ctx.maybe_present(
        ctx.receipt, GRANT_RECORD_FILENAME, stage="child_grant"
    )
    bootstrap = terminal.get("bootstrap_grant")
    if not present:
        if ctx.frozen.expected_child_grant_bindings is not None:
            raise _Reject(
                "child_grant", "expected child bindings were supplied without a durable grant record"
            )
        if bootstrap is not None and (
            type(bootstrap) is not dict
            or set(bootstrap) != _BOOTSTRAP_GRANT_FIELDS
            or bootstrap.get("transport") != "inherited-anonymous-unix-stream-socket"
            or bootstrap.get("environment_locator") != "ABC6_GRANT_FD"
            or type(bootstrap.get("descriptor_fd")) is not int
            or bootstrap.get("grant_sha256") is not None
            or bootstrap.get("grant_record_sha256") is not None
            or bootstrap.get("delivery_status") != "not_attempted"
            or bootstrap.get("digest_receipt_leaf") != GRANT_RECORD_FILENAME
        ):
            raise _Reject("child_grant", "terminal claims a grant without a durable grant record")
        return
    raw, grant, _identity = ctx.read(
        ctx.receipt,
        GRANT_RECORD_FILENAME,
        maximum_bytes=MAX_CLAIM_BYTES,
        label="durable child grant record",
        stage="child_grant",
    )
    bindings = grant.get("bindings")
    expected_bindings = ctx.frozen.expected_child_grant_bindings
    if (
        set(grant)
        != {"schema_version", "grant_sha256", "grant_fd", "bindings_sha256", "bindings"}
        or type(grant.get("schema_version")) is not int
        or grant.get("schema_version") != 1
        or not _is_sha256(grant.get("grant_sha256"))
        or type(grant.get("grant_fd")) is not int
        or grant["grant_fd"] < 0
        or not _is_sha256(grant.get("bindings_sha256"))
        or type(bindings) is not dict
        or hashlib.sha256(_canonical_json(bindings)).hexdigest()
        != grant.get("bindings_sha256")
        or type(bootstrap) is not dict
        or set(bootstrap) != _BOOTSTRAP_GRANT_FIELDS
        or bootstrap.get("transport") != "inherited-anonymous-unix-stream-socket"
        or bootstrap.get("environment_locator") != "ABC6_GRANT_FD"
        or type(bootstrap.get("descriptor_fd")) is not int
        or bootstrap["descriptor_fd"] < 0
        or grant.get("grant_fd") != bootstrap.get("descriptor_fd")
        or bootstrap.get("grant_sha256") != grant.get("grant_sha256")
        or not _is_sha256(bootstrap.get("grant_record_sha256"))
        or hashlib.sha256(raw).hexdigest() != bootstrap.get("grant_record_sha256")
        or bootstrap.get("delivery_status") != "delivered"
        or bootstrap.get("digest_receipt_leaf") != GRANT_RECORD_FILENAME
    ):
        raise _Reject("child_grant", "durable child grant record does not match its terminal link")
    if expected_bindings is None:
        raise _Reject(
            "child_grant", "durable grant record has no separately frozen expected bindings"
        )
    try:
        typed_bindings = grant_authority.ABC6AuthorityBindings.from_payload(bindings)
    except Exception as error:
        raise _Reject("child_grant", "grant bindings failed the authority schema") from error
    if typed_bindings.payload() != expected_bindings.payload():
        raise _Reject(
            "child_grant", "durable child grant differs from separately frozen expected bindings"
        )
    watchdog_process = terminal.get("watchdog_process_excluded_from_runner_tree")
    launch = terminal.get("child_launch")
    observed = launch.get("observed_process") if type(launch) is dict else None
    reap = terminal.get("kill_and_reap")
    if (
        type(expected_bindings.watchdog_pid) is not int
        or type(expected_bindings.child_pid) is not int
        or type(watchdog_process) is not dict
        or watchdog_process.get("excluded") is not True
        or type(watchdog_process.get("pid")) is not int
        or watchdog_process["pid"] != expected_bindings.watchdog_pid
        or type(launch) is not dict
        or type(launch.get("declared_executable_path")) is not str
        or launch["declared_executable_path"]
        != expected_bindings.child_launch_image_path
        or not _is_sha256(launch.get("declared_executable_sha256"))
        or launch["declared_executable_sha256"]
        != expected_bindings.child_launch_image_sha256
        or type(observed) is not dict
        or type(observed.get("pid")) is not int
        or observed["pid"] != expected_bindings.child_pid
        or type(reap) is not dict
        or type(reap.get("process_group_id")) is not int
        or reap["process_group_id"] != expected_bindings.child_pid
    ):
        raise _Reject(
            "child_grant",
            "terminal watchdog, child, process-group, or launcher identity differs from expected grant bindings",
        )
    root_identity = ctx.frozen.receipt_root_identity
    expected_source_hashes = dict(expected_bindings.source_hashes)
    if (
        expected_bindings.protocol_id != ctx.frozen.protocol_id
        or expected_bindings.run_id != ctx.frozen.run_id
        or expected_bindings.manifest_sha256 != ctx.frozen.manifest_sha256
        or expected_bindings.approval_sha256 != ctx.frozen.approval_record_sha256
        or expected_bindings.reviewed_git_head != ctx.frozen.reviewed_git_head
        or expected_bindings.watchdog_claim_sha256 != ctx.frozen.watchdog_claim_sha256
        or expected_bindings.watchdog_pid == expected_bindings.child_pid
        or expected_bindings.physical_root != root_identity.repository_root_realpath
        or expected_bindings.root_device != root_identity.repository_root_device
        or expected_bindings.root_inode != root_identity.repository_root_inode
        or expected_bindings.receipt_root_relative != root_identity.receipt_root_relative
        or expected_bindings.receipt_device != root_identity.receipt_root_device
        or expected_bindings.receipt_inode != root_identity.receipt_root_inode
        or tuple(
            (path, expected_source_hashes.get(path))
            for path in _INTEGRATED_SOURCE_PATHS
        )
        != ctx.frozen.integrated_source_hashes
    ):
        raise _Reject(
            "child_grant", "expected child bindings differ from the frozen run identity"
        )
    if (
        observed.get("observed_live_argv") != list(expected_bindings.child_vector)
        or observed.get("observed_executable_path") != expected_bindings.child_image_path
        or observed.get("observed_executable_sha256")
        != expected_bindings.child_image_sha256
        or observed.get("verified_image_role") != "observed_python_app_image"
    ):
        raise _Reject(
            "child_grant", "terminal process attestation differs from expected child bindings"
        )


def _classify_no_campaign_claim(
    terminal: dict[str, object],
    child_state: Literal["prechild", "child"],
) -> tuple[Literal["failed", "incomplete"], str, str]:
    """Classify only fully evidenced watchdog-only/pre-campaign outcomes."""

    capture = terminal["intervention_capture"]
    status = terminal["status"]
    return_code = terminal.get("process_return_code")
    if capture["status"] == "recorded" and capture["attempted_type"] in {
        "budget_cap",
        "operator_stop",
    }:
        monitor_error = terminal.get("monitor_error")
        expected_operator_stop = (
            capture["attempted_type"] == "operator_stop"
            and capture["attempted_stage"] == "watchdog_interrupted"
            and monitor_error == "KeyboardInterrupt: "
        )
        if monitor_error is not None and not expected_operator_stop:
            raise _Reject(
                "terminal_monitor_error",
                "watchdog intervention has an unresolved monitor error",
            )
        if status not in {"capped", "failed"}:
            raise _Reject(
                "terminal_intervention", "recorded stop event contradicts terminal status"
            )
        stage = str(capture["attempted_stage"])
        return (
            "incomplete",
            stage,
            "A recorded budget or operator stop verified before the campaign claim.",
        )
    if capture["status"] == "not_attempted" and child_state == "prechild":
        raise _Reject(
            "prechild_failure_unstructured",
            "a prechild failure has no structured failure receipt and is unreplayable",
        )
    if (
        capture["status"] == "not_attempted"
        and child_state == "child"
        and status == "failed"
        and type(return_code) is int
        and return_code != 0
    ):
        return (
            "failed",
            str(terminal["stop_reason"]),
            "An attested child exited nonzero before creating a campaign claim.",
        )
    raise _Reject(
        "terminal_score_chain",
        "watchdog terminal does not support a pre-campaign classification",
    )


def _verify_statuses_and_summary(
    ctx: _ReadSession,
    campaign_claim_sha: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, object], str, bool]:
    statuses: list[dict[str, object]] = []
    verified_receipts: list[tuple[str, str]] = []
    for case in cases.CASE_ROSTER:
        for component in ("fit", "baseline"):
            filename = f"case-{case.case_index:02d}.{component}-status.json"
            raw, payload, _ = ctx.read(
                ctx.receipt,
                filename,
                maximum_bytes=MAX_STATUS_BYTES,
                label=f"{component} status receipt {case.case_index}",
                stage="status_receipts",
            )
            status, digest = receipt_io._validate_status_receipt_bytes(
                raw,
                filename,
                case_index=case.case_index,
                case_id=case.case_id,
                component=component,
                protocol_id=ctx.frozen.protocol_id,
                run_id=ctx.frozen.run_id,
            )
            statuses.append(
                {
                    "case_index": case.case_index,
                    "case_id": case.case_id,
                    "component": component,
                    "status": status,
                    "filename": filename,
                    "sha256": digest,
                }
            )
            verified_receipts.append((filename, digest))
            if payload.get("status") != status:
                raise _Reject("status_receipts", "status receipt parsing mismatch")

    summary_raw, summary, _ = ctx.read(
        ctx.receipt,
        campaign_fit.SUMMARY_FILENAME,
        maximum_bytes=MAX_TRAINING_SUMMARY_BYTES,
        label="campaign training summary",
        stage="training_summary",
    )
    _payload_hash_valid(summary, stage="training_summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("protocol_id") != ctx.frozen.protocol_id
        or summary.get("run_id") != ctx.frozen.run_id
        or summary.get("manifest_sha256") != ctx.frozen.manifest_sha256
        or summary.get("claim_sha256") != campaign_claim_sha
        or summary.get("case_count") != len(cases.CASE_ROSTER)
        or summary.get("training_only") is not True
        or summary.get("prospective_targets_generated") is not False
        or summary.get("forecasts_run") is not False
        or summary.get("postfit_target_gate_opened") is not False
    ):
        raise _Reject("training_summary", "training summary identity or target boundary is invalid")
    expected_cases: list[dict[str, object]] = []
    for index, case in enumerate(cases.CASE_ROSTER):
        fit = statuses[index * 2]
        baseline = statuses[index * 2 + 1]
        expected_cases.append(
            {
                "case_index": index,
                "case_id": case.case_id,
                "fit_status": fit["status"],
                "baseline_status": baseline["status"],
                "fit_receipt_sha256": fit["sha256"],
                "baseline_receipt_sha256": baseline["sha256"],
            }
        )
    if summary.get("case_statuses") != expected_cases:
        raise _Reject("training_summary", "summary does not link the ordered 48 statuses")
    expected_status = (
        "complete"
        if all(row["status"] == "complete" for row in statuses)
        else "incomplete"
    )
    if summary.get("status") != expected_status:
        raise _Reject("training_summary", "summary status differs from the 48 durable receipts")
    for component in ("fit", "baseline"):
        expected_counts: dict[str, int] = {}
        for row in statuses:
            if row["component"] == component:
                state = str(row["status"])
                expected_counts[state] = expected_counts.get(state, 0) + 1
        if summary.get(f"{component}_status_counts") != dict(sorted(expected_counts.items())):
            raise _Reject("training_summary", f"summary {component} counts differ from receipts")
    summary_digest = hashlib.sha256(summary_raw).hexdigest()
    return tuple(statuses), summary, summary_digest, expected_status == "complete"


def _verify_evidence_artifacts(
    ctx: _ReadSession,
    statuses: tuple[dict[str, object], ...],
    summary: dict[str, object],
    summary_sha256: str,
    campaign_claim_sha: str,
) -> tuple[dict[str, object], str]:
    raw, manifest, _ = ctx.read(
        ctx.receipt,
        campaign_fit.EVIDENCE_MANIFEST_FILENAME,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="training evidence manifest",
        stage="training_evidence_manifest",
    )
    _payload_hash_valid(manifest, stage="training_evidence_manifest")
    case_ids = tuple(case.case_id for case in cases.CASE_ROSTER)
    expected_status_entries = [
        {
            "filename": row["filename"],
            "sha256": row["sha256"],
            "case_index": row["case_index"],
            "case_id": row["case_id"],
            "component": row["component"],
            "status": row["status"],
        }
        for row in statuses
    ]
    if (
        set(manifest)
        != {
            "schema_version", "protocol_id", "run_id", "manifest_sha256",
            "claim_sha256", "training_summary_filename", "training_summary_sha256",
            "roster_sha256", "ordered_case_identities", "status_receipts",
            "bundle_artifact", "case_artifacts", "payload_sha256",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != ctx.frozen.protocol_id
        or manifest.get("run_id") != ctx.frozen.run_id
        or manifest.get("manifest_sha256") != ctx.frozen.manifest_sha256
        or manifest.get("claim_sha256") != campaign_claim_sha
        or manifest.get("training_summary_filename") != campaign_fit.SUMMARY_FILENAME
        or manifest.get("training_summary_sha256") != summary_sha256
        or manifest.get("roster_sha256") != campaign_fit._roster_sha256()
        or manifest.get("ordered_case_identities") != campaign_fit._roster_identities()
        or manifest.get("status_receipts") != expected_status_entries
        or case_ids != tuple(case.case_id for case in cases.CASE_ROSTER)
    ):
        raise _Reject("training_evidence_manifest", "evidence manifest chain is invalid")
    evidence_manifest_sha = hashlib.sha256(raw).hexdigest()
    bundle_entry = manifest.get("bundle_artifact")
    case_entries = manifest.get("case_artifacts")
    if (
        type(bundle_entry) is not dict
        or set(bundle_entry) != {"filename", "sha256"}
        or bundle_entry.get("filename") != "training-bundle.evidence.json"
        or not _is_sha256(bundle_entry.get("sha256"))
        or type(case_entries) is not list
        or len(case_entries) != len(cases.CASE_ROSTER)
    ):
        raise _Reject("training_evidence_manifest", "evidence artifact roster is invalid")
    evidence_anchor = ctx.open_directory(
        campaign_fit.EVIDENCE_DIRECTORY_NAME,
        label="training evidence directory",
        base=ctx.receipt,
    )
    total_size = 0
    bundle_raw, bundle, _ = ctx.read(
        evidence_anchor,
        str(bundle_entry["filename"]),
        maximum_bytes=MAX_EVIDENCE_FILE_BYTES,
        label="training bundle evidence",
        stage="training_evidence_bundle",
    )
    total_size += len(bundle_raw)
    if (
        hashlib.sha256(bundle_raw).hexdigest() != bundle_entry["sha256"]
        or bundle.get("schema_version") != 1
        or bundle.get("kind") != "abc6_training_bundle"
        or bundle.get("protocol_id") != ctx.frozen.protocol_id
        or bundle.get("run_id") != ctx.frozen.run_id
        or bundle.get("ordered_case_identities") != campaign_fit._roster_identities()
    ):
        raise _Reject("training_evidence_bundle", "training bundle evidence is invalid")

    for index, (case, entry) in enumerate(zip(cases.CASE_ROSTER, case_entries, strict=True)):
        expected_name = f"case-{index:02d}.training-evidence.json"
        fit_status = statuses[index * 2]["status"]
        baseline_status = statuses[index * 2 + 1]["status"]
        case_identity = campaign_fit._case_identity(case)
        if (
            type(entry) is not dict
            or set(entry)
            != {
                "filename", "sha256", "roster_index", "case_identity",
                "fit_status", "baseline_status",
            }
            or entry.get("filename") != expected_name
            or entry.get("roster_index") != index
            or entry.get("case_identity") != case_identity
            or entry.get("fit_status") != fit_status
            or entry.get("baseline_status") != baseline_status
            or not _is_sha256(entry.get("sha256"))
        ):
            raise _Reject("training_evidence_manifest", f"case evidence link {index} is invalid")
        case_raw, case_value, _ = ctx.read(
            evidence_anchor,
            expected_name,
            maximum_bytes=MAX_EVIDENCE_FILE_BYTES,
            label=f"training case evidence {index}",
            stage="training_case_evidence",
        )
        total_size += len(case_raw)
        if total_size > MAX_EVIDENCE_TOTAL_BYTES:
            raise _Reject("training_case_evidence", "training evidence exceeds its total byte limit")
        if (
            hashlib.sha256(case_raw).hexdigest() != entry["sha256"]
            or set(case_value)
            != {
                "schema_version", "protocol_id", "run_id", "kind",
                "roster_index", "case_identity", "fit_status", "baseline_status", "result",
            }
            or case_value.get("schema_version") != 1
            or case_value.get("protocol_id") != ctx.frozen.protocol_id
            or case_value.get("run_id") != ctx.frozen.run_id
            or case_value.get("kind") != "abc6_training_case_result"
            or case_value.get("roster_index") != index
            or case_value.get("case_identity") != case_identity
            or case_value.get("fit_status") != fit_status
            or case_value.get("baseline_status") != baseline_status
        ):
            raise _Reject("training_case_evidence", f"training case evidence {index} is invalid")
    return manifest, evidence_manifest_sha


def _expected_case_identities() -> list[dict[str, object]]:
    return [
        {
            "case_index": case.case_index,
            "case_id": case.case_id,
            "truth_id": case.truth_id,
            "input_window": case.input_window,
            "replicate": case.replicate,
            "fit_model": case.fit_model.value,
        }
        for case in cases.CASE_ROSTER
    ]


def _verify_forecast(
    ctx: _ReadSession,
    statuses: tuple[dict[str, object], ...],
    campaign_claim_sha: str,
    summary_sha256: str,
    evidence_manifest_sha256: str,
    source_hashes: tuple[tuple[str, str], ...],
) -> tuple[dict[str, object], str, str, tuple[dict[str, object], ...], bool]:
    filename = "campaign.target-free-forecasts.json"
    raw, forecast, _ = ctx.read(
        ctx.receipt,
        filename,
        maximum_bytes=MAX_JSON_BYTES,
        label="target-free forecast artifact",
        stage="target_free_forecast",
    )
    _payload_hash_valid(forecast, stage="target_free_forecast")
    expected_sources = [{"path": path, "sha256": digest} for path, digest in source_hashes]
    expected_status_links = [
        {"filename": row["filename"], "sha256": row["sha256"]}
        for row in statuses
    ]
    forecast_rows = forecast.get("forecasts")
    case_digests = forecast.get("forecast_case_sha256")
    if (
        set(forecast)
        != {
            "schema_version", "protocol_id", "run_id", "training_manifest_sha256",
            "training_claim_sha256", "training_summary_filename", "training_summary_sha256",
            "training_evidence_manifest_sha256", "integrated_source_hashes",
            "ordered_case_identities", "status_receipts", "forecast_case_sha256",
            "forecast_roster_sha256", "forecasts", "target_free",
            "prospective_targets_generated_by_runner", "retry_allowed", "payload_sha256",
        }
        or forecast.get("schema_version") != 1
        or forecast.get("protocol_id") != ctx.frozen.protocol_id
        or forecast.get("run_id") != ctx.frozen.run_id
        or forecast.get("training_manifest_sha256") != ctx.frozen.manifest_sha256
        or forecast.get("training_claim_sha256") != campaign_claim_sha
        or forecast.get("training_summary_filename") != campaign_fit.SUMMARY_FILENAME
        or forecast.get("training_summary_sha256") != summary_sha256
        or forecast.get("training_evidence_manifest_sha256") != evidence_manifest_sha256
        or forecast.get("integrated_source_hashes") != expected_sources
        or forecast.get("ordered_case_identities") != _expected_case_identities()
        or forecast.get("status_receipts") != expected_status_links
        or type(forecast_rows) is not list
        or len(forecast_rows) != len(cases.CASE_ROSTER)
        or type(case_digests) is not list
        or len(case_digests) != len(cases.CASE_ROSTER)
        or forecast.get("target_free") is not True
        or forecast.get("prospective_targets_generated_by_runner") is not False
        or forecast.get("retry_allowed") is not False
    ):
        raise _Reject("target_free_forecast", "target-free forecast links are invalid")
    computed_case_digests: list[str] = []
    forecast_records: list[dict[str, object]] = []
    scientific_ready = True
    eligible_same_family_count = 0
    for index, (case, row, declared_digest) in enumerate(
        zip(cases.CASE_ROSTER, forecast_rows, case_digests, strict=True)
    ):
        required_fields = {
            "status", "case_index", "case_id", "fit_window", "prospective_inputs",
            "common_state_index", "parameter_order", "particles", "weights",
            "particle_trajectories", "aggregate_status", "pointwise_weighted_mean",
            "pointwise_weighted_median", "pointwise_q05", "pointwise_q95",
            "effective_sample_size", "quantile_convention",
            "pointwise_summaries_are_coherent_trajectories", "baseline_parameter_values",
            "baseline_common_time_state", "baseline_trajectory", "baseline_failure",
        }
        if (
            type(row) is not dict
            or set(row) != required_fields
            or row.get("case_index") != index
            or row.get("case_id") != case.case_id
            or row.get("fit_window") != case.input_window
            or row.get("status") not in {"complete", "abstained_n", "incomplete_abc_fit"}
            or (case.truth_id == "N" and row.get("status") != "abstained_n")
            or (case.truth_id != "N" and row.get("status") == "abstained_n")
            or not _is_sha256(declared_digest)
        ):
            raise _Reject("target_free_forecast", f"forecast roster case {index} is invalid")
        case_payload = {
            "protocol_id": ctx.frozen.protocol_id,
            "run_id": ctx.frozen.run_id,
            "case_index": index,
            "case_id": case.case_id,
            "truth_id": case.truth_id,
            "input_window": case.input_window,
            "replicate": case.replicate,
            "fit_model": case.fit_model.value,
            "forecast": row,
        }
        case_digest = hashlib.sha256(_canonical_json(case_payload)).hexdigest()
        if case_digest != declared_digest:
            raise _Reject("target_free_forecast", f"forecast case digest differs at index {index}")
        computed_case_digests.append(case_digest)
        forecast_records.append(row)
        is_scientific_case = (
            case.truth_id in {"A", "B"}
            and case.fit_model.value == ("O2" if case.truth_id == "A" else "C2")
        )
        if is_scientific_case:
            eligible_same_family_count += 1
            particles = row.get("particles")
            trajectories = row.get("particle_trajectories")
            if (
                statuses[index * 2]["status"] != "complete"
                or row.get("status") != "complete"
                or row.get("aggregate_status") != "complete"
                or type(particles) is not list
                or len(particles) != 48
                or type(trajectories) is not list
                or len(trajectories) != 48
                or any(type(path) is not list or len(path) != cases.PROSPECTIVE_LENGTH for path in trajectories)
            ):
                scientific_ready = False
    roster_digest = hashlib.sha256(
        _canonical_json(
            {
                "protocol_id": ctx.frozen.protocol_id,
                "run_id": ctx.frozen.run_id,
                "case_sha256": computed_case_digests,
            }
        )
    ).hexdigest()
    if forecast.get("forecast_roster_sha256") != roster_digest:
        raise _Reject("target_free_forecast", "forecast roster digest is invalid")
    if eligible_same_family_count != 16:
        raise _Reject("target_free_forecast", "same-family S/L roster does not contain 16 fits")
    forecast_array_digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": 1,
                "protocol_id": ctx.frozen.protocol_id,
                "run_id": ctx.frozen.run_id,
                "forecast_roster_sha256": roster_digest,
                "ordered_case_sha256": computed_case_digests,
                "forecasts": forecast_records,
            }
        )
    ).hexdigest()
    return forecast, hashlib.sha256(raw).hexdigest(), forecast_array_digest, tuple(forecast_records), scientific_ready


def _source_hash_rows(value: object, *, label: str, stage: str) -> tuple[tuple[str, str], ...]:
    if type(value) is not list or len(value) != len(_INTEGRATED_SOURCE_PATHS):
        raise _Reject(stage, f"{label} does not contain the frozen source list")
    rows: list[tuple[str, str]] = []
    for expected_path, row in zip(_INTEGRATED_SOURCE_PATHS, value, strict=True):
        if type(row) is list and len(row) == 2:
            path, digest = row
        elif type(row) is dict and set(row) == {"path", "sha256"}:
            path, digest = row["path"], row["sha256"]
        else:
            raise _Reject(stage, f"{label} source entry is malformed")
        if path != expected_path or not _is_sha256(digest):
            raise _Reject(stage, f"{label} source entry differs from frozen pins")
        rows.append((path, digest))
    return tuple(rows)


def _validate_event(value: object, *, stage: str) -> tuple[str, int, str]:
    if type(value) is not dict or set(value) != {
        "stage", "monotonic_ns", "occurred_at_utc", "clock"
    }:
        raise _Reject(stage, "score event schema is invalid")
    event_stage = value.get("stage")
    monotonic_ns = value.get("monotonic_ns")
    utc = value.get("occurred_at_utc")
    if (
        type(event_stage) is not str
        or not event_stage
        or type(monotonic_ns) is not int
        or not 0 <= monotonic_ns <= 2**63 - 1
        or type(utc) is not str
        or len(utc) > 32
        or not utc.endswith("Z")
        or value.get("clock") != "host-local-monotonic-ns"
    ):
        raise _Reject(stage, "score event values are invalid")
    try:
        parsed = datetime.fromisoformat(utc[:-1] + "+00:00")
        if parsed.utcoffset() is None:
            raise ValueError("UTC timestamp has no offset")
    except ValueError as error:
        raise _Reject(stage, "score event UTC timestamp is invalid") from error
    return event_stage, monotonic_ns, utc


def _valid_utc_text(value: object) -> bool:
    if type(value) is not str or len(value) > 32 or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _verify_score_result(
    score: object,
    *,
    frozen: ABC6FrozenReplayIdentity,
    marker_sha: str,
    expected_target_hashes: tuple[tuple[str, str], ...],
    stage: str,
) -> str:
    if type(score) is not dict or set(score) != {
        "protocol_id", "run_id", "reveal_marker_sha256", "target_sha256_by_truth",
        "case_scores", "paired_horizon_contrasts", "parameter_inclusion_counts",
        "no_crossing_diagnostics", "m_training_residual_means", "model_choice", "bayes_factors",
    }:
        raise _Reject(stage, "score result diagnostics schema is invalid")
    if (
        score.get("protocol_id") != frozen.protocol_id
        or score.get("run_id") != frozen.run_id
        or score.get("reveal_marker_sha256") != marker_sha
        or score.get("target_sha256_by_truth") != [list(row) for row in expected_target_hashes]
        or score.get("model_choice") is not None
        or score.get("bayes_factors") is not None
    ):
        raise _Reject(stage, "score result identity/hash links are invalid")
    case_scores = score.get("case_scores")
    pairs = score.get("paired_horizon_contrasts")
    inclusions = score.get("parameter_inclusion_counts")
    no_crossings = score.get("no_crossing_diagnostics")
    m_residuals = score.get("m_training_residual_means")
    if (
        type(case_scores) is not list
        or len(case_scores) != 24
        or type(pairs) is not list
        or len(pairs) != 8
        or type(inclusions) is not list
        or len(inclusions) != 24
        or type(no_crossings) is not list
        or len(no_crossings) != 2
        or type(m_residuals) is not list
        or len(m_residuals) != 2
    ):
        raise _Reject(stage, "score result omitted a fixed diagnostics group")
    for index, (case, row) in enumerate(zip(cases.CASE_ROSTER, case_scores, strict=True)):
        if (
            type(row) is not dict
            or row.get("case_index") != index
            or row.get("case_id") != case.case_id
            or row.get("truth_id") != case.truth_id
            or row.get("fit_model") != case.fit_model.value
            or row.get("input_window") != case.input_window
            or row.get("replicate") != case.replicate
        ):
            raise _Reject(stage, f"case score identity differs at roster index {index}")
        if case.truth_id == "N" and any(
            row.get(name) is not None
            for name in (
                "abc_weighted_mean_rmse", "abc_weighted_median_rmse",
                "baseline_coherent_rmse", "abc_q05_q95_envelope_inclusion_fraction",
                "terminal_effective_sample_size", "parameter_intervals",
            )
        ):
            raise _Reject(stage, "N score must abstain with no prospective score")
    if [row.get("case_index") for row in no_crossings] != [8, 9]:
        raise _Reject(stage, "N ceiling diagnostics are missing or reordered")
    if [row.get("case_index") for row in m_residuals] != [10, 11]:
        raise _Reject(stage, "M training residual diagnostics are missing or reordered")
    if any(row.get("truth_inclusion_claim") is not False for row in m_residuals):
        raise _Reject(stage, "M residual diagnostics make a forbidden truth-inclusion claim")
    try:
        return hashlib.sha256(_canonical_json(score)).hexdigest()
    except ValueError as error:
        raise _Reject(stage, "score result contains non-finite diagnostics") from error


def _verify_target_arrays(
    ctx: _ReadSession,
    *,
    score_receipt: dict[str, object],
) -> tuple[tuple[tuple[str, str], ...], bool]:
    prospective = score_receipt.get("prospective_targets")
    if type(prospective) is not dict:
        raise _Reject("target_arrays", "prospective target section is missing")
    target_name = prospective.get("target_array_artifact_filename")
    target_sha = prospective.get("target_array_artifact_sha256")
    target_rows = prospective.get("target_sha256_by_truth")
    has_artifact = ctx.maybe_present(ctx.receipt, scoring.TARGET_ARRAYS_ARTIFACT_FILENAME, stage="target_arrays")
    expected_n = {
        "prospective_target_generated": False,
        "prospective_score_computed": False,
        "mechanism_abstention": True,
        "status": "mechanism_abstention_no_target_no_score",
    }
    if prospective.get("N") != expected_n:
        raise _Reject("target_arrays", "N must remain an explicit no-target abstention")
    if target_name is None:
        if target_sha is not None or has_artifact:
            raise _Reject("target_arrays", "unexpected target-array artifact without a receipt link")
        if (
            prospective.get("truth_ids") != []
            or target_rows != [["A", None], ["B", None], ["M", None]]
        ):
            raise _Reject("target_arrays", "failed score claims targets before target publication")
        return (), False
    if (
        target_name != scoring.TARGET_ARRAYS_ARTIFACT_FILENAME
        or not _is_sha256(target_sha)
        or not has_artifact
        or prospective.get("truth_ids") != ["A", "B", "M"]
        or prospective.get("target_sha256_encoding") != "float64-little-endian-c-order"
        or type(target_rows) is not list
        or len(target_rows) != 3
    ):
        raise _Reject("target_arrays", "target-array artifact link is invalid")
    expected_hashes: list[tuple[str, str]] = []
    for truth, row in zip(("A", "B", "M"), target_rows, strict=True):
        if type(row) is not list or len(row) != 2 or row[0] != truth or not _is_sha256(row[1]):
            raise _Reject("target_arrays", "A/B/M target digest list is malformed")
        expected_hashes.append((truth, row[1]))
    raw, artifact, _ = ctx.read(
        ctx.receipt,
        scoring.TARGET_ARRAYS_ARTIFACT_FILENAME,
        maximum_bytes=MAX_JSON_BYTES,
        label="A/B/M target-array artifact",
        stage="target_arrays",
    )
    _payload_hash_valid(artifact, stage="target_arrays")
    if (
        hashlib.sha256(raw).hexdigest() != target_sha
        or set(artifact)
        != {
            "schema_version", "record_type", "protocol_id", "run_id", "truth_ids",
            "target_sha256_encoding", "target_sha256_by_truth", "targets", "N", "payload_sha256",
        }
        or artifact.get("schema_version") != 1
        or artifact.get("record_type") != "cascaded_tanks_abc6_prospective_target_arrays"
        or artifact.get("protocol_id") != ctx.frozen.protocol_id
        or artifact.get("run_id") != ctx.frozen.run_id
        or artifact.get("truth_ids") != ["A", "B", "M"]
        or artifact.get("target_sha256_encoding") != "float64-little-endian-c-order"
        or artifact.get("target_sha256_by_truth") != target_rows
        or artifact.get("N") != {
            "prospective_target_generated": False,
            "prospective_score_computed": False,
            "status": "mechanism_abstention_no_target_no_score",
        }
    ):
        raise _Reject("target_arrays", "A/B/M target-array artifact does not match score receipt")
    values = artifact.get("targets")
    if type(values) is not list or len(values) != 3:
        raise _Reject("target_arrays", "target arrays do not contain the fixed A/B/M roster")
    for truth, row, expected in zip(("A", "B", "M"), values, expected_hashes, strict=True):
        if type(row) is not dict or set(row) != {"truth_id", "values"} or row.get("truth_id") != truth:
            raise _Reject("target_arrays", f"{truth} target-array entry is malformed")
        vector = row.get("values")
        if type(vector) is not list or len(vector) != cases.PROSPECTIVE_LENGTH or any(type(x) not in (int, float) or not math.isfinite(float(x)) for x in vector):
            raise _Reject("target_arrays", f"{truth} target vector is invalid")
        binary = struct.pack("<" + "d" * len(vector), *(float(x) for x in vector))
        if hashlib.sha256(binary).hexdigest() != expected[1]:
            raise _Reject("target_arrays", f"{truth} target array digest differs")
    if artifact.get("N") != {
        "prospective_target_generated": False,
        "prospective_score_computed": False,
        "status": "mechanism_abstention_no_target_no_score",
    }:
        raise _Reject("target_arrays", "N no-target abstention record differs")
    return tuple(expected_hashes), True


def _verify_marker(
    ctx: _ReadSession,
    marker_raw: bytes,
    marker_value: dict[str, object],
    marker_identity: _FileIdentity,
    score: dict[str, object],
    *,
    source_hashes: tuple[tuple[str, str], ...],
    campaign_claim_sha: str,
    summary_sha256: str,
    evidence_manifest_sha256: str,
    forecast_sha256: str,
    forecast_roster_sha256: str,
    statuses: tuple[dict[str, object], ...],
) -> str:
    digest = hashlib.sha256(marker_raw).hexdigest()
    marker = score.get("reveal_marker")
    if type(marker) is not dict:
        raise _Reject("reveal_marker", "score receipt marker identity is missing")
    expected_path = (
        "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims/"
        f"{ctx.frozen.run_id}.claim"
    )
    expected_marker_identity = {
        "relative_path": expected_path,
        "sha256": digest,
        "parent_device": ctx.scorer_claim_parent.device,
        "parent_inode": ctx.scorer_claim_parent.inode,
        "device": marker_identity.device,
        "inode": marker_identity.inode,
        "size_bytes": marker_identity.size,
        "mtime_ns": marker_identity.mtime_ns,
        "ctime_ns": marker_identity.ctime_ns,
    }
    if marker != expected_marker_identity:
        raise _Reject("reveal_marker", "score receipt marker identity does not match pinned marker")
    digest = _verify_marker_pre_score_chain(
        ctx,
        marker_raw,
        marker_value,
        source_hashes=source_hashes,
        summary_sha256=summary_sha256,
        evidence_manifest_sha256=evidence_manifest_sha256,
        forecast_sha256=forecast_sha256,
        forecast_roster_sha256=forecast_roster_sha256,
        statuses=statuses,
    )
    if score.get("run_id") != ctx.frozen.run_id or score.get("protocol_id") != ctx.frozen.protocol_id:
        raise _Reject("deferred_score_receipt", "score receipt protocol/run identity is invalid")
    training = score.get("training")
    if type(training) is not dict or training.get("claim_sha256") != campaign_claim_sha:
        raise _Reject("deferred_score_receipt", "score receipt campaign claim link is invalid")
    return digest


def _verify_marker_pre_score_chain(
    ctx: _ReadSession,
    marker_raw: bytes,
    marker_value: dict[str, object],
    *,
    source_hashes: tuple[tuple[str, str], ...],
    summary_sha256: str,
    evidence_manifest_sha256: str,
    forecast_sha256: str,
    forecast_roster_sha256: str,
    statuses: tuple[dict[str, object], ...],
) -> str:
    """Verify the durable reveal claim against the target-free pre-score chain."""

    digest = hashlib.sha256(marker_raw).hexdigest()
    expected_marker_payload = {
        "schema": "cascaded-tanks-abc6-synthetic-reveal-v1",
        "protocol_id": ctx.frozen.protocol_id,
        "run_id": ctx.frozen.run_id,
        "condition": "synthetic-prospective-target-confirmation-v1",
        "semantics": "consumed-on-create; success-or-failure; no-retry",
        "forecast_roster_sha256": forecast_roster_sha256,
        "forecast_artifact_sha256": forecast_sha256,
        "training_manifest_sha256": ctx.frozen.manifest_sha256,
        "training_evidence_manifest_sha256": evidence_manifest_sha256,
        "integrated_source_hashes": [[path, sha] for path, sha in source_hashes],
        "status_receipt_sha256": [[row["filename"], row["sha256"]] for row in statuses],
        "training_summary_sha256": summary_sha256,
    }
    if any(marker_value.get(key) != value for key, value in expected_marker_payload.items()):
        raise _Reject("reveal_marker", "durable marker does not bind the frozen pre-score chain")
    if set(marker_value) != set(expected_marker_payload) | {"created_at_utc"}:
        raise _Reject("reveal_marker", "durable marker schema is unexpected")
    timestamp = marker_value.get("created_at_utc")
    if type(timestamp) is not str or len(timestamp) > 32:
        raise _Reject("reveal_marker", "durable marker timestamp is missing")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("marker timestamp is not UTC")
    except ValueError as error:
        raise _Reject("reveal_marker", "durable marker timestamp is invalid") from error
    return digest


def _verify_terminal(
    ctx: _ReadSession,
    *,
    expected_campaign_claim_sha: str | None = None,
    expected_summary_sha: str | None = None,
    expected_evidence_sha: str | None = None,
    expected_forecast_sha: str | None = None,
    statuses: tuple[dict[str, object], ...] | None = None,
) -> tuple[dict[str, object], int | None]:
    if ctx.scorer_claim_parent is None:
        raise _Reject(
            "watchdog_claim", "pinned scorer claim parent is unavailable"
        )
    try:
        os.fsync(ctx.receipt.descriptor)
    except OSError as error:
        raise _Reject("terminal_receipt", "could not sync the pinned receipt root") from error
    terminal_raw, terminal, terminal_file_identity = ctx.read(
        ctx.receipt,
        TERMINAL_FILENAME,
        maximum_bytes=MAX_JSON_BYTES,
        label="watchdog terminal receipt",
        stage="terminal_receipt",
    )
    ack_raw, ack, _ = ctx.read(
        ctx.receipt,
        TERMINAL_ACK_FILENAME,
        maximum_bytes=64 * 1024,
        label="terminal readback acknowledgment",
        stage="terminal_acknowledgment",
    )
    root_binding = {
        "repository_root_realpath": str(ctx.root.path),
        "repository_root_device": ctx.root.device,
        "repository_root_inode": ctx.root.inode,
        "receipt_root_relative": ctx.receipt.path.relative_to(ctx.root.path).as_posix(),
        "receipt_root_device": ctx.receipt.device,
        "receipt_root_inode": ctx.receipt.inode,
    }
    claim_path = (
        ctx.root.path
        / campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{ctx.frozen.run_id}.watchdog.claim"
    )
    ack_terminal = ack.get("terminal_receipt")
    if (
        set(ack)
        != {
            "schema_version", "kind", "protocol_id", "run_id", "manifest_sha256",
            "watchdog_claim_path", "watchdog_claim_sha256", "root_binding", "terminal_receipt",
        }
        or ack.get("schema_version") != 1
        or ack.get("kind") != "abc6-watchdog-terminal-readback-acknowledgment"
        or ack.get("protocol_id") != ctx.frozen.protocol_id
        or ack.get("run_id") != ctx.frozen.run_id
        or ack.get("manifest_sha256") != ctx.frozen.manifest_sha256
        or ack.get("watchdog_claim_path") != str(claim_path)
        or ack.get("watchdog_claim_sha256") != ctx.frozen.watchdog_claim_sha256
        or ack.get("root_binding") != root_binding
        or type(ack_terminal) is not dict
        or ack_terminal.get("leaf_name") != TERMINAL_FILENAME
        or ack_terminal.get("sha256") != hashlib.sha256(terminal_raw).hexdigest()
        or ack_terminal.get("file_identity")
        != {
            "device": terminal_file_identity.device,
            "inode": terminal_file_identity.inode,
            "size": terminal_file_identity.size,
            "mtime_ns": terminal_file_identity.mtime_ns,
            "ctime_ns": terminal_file_identity.ctime_ns,
        }
    ):
        raise _Reject("terminal_acknowledgment", "readback acknowledgment does not bind terminal receipt")
    terminal_root = {
        key: terminal.get(key)
        for key in root_binding
    }
    terminal_status = terminal.get("status")
    stop_reason = terminal.get("stop_reason")
    return_code = terminal.get("process_return_code")
    capture = terminal.get("intervention_capture")
    if (
        type(terminal.get("schema_version")) is not int
        or terminal.get("schema_version") != 3
        or terminal.get("protocol_id") != ctx.frozen.protocol_id
        or terminal.get("run_id") != ctx.frozen.run_id
        or terminal.get("manifest_sha256") != ctx.frozen.manifest_sha256
        or terminal.get("approval_record_sha256") != ctx.frozen.approval_record_sha256
        or terminal.get("reviewed_git_head") != ctx.frozen.reviewed_git_head
        or terminal_root != root_binding
        or terminal.get("watchdog_claim_path") != str(claim_path)
        or terminal.get("watchdog_claim_sha256") != ctx.frozen.watchdog_claim_sha256
        or terminal.get("watchdog_claim_parent_device") != ctx.campaign_claim_parent.device
        or terminal.get("watchdog_claim_parent_inode") != ctx.campaign_claim_parent.inode
        or terminal.get("scorer_claim_parent_device") != ctx.scorer_claim_parent.device
        or terminal.get("scorer_claim_parent_inode") != ctx.scorer_claim_parent.inode
        or terminal_status not in {"completed", "capped", "failed"}
        or type(stop_reason) is not str
        or not stop_reason
        or len(stop_reason) > 256
        or (return_code is not None and type(return_code) is not int)
        or (terminal_status == "completed" and (stop_reason != "child_exited" or return_code != 0))
        or (
            terminal_status == "capped"
            and stop_reason not in {"sampled_rss_limit_exceeded", "wall_clock_limit_exceeded"}
        )
        or (
            terminal_status == "failed"
            and stop_reason in {"sampled_rss_limit_exceeded", "wall_clock_limit_exceeded"}
        )
        or (
            terminal_status == "failed"
            and stop_reason == "child_exited"
            and return_code == 0
        )
    ):
        raise _Reject("terminal_receipt", "terminal identity differs from frozen watchdog claim")
    # Bind any durable grant before operational classification on both the
    # preclaim and post-marker paths. The separately supplied expected object
    # is the authority for every value, never the terminal or grant record.
    _verify_child_grant_record(ctx, terminal)
    fit_gate = terminal.get("fit_phase_gate")
    if type(fit_gate) is not dict:
        raise _Reject("terminal_fit_gate", "terminal receipt has no fit-phase gate")
    supplied_chain = (
        expected_campaign_claim_sha,
        expected_summary_sha,
        expected_evidence_sha,
        expected_forecast_sha,
        statuses,
    )
    if any(value is not None for value in supplied_chain):
        if any(value is None for value in supplied_chain):
            raise _Reject("terminal_fit_gate", "partial expected fit-chain binding was supplied")
        assert statuses is not None
        expected_gate_statuses = [
            {
                "case_index": row["case_index"],
                "case_id": row["case_id"],
                "component": row["component"],
                "status": row["status"],
                "filename": row["filename"],
                "path": str(ctx.receipt.path / str(row["filename"])),
                "sha256": row["sha256"],
            }
            for row in statuses
        ]
        if (
            fit_gate.get("status_receipt_count") != 48
            or fit_gate.get("status_receipts") != expected_gate_statuses
            or fit_gate.get("summary_sha256") != expected_summary_sha
            or fit_gate.get("evidence_manifest_sha256") != expected_evidence_sha
            or fit_gate.get("target_free_forecast_sha256") != expected_forecast_sha
            or fit_gate.get("pre_score_artifact_chain_valid") is not True
            or fit_gate.get("all_48_status_receipts_present_and_linked") is not True
            or fit_gate.get("training_evidence_chain_valid") is not True
            or fit_gate.get("target_free_forecast_chain_valid") is not True
        ):
            raise _Reject("terminal_fit_gate", "terminal receipt does not bind the verified pre-score chain")
    if type(capture) is not dict or set(capture) != {
        "status", "attempted_type", "attempted_stage", "failure_kind"
    }:
        raise _Reject("terminal_intervention_capture", "intervention capture schema is invalid")
    capture_status = capture.get("status")
    attempted_type = capture.get("attempted_type")
    attempted_stage = capture.get("attempted_stage")
    failure_kind = capture.get("failure_kind")
    intervention = terminal.get("watchdog_intervention")
    intervention_ns: int | None = None
    if capture_status == "not_attempted":
        if (
            attempted_type is not None
            or attempted_stage is not None
            or failure_kind is not None
            or intervention is not None
        ):
            raise _Reject(
                "terminal_intervention_capture",
                "not_attempted capture contradicts its event or attempted cause",
            )
    elif capture_status in {"recorded", "unavailable"}:
        if (
            type(attempted_type) is not str
            or type(attempted_stage) is not str
            or attempted_type not in _WATCHDOG_INTERVENTION_REASONS
            or attempted_stage not in _WATCHDOG_INTERVENTION_REASONS[attempted_type]
        ):
            raise _Reject(
                "terminal_intervention_capture", "attempted intervention cause is invalid"
            )
        if capture_status == "unavailable":
            if (
                failure_kind not in {"monotonic_unavailable", "event_snapshot_invalid"}
                or intervention is not None
            ):
                raise _Reject(
                    "terminal_intervention_capture",
                    "unavailable capture contradicts its bounded failure kind or event",
                )
        else:
            if failure_kind is not None:
                raise _Reject(
                    "terminal_intervention_capture",
                    "recorded capture cannot carry a failure kind",
                )
            if (
                type(intervention) is not dict
                or set(intervention)
                != {"type", "stage", "monotonic_ns", "occurred_at_utc", "clock"}
                or intervention.get("type") != attempted_type
                or intervention.get("stage") != attempted_stage
                or type(intervention.get("monotonic_ns")) is not int
                or not 0 <= intervention["monotonic_ns"] <= 2**63 - 1
                or intervention.get("clock") != "host-local-monotonic-ns"
                or (
                    intervention.get("occurred_at_utc") is not None
                    and not _valid_utc_text(intervention.get("occurred_at_utc"))
                )
            ):
                raise _Reject("terminal_intervention", "watchdog intervention event is invalid")
            intervention_ns = intervention["monotonic_ns"]
    else:
        raise _Reject("terminal_intervention_capture", "intervention capture status is invalid")
    if terminal_status == "completed" and capture_status != "not_attempted":
        raise _Reject(
            "terminal_intervention_capture", "completed terminal has an attempted intervention"
        )
    if terminal_status == "capped" and (
        stop_reason not in _WATCHDOG_INTERVENTION_REASONS["budget_cap"]
        or attempted_type != "budget_cap"
        or attempted_stage != stop_reason
        or capture_status == "not_attempted"
    ):
        raise _Reject("terminal_intervention", "capped terminal lacks its budget-cap cause")
    if terminal_status == "capped" and capture_status == "recorded" and (
        intervention is None or intervention.get("type") != "budget_cap"
    ):
        raise _Reject("terminal_intervention", "capped terminal lacks a recorded budget-cap event")
    try:
        os.fsync(ctx.receipt.descriptor)
    except OSError as error:
        raise _Reject("terminal_receipt", "could not sync the receipt root after readback") from error
    return terminal, intervention_ns


def _terminal_intervention_fields(
    terminal: dict[str, object],
) -> dict[str, object]:
    event = terminal.get("watchdog_intervention")
    if type(event) is not dict:
        return {
            "watchdog_intervention_type": None,
            "watchdog_intervention_stage": None,
            "watchdog_intervention_utc": None,
            "watchdog_intervention_clock": None,
        }
    return {
        "watchdog_intervention_type": event.get("type"),
        "watchdog_intervention_stage": event.get("stage"),
        "watchdog_intervention_utc": event.get("occurred_at_utc"),
        "watchdog_intervention_clock": event.get("clock"),
    }


def _terminal_capture_fields(terminal: dict[str, object]) -> dict[str, object]:
    capture = terminal.get("intervention_capture")
    if type(capture) is not dict:
        return {
            "watchdog_intervention_capture_status": None,
            "watchdog_intervention_attempted_type": None,
            "watchdog_intervention_attempted_stage": None,
            "watchdog_intervention_failure_kind": None,
        }
    return {
        "watchdog_intervention_capture_status": capture.get("status"),
        "watchdog_intervention_attempted_type": capture.get("attempted_type"),
        "watchdog_intervention_attempted_stage": capture.get("attempted_stage"),
        "watchdog_intervention_failure_kind": capture.get("failure_kind"),
    }


def _require_verified_child_reap(terminal: dict[str, object]) -> None:
    """Require retained direct-child attestation and a verified empty group."""

    launch = terminal.get("child_launch")
    observed = launch.get("observed_process") if type(launch) is dict else None
    declared_vector = launch.get("declared_launch_vector") if type(launch) is dict else None
    observations = launch.get("image_observations") if type(launch) is dict else None
    if (
        type(launch) is not dict
        or type(declared_vector) is not list
        or not declared_vector
        or any(type(value) is not str for value in declared_vector)
        or type(observed) is not dict
        or type(observed.get("observed_live_argv")) is not list
        or not observed["observed_live_argv"]
        or any(type(value) is not str for value in observed["observed_live_argv"])
        or type(observed.get("observed_executable_path")) is not str
        or not Path(observed["observed_executable_path"]).is_absolute()
        or not _is_sha256(observed.get("observed_executable_sha256"))
        or observed.get("verified_image_role") != "observed_python_app_image"
        or not _valid_utc_text(observed.get("observed_at_utc"))
        or type(observations) is not list
        or not any(
            type(row) is dict
            and row.get("path") == observed.get("observed_executable_path")
            and row.get("sha256") == observed.get("observed_executable_sha256")
            and row.get("phase") == "observed_python_app_image"
            for row in observations
        )
        or observed["observed_live_argv"][1:] != declared_vector[1:]
        or observed["observed_live_argv"][0]
        not in {declared_vector[0], observed.get("observed_executable_path")}
    ):
        raise _Reject(
            "terminal_child_attestation",
            "terminal does not retain a consistent observed child launch identity",
        )
    reap = terminal.get("kill_and_reap")
    if (
        type(reap) is not dict
        or reap.get("child_reaped") is not True
        or reap.get("tracked_process_group_reaped") is not True
        or reap.get("membership_verification") != "verified_empty"
        or reap.get("membership_enumeration_errors") != []
        or reap.get("unreaped_process_pids") != []
    ):
        raise _Reject(
            "terminal_child_reap",
            "terminal does not verify direct-child reap and an empty tracked process group",
        )


def _require_no_prior_terminal_error(terminal: dict[str, object]) -> None:
    fit_gate = terminal.get("fit_phase_gate")
    if type(fit_gate) is not dict or fit_gate.get("problems") != []:
        raise _Reject(
            "terminal_fit_gate", "terminal retains an earlier or unresolved fit-stage error"
        )
    monitor_error = terminal.get("monitor_error")
    capture = terminal.get("intervention_capture")
    expected_operator_stop = (
        type(capture) is dict
        and capture.get("status") == "recorded"
        and capture.get("attempted_type") == "operator_stop"
        and capture.get("attempted_stage") == "watchdog_interrupted"
        and monitor_error == "KeyboardInterrupt: "
    )
    if monitor_error is not None and not expected_operator_stop:
        raise _Reject(
            "terminal_monitor_error", "terminal retains an error with no monotonic order evidence"
        )


def _verify_score_receipt(
    ctx: _ReadSession,
    *,
    campaign_claim_sha: str,
    receipt_claim: dict[str, object],
    statuses: tuple[dict[str, object], ...],
    summary: dict[str, object],
    summary_sha256: str,
    evidence_manifest: dict[str, object],
    evidence_manifest_sha256: str,
    forecast: dict[str, object],
    forecast_sha256: str,
    forecast_array_sha256: str,
    source_hashes: tuple[tuple[str, str], ...],
    marker_raw: bytes,
    marker_value: dict[str, object],
    marker_identity: _FileIdentity,
) -> tuple[dict[str, object], str, tuple[tuple[str, str], ...], bool]:
    raw, score, _ = ctx.read(
        ctx.receipt,
        scoring.SCORE_RECEIPT_FILENAME,
        maximum_bytes=MAX_JSON_BYTES,
        label="deferred score receipt",
        stage="deferred_score_receipt",
    )
    _payload_hash_valid(score, stage="deferred_score_receipt")
    expected_receipt_rows = [
        {"filename": row["filename"], "sha256": row["sha256"]} for row in statuses
    ]
    case_status_rows = [
        {
            "case_index": case.case_index,
            "case_id": case.case_id,
            "fit_status": statuses[index * 2]["status"],
            "baseline_status": statuses[index * 2 + 1]["status"],
            "fit_receipt_sha256": statuses[index * 2]["sha256"],
            "baseline_receipt_sha256": statuses[index * 2 + 1]["sha256"],
        }
        for index, case in enumerate(cases.CASE_ROSTER)
    ]
    training = score.get("training")
    target_free = score.get("target_free_forecast")
    evidence_links = None if type(training) is not dict else training.get("evidence")
    expected_evidence_links = {
        "manifest_filename": campaign_fit.EVIDENCE_MANIFEST_FILENAME,
        "manifest_sha256": evidence_manifest_sha256,
        "artifact_directory": campaign_fit.EVIDENCE_DIRECTORY_NAME,
        "bundle_artifact": evidence_manifest.get("bundle_artifact"),
        "case_artifacts": [
            {
                "roster_index": row["roster_index"],
                "filename": row["filename"],
                "sha256": row["sha256"],
                "fit_status": row["fit_status"],
                "baseline_status": row["baseline_status"],
            }
            for row in evidence_manifest["case_artifacts"]
        ],
        "diagnostics_location": "case_artifacts contain complete fit/baseline costs and failure evidence",
    }
    if (
        set(score)
        != {
            "schema_version", "record_type", "outcome", "protocol_id", "run_id",
            "training", "target_free_forecast", "reveal_marker", "prospective_targets",
            "score_result", "score_result_sha256", "score_result_digest_encoding",
            "score_event", "failure_checkpoint", "retry_allowed", "payload_sha256",
        }
        or score.get("schema_version") != 1
        or score.get("record_type") != "cascaded_tanks_abc6_deferred_score_receipt"
        or score.get("protocol_id") != ctx.frozen.protocol_id
        or score.get("run_id") != ctx.frozen.run_id
        or score.get("retry_allowed") is not False
        or score.get("score_result_digest_encoding") != "canonical-json-score-result-body-v1"
        or type(training) is not dict
        or training.get("manifest_sha256") != ctx.frozen.manifest_sha256
        or training.get("claim_sha256") != campaign_claim_sha
        or training.get("status") != summary.get("status")
        or training.get("summary_filename") != campaign_fit.SUMMARY_FILENAME
        or training.get("summary_sha256") != summary_sha256
        or training.get("status_receipts") != expected_receipt_rows
        or training.get("case_statuses") != case_status_rows
        or evidence_links != expected_evidence_links
        or type(target_free) is not dict
        or target_free.get("artifact_filename") != "campaign.target-free-forecasts.json"
        or target_free.get("artifact_sha256") != forecast_sha256
        or target_free.get("forecast_roster_sha256") != forecast.get("forecast_roster_sha256")
        or target_free.get("ordered_case_sha256") != forecast.get("forecast_case_sha256")
        or target_free.get("forecast_array_sha256") != forecast_array_sha256
        or target_free.get("forecast_array_digest_encoding")
        != "canonical-json-ordered-full-forecast-records-v1"
    ):
        raise _Reject("deferred_score_receipt", "score receipt does not link the verified pre-score chain")
    marker_sha = _verify_marker(
        ctx,
        marker_raw,
        marker_value,
        marker_identity,
        score,
        source_hashes=source_hashes,
        campaign_claim_sha=campaign_claim_sha,
        summary_sha256=summary_sha256,
        evidence_manifest_sha256=evidence_manifest_sha256,
        forecast_sha256=forecast_sha256,
        forecast_roster_sha256=str(target_free["forecast_roster_sha256"]),
        statuses=statuses,
    )
    if score.get("score_event_monotonic_ns") is not None:
        raise _Reject("deferred_score_receipt", "score event must use the frozen nested event schema")
    outcome = score.get("outcome")
    if outcome not in {"complete", "failed"}:
        raise _Reject("deferred_score_receipt", "score receipt outcome is invalid")
    event_stage, _event_ns, _event_utc = _validate_event(
        score.get("score_event"), stage="score_event"
    )
    score_result = score.get("score_result")
    result_digest = score.get("score_result_sha256")
    if score_result is None:
        if result_digest is not None:
            raise _Reject("score_result", "score result digest exists without a result")
    else:
        computed_result_sha = hashlib.sha256(_canonical_json(score_result)).hexdigest()
        if result_digest != computed_result_sha:
            raise _Reject("score_result", "score result digest is invalid")
    target_hashes, target_artifact_verified = _verify_target_arrays(ctx, score_receipt=score)
    if outcome == "complete":
        if (
            event_stage != "score_complete"
            or score.get("failure_checkpoint") is not None
            or score_result is None
            or not target_artifact_verified
        ):
            raise _Reject("score_result", "complete score receipt lacks terminal score evidence")
        _verify_score_result(
            score_result,
            frozen=ctx.frozen,
            marker_sha=marker_sha,
            expected_target_hashes=target_hashes,
            stage="score_result",
        )
    else:
        checkpoint = score.get("failure_checkpoint")
        if type(checkpoint) is not dict or set(checkpoint) != {
            "stage", "occurred_at_monotonic_ns", "occurred_at_utc", "exception_type",
            "message", "retry_forbidden", "condition_consumed",
        }:
            raise _Reject("failure_checkpoint", "post-marker failure checkpoint is missing")
        if (
            checkpoint.get("stage") != event_stage
            or checkpoint.get("occurred_at_monotonic_ns") != score["score_event"]["monotonic_ns"]
            or checkpoint.get("occurred_at_utc") != score["score_event"]["occurred_at_utc"]
            or checkpoint.get("retry_forbidden") is not True
            or checkpoint.get("condition_consumed") is not True
            or type(checkpoint.get("exception_type")) is not str
            or type(checkpoint.get("message")) is not str
        ):
            raise _Reject("failure_checkpoint", "post-marker failure event does not match checkpoint")
        if score_result is not None:
            _verify_score_result(
                score_result,
                frozen=ctx.frozen,
                marker_sha=marker_sha,
                expected_target_hashes=target_hashes,
                stage="score_result",
            )
    return score, hashlib.sha256(raw).hexdigest(), target_hashes, target_artifact_verified


def verify_abc6_postscore_evidence(
    frozen: ABC6FrozenReplayIdentity,
) -> ABC6ReplayGate:
    """Read and revalidate the frozen post-score evidence chain without replay.

    No target, fit, forecast, simulator, source, or model computation is
    performed.  A valid result reports evidence classification, not scientific
    pass/fail.  Missing/tampered or identity-unstable post-marker evidence is
    always ``unreplayable`` and the function never repairs or retries it.
    """

    if not isinstance(frozen, ABC6FrozenReplayIdentity):
        raise TypeError("frozen must be ABC6FrozenReplayIdentity")
    ctx: _ReadSession | None = None
    state: dict[str, object] = {}
    try:
        ctx = _ReadSession(frozen)
        marker_present = (
            False
            if ctx.scorer_claim_parent is None
            else _open_presence(
                ctx, ctx.scorer_claim_parent, f"{frozen.run_id}.claim"
            )
        )
        state["marker_present"] = marker_present
        score_present = _open_presence(
            ctx, ctx.receipt, scoring.SCORE_RECEIPT_FILENAME
        )
        targets_present = _open_presence(
            ctx, ctx.receipt, scoring.TARGET_ARRAYS_ARTIFACT_FILENAME
        )
        watchdog_claim_present = _open_presence(
            ctx,
            ctx.campaign_claim_parent,
            f"{frozen.run_id}.watchdog.claim",
        )
        receipt_claim_present = _open_presence(
            ctx, ctx.receipt, campaign_fit.CLAIM_FILENAME
        )
        global_claim_present = _open_presence(
            ctx,
            ctx.campaign_claim_parent,
            f"{frozen.run_id}.claim",
        )
        terminal_present = _open_presence(ctx, ctx.receipt, TERMINAL_FILENAME)
        terminal_ack_present = _open_presence(ctx, ctx.receipt, TERMINAL_ACK_FILENAME)
        if (
            not marker_present
            and not score_present
            and not targets_present
            and not watchdog_claim_present
            and not terminal_present
            and not terminal_ack_present
        ):
            if receipt_claim_present or global_claim_present:
                raise _Reject(
                    "watchdog_claim",
                    "campaign claim exists without a verifiable supervised watchdog claim and terminal",
                )
            ctx.verify_stable()
            return ABC6ReplayGate(
                classification="premarker_absent",
                marker_present=False,
                failure_stage="reveal_marker_absent",
                detail="No durable reveal marker or post-score artifact exists.",
            )
        # Once the watchdog's one-use claim exists, canonical terminal and ACK
        # validation precedes every operational classification, including a
        # no-score outcome. A dangling claim or terminal cannot be mistaken for
        # work that never started.
        terminal, intervention_ns = _verify_terminal(ctx)
        state.update(
            terminal_status=terminal.get("status"),
            watchdog_intervention_monotonic_ns=intervention_ns,
            **_terminal_intervention_fields(terminal),
            **_terminal_capture_fields(terminal),
        )
        if terminal["intervention_capture"]["status"] == "unavailable":
            raise _Reject(
                "terminal_intervention_unavailable",
                "the intervention was attempted but no trustworthy monotonic event was captured",
            )
        if (
            not marker_present
            and not score_present
            and not targets_present
            and not receipt_claim_present
            and not global_claim_present
        ):
            _verify_watchdog_claim(ctx)
            _verify_no_campaign_chain(ctx)
            child_state = _verify_pre_campaign_child_state(ctx, terminal)
            classification, failure_stage, detail = _classify_no_campaign_claim(
                terminal, child_state
            )
            ctx.verify_stable()
            return ABC6ReplayGate(
                classification=classification,
                marker_present=False,
                failure_stage=failure_stage,
                detail=detail,
                terminal_status=str(terminal["status"]),
                watchdog_intervention_monotonic_ns=intervention_ns,
                **_terminal_intervention_fields(terminal),
                **_terminal_capture_fields(terminal),
                status_receipt_count=0,
                all_statuses_complete=False,
            )
        if marker_present and not score_present:
            raise _Reject(
                "post_marker_chain",
                "post-marker runs without a score receipt remain outside this staged verifier",
            )
        if not score_present and not targets_present:
            if not watchdog_claim_present:
                raise _Reject("watchdog_claim", "terminal evidence exists without its one-use claim")
            marker_raw: bytes | None = None
            marker_value: dict[str, object] | None = None
            source_hashes = frozen.integrated_source_hashes
            if marker_present:
                marker_raw, marker_value, _marker_identity = ctx.read(
                    ctx.scorer_claim_parent,
                    f"{frozen.run_id}.claim",
                    maximum_bytes=MAX_MARKER_BYTES,
                    label="fixed one-use reveal marker",
                    stage="reveal_marker",
                )
                state["marker_present"] = True
                source_hashes = _source_hash_rows(
                    marker_value.get("integrated_source_hashes"),
                    label="reveal marker",
                    stage="reveal_marker",
                )
                if source_hashes != frozen.integrated_source_hashes:
                    raise _Reject(
                        "reveal_marker",
                        "marker integrated source pins differ from frozen identity",
                    )
            campaign_claim_sha, _receipt_claim, _watchdog_claim = _verify_claims(ctx)
            statuses, summary, summary_sha, all_statuses_complete = _verify_statuses_and_summary(
                ctx, campaign_claim_sha
            )
            state.update(
                status_receipt_count=len(statuses),
                all_statuses_complete=all_statuses_complete,
            )
            evidence_manifest, evidence_sha = _verify_evidence_artifacts(
                ctx, statuses, summary, summary_sha, campaign_claim_sha
            )
            source_hashes = frozen.integrated_source_hashes
            forecast, forecast_sha, _forecast_array_sha, _forecast_records, scientific_ready = _verify_forecast(
                ctx,
                statuses,
                campaign_claim_sha,
                summary_sha,
                evidence_sha,
                source_hashes,
            )
            if marker_present:
                assert marker_raw is not None and marker_value is not None
                _verify_marker_pre_score_chain(
                    ctx,
                    marker_raw,
                    marker_value,
                    source_hashes=source_hashes,
                    summary_sha256=summary_sha,
                    evidence_manifest_sha256=evidence_sha,
                    forecast_sha256=forecast_sha,
                    forecast_roster_sha256=str(forecast["forecast_roster_sha256"]),
                    statuses=statuses,
                )
            terminal, intervention_ns = _verify_terminal(
                ctx,
                expected_campaign_claim_sha=campaign_claim_sha,
                expected_summary_sha=summary_sha,
                expected_evidence_sha=evidence_sha,
                expected_forecast_sha=forecast_sha,
                statuses=statuses,
            )
            state.update(
                terminal_status=terminal.get("status"),
                watchdog_intervention_monotonic_ns=intervention_ns,
                scientifically_ready=scientific_ready,
                **_terminal_intervention_fields(terminal),
                **_terminal_capture_fields(terminal),
            )
            status = terminal["status"]
            intervention = terminal.get("watchdog_intervention")
            stop_reason = terminal["stop_reason"]
            return_code = terminal.get("process_return_code")
            capture_status = terminal["intervention_capture"]["status"]
            attempted_type = terminal["intervention_capture"]["attempted_type"]
            if status == "completed":
                raise _Reject(
                    "terminal_score_chain",
                    "completed terminal has no score receipt",
                )
            if capture_status == "recorded" and attempted_type in {
                "budget_cap", "operator_stop"
            }:
                _require_verified_child_reap(terminal)
                classification: ReplayClassification = "incomplete"
                failure_stage = str(stop_reason)
                detail = (
                    "A verified watchdog intervention stopped the linked pre-score chain "
                    "after marker consumption."
                    if marker_present
                    else "A verified watchdog intervention stopped a fully linked pre-score chain."
                )
            elif (
                status == "failed"
                and capture_status == "not_attempted"
                and type(return_code) is int
                and return_code != 0
            ):
                _require_verified_child_reap(terminal)
                classification = "failed"
                failure_stage = str(stop_reason)
                detail = (
                    "A verified watchdog terminal records failure after marker consumption."
                    if marker_present
                    else "A verified watchdog terminal records failure before score handoff."
                )
            else:
                raise _Reject(
                    "terminal_score_chain",
                    "terminal outcome does not support a no-score classification",
                )
            ctx.verify_stable()
            return ABC6ReplayGate(
                classification=classification,
                marker_present=marker_present,
                failure_stage=failure_stage,
                detail=detail,
                terminal_status=str(status),
                watchdog_intervention_monotonic_ns=intervention_ns,
                **_terminal_intervention_fields(terminal),
                **_terminal_capture_fields(terminal),
                status_receipt_count=len(statuses),
                all_statuses_complete=all_statuses_complete,
                scientifically_ready=scientific_ready,
            )
        if not marker_present:
            raise _Reject("reveal_marker", "score or target artifact exists without reveal marker")
        if not score_present:
            raise _Reject("deferred_score_receipt", "marker exists but score receipt is missing")
        marker_raw, marker_value, marker_file_identity = ctx.read(
            ctx.scorer_claim_parent,
            f"{frozen.run_id}.claim",
            maximum_bytes=MAX_MARKER_BYTES,
            label="fixed one-use reveal marker",
            stage="reveal_marker",
        )
        source_hashes = _source_hash_rows(
            marker_value.get("integrated_source_hashes"),
            label="reveal marker",
            stage="reveal_marker",
        )
        if source_hashes != frozen.integrated_source_hashes:
            raise _Reject("reveal_marker", "marker integrated source pins differ from frozen identity")
        campaign_claim_sha, receipt_claim, watchdog_claim = _verify_claims(ctx)
        statuses, summary, summary_sha, all_statuses_complete = _verify_statuses_and_summary(
            ctx, campaign_claim_sha
        )
        state.update(
            status_receipt_count=len(statuses),
            all_statuses_complete=all_statuses_complete,
        )
        evidence_manifest, evidence_sha = _verify_evidence_artifacts(
            ctx, statuses, summary, summary_sha, campaign_claim_sha
        )
        forecast, forecast_sha, forecast_array_sha, _forecast_records, scientific_ready = _verify_forecast(
            ctx,
            statuses,
            campaign_claim_sha,
            summary_sha,
            evidence_sha,
            source_hashes,
        )
        state["scientifically_ready"] = scientific_ready
        score, score_sha, target_hashes, _target_artifact_verified = _verify_score_receipt(
            ctx,
            campaign_claim_sha=campaign_claim_sha,
            receipt_claim=receipt_claim,
            statuses=statuses,
            summary=summary,
            summary_sha256=summary_sha,
            evidence_manifest=evidence_manifest,
            evidence_manifest_sha256=evidence_sha,
            forecast=forecast,
            forecast_sha256=forecast_sha,
            forecast_array_sha256=forecast_array_sha,
            source_hashes=source_hashes,
            marker_raw=marker_raw,
            marker_value=marker_value,
            marker_identity=marker_file_identity,
        )
        score_event_stage, score_event_ns, score_event_utc = _validate_event(
            score.get("score_event"), stage="score_event"
        )
        state.update(
            marker_present=True,
            score_receipt_sha256=score_sha,
            score_outcome=score["outcome"],
            score_event_stage=score_event_stage,
            score_event_monotonic_ns=score_event_ns,
            score_event_utc=score_event_utc,
            score_event_clock=score["score_event"].get("clock"),
            verified_target_hashes=target_hashes,
        )
        terminal, intervention_ns = _verify_terminal(
            ctx,
            expected_campaign_claim_sha=campaign_claim_sha,
            expected_summary_sha=summary_sha,
            expected_evidence_sha=evidence_sha,
            expected_forecast_sha=forecast_sha,
            statuses=statuses,
        )
        state.update(
            marker_present=True,
            score_receipt_sha256=score_sha,
            score_outcome=score["outcome"],
            score_event_stage=score_event_stage,
            score_event_monotonic_ns=score_event_ns,
            score_event_utc=score_event_utc,
            score_event_clock=score["score_event"].get("clock"),
            terminal_status=terminal.get("status"),
            watchdog_intervention_monotonic_ns=intervention_ns,
            verified_target_hashes=target_hashes,
            **_terminal_intervention_fields(terminal),
            **_terminal_capture_fields(terminal),
        )
        if intervention_ns is not None and intervention_ns == score_event_ns:
            raise _Reject(
                "ambiguous_event_order",
                "score and watchdog events share monotonic_ns; raw events are retained on the gate",
            )
        intervention_preceded_event = (
            intervention_ns is not None and intervention_ns < score_event_ns
        )
        intervention_followed_event = (
            intervention_ns is not None and intervention_ns > score_event_ns
        )
        capture = terminal["intervention_capture"]
        if score["outcome"] == "failed":
            if intervention_preceded_event:
                classification: ReplayClassification = "incomplete"
                failure_stage = f"watchdog_intervention_before_{score_event_stage}"
                detail = "Watchdog intervention preceded the recorded score failure."
            elif intervention_followed_event:
                if terminal.get("status") not in {"capped", "failed"}:
                    raise _Reject(
                        "terminal_score_chain",
                        "post-score intervention contradicts the watchdog terminal status",
                    )
                _require_verified_child_reap(terminal)
                classification = "failed"
                checkpoint = score.get("failure_checkpoint")
                failure_stage = str(checkpoint["stage"])
                detail = "The recorded score failure preceded any watchdog intervention."
            elif (
                capture["status"] == "not_attempted"
                and terminal.get("status") == "failed"
                and type(terminal.get("process_return_code")) is int
                and terminal["process_return_code"] != 0
            ):
                _require_verified_child_reap(terminal)
                classification = "failed"
                checkpoint = score.get("failure_checkpoint")
                failure_stage = str(checkpoint["stage"])
                detail = "The validated score failure checkpoint matches a failed child terminal."
            else:
                raise _Reject(
                    "terminal_receipt",
                    "terminal fields do not establish the order of this failed score",
                )
        elif score["outcome"] == "complete":
            if intervention_preceded_event:
                classification = "incomplete"
                failure_stage = f"watchdog_intervention_before_{score_event_stage}"
                detail = "Watchdog intervention preceded the recorded score event."
            elif intervention_followed_event:
                if (
                    capture.get("attempted_type") not in {"budget_cap", "operator_stop"}
                    or terminal.get("status") not in {"capped", "failed"}
                ):
                    raise _Reject(
                        "terminal_score_chain",
                        "post-score watchdog stop lacks a supported complete-run predicate",
                    )
                _require_no_prior_terminal_error(terminal)
                _require_verified_child_reap(terminal)
                classification = "complete"
                failure_stage = None
                detail = (
                    "The score completed before a recorded budget or operator stop, and child cleanup verified."
                )
            elif terminal.get("status") == "completed":
                _require_verified_child_reap(terminal)
                classification = "complete"
                failure_stage = None
                detail = "The completed terminal and every deferred score link verified."
            else:
                raise _Reject(
                    "terminal_receipt",
                    "a missing intervention event cannot order a failed watchdog terminal against the score",
                )
        else:
            raise _Reject("score_result", "score outcome cannot be operationally classified")
        ctx.verify_stable()
        return ABC6ReplayGate(
            classification=classification,
            marker_present=True,
            score_receipt_sha256=score_sha,
            score_outcome=score["outcome"],
            score_event_stage=score_event_stage,
            score_event_monotonic_ns=score_event_ns,
            score_event_utc=score_event_utc,
            score_event_clock=score["score_event"].get("clock"),
            failure_stage=failure_stage,
            detail=detail,
            terminal_status=str(terminal.get("status")),
            watchdog_intervention_monotonic_ns=intervention_ns,
            **_terminal_intervention_fields(terminal),
            **_terminal_capture_fields(terminal),
            status_receipt_count=len(statuses),
            all_statuses_complete=all_statuses_complete,
            scientifically_ready=scientific_ready,
            verified_target_hashes=target_hashes,
        )
    except _Reject as error:
        return ABC6ReplayGate(
            classification="unreplayable",
            marker_present=bool(state.get("marker_present", False)),
            score_receipt_sha256=state.get("score_receipt_sha256"),
            score_outcome=state.get("score_outcome"),
            score_event_stage=state.get("score_event_stage"),
            score_event_monotonic_ns=state.get("score_event_monotonic_ns"),
            score_event_utc=state.get("score_event_utc"),
            score_event_clock=state.get("score_event_clock"),
            failure_stage=error.stage,
            detail=str(error),
            terminal_status=state.get("terminal_status"),
            watchdog_intervention_type=state.get("watchdog_intervention_type"),
            watchdog_intervention_stage=state.get("watchdog_intervention_stage"),
            watchdog_intervention_monotonic_ns=state.get("watchdog_intervention_monotonic_ns"),
            watchdog_intervention_utc=state.get("watchdog_intervention_utc"),
            watchdog_intervention_clock=state.get("watchdog_intervention_clock"),
            watchdog_intervention_capture_status=state.get("watchdog_intervention_capture_status"),
            watchdog_intervention_attempted_type=state.get("watchdog_intervention_attempted_type"),
            watchdog_intervention_attempted_stage=state.get("watchdog_intervention_attempted_stage"),
            watchdog_intervention_failure_kind=state.get("watchdog_intervention_failure_kind"),
            status_receipt_count=int(state.get("status_receipt_count", 0)),
            all_statuses_complete=state.get("all_statuses_complete"),
            scientifically_ready=state.get("scientifically_ready"),
        )
    except Exception as error:  # Fail closed on unforeseen parser/OS errors.
        return ABC6ReplayGate(
            classification="unreplayable",
            marker_present=bool(state.get("marker_present", False)),
            failure_stage="unexpected_verification_error",
            detail=f"{type(error).__name__}: {str(error)[:512]}",
            score_event_stage=state.get("score_event_stage"),
            score_event_monotonic_ns=state.get("score_event_monotonic_ns"),
            score_event_utc=state.get("score_event_utc"),
            score_event_clock=state.get("score_event_clock"),
            terminal_status=state.get("terminal_status"),
            watchdog_intervention_type=state.get("watchdog_intervention_type"),
            watchdog_intervention_stage=state.get("watchdog_intervention_stage"),
            watchdog_intervention_monotonic_ns=state.get("watchdog_intervention_monotonic_ns"),
            watchdog_intervention_utc=state.get("watchdog_intervention_utc"),
            watchdog_intervention_clock=state.get("watchdog_intervention_clock"),
            watchdog_intervention_capture_status=state.get("watchdog_intervention_capture_status"),
            watchdog_intervention_attempted_type=state.get("watchdog_intervention_attempted_type"),
            watchdog_intervention_attempted_stage=state.get("watchdog_intervention_attempted_stage"),
            watchdog_intervention_failure_kind=state.get("watchdog_intervention_failure_kind"),
            status_receipt_count=int(state.get("status_receipt_count", 0)),
            all_statuses_complete=state.get("all_statuses_complete"),
            scientifically_ready=state.get("scientifically_ready"),
        )
    finally:
        if ctx is not None:
            ctx.close()


__all__ = (
    "ABC6FrozenReplayIdentity",
    "ABC6ReplayGate",
    "verify_abc6_postscore_evidence",
)

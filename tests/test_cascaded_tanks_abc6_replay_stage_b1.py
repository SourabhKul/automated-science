"""Fake-only adversarial tests for the isolated ABC6 replay Stage B1 helper."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from core.real_data import (
    cascaded_tanks_abc6_campaign_fit as campaign_fit,
    cascaded_tanks_abc6_cases as cases,
    cascaded_tanks_abc6_replay as replay,
)


_PROTOCOL = "private-fake-abc6-stage-b1"
_RUN_ID = campaign_fit.RUN_ID
_HEAD = "a" * 40
_CLAIM_PARENT_RELATIVE = campaign_fit.CAMPAIGN_CLAIM_PARENT_RELATIVE
_SCORER_PARENT_RELATIVE = replay._STAGE_B1_SCORER_CLAIM_PARENT_RELATIVE
_USE_FIXTURE_EXPECTATION = object()


class _StageB1Fixture:
    def __init__(self, root_path: Path):
        root_path.mkdir()
        root_path = root_path.resolve(strict=True)
        self.root = cases._open_directory_anchor(root_path, label="fake Stage B1 root")
        self.receipt = cases._open_relative_directory_anchor(
            self.root,
            campaign_fit.RECEIPT_ROOT_RELATIVE,
            label="fake Stage B1 receipt root",
            create=True,
        )
        self.claim_parent = cases._open_relative_directory_anchor(
            self.root,
            _CLAIM_PARENT_RELATIVE,
            label="fake Stage B1 watchdog claim parent",
            create=True,
        )
        self.scorer_parent = cases._open_relative_directory_anchor(
            self.root,
            _SCORER_PARENT_RELATIVE,
            label="fake Stage B1 scorer claim parent",
            create=True,
        )
        self.claim_path = self.claim_parent.path / f"{_RUN_ID}.watchdog.claim"
        self.sidecar_path = self.claim_parent.path / f"{_RUN_ID}.launch-attestation.json"
        self.fixture_review_bytes = b"private fake Stage B1 approval review"
        self.fixture_review_sha256 = hashlib.sha256(self.fixture_review_bytes).hexdigest()
        self.fixture_manifest = {
            "protocol_id": _PROTOCOL,
            "run_id": _RUN_ID,
            "fixture_kind": "fake-stage-b1-only",
        }
        self.fixture_manifest_sha256 = hashlib.sha256(
            replay._canonical_json(self.fixture_manifest)
        ).hexdigest()
        self.fixture_approval = {
            "record_type": "private_fake_approval",
            "protocol_id": _PROTOCOL,
            "run_id": _RUN_ID,
            "manifest_sha256": self.fixture_manifest_sha256,
            "review_report_sha256": self.fixture_review_sha256,
            "reviewed_git_head": _HEAD,
        }
        self.fixture_approval_sha256 = hashlib.sha256(
            replay._canonical_json(self.fixture_approval)
        ).hexdigest()
        self.child_vector = ("/private/fake-python-bin", "-m", "fake_runner", "--private")
        self.child_image = "/private/fake-Python.app/Contents/MacOS/Python"
        self.child_resolved = "/private/fake-python-framework/bin/python3.14"
        self.watchdog_runtime = replay.ABC6StageB1WatchdogRuntimeExpectation(
            declared_launch_vector=(
                "/private/fake-python-bin",
                "/private/fake-watch.py",
                "--fake",
            ),
            declared_python_bin="/private/fake-python-bin",
            declared_python_bin_resolved_path="/private/fake-python-framework/bin/python3.14",
            declared_python_bin_sha256="4" * 64,
            observed_live_argv=(
                "/private/fake-Watchdog.app/Contents/MacOS/Python",
                "/private/fake-watch.py",
                "--fake",
            ),
            observed_python_app_image_path="/private/fake-Watchdog.app/Contents/MacOS/Python",
            observed_python_app_image_sha256="5" * 64,
            python_version="3.14.3-fake",
            psutil_version="7.2.2-fake",
            psutil_module_path="/private/fake-site/psutil/__init__.py",
            psutil_module_sha256="6" * 64,
        )
        self.expected = replay.ABC6StageB1Expectation(
            protocol_id=self.fixture_manifest["protocol_id"],
            run_id=self.fixture_manifest["run_id"],
            manifest_sha256=self.fixture_manifest_sha256,
            approval_record_sha256=self.fixture_approval_sha256,
            review_sha256=self.fixture_approval["review_report_sha256"],
            reviewed_git_head=self.fixture_approval["reviewed_git_head"],
            watchdog_claim_sha256="0" * 64,
            repository_root_realpath=str(self.root.path),
            repository_root_device=self.root.device,
            repository_root_inode=self.root.inode,
            receipt_root_relative=campaign_fit.RECEIPT_ROOT_RELATIVE,
            receipt_root_device=self.receipt.device,
            receipt_root_inode=self.receipt.inode,
            watchdog_claim_parent_device=self.claim_parent.device,
            watchdog_claim_parent_inode=self.claim_parent.inode,
            scorer_claim_parent_device=self.scorer_parent.device,
            scorer_claim_parent_inode=self.scorer_parent.inode,
            child_declared_launch_vector=self.child_vector,
            child_declared_executable_path=self.child_vector[0],
            child_declared_executable_resolved_path=self.child_resolved,
            child_declared_executable_sha256="7" * 64,
            child_accepted_observed_argv0=(self.child_vector[0], self.child_image),
            child_expected_observed_argv_tail=self.child_vector[1:],
            child_expected_observed_image_path=self.child_image,
            child_expected_observed_image_sha256="8" * 64,
            watchdog_runtime=self.watchdog_runtime,
            watchdog_process=replay.ABC6StageB1ProcessExpectation(
                pid=8101, start_identity="watchdog-start-8101"
            ),
            child_process=replay.ABC6StageB1ProcessExpectation(
                pid=8102, start_identity="child-start-8102"
            ),
        )
        claim = self._claim_object()
        sidecar = self._sidecar_object("0" * 64)
        self.terminal = self._publish(claim, sidecar)

    def _claim_object(self) -> dict[str, object]:
        expected = self.expected
        runtime = expected.watchdog_runtime
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
            "claimed_at_utc": "2026-09-30T12:00:00.000000Z",
            "declared_child_launch_vector": list(expected.child_declared_launch_vector),
            "declared_child_executable_path": expected.child_declared_executable_path,
            "declared_child_executable_resolved_path": expected.child_declared_executable_resolved_path,
            "declared_child_executable_sha256": expected.child_declared_executable_sha256,
            "accepted_observed_child_argv0": list(expected.child_accepted_observed_argv0),
            "expected_observed_child_argv_tail": list(expected.child_expected_observed_argv_tail),
            "expected_observed_child_image_path": expected.child_expected_observed_image_path,
            "expected_observed_child_image_sha256": expected.child_expected_observed_image_sha256,
            "watchdog_attestation": {
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
            },
            "claim_semantics": "consumed_once_no_resume",
        }

    def _sidecar_object(self, claim_sha256: str) -> dict[str, object]:
        expected = self.expected
        runtime = expected.watchdog_runtime
        return {
            "schema_version": 1,
            "protocol_id": expected.protocol_id,
            "run_id": expected.run_id,
            "manifest_sha256": expected.manifest_sha256,
            "approval_sha256": expected.approval_record_sha256,
            "review_sha256": expected.review_sha256,
            "reviewed_git_head": expected.reviewed_git_head,
            "watchdog_claim_sha256": claim_sha256,
            "physical_root": expected.repository_root_realpath,
            "root_device": expected.repository_root_device,
            "root_inode": expected.repository_root_inode,
            "receipt_root_relative": expected.receipt_root_relative,
            "receipt_device": expected.receipt_root_device,
            "receipt_inode": expected.receipt_root_inode,
            "claim_parent_device": expected.watchdog_claim_parent_device,
            "claim_parent_inode": expected.watchdog_claim_parent_inode,
            "observed_monotonic_ns": 234567890,
            "watchdog_process": {
                "pid": 8101,
                "start_identity": "watchdog-start-8101",
                "argv": list(runtime.observed_live_argv),
                "image_path": runtime.observed_python_app_image_path,
                "image_sha256": runtime.observed_python_app_image_sha256,
            },
            "child_process": {
                "pid": 8102,
                "start_identity": "child-start-8102",
                "argv": [self.child_image, *self.child_vector[1:]],
                "image_path": self.child_image,
                "image_sha256": expected.child_expected_observed_image_sha256,
                "declared_launch_vector": list(self.child_vector),
                "declared_launch_image_path": self.child_resolved,
                "declared_launch_image_sha256": expected.child_declared_executable_sha256,
            },
        }

    def _publish(
        self,
        claim: dict[str, object],
        sidecar: dict[str, object],
        *,
        claim_raw: bytes | None = None,
        sidecar_raw: bytes | None = None,
        canonical_sidecar: bool = True,
    ) -> dict[str, object]:
        claim_bytes = replay._canonical_json(claim) if claim_raw is None else claim_raw
        claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
        self.expected = replace(self.expected, watchdog_claim_sha256=claim_sha256)
        sidecar = dict(sidecar)
        sidecar["watchdog_claim_sha256"] = claim_sha256
        sidecar_bytes = (
            replay._canonical_json(sidecar)
            if canonical_sidecar
            else json.dumps(sidecar, indent=2, ensure_ascii=True).encode("ascii")
        )
        if sidecar_raw is not None:
            sidecar_bytes = sidecar_raw
        self.claim_path.write_bytes(claim_bytes)
        self.sidecar_path.write_bytes(sidecar_bytes)
        info = os.stat(self.sidecar_path, follow_symlinks=False)
        self.terminal = {
            "schema_version": 4,
            "watchdog_claim_sha256": claim_sha256,
            "launch_attestation": {
                "leaf": self.sidecar_path.name,
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                "device": info.st_dev,
                "inode": info.st_ino,
                "mode": info.st_mode,
                "size_bytes": info.st_size,
            },
        }
        return self.terminal

    def verify(self, terminal: dict[str, object] | None = None, expected=_USE_FIXTURE_EXPECTATION):
        return replay.verify_abc6_stage_b1_launch_evidence(
            self.root,
            self.receipt,
            self.claim_parent,
            self.scorer_parent,
            self.terminal if terminal is None else terminal,
            self.expected if expected is _USE_FIXTURE_EXPECTATION else expected,
        )

    def close(self) -> None:
        for anchor in (self.scorer_parent, self.claim_parent, self.receipt, self.root):
            anchor.close()


@pytest.fixture
def b1_case(tmp_path):
    case = _StageB1Fixture(tmp_path / "fake-b1-root")
    try:
        yield case
    finally:
        case.close()


def _require_unreplayable(result):
    assert result.status == "unreplayable"
    assert result.detail
    assert result.claim_sha256 is None
    assert result.launch_attestation_sha256 is None


def test_stage_b1_links_exact_fake_claim_and_sidecar(b1_case):
    result = b1_case.verify()
    assert result.status == "evidence_linked"
    assert result.claim_sha256 == b1_case.expected.watchdog_claim_sha256
    assert result.launch_attestation_sha256 == b1_case.terminal["launch_attestation"]["sha256"]


def test_stage_b1_missing_independent_expectation_is_explicitly_unreplayable(b1_case):
    _require_unreplayable(b1_case.verify(expected=None))


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "boolean_integer",
        "duplicate",
        "noncanonical",
        "nested_extra",
        "nested_missing",
        "nested_duplicate",
    ],
)
def test_stage_b1_rejects_nonexact_claim_schema_even_when_hash_links_match(b1_case, mutation):
    claim = b1_case._claim_object()
    raw = None
    if mutation == "extra":
        claim["unreviewed"] = "extra"
    elif mutation == "missing":
        claim.pop("approval_record_sha256")
    elif mutation == "boolean_integer":
        claim["repository_root_inode"] = True
    elif mutation == "duplicate":
        canonical = replay._canonical_json(claim)
        raw = canonical[:-1] + b',"schema_version":1}'
    elif mutation == "noncanonical":
        raw = json.dumps(claim, indent=2, ensure_ascii=True).encode("ascii")
    elif mutation == "nested_extra":
        claim["watchdog_attestation"]["unreviewed"] = "extra"
    elif mutation == "nested_missing":
        claim["watchdog_attestation"].pop("psutil_version")
    elif mutation == "nested_duplicate":
        canonical = replay._canonical_json(claim)
        field = b'"psutil_version":"7.2.2-fake"'
        raw = canonical.replace(field, field + b',"psutil_version":"7.2.2-fake"', 1)
    terminal = b1_case._publish(claim, b1_case._sidecar_object("0" * 64), claim_raw=raw)
    _require_unreplayable(b1_case.verify(terminal))


def test_stage_b1_rejects_self_consistent_but_wrong_nested_watchdog_runtime(b1_case):
    claim = b1_case._claim_object()
    wrong_argv = ["/private/other-watchdog", "/private/fake-watch.py", "--fake"]
    claim["watchdog_attestation"]["observed_live_argv"] = wrong_argv
    sidecar = b1_case._sidecar_object("0" * 64)
    sidecar["watchdog_process"]["argv"] = wrong_argv
    terminal = b1_case._publish(claim, sidecar)
    _require_unreplayable(b1_case.verify(terminal))


def test_stage_b1_rejects_self_consistent_but_wrong_child_argv_and_image(b1_case):
    sidecar = b1_case._sidecar_object(b1_case.expected.watchdog_claim_sha256)
    sidecar["child_process"]["argv"] = ["/private/other-python", "-m", "fake_runner", "--private"]
    sidecar["child_process"]["image_path"] = "/private/other-image"
    sidecar["child_process"]["image_sha256"] = "9" * 64
    terminal = b1_case._publish(b1_case._claim_object(), sidecar)
    _require_unreplayable(b1_case.verify(terminal))


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "boolean_integer",
        "duplicate",
        "noncanonical",
        "nested_extra",
        "nested_missing",
        "nested_duplicate",
    ],
)
def test_stage_b1_rejects_nonexact_sidecar_schema_even_when_terminal_hash_matches(
    b1_case, mutation
):
    sidecar = b1_case._sidecar_object(b1_case.expected.watchdog_claim_sha256)
    raw = None
    if mutation == "extra":
        sidecar["unreviewed"] = "extra"
    elif mutation == "missing":
        sidecar.pop("review_sha256")
    elif mutation == "boolean_integer":
        sidecar["watchdog_process"]["pid"] = True
    elif mutation == "duplicate":
        canonical = replay._canonical_json(sidecar)
        raw = canonical[:-1] + b',"schema_version":1}'
    elif mutation == "noncanonical":
        raw = json.dumps(sidecar, indent=2, ensure_ascii=True).encode("ascii")
    elif mutation == "nested_extra":
        sidecar["watchdog_process"]["extra"] = "extra"
    elif mutation == "nested_missing":
        sidecar["child_process"].pop("start_identity")
    elif mutation == "nested_duplicate":
        canonical = replay._canonical_json(sidecar)
        field = b'"start_identity":"watchdog-start-8101"'
        raw = canonical.replace(field, field + b',"start_identity":"watchdog-start-8101"', 1)
    terminal = b1_case._publish(
        b1_case._claim_object(), sidecar, sidecar_raw=raw, canonical_sidecar=raw is None
    )
    _require_unreplayable(b1_case.verify(terminal))


@pytest.mark.parametrize("field", ["sha256", "device", "inode", "mode", "size_bytes"])
def test_stage_b1_rejects_terminal_sidecar_reference_mismatch(b1_case, field):
    terminal = dict(b1_case.terminal)
    reference = dict(terminal["launch_attestation"])
    if field == "sha256":
        reference[field] = "f" * 64
    elif field == "mode":
        reference[field] = stat.S_IFDIR | 0o600
    else:
        reference[field] = reference[field] + 1
    terminal["launch_attestation"] = reference
    _require_unreplayable(b1_case.verify(terminal))


def test_stage_b1_rejects_boolean_terminal_reference_integer(b1_case):
    terminal = dict(b1_case.terminal)
    reference = dict(terminal["launch_attestation"])
    reference["inode"] = True
    terminal["launch_attestation"] = reference
    _require_unreplayable(b1_case.verify(terminal))


@pytest.mark.parametrize("leaf_kind", ["claim", "sidecar"])
def test_stage_b1_rejects_claim_or_sidecar_swap_from_another_fixture(
    tmp_path, b1_case, leaf_kind
):
    other = _StageB1Fixture(tmp_path / "other-fake-b1-root")
    try:
        if leaf_kind == "claim":
            shutil.copyfile(other.claim_path, b1_case.claim_path)
            terminal = b1_case.terminal
        else:
            shutil.copyfile(other.sidecar_path, b1_case.sidecar_path)
            info = os.stat(b1_case.sidecar_path, follow_symlinks=False)
            terminal = dict(b1_case.terminal)
            reference = dict(terminal["launch_attestation"])
            reference.update(
                sha256=hashlib.sha256(b1_case.sidecar_path.read_bytes()).hexdigest(),
                device=info.st_dev,
                inode=info.st_ino,
                mode=info.st_mode,
                size_bytes=info.st_size,
            )
            terminal["launch_attestation"] = reference
        _require_unreplayable(b1_case.verify(terminal))
    finally:
        other.close()


def test_stage_b1_rejects_symlink_claim_and_sidecar_leaves(b1_case, tmp_path):
    claim_target = tmp_path / "claim-copy.json"
    shutil.copyfile(b1_case.claim_path, claim_target)
    b1_case.claim_path.unlink()
    b1_case.claim_path.symlink_to(claim_target)
    _require_unreplayable(b1_case.verify())
    b1_case.claim_path.unlink()
    b1_case.claim_path.write_bytes(claim_target.read_bytes())

    sidecar_target = tmp_path / "sidecar-copy.json"
    shutil.copyfile(b1_case.sidecar_path, sidecar_target)
    b1_case.sidecar_path.unlink()
    b1_case.sidecar_path.symlink_to(sidecar_target)
    _require_unreplayable(b1_case.verify())


def test_stage_b1_requires_this_runs_fixed_sidecar_leaf(b1_case, tmp_path):
    foreign_leaf = b1_case.claim_parent.path / "another-run.launch-attestation.json"
    shutil.copyfile(b1_case.sidecar_path, foreign_leaf)
    b1_case.sidecar_path.unlink()
    _require_unreplayable(b1_case.verify())


@pytest.mark.parametrize("process_name", ["watchdog_process", "child_process"])
def test_stage_b1_compares_available_independent_start_identity_without_pid_sampling(
    b1_case, process_name
):
    original = (
        b1_case.expected.watchdog_process
        if process_name == "watchdog_process"
        else b1_case.expected.child_process
    )
    mismatched = replace(original, start_identity="different-fixture-start")
    expected = replace(b1_case.expected, **{process_name: mismatched})
    _require_unreplayable(b1_case.verify(expected=expected))


def test_stage_b1_rejects_claim_parent_from_a_copied_ancestor(tmp_path, b1_case):
    other = _StageB1Fixture(tmp_path / "copied-fake-b1-root")
    try:
        _require_unreplayable(
            replay.verify_abc6_stage_b1_launch_evidence(
                b1_case.root,
                b1_case.receipt,
                other.claim_parent,
                b1_case.scorer_parent,
                b1_case.terminal,
                b1_case.expected,
            )
        )
    finally:
        other.close()

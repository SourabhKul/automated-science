"""Fake-process checks for the external ABC6 watchdog seam."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import psutil
import pytest

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign
from scripts import watch_cascaded_tanks_abc6 as watchdog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_fake_child(
    tmp_path: Path,
    code: str,
    *,
    wall_seconds: float = 2.0,
    rss_limit_bytes: int = 2 * 1024**3,
    sample_seconds: float = 0.02,
    claim_path: Path | None = None,
    claim_project_root: Path | None = None,
) -> dict[str, object]:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    root_info = tmp_path.stat()
    receipt_info = receipt_directory.stat()
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    vector = [str(executable), "-c", code]
    return watchdog._supervise_command(
        launch_vector=vector,
        observed_executable_path=observed_image,
        observed_executable_sha256=_sha256(observed_image),
        claim_path=claim_path or tmp_path / "claims" / "fake-run.watchdog.claim",
        receipt_path=receipt_directory / "watchdog-terminal-receipt.json",
        claim_project_root=claim_project_root,
        run_identity={
            "protocol_id": "fake-protocol",
            "run_id": "fake-run",
            "manifest_sha256": "a" * 64,
            "repository_root_realpath": str(tmp_path),
            "repository_root_device": root_info.st_dev,
            "repository_root_inode": root_info.st_ino,
            "receipt_root_relative": "receipts",
            "receipt_root_device": receipt_info.st_dev,
            "receipt_root_inode": receipt_info.st_ino,
            "watchdog_image_path": str(observed_image),
            "watchdog_image_sha256": _sha256(observed_image),
            "watchdog_attestation": {"fixture": True},
        },
        wall_limit_seconds=wall_seconds,
        rss_limit_bytes=rss_limit_bytes,
        sample_interval_seconds=sample_seconds,
        terminate_grace_seconds=0.15,
        kill_reap_grace_seconds=0.20,
    )


def _private_preflight_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    child_vector_override: list[str] | None = None,
) -> dict[str, object]:
    root = tmp_path / "fake-physical-checkout"
    root.mkdir(parents=True)
    run_directory = root / watchdog._RUN_DIRECTORY_RELATIVE
    run_directory.mkdir(parents=True)
    review_report_path = run_directory / watchdog._REVIEW_REPORT_FILENAME
    review_report_raw = b"# Private fake independent review\n\nStatus: fake-only fixture.\n"
    review_report_path.write_bytes(review_report_raw)
    review_report_sha256 = hashlib.sha256(review_report_raw).hexdigest()
    receipt_directory = root / watchdog._RECEIPT_ROOT_RELATIVE
    receipt_directory.mkdir(parents=True)
    (root / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE).mkdir(parents=True)
    (root / watchdog._SCORER_CLAIM_PARENT_RELATIVE).mkdir(parents=True)
    script_path = root / "scripts" / "watch_cascaded_tanks_abc6.py"
    script_path.parent.mkdir()
    script_path.write_bytes(Path(watchdog.__file__).read_bytes())

    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    monkeypatch.setattr(watchdog, "_SCRIPT_PATH", script_path)
    head = "d" * 40
    monkeypatch.setattr(watchdog, "_current_git_head", lambda: head)

    manifest: dict[str, object] = {
        "schema_version": 2,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "repository_root_realpath": str(root),
        "receipt_root_relative": watchdog._RECEIPT_ROOT_RELATIVE,
        "reviewed_git_head": head,
        "reviewed_proposal": {"path": "proposal.md", "sha256": "a" * 64},
        "training_control_manifest_sha256": "b" * 64,
        "ordered_cases": [],
        "source_hashes": {},
        "runtime_fingerprint": {
            "declared_launch_interpreter_path": watchdog.PINNED_INTERPRETER_PATH,
            "resolved_running_interpreter_path": watchdog.PINNED_INTERPRETER_PATH,
            "python_version": watchdog.PINNED_PYTHON_VERSION,
            "interpreter_sha256": watchdog.PINNED_INTERPRETER_SHA256,
            "numpy_version": watchdog.PINNED_NUMPY_VERSION,
            "numpy_module_path": watchdog.PINNED_NUMPY_PATH,
            "numpy_module_sha256": watchdog.PINNED_NUMPY_SHA256,
            "psutil_version": watchdog.PINNED_PSUTIL_VERSION,
            "psutil_module_path": watchdog.PINNED_PSUTIL_PATH,
            "psutil_module_sha256": watchdog.PINNED_PSUTIL_SHA256,
        },
        "execution_contract": {
            "wall_clock_limit_seconds": int(watchdog.WALL_LIMIT_SECONDS),
            "runner_tree_rss_limit_bytes": watchdog.RSS_LIMIT_BYTES,
            "worker_count": 1,
            "watchdog_enforcement": "external",
            "independent_manifest_approval_required": True,
            "prospective_targets_before_receipts": False,
            "projected_artifact_bytes": 1024,
        },
    }
    manifest_path = root / watchdog._MANIFEST_RELATIVE
    manifest_raw = watchdog._canonical_json(manifest)
    manifest_path.write_bytes(manifest_raw)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    approval_path = root / watchdog._APPROVAL_RELATIVE
    launch_vector = child_vector_override or watchdog._build_campaign_command(
        manifest_path, manifest_sha256, receipt_directory
    )
    actual_args = [
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--receipt-directory",
        str(receipt_directory),
        "--approval-record",
        str(approval_path),
        "--approval-record-sha256",
        "e" * 64,
    ]
    approval: dict[str, object] = {
        "schema_version": 2,
        "record_type": "abc6_independent_approval_and_launch_v2",
        "status": "approved",
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": manifest_sha256,
        "reviewed_git_head": head,
        "reviewer_id": "private-fake-reviewer",
        "approved_at_utc": "2026-09-29T00:00:00Z",
        "watchdog_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path),
        "receipt_directory": str(receipt_directory),
        "watchdog_launch_vector_template": watchdog._watchdog_launch_vector_template(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            receipt_directory=receipt_directory,
            approval_path=approval_path,
        ),
        "campaign_child_launch_vector": launch_vector,
        "review_report_path": str(review_report_path),
        "review_report_sha256": review_report_sha256,
    }
    approval_raw = watchdog._canonical_json(approval)
    approval_path.write_bytes(approval_raw)
    approval_sha256 = hashlib.sha256(approval_raw).hexdigest()
    actual_args[-1] = approval_sha256
    monkeypatch.setattr(
        watchdog,
        "_campaign_strict_preflight",
        lambda _path, _sha, *, pinned_manifest_bytes: manifest_sha256,
    )
    return {
        "root": root,
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "receipt": receipt_directory,
        "approval": approval_path,
        "approval_sha256": approval_sha256,
        "review_report": review_report_path,
        "review_report_sha256": review_report_sha256,
        "actual_args": actual_args,
        "child_vector": launch_vector,
        "manifest_body": manifest,
    }


def test_fake_child_samples_pid_rss_hashes_receipt_and_excludes_watchdog(
    tmp_path: Path,
) -> None:
    result = _run_fake_child(tmp_path, "import time; time.sleep(0.12)")

    receipt_path = Path(result["receipt_path"])
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("ascii"))
    assert result["receipt"]["status"] == "failed"
    assert result["receipt"]["process_return_code"] == 0
    assert result["receipt"]["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
    assert result["receipt"]["fit_phase_gate"]["pre_score_artifact_chain_valid"] is False
    assert receipt["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
    assert receipt["watchdog_process_excluded_from_runner_tree"]["excluded"] is True
    watchdog_pid = receipt["watchdog_process_excluded_from_runner_tree"]["pid"]
    assert all(
        member["pid"] != watchdog_pid
        for row in receipt["rss_sampling"]["raw_samples"]
        for member in row["members"]
    )
    samples = receipt["rss_sampling"]["raw_samples"]
    assert len(samples) >= 2
    assert all("timestamp_utc" in row and "members" in row for row in samples)
    assert all(member["rss_bytes"] > 0 for row in samples for member in row["members"])
    expected_samples_hash = hashlib.sha256(
        watchdog._canonical_json(samples)
    ).hexdigest()
    assert receipt["rss_sampling"]["raw_samples_sha256"] == expected_samples_hash
    assert receipt["rss_sampling"]["peak_sampled_runner_tree_rss_bytes"] > 0
    assert receipt["terminal_receipt_publication"]["performed_by_this_writer"] is True
    assert receipt["terminal_receipt_publication"]["readback_required_for_successful_return"] is True
    assert receipt["terminal_receipt_publication"]["readback_ack_leaf"] == watchdog._TERMINAL_READBACK_ACK_FILENAME
    assert receipt["terminal_receipt_publication"]["readback_ack_required_for_replay"] is True
    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True
    assert result["receipt_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    ack_path = Path(result["terminal_readback_ack_path"])
    assert ack_path.name == watchdog._TERMINAL_READBACK_ACK_FILENAME
    ack = json.loads(ack_path.read_text(encoding="ascii"))
    assert ack["kind"] == "abc6-watchdog-terminal-readback-acknowledgment"
    assert ack["run_id"] == receipt["run_id"]
    assert ack["watchdog_claim_sha256"] == receipt["watchdog_claim_sha256"]
    assert ack["manifest_sha256"] == receipt["manifest_sha256"]
    assert ack["terminal_receipt"]["sha256"] == result["receipt_sha256"]
    terminal_info = receipt_path.stat()
    assert ack["terminal_receipt"]["file_identity"] == {
        "device": terminal_info.st_dev,
        "inode": terminal_info.st_ino,
        "size": terminal_info.st_size,
        "mtime_ns": terminal_info.st_mtime_ns,
        "ctime_ns": terminal_info.st_ctime_ns,
    }
    returned_receipt = dict(result["receipt"])
    returned_receipt.pop("terminal_receipt_sha256")
    assert receipt == returned_receipt
    assert not tuple(receipt_path.parent.glob("*.tmp"))
    assert Path(result["receipt_path"]).is_file()


def test_wall_budget_terminates_and_reaps_fake_sleeper_with_bounded_grace(
    tmp_path: Path,
) -> None:
    fake_score_event_path = tmp_path / "fake-score-event-monotonic-ns.txt"
    result = _run_fake_child(
        tmp_path,
        "import time; "
        f"open({str(fake_score_event_path)!r}, 'w').write(str(time.monotonic_ns())); "
        "time.sleep(10)",
        wall_seconds=0.12,
        sample_seconds=0.01,
    )

    receipt = result["receipt"]
    assert receipt["status"] == "capped"
    assert receipt["stop_reason"] == "wall_clock_limit_exceeded"
    kill_reap = receipt["kill_and_reap"]
    assert kill_reap["child_reaped"] is True
    assert kill_reap["tracked_process_group_reaped"] is True
    assert kill_reap["all_descendants_reaped_claimed"] is False
    assert "runner_tree_reaped" not in kill_reap
    assert kill_reap["bounded_grace_seconds"] == pytest.approx(0.45)
    assert kill_reap["elapsed_seconds"] < 1.0
    assert receipt["elapsed_wall_seconds"] < 1.0
    intervention = receipt["watchdog_intervention"]
    assert intervention["type"] == "budget_cap"
    assert intervention["stage"] == "wall_clock_limit_exceeded"
    assert intervention["clock"] == "host-local-monotonic-ns"
    assert type(intervention["monotonic_ns"]) is int
    assert intervention["occurred_at_utc"].endswith("Z")
    fake_score_event_monotonic_ns = int(
        fake_score_event_path.read_text(encoding="ascii")
    )
    assert fake_score_event_monotonic_ns < intervention["monotonic_ns"]


def test_child_error_before_budget_cap_has_no_watchdog_intervention(
    tmp_path: Path,
) -> None:
    result = _run_fake_child(
        tmp_path,
        "import time,sys; time.sleep(0.04); sys.exit(7)",
        wall_seconds=0.8,
        sample_seconds=0.01,
    )

    receipt = result["receipt"]
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "child_exited"
    assert receipt["process_return_code"] == 7
    assert receipt["watchdog_intervention"] is None


def test_operator_stop_records_first_comparable_intervention_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_snapshot = watchdog._snapshot_process_group
    interrupted = False

    def interrupt_first_sample(process_group_id: int):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_snapshot(process_group_id)

    monkeypatch.setattr(watchdog, "_snapshot_process_group", interrupt_first_sample)
    result = _run_fake_child(
        tmp_path,
        "import time; time.sleep(10)",
        wall_seconds=1.0,
        sample_seconds=0.01,
    )

    receipt = result["receipt"]
    intervention = receipt["watchdog_intervention"]
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "watchdog_interrupted"
    assert intervention["type"] == "operator_stop"
    assert intervention["stage"] == "watchdog_interrupted"
    assert intervention["clock"] == "host-local-monotonic-ns"
    assert type(intervention["monotonic_ns"]) is int
    assert intervention["occurred_at_utc"].endswith("Z")


@pytest.mark.parametrize(
    ("fault", "leaf_is_published"),
    [
        ("temporary_write", False),
        ("file_fsync", False),
        ("directory_fsync", True),
        ("hard_link", False),
        ("readback", True),
    ],
)
def test_terminal_receipt_publication_faults_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    leaf_is_published: bool,
) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private terminal-receipt test root"
    )
    root_info = tmp_path.stat()
    receipt_info = os.fstat(receipt_fd)
    original_fsync = watchdog.os.fsync
    original_link = watchdog.os.link
    original_readback = watchdog._read_canonical_object_at
    fsync_calls = 0

    if fault == "temporary_write":
        def fail_write(_fd: int, _payload: bytes) -> None:
            raise OSError("private injected terminal write failure")

        monkeypatch.setattr(watchdog, "_write_all", fail_write)
    elif fault in {"file_fsync", "directory_fsync"}:
        def fail_selected_fsync(fd: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            expected_call = 1 if fault == "file_fsync" else 2
            if fsync_calls == expected_call:
                raise OSError("private injected terminal fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(watchdog.os, "fsync", fail_selected_fsync)
    elif fault == "hard_link":
        def fail_link(*_args: object, **_kwargs: object) -> None:
            raise OSError("private injected terminal hard-link failure")

        monkeypatch.setattr(watchdog.os, "link", fail_link)
    elif fault == "readback":
        def fail_readback(
            directory_fd: int,
            leaf_name: str,
            *,
            maximum_bytes: int,
            label: str,
            snapshot: watchdog._GateSnapshot | None = None,
        ):
            if label == "watchdog terminal receipt readback":
                raise watchdog.WatchdogError("private injected terminal readback failure")
            return original_readback(
                directory_fd,
                leaf_name,
                maximum_bytes=maximum_bytes,
                label=label,
                snapshot=snapshot,
            )

        monkeypatch.setattr(watchdog, "_read_canonical_object_at", fail_readback)
    else:  # pragma: no cover - parameter list is exhaustive.
        raise AssertionError(f"unexpected terminal publication fault {fault}")

    terminal_path = receipt_directory / "watchdog-terminal-receipt.json"
    try:
        with pytest.raises(watchdog.WatchdogError):
            watchdog._publish_and_verify_terminal_receipt_at(
                receipt_fd,
                terminal_path.name,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "fake_only": True,
                    "protocol_id": "fake-protocol",
                    "run_id": "fake-run",
                    "manifest_sha256": "a" * 64,
                    "watchdog_claim_path": str(
                        tmp_path / "claims" / "fake-run.watchdog.claim"
                    ),
                    "watchdog_claim_sha256": "b" * 64,
                    "repository_root_realpath": str(tmp_path),
                    "repository_root_device": root_info.st_dev,
                    "repository_root_inode": root_info.st_ino,
                    "receipt_root_relative": "receipts",
                    "receipt_root_device": receipt_info.st_dev,
                    "receipt_root_inode": receipt_info.st_ino,
                },
            )
    finally:
        os.close(receipt_fd)

    assert terminal_path.exists() is leaf_is_published
    assert not (receipt_directory / watchdog._TERMINAL_READBACK_ACK_FILENAME).exists()
    assert not tuple(receipt_directory.glob(".*.tmp"))


@pytest.mark.parametrize("fault", ["file_fsync", "directory_fsync", "hard_link"])
def test_terminal_readback_ack_publication_fault_is_not_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private terminal-ack test root"
    )
    root_info = tmp_path.stat()
    receipt_info = os.fstat(receipt_fd)
    original_fsync = watchdog.os.fsync
    original_link = watchdog.os.link
    fsync_calls = 0

    def fail_ack_fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        expected_call = 4 if fault == "file_fsync" else 5
        if fault in {"file_fsync", "directory_fsync"} and fsync_calls == expected_call:
            raise OSError("private injected acknowledgment fsync failure")
        original_fsync(fd)

    def fail_ack_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if destination == watchdog._TERMINAL_READBACK_ACK_FILENAME:
            raise OSError("private injected acknowledgment hard-link failure")
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    if fault in {"file_fsync", "directory_fsync"}:
        monkeypatch.setattr(watchdog.os, "fsync", fail_ack_fsync)
    else:
        monkeypatch.setattr(watchdog.os, "link", fail_ack_link)

    terminal_path = receipt_directory / "watchdog-terminal-receipt.json"
    terminal_value = {
        "schema_version": 1,
        "status": "completed",
        "protocol_id": "fake-protocol",
        "run_id": "fake-run",
        "manifest_sha256": "a" * 64,
        "watchdog_claim_path": str(tmp_path / "claims" / "fake-run.watchdog.claim"),
        "watchdog_claim_sha256": "b" * 64,
        "repository_root_realpath": str(tmp_path),
        "repository_root_device": root_info.st_dev,
        "repository_root_inode": root_info.st_ino,
        "receipt_root_relative": "receipts",
        "receipt_root_device": receipt_info.st_dev,
        "receipt_root_inode": receipt_info.st_ino,
    }
    try:
        with pytest.raises(watchdog.WatchdogError, match="acknowledgment"):
            terminal_sha256, terminal_info = watchdog._publish_and_verify_terminal_receipt_at(
                receipt_fd,
                terminal_path.name,
                terminal_value,
            )
            watchdog._publish_terminal_readback_ack_at(
                receipt_fd,
                terminal_path.name,
                terminal_value,
                terminal_sha256,
                terminal_info,
            )
    finally:
        os.close(receipt_fd)

    assert terminal_path.is_file()
    assert not (receipt_directory / watchdog._TERMINAL_READBACK_ACK_FILENAME).exists()
    assert json.loads(terminal_path.read_text(encoding="ascii"))["status"] == "completed"
    assert not tuple(receipt_directory.glob(".*.tmp"))


def test_terminal_publication_failure_keeps_claim_and_prevents_second_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    claim_path = tmp_path / "claims" / "fake-run.watchdog.claim"
    receipt_path = receipt_directory / "watchdog-terminal-receipt.json"
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    launch_vector = [str(executable), "-c", "import time; time.sleep(0.05)"]
    original_popen = watchdog.subprocess.Popen
    child_start_count = 0

    def counted_popen(*args: object, **kwargs: object):
        nonlocal child_start_count
        child_start_count += 1
        return original_popen(*args, **kwargs)

    def fail_terminal_publication(*_args: object, **_kwargs: object) -> str:
        raise watchdog.WatchdogError("private injected terminal temp-write failure")

    monkeypatch.setattr(watchdog.subprocess, "Popen", counted_popen)
    monkeypatch.setattr(
        watchdog,
        "_publish_and_verify_terminal_receipt_at",
        fail_terminal_publication,
    )

    def supervise_once() -> dict[str, object]:
        return watchdog._supervise_command(
            launch_vector=launch_vector,
            observed_executable_path=observed_image,
            observed_executable_sha256=_sha256(observed_image),
            claim_path=claim_path,
            receipt_path=receipt_path,
            run_identity={
                "protocol_id": "fake-protocol",
                "run_id": "fake-run",
                "manifest_sha256": "a" * 64,
                "watchdog_attestation": {"private_fake": True},
            },
            wall_limit_seconds=1.0,
            rss_limit_bytes=32 * 1024**2,
            sample_interval_seconds=0.01,
        )

    with pytest.raises(watchdog.WatchdogError, match="terminal temp-write"):
        supervise_once()
    assert child_start_count == 1
    assert claim_path.is_file()
    assert tuple(receipt_directory.iterdir()) == ()

    with pytest.raises(watchdog.AlreadyClaimedError):
        supervise_once()
    assert child_start_count == 1


def test_post_link_terminal_readback_failure_leaves_completed_receipt_unacknowledged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watchdog, "RUN_ID", "fake-run")
    monkeypatch.setattr(watchdog, "PROTOCOL_ID", "fake-protocol")
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    root_info = tmp_path.stat()
    receipt_info = receipt_directory.stat()
    claim_path = tmp_path / "claims" / "fake-run.watchdog.claim"
    receipt_path = receipt_directory / "watchdog-terminal-receipt.json"
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    original_inspect = watchdog._inspect_fit_phase_gate
    original_readback = watchdog._read_canonical_object_at
    original_popen = watchdog.subprocess.Popen
    child_start_count = 0

    def inspect_with_private_hashes(
        receipt_directory_fd: int,
        *,
        expected_receipt_directory: Path,
        expected_manifest_sha256: str,
        sync_directory: bool,
        preflight_anchors: watchdog._PreflightAnchors | None = None,
        expected_integrated_source_hashes: dict[str, str] | None = None,
        retain_snapshot: bool = False,
    ) -> dict[str, object]:
        source_hashes = _write_private_fit_artifact_chain(receipt_directory)
        return original_inspect(
            receipt_directory_fd,
            expected_receipt_directory=expected_receipt_directory,
            expected_manifest_sha256=expected_manifest_sha256,
            sync_directory=sync_directory,
            preflight_anchors=preflight_anchors,
            expected_integrated_source_hashes=source_hashes,
            retain_snapshot=retain_snapshot,
        )

    def fail_after_terminal_link(
        directory_fd: int,
        leaf_name: str,
        *,
        maximum_bytes: int,
        label: str,
        snapshot: watchdog._GateSnapshot | None = None,
    ):
        if label == "watchdog terminal receipt readback":
            raise watchdog.WatchdogError("private injected post-link readback failure")
        return original_readback(
            directory_fd,
            leaf_name,
            maximum_bytes=maximum_bytes,
            label=label,
            snapshot=snapshot,
        )

    def counted_popen(*args: object, **kwargs: object):
        nonlocal child_start_count
        child_start_count += 1
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(watchdog, "_inspect_fit_phase_gate", inspect_with_private_hashes)
    monkeypatch.setattr(watchdog, "_read_canonical_object_at", fail_after_terminal_link)
    monkeypatch.setattr(watchdog.subprocess, "Popen", counted_popen)

    def supervise_once(
        target_receipt_path: Path = receipt_path,
    ) -> dict[str, object]:
        target_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        target_receipt_info = target_receipt_path.parent.stat()
        return watchdog._supervise_command(
            launch_vector=[str(executable), "-c", "import time; time.sleep(0.05)"],
            observed_executable_path=observed_image,
            observed_executable_sha256=_sha256(observed_image),
            claim_path=claim_path,
            receipt_path=target_receipt_path,
            run_identity={
                "protocol_id": "fake-protocol",
                "run_id": "fake-run",
                "manifest_sha256": "c" * 64,
                "repository_root_realpath": str(tmp_path),
                "repository_root_device": root_info.st_dev,
                "repository_root_inode": root_info.st_ino,
                "receipt_root_relative": target_receipt_path.parent.name,
                "receipt_root_device": target_receipt_info.st_dev,
                "receipt_root_inode": target_receipt_info.st_ino,
                "watchdog_attestation": {"private_fake": True},
            },
            wall_limit_seconds=1.0,
            rss_limit_bytes=32 * 1024**2,
            sample_interval_seconds=0.01,
        )

    with pytest.raises(watchdog.WatchdogError, match="post-link readback"):
        supervise_once()
    assert child_start_count == 1
    assert claim_path.is_file()
    assert receipt_path.is_file()
    terminal = json.loads(receipt_path.read_text(encoding="ascii"))
    assert terminal["status"] == "completed"
    assert terminal["terminal_receipt_publication"][
        "readback_ack_required_for_replay"
    ] is True
    assert not (receipt_directory / watchdog._TERMINAL_READBACK_ACK_FILENAME).exists()
    assert not tuple(receipt_directory.glob(".*.tmp"))

    retry_directory = tmp_path / "retry-receipts"
    with pytest.raises(watchdog.AlreadyClaimedError):
        supervise_once(retry_directory / "watchdog-terminal-receipt.json")
    assert child_start_count == 1


def test_post_effect_ack_link_error_preserves_canonical_readback_linearization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A surviving canonical ack records the on-disk readback linearization point."""

    monkeypatch.setattr(watchdog, "RUN_ID", "fake-run")
    monkeypatch.setattr(watchdog, "PROTOCOL_ID", "fake-protocol")
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    root_info = tmp_path.stat()
    receipt_info = receipt_directory.stat()
    claim_path = tmp_path / "claims" / "fake-run.watchdog.claim"
    receipt_path = receipt_directory / "watchdog-terminal-receipt.json"
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    original_inspect = watchdog._inspect_fit_phase_gate
    original_link = watchdog.os.link
    original_popen = watchdog.subprocess.Popen
    child_start_count = 0
    post_effect_link_error = False

    def inspect_with_private_hashes(
        receipt_directory_fd: int,
        *,
        expected_receipt_directory: Path,
        expected_manifest_sha256: str,
        sync_directory: bool,
        preflight_anchors: watchdog._PreflightAnchors | None = None,
        expected_integrated_source_hashes: dict[str, str] | None = None,
        retain_snapshot: bool = False,
    ) -> dict[str, object]:
        source_hashes = _write_private_fit_artifact_chain(receipt_directory)
        return original_inspect(
            receipt_directory_fd,
            expected_receipt_directory=expected_receipt_directory,
            expected_manifest_sha256=expected_manifest_sha256,
            sync_directory=sync_directory,
            preflight_anchors=preflight_anchors,
            expected_integrated_source_hashes=source_hashes,
            retain_snapshot=retain_snapshot,
        )

    def link_then_raise_after_ack(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal post_effect_link_error
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if destination == watchdog._TERMINAL_READBACK_ACK_FILENAME:
            post_effect_link_error = True
            raise OSError("private injected link error after acknowledgment creation")

    def counted_popen(*args: object, **kwargs: object):
        nonlocal child_start_count
        child_start_count += 1
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(watchdog, "_inspect_fit_phase_gate", inspect_with_private_hashes)
    monkeypatch.setattr(watchdog.os, "link", link_then_raise_after_ack)
    monkeypatch.setattr(watchdog.subprocess, "Popen", counted_popen)

    def supervise_once(target_receipt_path: Path = receipt_path) -> dict[str, object]:
        target_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        target_receipt_info = target_receipt_path.parent.stat()
        return watchdog._supervise_command(
            launch_vector=[str(executable), "-c", "import time; time.sleep(0.05)"],
            observed_executable_path=observed_image,
            observed_executable_sha256=_sha256(observed_image),
            claim_path=claim_path,
            receipt_path=target_receipt_path,
            run_identity={
                "protocol_id": "fake-protocol",
                "run_id": "fake-run",
                "manifest_sha256": "c" * 64,
                "repository_root_realpath": str(tmp_path),
                "repository_root_device": root_info.st_dev,
                "repository_root_inode": root_info.st_ino,
                "receipt_root_relative": target_receipt_path.parent.name,
                "receipt_root_device": target_receipt_info.st_dev,
                "receipt_root_inode": target_receipt_info.st_ino,
                "watchdog_attestation": {"private_fake": True},
            },
            wall_limit_seconds=1.0,
            rss_limit_bytes=32 * 1024**2,
            sample_interval_seconds=0.01,
        )

    with pytest.raises(watchdog.WatchdogError, match="acknowledgment.*durably published"):
        supervise_once()

    assert post_effect_link_error is True
    assert child_start_count == 1
    assert claim_path.is_file()
    terminal_bytes = receipt_path.read_bytes()
    terminal = json.loads(terminal_bytes.decode("ascii"))
    assert terminal["status"] == "completed"
    assert terminal["terminal_receipt_publication"][
        "readback_ack_required_for_replay"
    ] is True
    ack_path = receipt_directory / watchdog._TERMINAL_READBACK_ACK_FILENAME
    assert ack_path.is_file()
    ack_bytes = ack_path.read_bytes()
    ack = json.loads(ack_bytes.decode("ascii"))
    assert watchdog._canonical_json(ack) == ack_bytes
    assert ack["kind"] == "abc6-watchdog-terminal-readback-acknowledgment"
    assert ack["run_id"] == terminal["run_id"]
    assert ack["manifest_sha256"] == terminal["manifest_sha256"]
    assert ack["watchdog_claim_path"] == terminal["watchdog_claim_path"]
    assert ack["watchdog_claim_sha256"] == terminal["watchdog_claim_sha256"]
    assert ack["root_binding"]["repository_root_realpath"] == str(tmp_path)
    assert ack["root_binding"]["receipt_root_device"] == receipt_info.st_dev
    assert ack["root_binding"]["receipt_root_inode"] == receipt_info.st_ino
    assert ack["terminal_receipt"]["sha256"] == hashlib.sha256(
        terminal_bytes
    ).hexdigest()
    terminal_info = receipt_path.stat()
    assert ack["terminal_receipt"]["file_identity"] == {
        "device": terminal_info.st_dev,
        "inode": terminal_info.st_ino,
        "size": terminal_info.st_size,
        "mtime_ns": terminal_info.st_mtime_ns,
        "ctime_ns": terminal_info.st_ctime_ns,
    }
    assert not tuple(receipt_directory.glob(".*.tmp"))

    retry_directory = tmp_path / "retry-receipts"
    with pytest.raises(watchdog.AlreadyClaimedError):
        supervise_once(retry_directory / "watchdog-terminal-receipt.json")
    assert child_start_count == 1


def test_sampled_rss_budget_terminates_fake_allocator(tmp_path: Path) -> None:
    result = _run_fake_child(
        tmp_path,
        "import time; blob=bytearray(32*1024*1024); time.sleep(10)",
        wall_seconds=2.0,
        rss_limit_bytes=512 * 1024,
        sample_seconds=0.02,
    )

    receipt = result["receipt"]
    assert receipt["status"] == "capped"
    assert receipt["stop_reason"] == "sampled_rss_limit_exceeded"
    assert receipt["rss_sampling"]["peak_sampled_runner_tree_rss_bytes"] > 512 * 1024
    assert receipt["kill_and_reap"]["child_reaped"] is True


def test_unexpected_descendant_is_sampled_and_terminated_with_child_tree(
    tmp_path: Path,
) -> None:
    child_code = "import time; time.sleep(10)"
    parent_code = (
        "import subprocess,time; "
        f"subprocess.Popen([{watchdog.PINNED_PYTHON_BIN!r}, '-c', {child_code!r}]); "
        "time.sleep(10)"
    )
    result = _run_fake_child(
        tmp_path,
        parent_code,
        wall_seconds=2.0,
        sample_seconds=0.01,
    )

    receipt = result["receipt"]
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "unexpected_descendant_process"
    assert any(len(row["members"]) > 1 for row in receipt["rss_sampling"]["raw_samples"])
    assert receipt["kill_and_reap"]["child_reaped"] is True


def test_orphaned_group_member_is_found_after_direct_child_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "orphan-pids.json"
    sleeper = "import time; time.sleep(30)"
    code = (
        "import json,os,subprocess; "
        f"p=subprocess.Popen([{watchdog.PINNED_PYTHON_BIN!r}, '-c', {sleeper!r}]); "
        f"open({str(marker)!r}, 'w').write(json.dumps([os.getpid(),p.pid])); "
        "os._exit(0)"
    )
    original_snapshot = watchdog._snapshot_process_group
    first_snapshot = True

    def wait_until_root_exited(process_group_id: int):
        nonlocal first_snapshot
        if first_snapshot:
            first_snapshot = False
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.002)
            # The direct child writes the marker immediately before os._exit.
            # Delay the first membership walk so the regression specifically
            # exercises an orphan after its root is gone.
            time.sleep(0.06)
        return original_snapshot(process_group_id)

    monkeypatch.setattr(watchdog, "_snapshot_process_group", wait_until_root_exited)
    result = _run_fake_child(tmp_path, code, wall_seconds=1.0, sample_seconds=0.01)
    receipt = result["receipt"]
    pids = json.loads(marker.read_text(encoding="ascii"))

    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "child_exited_with_live_process_group_members"
    assert receipt["kill_and_reap"]["tracked_process_group_reaped"] is True
    assert receipt["kill_and_reap"]["membership_verification"] == "verified_empty"
    assert any(
        member["pid"] == pids[1]
        for row in receipt["rss_sampling"]["raw_samples"]
        for member in row["members"]
    )
    assert not psutil.pid_exists(pids[1])


def test_access_denied_kills_group_but_receipt_stays_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "member-pids.json"
    sleeper = "import time; time.sleep(30)"
    code = (
        "import json,os,subprocess,time; "
        f"p=subprocess.Popen([{watchdog.PINNED_PYTHON_BIN!r}, '-c', {sleeper!r}]); "
        f"open({str(pid_file)!r}, 'w').write(json.dumps([os.getpid(),p.pid])); "
        "time.sleep(30)"
    )
    original_memory_info = psutil.Process.memory_info

    def deny_runner_memory(self: psutil.Process):
        try:
            group_id = os.getpgid(self.pid)
        except ProcessLookupError:
            return original_memory_info(self)
        if pid_file.exists() and group_id == self.pid:
            members = json.loads(pid_file.read_text(encoding="ascii"))
            if self.pid not in members:
                return original_memory_info(self)
            # A fresh-session runner has PGID == direct-child PID.  Raising
            # psutil's actual AccessDenied exercises enumeration uncertainty.
            raise psutil.AccessDenied(pid=self.pid)
        return original_memory_info(self)

    # psutil's oneshot context expects its decorator cache hooks on this
    # method; preserve them while replacing only the process-specific body.
    for hook in ("cache_activate", "cache_deactivate"):
        if hasattr(original_memory_info, hook):
            setattr(deny_runner_memory, hook, getattr(original_memory_info, hook))

    monkeypatch.setattr(psutil.Process, "memory_info", deny_runner_memory)
    result = _run_fake_child(tmp_path, code, wall_seconds=1.0, sample_seconds=0.01)
    receipt = result["receipt"]
    pids = json.loads(pid_file.read_text(encoding="ascii"))
    kill_reap = receipt["kill_and_reap"]

    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "process_group_membership_unknown"
    assert kill_reap["child_reaped"] is True
    assert kill_reap["tracked_process_group_reaped"] is False
    assert "runner_tree_reaped" not in kill_reap
    assert kill_reap["membership_verification"] == "unknown_after_enumeration_error"
    assert kill_reap["unreaped_process_pids"] is None
    assert any("AccessDenied" in error for error in kill_reap["membership_enumeration_errors"])
    assert not psutil.pid_exists(pids[1])


def test_detached_child_receipt_only_claims_group_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "detached-pid.txt"
    release_marker = tmp_path / "detached-child-release.txt"
    post_detach_samples = 0
    child_processes: list[object] = []
    real_popen = watchdog.subprocess.Popen
    real_snapshot = watchdog._snapshot_process_group

    def track_fake_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child_processes.append(child)
        return child

    def snapshot_then_release_detached_child(process_group_id: int):
        nonlocal post_detach_samples
        snapshot = real_snapshot(process_group_id)
        members, _rss, _vanished, errors = snapshot
        if (
            marker.is_file()
            and child_processes
            and process_group_id == child_processes[0].pid
            and not errors
            and any(member.get("pid") == process_group_id for member in members)
            and post_detach_samples < 5
        ):
            post_detach_samples += 1
            if post_detach_samples == 5:
                release_marker.touch()
                deadline = time.monotonic() + 1.0
                while (
                    child_processes[0].poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                if child_processes[0].poll() is not None:
                    snapshot = real_snapshot(process_group_id)
        return snapshot

    monkeypatch.setattr(watchdog.subprocess, "Popen", track_fake_child)
    monkeypatch.setattr(
        watchdog, "_snapshot_process_group", snapshot_then_release_detached_child
    )
    code = (
        "import os,time\n"
        "time.sleep(0.08)\n"
        "pid=os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        f"    open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        f"while not os.path.exists({str(marker)!r}): time.sleep(0.001)\n"
        f"while not os.path.exists({str(release_marker)!r}): time.sleep(0.001)\n"
        "os._exit(0)\n"
    )

    result = _run_fake_child(tmp_path, code, wall_seconds=2.0, sample_seconds=0.01)
    receipt = result["receipt"]
    detached_pid = int(marker.read_text(encoding="ascii"))
    try:
        assert psutil.pid_exists(detached_pid)
        assert receipt["status"] == "failed"
        assert receipt["process_return_code"] == 0
        assert receipt["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
        assert release_marker.is_file() and post_detach_samples >= 5
        assert len(child_processes) == 1
        assert receipt["kill_and_reap"]["tracked_process_group_reaped"] is True
        assert receipt["kill_and_reap"]["membership_scope"] == "isolated_process_group_only"
        assert receipt["kill_and_reap"]["all_descendants_reaped_claimed"] is False
        assert "runner_tree_reaped" not in receipt["kill_and_reap"]
        assert receipt["process_completion_scope"]["all_descendants_reaped_claimed"] is False
        assert "setsid" in receipt["process_completion_scope"]["detached_session_limitation"]
    finally:
        try:
            psutil.Process(detached_pid).kill()
        except psutil.Error:
            pass
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and psutil.pid_exists(detached_pid):
            time.sleep(0.01)
        assert not psutil.pid_exists(detached_pid)


def test_unexpected_runtime_fails_before_lock_or_child_start(tmp_path: Path) -> None:
    output = tmp_path / "receipts"
    output.mkdir()
    claim = tmp_path / "claims" / "fake-run.watchdog.claim"
    marker = tmp_path / "child-started"
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    command = [str(executable), "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]

    with pytest.raises(watchdog.WatchdogError, match="hash differs"):
        watchdog._supervise_command(
            launch_vector=command,
            observed_executable_path=observed_image,
            observed_executable_sha256="0" * 64,
            claim_path=claim,
            receipt_path=output / "watchdog-terminal-receipt.json",
            run_identity={
                "protocol_id": "fake-protocol",
                "run_id": "fake-run",
                "manifest_sha256": "b" * 64,
            },
            wall_limit_seconds=0.5,
            rss_limit_bytes=32 * 1024**2,
            sample_interval_seconds=0.01,
        )

    assert not claim.exists()
    assert not marker.exists()
    assert tuple(output.iterdir()) == ()


def test_project_claim_rejects_symlinked_ancestor_before_fake_child(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "fake-project"
    (project_root / "artifacts").mkdir(parents=True)
    outside_root = tmp_path / "outside-artifacts"
    outside_root.mkdir()
    (project_root / "artifacts" / "evaluations").symlink_to(
        outside_root,
        target_is_directory=True,
    )
    claim_path = watchdog._expected_project_claim_path(project_root, "fake-run")
    marker = tmp_path / "child-started"
    command = (
        "from pathlib import Path; "
        f"Path({str(marker)!r}).touch()"
    )

    with pytest.raises(watchdog.WatchdogError, match="no-follow project ancestry"):
        _run_fake_child(
            tmp_path,
            command,
            claim_path=claim_path,
            claim_project_root=project_root,
        )

    assert not marker.exists()
    assert not (outside_root / "cascaded_tanks_abc6_campaign_fit").exists()


def test_project_claim_rejects_traversal_run_id_before_filesystem_writes(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "fake-project"
    project_root.mkdir()
    run_id = "../../../../../escape"
    claim_path = (
        project_root
        / "artifacts"
        / "evaluations"
        / "cascaded_tanks_abc6_campaign_fit"
        / "claims"
        / f"{run_id}.watchdog.claim"
    )

    with pytest.raises(watchdog.WatchdogError, match="canonical single basename"):
        watchdog._claim_once(
            claim_path,
            {"run_id": run_id, "protocol_id": "fake-protocol"},
            project_root=project_root,
        )

    assert tuple(project_root.iterdir()) == ()
    assert not (tmp_path / "escape.watchdog.claim").exists()


def test_strict_campaign_preflight_failure_precedes_approval_claim_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    launch_called = False

    def reject_manifest(
        _path: Path, _sha: str, *, pinned_manifest_bytes: bytes
    ) -> str:
        raise RuntimeError("strict source/roster pin rejected")

    def should_not_launch(**_kwargs: object):
        nonlocal launch_called
        launch_called = True
        raise AssertionError("launch reached before strict preflight")

    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", reject_manifest)
    monkeypatch.setattr(watchdog, "_supervise_command", should_not_launch)
    args = [
        "--manifest",
        str(fixture["manifest"]),
        "--manifest-sha256",
        str(fixture["manifest_sha256"]),
        "--receipt-directory",
        str(fixture["receipt"]),
        "--approval-record",
        str(fixture["approval"]),
        "--approval-record-sha256",
        str(fixture["approval_sha256"]),
    ]

    with pytest.raises(watchdog.WatchdogError, match="strict manifest/source/runtime/roster"):
        watchdog._main(args)
    assert launch_called is False
    root = fixture["root"]
    claim = (
        root
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()


def test_approval_launch_record_binds_strict_hash_head_script_and_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    manifest_path = fixture["manifest"]
    receipt_directory = fixture["receipt"]
    approval_path = fixture["approval"]
    manifest_sha256 = fixture["manifest_sha256"]
    head = "d" * 40
    campaign_vector = fixture["child_vector"]
    actual_args = fixture["actual_args"]

    def write_record(extra: dict[str, object] | None = None) -> str:
        record: dict[str, object] = {
            "schema_version": 2,
            "record_type": "abc6_independent_approval_and_launch_v2",
            "status": "approved",
            "protocol_id": watchdog.PROTOCOL_ID,
            "run_id": watchdog.RUN_ID,
            "manifest_sha256": manifest_sha256,
            "reviewed_git_head": head,
            "reviewer_id": "independent-review-fixture",
            "approved_at_utc": "2026-09-28T00:00:00Z",
            "watchdog_script_sha256": hashlib.sha256(
                Path(watchdog._SCRIPT_PATH).read_bytes()
            ).hexdigest(),
            "manifest_path": str(manifest_path),
            "receipt_directory": str(receipt_directory),
            "watchdog_launch_vector_template": watchdog._watchdog_launch_vector_template(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                receipt_directory=receipt_directory,
                approval_path=approval_path,
            ),
            "campaign_child_launch_vector": campaign_vector,
            "review_report_path": str(fixture["review_report"]),
            "review_report_sha256": fixture["review_report_sha256"],
        }
        if extra:
            record.update(extra)
        raw = watchdog._canonical_json(record)
        approval_path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    approval_sha256 = write_record()
    actual_args[-1] = approval_sha256
    manifest, verified_sha, record, anchors = watchdog._production_preflight(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt_directory=receipt_directory,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        actual_watchdog_args=actual_args,
    )
    assert verified_sha == manifest_sha256
    assert record["reviewed_git_head"] == head
    assert record["review_report_path"] == str(fixture["review_report"])
    assert record["review_report_sha256"] == fixture["review_report_sha256"]
    assert manifest["protocol_id"] == watchdog.PROTOCOL_ID
    assert campaign_vector[1] == str(
        Path(watchdog._REPO_ROOT) / "scripts/run_cascaded_tanks_abc6_synthetic.py"
    )
    assert "-c" not in campaign_vector
    anchors.close()

    unknown_sha = write_record({"unreviewed_extra": True})
    with pytest.raises(watchdog.WatchdogError, match="schema is incomplete"):
        watchdog._production_preflight(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            receipt_directory=receipt_directory,
            approval_path=approval_path,
            approval_sha256=unknown_sha,
            actual_watchdog_args=[*actual_args[:-1], unknown_sha],
        )

    v1_sha = write_record(
        {
            "schema_version": 1,
            "record_type": "abc6_independent_approval_and_launch_v1",
        }
    )
    with pytest.raises(watchdog.WatchdogError, match="schema is incomplete"):
        _production_preflight_from_fixture(
            {**fixture, "approval_sha256": v1_sha, "actual_args": [*actual_args[:-1], v1_sha]}
        )

    missing_review_sha = write_record()
    record_without_path = json.loads(approval_path.read_text())
    record_without_path.pop("review_report_path")
    raw_without_path = watchdog._canonical_json(record_without_path)
    approval_path.write_bytes(raw_without_path)
    missing_review_sha = hashlib.sha256(raw_without_path).hexdigest()
    with pytest.raises(watchdog.WatchdogError, match="schema is incomplete"):
        _production_preflight_from_fixture(
            {**fixture, "approval_sha256": missing_review_sha, "actual_args": [*actual_args[:-1], missing_review_sha]}
        )

    missing_review_sha = write_record()
    record_without_digest = json.loads(approval_path.read_text())
    record_without_digest.pop("review_report_sha256")
    raw_without_digest = watchdog._canonical_json(record_without_digest)
    approval_path.write_bytes(raw_without_digest)
    missing_review_sha = hashlib.sha256(raw_without_digest).hexdigest()
    with pytest.raises(watchdog.WatchdogError, match="schema is incomplete"):
        _production_preflight_from_fixture(
            {**fixture, "approval_sha256": missing_review_sha, "actual_args": [*actual_args[:-1], missing_review_sha]}
        )

    bad_path_sha = write_record({"review_report_path": str(fixture["root"] / "elsewhere.md")})
    with pytest.raises(watchdog.WatchdogError, match="does not bind the fixed independent review report"):
        _production_preflight_from_fixture(
            {**fixture, "approval_sha256": bad_path_sha, "actual_args": [*actual_args[:-1], bad_path_sha]}
        )

    bad_digest_sha = write_record({"review_report_sha256": "A" * 64})
    with pytest.raises(watchdog.WatchdogError, match="does not bind the fixed independent review report"):
        _production_preflight_from_fixture(
            {**fixture, "approval_sha256": bad_digest_sha, "actual_args": [*actual_args[:-1], bad_digest_sha]}
        )

    mismatched_digest = write_record({"review_report_sha256": "f" * 64})
    with pytest.raises(watchdog.WatchdogError, match="does not bind the fixed independent review report"):
        _production_preflight_from_fixture(
            {**fixture, "approval_sha256": mismatched_digest, "actual_args": [*actual_args[:-1], mismatched_digest]}
        )

    wrong_head_sha = write_record({"reviewed_git_head": "f" * 40})
    with pytest.raises(watchdog.WatchdogError, match="does not bind current manifest"):
        watchdog._production_preflight(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            receipt_directory=receipt_directory,
            approval_path=approval_path,
            approval_sha256=wrong_head_sha,
            actual_watchdog_args=[*actual_args[:-1], wrong_head_sha],
        )


def _production_preflight_from_fixture(fixture: dict[str, object]):
    return watchdog._production_preflight(
        manifest_path=fixture["manifest"],
        manifest_sha256=fixture["manifest_sha256"],
        receipt_directory=fixture["receipt"],
        approval_path=fixture["approval"],
        approval_sha256=fixture["approval_sha256"],
        actual_watchdog_args=fixture["actual_args"],
    )


def test_wrong_receipt_root_path_fails_before_campaign_preflight_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    strict_called = False
    child_called = False

    def strict(*_args: object, **_kwargs: object) -> str:
        nonlocal strict_called
        strict_called = True
        return str(fixture["manifest_sha256"])

    def child(**_kwargs: object):
        nonlocal child_called
        child_called = True
        raise AssertionError("child start reached before root rejection")

    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", strict)
    monkeypatch.setattr(watchdog, "_supervise_command", child)
    wrong_receipt = tmp_path / "redirected-receipts"
    wrong_receipt.mkdir()
    args = list(fixture["actual_args"])
    args[args.index("--receipt-directory") + 1] = str(wrong_receipt)
    with pytest.raises(watchdog.WatchdogError, match="receipt directory"):
        watchdog._main(args)
    assert not strict_called
    assert not child_called
    root = fixture["root"]
    claim = (
        root
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()


def test_wrong_manifest_path_fails_before_campaign_preflight_or_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    strict_called = False

    def strict(*_args: object, **_kwargs: object) -> str:
        nonlocal strict_called
        strict_called = True
        return str(fixture["manifest_sha256"])

    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", strict)
    args = list(fixture["actual_args"])
    alternate_manifest = tmp_path / "alternate-manifest-v2.json"
    alternate_manifest.write_bytes(Path(fixture["manifest"]).read_bytes())
    args[args.index("--manifest") + 1] = str(alternate_manifest)
    with pytest.raises(watchdog.WatchdogError, match="fixed run root"):
        watchdog._main(args)
    assert not strict_called
    root = fixture["root"]
    claim = (
        root
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()


def test_manifest_root_mismatch_and_training_only_vector_fail_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(
        tmp_path,
        monkeypatch,
        child_vector_override=[
            watchdog.PINNED_PYTHON_BIN,
            "-c",
            (
                "from core.real_data.cascaded_tanks_abc6_campaign_fit import "
                "run_abc6_training_campaign"
            ),
        ],
    )
    with pytest.raises(watchdog.WatchdogError, match="does not bind current manifest"):
        _production_preflight_from_fixture(fixture)
    root = fixture["root"]
    claim = (
        root
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()

    valid_fixture = _private_preflight_fixture(tmp_path / "wrong-root", monkeypatch)
    manifest_path = valid_fixture["manifest"]
    manifest = dict(valid_fixture["manifest_body"])
    manifest["repository_root_realpath"] = str(tmp_path / "different-checkout")
    raw = watchdog._canonical_json(manifest)
    manifest_path.write_bytes(raw)
    valid_fixture["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(watchdog.WatchdogError, match="physical-root identity"):
        _production_preflight_from_fixture(valid_fixture)


@pytest.mark.parametrize("control_file", ["manifest", "approval", "review_report"])
@pytest.mark.parametrize("leaf_kind", ["fifo", "symlink"])
def test_control_file_fifo_or_symlink_fails_nonblocking_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_file: str,
    leaf_kind: str,
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    path = fixture[control_file]
    assert isinstance(path, Path)
    original = path.read_bytes()
    path.unlink()
    if leaf_kind == "fifo":
        os.mkfifo(path)
    else:
        target = tmp_path / f"{control_file}-outside.json"
        target.write_bytes(original)
        path.symlink_to(target)
    with pytest.raises(watchdog.WatchdogError):
        _production_preflight_from_fixture(fixture)
    root = fixture["root"]
    claim = (
        root
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()


def test_tampered_review_report_bytes_fail_before_one_use_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    report = fixture["review_report"]
    assert isinstance(report, Path)
    report.write_bytes(report.read_bytes() + b"tampered after approval\n")
    with pytest.raises(watchdog.WatchdogError, match="does not bind the fixed independent review report"):
        _production_preflight_from_fixture(fixture)
    claim = (
        fixture["root"]
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()


def test_missing_review_report_fails_before_one_use_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    report = fixture["review_report"]
    assert isinstance(report, Path)
    report.unlink()
    with pytest.raises(watchdog.WatchdogError, match="independent review report is missing or unsafe"):
        _production_preflight_from_fixture(fixture)
    claim = (
        fixture["root"]
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    )
    assert not claim.exists()


def test_review_report_swap_after_preflight_fails_at_immediate_preclaim_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    _manifest, _verified_sha, _approval, anchors = _production_preflight_from_fixture(fixture)
    report = fixture["review_report"]
    assert isinstance(report, Path)
    claim_reached = False
    original_utc_now = watchdog._utc_now

    def swap_report_before_final_anchor_check() -> str:
        replacement = report.with_name("review-report-replacement.md")
        replacement.write_bytes(report.read_bytes())
        report.unlink()
        replacement.rename(report)
        return original_utc_now()

    def forbidden_claim(*_args: object, **_kwargs: object) -> str:
        nonlocal claim_reached
        claim_reached = True
        raise AssertionError("claim attempted before independent report revalidation")

    monkeypatch.setattr(watchdog, "_utc_now", swap_report_before_final_anchor_check)
    monkeypatch.setattr(watchdog, "_claim_once", forbidden_claim)
    executable = Path(os.sys.executable)
    run_identity = {
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": fixture["manifest_sha256"],
        "approval_record_sha256": fixture["approval_sha256"],
    }
    try:
        with pytest.raises(watchdog.WatchdogError, match="independent review report path or bytes changed"):
            watchdog._supervise_command(
                launch_vector=[str(executable), "private-fake-child"],
                observed_executable_path=executable,
                observed_executable_sha256=watchdog._sha256_file(executable),
                claim_path=watchdog._expected_project_claim_path(
                    fixture["root"], watchdog.RUN_ID
                ),
                receipt_path=report.parent / "receipts" / "watchdog-terminal-receipt.json",
                run_identity=run_identity,
                claim_project_root=fixture["root"],
                preflight_anchors=anchors,
            )
    finally:
        anchors.close()
    assert claim_reached is False
    assert not (
        fixture["root"]
        / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{watchdog.RUN_ID}.watchdog.claim"
    ).exists()


def test_manifest_swap_to_fifo_after_watchdog_pin_fails_before_claim_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    manifest_path = fixture["manifest"]
    assert isinstance(manifest_path, Path)
    pinned_bytes = manifest_path.read_bytes()
    original_preflight = campaign.preflight_abc6_campaign_manifest
    swap_reached_campaign = False
    child_called = False

    def swap_then_run_campaign_preflight(
        path: Path,
        manifest_sha256: str,
        *,
        pinned_manifest_bytes: bytes,
    ) -> str:
        nonlocal swap_reached_campaign
        assert pinned_manifest_bytes == pinned_bytes
        path.unlink()
        os.mkfifo(path)
        swap_reached_campaign = True
        return original_preflight(
            path,
            manifest_sha256,
            require_reviewed_runtime=True,
            pinned_manifest_bytes=pinned_manifest_bytes,
        )

    def forbidden_child(**_kwargs: object):
        nonlocal child_called
        child_called = True
        raise AssertionError("child launch reached after manifest FIFO swap")

    monkeypatch.setattr(
        watchdog, "_campaign_strict_preflight", swap_then_run_campaign_preflight
    )
    monkeypatch.setattr(watchdog, "_supervise_command", forbidden_child)
    with pytest.raises(
        watchdog.WatchdogError,
        match=(
            "strict manifest/source/runtime/roster preflight failed: "
            "manifest schema/type validation failed"
        ),
    ):
        watchdog._main(fixture["actual_args"])

    assert swap_reached_campaign
    assert not child_called
    root = fixture["root"]
    assert isinstance(root, Path)
    claim_parent = root / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
    assert not (claim_parent / f"{watchdog.RUN_ID}.claim").exists()
    assert not (claim_parent / f"{watchdog.RUN_ID}.watchdog.claim").exists()


@pytest.mark.parametrize(
    "occupied_path",
    [
        "campaign_claim",
        "watchdog_claim",
        "launch_attestation",
        "scorer_marker",
        "receipt_output",
    ],
)
def test_existing_claim_marker_or_output_fails_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied_path: str,
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    root = fixture["root"]
    if occupied_path == "campaign_claim":
        path = root / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE / f"{watchdog.RUN_ID}.claim"
    elif occupied_path == "watchdog_claim":
        path = root / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE / f"{watchdog.RUN_ID}.watchdog.claim"
    elif occupied_path == "launch_attestation":
        path = (
            root
            / watchdog._CAMPAIGN_CLAIM_PARENT_RELATIVE
            / f"{watchdog.RUN_ID}{watchdog._LAUNCH_ATTESTATION_SUFFIX}"
        )
    elif occupied_path == "scorer_marker":
        path = root / watchdog._SCORER_CLAIM_PARENT_RELATIVE / f"{watchdog.RUN_ID}.claim"
    else:
        path = Path(fixture["receipt"]) / "campaign.target-free-forecasts.json"
    path.write_bytes(b"already-used")
    strict_called = False

    def strict(*_args: object, **_kwargs: object) -> str:
        nonlocal strict_called
        strict_called = True
        return str(fixture["manifest_sha256"])

    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", strict)
    with pytest.raises(watchdog.WatchdogError, match="already exists|not empty"):
        _production_preflight_from_fixture(fixture)
    assert not strict_called


def test_preopen_copied_artifacts_ancestor_symlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    root = fixture["root"]
    artifacts = root / "artifacts"
    copied = tmp_path / "copied-artifacts"
    shutil.copytree(artifacts, copied)
    shutil.rmtree(artifacts)
    artifacts.symlink_to(copied, target_is_directory=True)
    strict_called = False

    def strict(*_args: object, **_kwargs: object) -> str:
        nonlocal strict_called
        strict_called = True
        return str(fixture["manifest_sha256"])

    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", strict)
    with pytest.raises(watchdog.WatchdogError, match="fixed run directory"):
        _production_preflight_from_fixture(fixture)
    assert not strict_called
    assert not (
        copied
        / "evaluations/cascaded_tanks_abc6_campaign_fit/claims"
        / f"{watchdog.RUN_ID}.watchdog.claim"
    ).exists()


def test_postopen_copied_ancestor_swap_stops_before_watchdog_claim_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    _manifest, _sha, _approval, anchors = _production_preflight_from_fixture(fixture)
    root = fixture["root"]
    artifacts = root / "artifacts"
    original_artifacts = root / "artifacts-pinned"
    decoy = tmp_path / "decoy-artifacts"
    shutil.copytree(artifacts, decoy)
    artifacts.rename(original_artifacts)
    artifacts.symlink_to(decoy, target_is_directory=True)
    marker = tmp_path / "child-started"
    child_calls = 0

    def forbidden_child(*_args: object, **_kwargs: object):
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("child process started after ancestor identity changed")

    monkeypatch.setattr(watchdog.subprocess, "Popen", forbidden_child)
    try:
        with pytest.raises(watchdog.WatchdogError, match="pinned directory|symlink"):
            watchdog._supervise_command(
                launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "pass"],
                observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
                observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
                claim_path=watchdog._expected_project_claim_path(root, watchdog.RUN_ID),
                receipt_path=(
                    root
                    / watchdog._RECEIPT_ROOT_RELATIVE
                    / "watchdog-terminal-receipt.json"
                ),
                run_identity={
                    "protocol_id": watchdog.PROTOCOL_ID,
                    "run_id": watchdog.RUN_ID,
                    "manifest_sha256": str(fixture["manifest_sha256"]),
                },
                claim_project_root=root,
                preflight_anchors=anchors,
                wall_limit_seconds=0.2,
                rss_limit_bytes=32 * 1024**2,
                sample_interval_seconds=0.01,
            )
    finally:
        anchors.close()
    assert child_calls == 0
    assert not marker.exists()
    assert not (
        decoy
        / "evaluations/cascaded_tanks_abc6_campaign_fit/claims"
        / f"{watchdog.RUN_ID}.watchdog.claim"
    ).exists()
    assert not (
        original_artifacts
        / "evaluations/cascaded_tanks_abc6_campaign_fit/claims"
        / f"{watchdog.RUN_ID}.watchdog.claim"
    ).exists()


def test_private_fake_launch_claim_and_terminal_write_use_pinned_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _private_preflight_fixture(tmp_path, monkeypatch)
    _manifest, _sha, _approval, anchors = _production_preflight_from_fixture(fixture)
    root = fixture["root"]
    receipt = Path(fixture["receipt"])
    claim_path = watchdog._expected_project_claim_path(root, watchdog.RUN_ID)
    receipt_path = receipt / "watchdog-terminal-receipt.json"
    root_info = os.fstat(anchors.root_fd)
    receipt_info = os.fstat(anchors.receipt_root_fd)
    watchdog_claim_info = os.fstat(anchors.watchdog_claim_parent_fd)
    scorer_claim_info = os.fstat(
        anchors.directories[watchdog._SCORER_CLAIM_PARENT_RELATIVE]
    )
    try:
        result = watchdog._supervise_command(
            launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "pass"],
            observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
            observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
            claim_path=claim_path,
            receipt_path=receipt_path,
            run_identity={
                "protocol_id": watchdog.PROTOCOL_ID,
                "run_id": watchdog.RUN_ID,
                "manifest_sha256": str(fixture["manifest_sha256"]),
                "approval_record_sha256": str(fixture["approval_sha256"]),
                "reviewed_git_head": "d" * 40,
                "repository_root_realpath": str(root),
                "receipt_root_relative": watchdog._RECEIPT_ROOT_RELATIVE,
                "repository_root_device": root_info.st_dev,
                "repository_root_inode": root_info.st_ino,
                "receipt_root_device": receipt_info.st_dev,
                "receipt_root_inode": receipt_info.st_ino,
                "watchdog_claim_parent_device": watchdog_claim_info.st_dev,
                "watchdog_claim_parent_inode": watchdog_claim_info.st_ino,
                "scorer_claim_parent_device": scorer_claim_info.st_dev,
                "scorer_claim_parent_inode": scorer_claim_info.st_ino,
            },
            claim_project_root=root,
            preflight_anchors=anchors,
            wall_limit_seconds=1.0,
            rss_limit_bytes=32 * 1024**2,
            sample_interval_seconds=0.01,
        )
    finally:
        anchors.close()

    receipt_body = json.loads(receipt_path.read_text(encoding="ascii"))
    claim_body = json.loads(claim_path.read_text(encoding="ascii"))
    assert result["receipt"]["status"] == "failed"
    assert result["receipt"]["process_return_code"] == 0
    assert result["receipt"]["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
    assert receipt_body["receipt_root_relative"] == watchdog._RECEIPT_ROOT_RELATIVE
    assert receipt_body["repository_root_realpath"] == str(root)
    assert receipt_body["watchdog_claim_parent_inode"] == watchdog_claim_info.st_ino
    assert claim_body["declared_child_launch_vector"] == [
        watchdog.PINNED_PYTHON_BIN,
        "-c",
        "pass",
    ]


def test_existing_one_use_claim_rejects_before_second_child_start(tmp_path: Path) -> None:
    first = _run_fake_child(tmp_path, "import time; time.sleep(0.04)")
    claim_path = tmp_path / "claims" / "fake-run.watchdog.claim"
    second_directory = tmp_path / "second-receipts"
    second_directory.mkdir()
    marker = tmp_path / "second-child-started"
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    command = [str(executable), "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]

    with pytest.raises(watchdog.AlreadyClaimedError):
        watchdog._supervise_command(
            launch_vector=command,
            observed_executable_path=observed_image,
            observed_executable_sha256=_sha256(observed_image),
            claim_path=claim_path,
            receipt_path=second_directory / "watchdog-terminal-receipt.json",
            run_identity={
                "protocol_id": "fake-protocol",
                "run_id": "fake-run",
                "manifest_sha256": "a" * 64,
            },
            wall_limit_seconds=0.5,
            rss_limit_bytes=32 * 1024**2,
            sample_interval_seconds=0.01,
        )

    assert first["receipt"]["status"] == "failed"
    assert first["receipt"]["process_return_code"] == 0
    assert first["receipt"]["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
    assert claim_path.is_file()
    assert not marker.exists()
    assert tuple(second_directory.iterdir()) == ()


def test_exact_vector_comparison_rejects_changed_flags_or_runner() -> None:
    watchdog._require_exact_vector(
        ["/bin/python", "scripts/watch.py", "--manifest", "/tmp/m.json"],
        ["/bin/python", "scripts/watch.py", "--manifest", "/tmp/m.json"],
    )
    with pytest.raises(watchdog.WatchdogError, match="exact declared vector"):
        watchdog._require_exact_vector(
            ["/bin/python", "scripts/watch.py", "--manifest", "/tmp/other.json"],
            ["/bin/python", "scripts/watch.py", "--manifest", "/tmp/m.json"],
        )


def _write_private_fit_artifact_chain(receipt_directory: Path) -> dict[str, str]:
    from core.real_data.cascaded_tanks_abc6_cases import (
        CASE_ROSTER,
        PARAMETER_ORDER,
        PROSPECTIVE_INPUT,
    )
    from core.real_data.cascaded_tanks_abc6_forecast import QUANTILE_CONVENTION

    manifest_sha256 = "c" * 64
    claim: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": manifest_sha256,
        "receipt_directory": str(receipt_directory),
        "claimed_at_utc": "2026-09-28T00:00:00Z",
        "claim_semantics": "consumed_once_no_resume",
    }
    claim_raw = watchdog._canonical_json(claim)
    (receipt_directory / "campaign.claim").write_bytes(claim_raw)
    claim_sha256 = hashlib.sha256(claim_raw).hexdigest()

    case_summaries: list[dict[str, object]] = []
    status_receipts: list[dict[str, object]] = []
    for case in CASE_ROSTER:
        links: dict[str, str] = {}
        for component in ("fit", "baseline"):
            filename = f"case-{case.case_index:02d}.{component}-status.json"
            status: dict[str, object] = {
                "schema_version": 1,
                "protocol_id": watchdog.PROTOCOL_ID,
                "run_id": watchdog.RUN_ID,
                "case_index": case.case_index,
                "case_id": case.case_id,
                "component": component,
                "status": "complete",
            }
            status["payload_sha256"] = hashlib.sha256(
                watchdog._canonical_json(status)
            ).hexdigest()
            raw = watchdog._canonical_json(status)
            (receipt_directory / filename).write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            links[f"{component}_receipt_sha256"] = digest
            status_receipts.append(
                {
                    "filename": filename,
                    "sha256": digest,
                    "case_index": case.case_index,
                    "case_id": case.case_id,
                    "component": component,
                    "status": "complete",
                }
            )
        case_summaries.append(
            {
                "case_index": case.case_index,
                "case_id": case.case_id,
                "fit_status": "complete",
                "baseline_status": "complete",
                **links,
            }
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": manifest_sha256,
        "claim_sha256": claim_sha256,
        "written_at_utc": "2026-09-28T00:01:00Z",
        "status": "complete",
        "case_count": 24,
        "fit_status_counts": {"complete": 24},
        "baseline_status_counts": {"complete": 24},
        "case_statuses": case_summaries,
        "training_only": True,
        "prospective_targets_generated": False,
        "forecasts_run": False,
        "postfit_target_gate_opened": False,
        "external_watchdog_enforced_here": False,
        "independent_manifest_approval_enforced_here": False,
        "resume_allowed": False,
    }
    summary["payload_sha256"] = hashlib.sha256(
        watchdog._canonical_json(summary)
    ).hexdigest()
    summary_raw = watchdog._canonical_json(summary)
    (receipt_directory / campaign.SUMMARY_FILENAME).write_bytes(summary_raw)
    summary_sha256 = hashlib.sha256(summary_raw).hexdigest()

    evidence_directory = receipt_directory / campaign.EVIDENCE_DIRECTORY_NAME
    evidence_directory.mkdir()
    roster_identities = campaign._roster_identities()
    bundle = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "kind": "abc6_training_bundle",
        "ordered_case_identities": roster_identities,
        "bundle": {"private_fake_fixture": True},
    }
    bundle_raw = watchdog._canonical_json(bundle)
    (evidence_directory / "training-bundle.evidence.json").write_bytes(bundle_raw)
    case_artifacts: list[dict[str, object]] = []
    for case in CASE_ROSTER:
        evidence_filename = f"case-{case.case_index:02d}.training-evidence.json"
        evidence = {
            "schema_version": 1,
            "protocol_id": watchdog.PROTOCOL_ID,
            "run_id": watchdog.RUN_ID,
            "kind": "abc6_training_case_result",
            "roster_index": case.case_index,
            "case_identity": campaign._case_identity(case),
            "fit_status": "complete",
            "baseline_status": "complete",
            "result": {"private_fake_fixture": True},
        }
        evidence_raw = watchdog._canonical_json(evidence)
        (evidence_directory / evidence_filename).write_bytes(evidence_raw)
        case_artifacts.append(
            {
                "filename": evidence_filename,
                "sha256": hashlib.sha256(evidence_raw).hexdigest(),
                "roster_index": case.case_index,
                "case_identity": campaign._case_identity(case),
                "fit_status": "complete",
                "baseline_status": "complete",
            }
        )
    evidence_manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": manifest_sha256,
        "claim_sha256": claim_sha256,
        "training_summary_filename": campaign.SUMMARY_FILENAME,
        "training_summary_sha256": summary_sha256,
        "roster_sha256": campaign._roster_sha256(),
        "ordered_case_identities": roster_identities,
        "status_receipts": [
            {
                key: entry[key]
                for key in (
                    "filename",
                    "sha256",
                    "case_index",
                    "case_id",
                    "component",
                    "status",
                )
            }
            for entry in status_receipts
        ],
        "bundle_artifact": {
            "filename": "training-bundle.evidence.json",
            "sha256": hashlib.sha256(bundle_raw).hexdigest(),
        },
        "case_artifacts": case_artifacts,
    }
    evidence_manifest["payload_sha256"] = hashlib.sha256(
        watchdog._canonical_json(evidence_manifest)
    ).hexdigest()
    evidence_manifest_raw = watchdog._canonical_json(evidence_manifest)
    (receipt_directory / campaign.EVIDENCE_MANIFEST_FILENAME).write_bytes(
        evidence_manifest_raw
    )
    evidence_manifest_sha256 = hashlib.sha256(evidence_manifest_raw).hexdigest()

    source_hashes = {
        path: hashlib.sha256(path.encode("ascii")).hexdigest()
        for path in watchdog._INTEGRATED_SOURCE_PATHS
    }
    short_identities = watchdog._expected_forecast_case_identities()
    forecasts: list[dict[str, object]] = []
    forecast_case_sha256: list[str] = []
    for case in CASE_ROSTER:
        is_n = case.truth_id == "N"
        particles = [] if is_n else [
            {
                "particle_index": particle_index,
                "parameter_values": [0.5, 0.4, 0.5, 0.5, 0.5, 3.0],
                "weight": 1.0 / 48.0,
                "common_time_state": {"x1": 0.5, "x2": 0.5},
                "trajectory": [1.0] * len(PROSPECTIVE_INPUT),
                "failure": None,
            }
            for particle_index in range(48)
        ]
        weights = [] if is_n else [1.0 / 48.0] * 48
        trajectories = [] if is_n else [[1.0] * len(PROSPECTIVE_INPUT)] * 48
        forecast = {
            "status": "abstained_n" if is_n else "complete",
            "case_index": case.case_index,
            "case_id": case.case_id,
            "fit_window": case.input_window,
            "prospective_inputs": None if is_n else list(PROSPECTIVE_INPUT),
            "common_state_index": None if is_n else case.input_length,
            "parameter_order": list(PARAMETER_ORDER),
            "particles": particles,
            "weights": weights,
            "particle_trajectories": trajectories,
            "aggregate_status": "unavailable" if is_n else "complete",
            "pointwise_weighted_mean": None if is_n else [1.0] * len(PROSPECTIVE_INPUT),
            "pointwise_weighted_median": None if is_n else [1.0] * len(PROSPECTIVE_INPUT),
            "pointwise_q05": None if is_n else [1.0] * len(PROSPECTIVE_INPUT),
            "pointwise_q95": None if is_n else [1.0] * len(PROSPECTIVE_INPUT),
            "effective_sample_size": None if is_n else 48.0,
            "quantile_convention": QUANTILE_CONVENTION,
            "pointwise_summaries_are_coherent_trajectories": False,
            "baseline_parameter_values": None if is_n else [0.5, 0.4, 0.5, 0.5, 0.5, 3.0],
            "baseline_common_time_state": None if is_n else {"x1": 0.5, "x2": 0.5},
            "baseline_trajectory": None if is_n else [1.0] * len(PROSPECTIVE_INPUT),
            "baseline_failure": None,
        }
        forecasts.append(forecast)
        case_identity = short_identities[case.case_index]
        case_payload = {
            "protocol_id": watchdog.PROTOCOL_ID,
            "run_id": watchdog.RUN_ID,
            "case_index": case.case_index,
            "case_id": case.case_id,
            "truth_id": case.truth_id,
            "input_window": case.input_window,
            "replicate": case.replicate,
            "fit_model": case.fit_model.value,
            "forecast": forecast,
        }
        forecast_case_sha256.append(
            hashlib.sha256(watchdog._canonical_json(case_payload)).hexdigest()
        )
    forecast_roster_sha256 = hashlib.sha256(
        watchdog._canonical_json(
            {
                "protocol_id": watchdog.PROTOCOL_ID,
                "run_id": watchdog.RUN_ID,
                "case_sha256": forecast_case_sha256,
            }
        )
    ).hexdigest()
    forecast_artifact: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "training_manifest_sha256": manifest_sha256,
        "training_claim_sha256": claim_sha256,
        "training_summary_filename": campaign.SUMMARY_FILENAME,
        "training_summary_sha256": summary_sha256,
        "training_evidence_manifest_sha256": evidence_manifest_sha256,
        "integrated_source_hashes": [
            {"path": path, "sha256": source_hashes[path]}
            for path in watchdog._INTEGRATED_SOURCE_PATHS
        ],
        "ordered_case_identities": short_identities,
        "status_receipts": [
            {"filename": row["filename"], "sha256": row["sha256"]}
            for row in status_receipts
        ],
        "forecast_case_sha256": forecast_case_sha256,
        "forecast_roster_sha256": forecast_roster_sha256,
        "forecasts": forecasts,
        "target_free": True,
        "prospective_targets_generated_by_runner": False,
        "retry_allowed": False,
    }
    forecast_artifact["payload_sha256"] = hashlib.sha256(
        watchdog._canonical_json(forecast_artifact)
    ).hexdigest()
    (receipt_directory / watchdog._TARGET_FREE_FORECAST).write_bytes(
        watchdog._canonical_json(forecast_artifact)
    )
    return source_hashes


def test_postfit_gate_validates_all_48_durable_status_links(tmp_path: Path) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    source_hashes = _write_private_fit_artifact_chain(receipt_directory)
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private test receipt root"
    )

    def inspect(*, sync: bool = False):
        return watchdog._inspect_fit_phase_gate(
            receipt_fd,
            expected_receipt_directory=receipt_directory,
            expected_manifest_sha256="c" * 64,
            sync_directory=sync,
            expected_integrated_source_hashes=source_hashes,
        )

    try:
        gate = inspect(sync=True)
        assert gate["status_receipt_count"] == 48
        assert gate["all_48_status_receipts_present_and_linked"] is True
        assert gate["all_48_fit_and_baseline_statuses_complete"] is True
        assert gate["training_evidence_chain_valid"] is True
        assert gate["target_free_forecast_chain_valid"] is True
        assert gate["pre_score_artifact_chain_valid"] is True
        assert gate["postfit_target_gate_opened"] is False
        assert gate["problems"] == []

        summary_path = receipt_directory / campaign.SUMMARY_FILENAME
        original_summary = json.loads(summary_path.read_text(encoding="ascii"))

        def write_tampered_summary(payload: dict[str, object]) -> None:
            payload.pop("payload_sha256", None)
            payload["payload_sha256"] = hashlib.sha256(
                watchdog._canonical_json(payload)
            ).hexdigest()
            summary_path.write_bytes(watchdog._canonical_json(payload))

        bool_schema = dict(original_summary)
        bool_schema["schema_version"] = True
        write_tampered_summary(bool_schema)
        bool_gate = inspect()
        assert bool_gate["status_receipt_count"] == 48
        assert bool_gate["all_48_fit_and_baseline_statuses_complete"] is False
        assert bool_gate["pre_score_artifact_chain_valid"] is False
        assert "training summary is not a valid durable campaign receipt" in bool_gate["problems"]

        wrong_case_id = dict(original_summary)
        wrong_case_statuses = [dict(entry) for entry in original_summary["case_statuses"]]
        wrong_case_statuses[12]["case_id"] = "wrong-case-id"
        wrong_case_id["case_statuses"] = wrong_case_statuses
        write_tampered_summary(wrong_case_id)
        wrong_case_gate = inspect()
        assert wrong_case_gate["status_receipt_count"] == 48
        assert wrong_case_gate["all_48_status_receipts_present_and_linked"] is False
        assert wrong_case_gate["all_48_fit_and_baseline_statuses_complete"] is False
        assert wrong_case_gate["pre_score_artifact_chain_valid"] is False
        assert "summary identity/status links do not match durable case receipts" in wrong_case_gate["problems"]
    finally:
        os.close(receipt_fd)


@pytest.mark.parametrize("swap_after_gate", [False, True])
def test_terminal_completion_rechecks_retained_fit_gate_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_after_gate: bool,
) -> None:
    monkeypatch.setattr(watchdog, "RUN_ID", "fake-run")
    monkeypatch.setattr(watchdog, "PROTOCOL_ID", "fake-protocol")
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    root_info = tmp_path.stat()
    receipt_info = receipt_directory.stat()
    claim_path = tmp_path / "claims" / "fake-run.watchdog.claim"
    receipt_path = receipt_directory / "watchdog-terminal-receipt.json"
    executable = Path(watchdog.PINNED_PYTHON_BIN)
    observed_image = Path(watchdog.PINNED_PYTHON_APP)
    original_inspect = watchdog._inspect_fit_phase_gate
    swapped_leaf = receipt_directory / "case-00.fit-status.json"
    copied_leaf = tmp_path / "copied-fit-status.json"

    def inspect_then_swap(
        receipt_directory_fd: int,
        *,
        expected_receipt_directory: Path,
        expected_manifest_sha256: str,
        sync_directory: bool,
        preflight_anchors: watchdog._PreflightAnchors | None = None,
        expected_integrated_source_hashes: dict[str, str] | None = None,
        retain_snapshot: bool = False,
    ) -> dict[str, object]:
        source_hashes = _write_private_fit_artifact_chain(receipt_directory)
        gate = original_inspect(
            receipt_directory_fd,
            expected_receipt_directory=expected_receipt_directory,
            expected_manifest_sha256=expected_manifest_sha256,
            sync_directory=sync_directory,
            preflight_anchors=preflight_anchors,
            expected_integrated_source_hashes=source_hashes,
            retain_snapshot=retain_snapshot,
        )
        assert gate["pre_score_artifact_chain_valid"] is True
        if swap_after_gate:
            shutil.copyfile(swapped_leaf, copied_leaf)
            swapped_leaf.unlink()
            swapped_leaf.symlink_to(copied_leaf)
        return gate

    monkeypatch.setattr(watchdog, "_inspect_fit_phase_gate", inspect_then_swap)
    result = watchdog._supervise_command(
        launch_vector=[str(executable), "-c", "import time; time.sleep(0.05)"],
        observed_executable_path=observed_image,
        observed_executable_sha256=_sha256(observed_image),
        claim_path=claim_path,
        receipt_path=receipt_path,
        run_identity={
            "protocol_id": "fake-protocol",
            "run_id": "fake-run",
            "manifest_sha256": "c" * 64,
            "repository_root_realpath": str(tmp_path),
            "repository_root_device": root_info.st_dev,
            "repository_root_inode": root_info.st_ino,
            "receipt_root_relative": "receipts",
            "receipt_root_device": receipt_info.st_dev,
            "receipt_root_inode": receipt_info.st_ino,
            "watchdog_attestation": {"private_fake": True},
        },
        wall_limit_seconds=1.0,
        rss_limit_bytes=32 * 1024**2,
        sample_interval_seconds=0.01,
    )

    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True
    persisted = json.loads(receipt_path.read_text(encoding="ascii"))
    if swap_after_gate:
        assert result["receipt"]["status"] == "failed"
        assert result["receipt"]["stop_reason"] == "fit_gate_changed_before_terminal_receipt"
        assert result["receipt"]["fit_phase_gate"]["pre_score_artifact_chain_valid"] is False
        assert persisted["status"] == "failed"
        assert persisted["terminal_receipt_publication"][
            "fit_gate_revalidated_before_completed"
        ] is False
    else:
        assert result["receipt"]["status"] == "completed"
        assert result["receipt"]["fit_phase_gate"]["pre_score_artifact_chain_valid"] is True
        assert persisted["status"] == "completed"
        assert persisted["terminal_receipt_publication"][
            "fit_gate_revalidated_before_completed"
        ] is True
    ack_path = Path(result["terminal_readback_ack_path"])
    assert ack_path.is_file()
    ack = json.loads(ack_path.read_text(encoding="ascii"))
    assert ack["run_id"] == "fake-run"
    assert ack["watchdog_claim_sha256"] == persisted["watchdog_claim_sha256"]
    assert ack["terminal_receipt"]["sha256"] == result["receipt_sha256"]
    assert ack["root_binding"]["receipt_root_device"] == receipt_info.st_dev
    assert ack["root_binding"]["receipt_root_inode"] == receipt_info.st_ino


@pytest.mark.parametrize(
    "relative_leaf",
    [
        "case-00.fit-status.json",
        "campaign.claim",
        campaign.SUMMARY_FILENAME,
        campaign.EVIDENCE_MANIFEST_FILENAME,
        "training-evidence/training-bundle.evidence.json",
        "training-evidence/case-07.training-evidence.json",
        watchdog._TARGET_FREE_FORECAST,
    ],
)
def test_postfit_gate_rejects_leaf_symlink_swap_after_validated_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_leaf: str,
) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    source_hashes = _write_private_fit_artifact_chain(receipt_directory)
    target = receipt_directory / relative_leaf
    copied = tmp_path / "copied-leaf.json"
    shutil.copyfile(target, copied)
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private post-read-swap receipt root"
    )
    original_read = watchdog._read_canonical_object_at
    swapped = False

    def read_then_swap(
        directory_fd: int,
        leaf_name: str,
        *,
        maximum_bytes: int,
        label: str,
        snapshot: watchdog._GateSnapshot | None = None,
    ):
        nonlocal swapped
        result = original_read(
            directory_fd,
            leaf_name,
            maximum_bytes=maximum_bytes,
            label=label,
            snapshot=snapshot,
        )
        if not swapped and leaf_name == target.name and label in {
            "status receipt case-00.fit-status.json",
            "campaign claim receipt",
            "training summary",
            "training evidence manifest",
            "training bundle evidence",
            "training evidence case 7",
            "target-free forecast artifact",
        }:
            target.unlink()
            target.symlink_to(copied)
            swapped = True
        return result

    monkeypatch.setattr(watchdog, "_read_canonical_object_at", read_then_swap)
    try:
        gate = watchdog._inspect_fit_phase_gate(
            receipt_fd,
            expected_receipt_directory=receipt_directory,
            expected_manifest_sha256="c" * 64,
            sync_directory=False,
            expected_integrated_source_hashes=source_hashes,
        )
    finally:
        os.close(receipt_fd)

    assert swapped is True
    assert gate["file_identity_revalidation_valid"] is False
    status_chain_leaf = relative_leaf in {
        "case-00.fit-status.json",
        "campaign.claim",
        campaign.SUMMARY_FILENAME,
    }
    assert gate["all_48_status_receipts_present_and_linked"] is (
        not status_chain_leaf
    )
    evidence_chain_leaf = status_chain_leaf or relative_leaf in {
        campaign.EVIDENCE_MANIFEST_FILENAME,
        "training-evidence/training-bundle.evidence.json",
        "training-evidence/case-07.training-evidence.json",
    }
    assert gate["training_evidence_chain_valid"] is (
        not evidence_chain_leaf
    )
    assert gate["pre_score_artifact_chain_valid"] is False
    assert any(
        "fit-gate file identity changed before return" in problem
        for problem in gate["problems"]
    )


def test_postfit_gate_rejects_fifo_status_without_blocking(tmp_path: Path) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    source_hashes = _write_private_fit_artifact_chain(receipt_directory)
    status_path = receipt_directory / "case-00.fit-status.json"
    status_path.unlink()
    os.mkfifo(status_path)
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private FIFO receipt root"
    )
    started = time.monotonic()
    try:
        gate = watchdog._inspect_fit_phase_gate(
            receipt_fd,
            expected_receipt_directory=receipt_directory,
            expected_manifest_sha256="c" * 64,
            sync_directory=False,
            expected_integrated_source_hashes=source_hashes,
        )
    finally:
        os.close(receipt_fd)
    assert time.monotonic() - started < 1.0
    assert gate["status_receipt_count"] == 47
    assert gate["pre_score_artifact_chain_valid"] is False
    assert any("cannot validate case-00.fit-status.json" in item for item in gate["problems"])


def test_postfit_gate_rejects_status_symlink_without_following_it(tmp_path: Path) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    source_hashes = _write_private_fit_artifact_chain(receipt_directory)
    victim = receipt_directory / "case-00.fit-status.json"
    copied = tmp_path / "copied-status.json"
    shutil.copyfile(victim, copied)
    victim.unlink()
    victim.symlink_to(copied)
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private symlink receipt root"
    )
    try:
        gate = watchdog._inspect_fit_phase_gate(
            receipt_fd,
            expected_receipt_directory=receipt_directory,
            expected_manifest_sha256="c" * 64,
            sync_directory=False,
            expected_integrated_source_hashes=source_hashes,
        )
    finally:
        os.close(receipt_fd)
    assert gate["status_receipt_count"] == 47
    assert gate["pre_score_artifact_chain_valid"] is False
    assert any("cannot validate case-00.fit-status.json" in item for item in gate["problems"])


def test_postfit_gate_rejects_copied_artifacts_ancestor_swap_after_open(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-checkout"
    receipt_directory = root / "artifacts" / "run" / "receipts"
    receipt_directory.mkdir(parents=True)
    source_hashes = _write_private_fit_artifact_chain(receipt_directory)
    receipt_fd = watchdog._open_absolute_directory_nofollow(
        receipt_directory, label="private anchored receipt root"
    )
    saved_artifacts = root / "artifacts.original"
    copied_artifacts = tmp_path / "copied-artifacts"
    shutil.copytree(root / "artifacts", copied_artifacts)
    (root / "artifacts").rename(saved_artifacts)
    (root / "artifacts").symlink_to(copied_artifacts, target_is_directory=True)
    try:
        gate = watchdog._inspect_fit_phase_gate(
            receipt_fd,
            expected_receipt_directory=receipt_directory,
            expected_manifest_sha256="c" * 64,
            sync_directory=False,
            expected_integrated_source_hashes=source_hashes,
        )
    finally:
        os.close(receipt_fd)
    assert gate["status_receipt_count"] == 48
    assert gate["receipt_root_identity_valid"] is False
    assert gate["all_48_status_receipts_present_and_linked"] is False
    assert gate["pre_score_artifact_chain_valid"] is False
    assert any("pinned receipt-root identity is invalid" in item for item in gate["problems"])

"""Fake-process checks for the external ABC6 watchdog seam."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import pytest

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


def test_fake_child_samples_pid_rss_hashes_receipt_and_excludes_watchdog(
    tmp_path: Path,
) -> None:
    result = _run_fake_child(tmp_path, "import time; time.sleep(0.12)")

    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    assert result["receipt"]["status"] == "completed"
    assert receipt["stop_reason"] == "child_exited"
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
    assert not tuple(receipt_path.parent.glob("*.tmp"))
    assert Path(result["receipt_path"]).is_file()


def test_wall_budget_terminates_and_reaps_fake_sleeper_with_bounded_grace(
    tmp_path: Path,
) -> None:
    result = _run_fake_child(
        tmp_path,
        "import time; time.sleep(10)",
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
    tmp_path: Path,
) -> None:
    marker = tmp_path / "detached-pid.txt"
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
        "time.sleep(0.08)\n"
        "os._exit(0)\n"
    )

    result = _run_fake_child(tmp_path, code, wall_seconds=1.0, sample_seconds=0.01)
    receipt = result["receipt"]
    detached_pid = int(marker.read_text(encoding="ascii"))
    try:
        assert psutil.pid_exists(detached_pid)
        assert receipt["status"] == "completed"
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
    manifest = tmp_path / "manifest.json"
    approval = tmp_path / "approval.json"
    receipts = tmp_path / "receipts"
    manifest.write_text("{}", encoding="ascii")
    receipts.mkdir()
    launch_called = False

    def reject_manifest(_path: Path, _sha: str) -> str:
        raise RuntimeError("strict source/roster pin rejected")

    def should_not_launch(**_kwargs: object):
        nonlocal launch_called
        launch_called = True
        raise AssertionError("launch reached before strict preflight")

    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", reject_manifest)
    monkeypatch.setattr(watchdog, "_supervise_command", should_not_launch)
    args = [
        "--manifest",
        str(manifest),
        "--manifest-sha256",
        "a" * 64,
        "--receipt-directory",
        str(receipts),
        "--approval-record",
        str(approval),
        "--approval-record-sha256",
        "b" * 64,
    ]

    with pytest.raises(watchdog.WatchdogError, match="strict manifest/source/runtime/roster"):
        watchdog._main(args)
    assert launch_called is False
    assert not watchdog._PRODUCTION_CLAIM_PATH.exists()


def test_approval_launch_record_binds_strict_hash_head_script_and_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    receipt_directory = tmp_path / "receipts"
    approval_path = tmp_path / "approval.json"
    receipt_directory.mkdir()
    manifest_path.write_text("{}", encoding="ascii")
    manifest_sha256 = "c" * 64
    head = "d" * 40
    campaign_vector = watchdog._build_campaign_command(
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
    monkeypatch.setattr(watchdog, "_campaign_strict_preflight", lambda *_: manifest_sha256)
    monkeypatch.setattr(
        watchdog,
        "_load_and_check_manifest",
        lambda *_: ({"protocol_id": watchdog.PROTOCOL_ID, "run_id": watchdog.RUN_ID}, manifest_sha256),
    )
    monkeypatch.setattr(watchdog, "_current_git_head", lambda: head)

    def write_record(extra: dict[str, object] | None = None) -> str:
        record: dict[str, object] = {
            "schema_version": 1,
            "record_type": "abc6_independent_approval_and_launch_v1",
            "status": "approved",
            "protocol_id": watchdog.PROTOCOL_ID,
            "run_id": watchdog.RUN_ID,
            "manifest_sha256": manifest_sha256,
            "reviewed_git_head": head,
            "reviewer_id": "independent-review-fixture",
            "approved_at_utc": "2026-09-28T00:00:00Z",
            "watchdog_script_sha256": _sha256(Path(watchdog.__file__).resolve()),
            "manifest_path": str(manifest_path),
            "receipt_directory": str(receipt_directory),
            "watchdog_launch_vector_template": watchdog._watchdog_launch_vector_template(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                receipt_directory=receipt_directory,
                approval_path=approval_path,
            ),
            "campaign_child_launch_vector": campaign_vector,
        }
        if extra:
            record.update(extra)
        raw = watchdog._canonical_json(record)
        approval_path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    approval_sha256 = write_record()
    actual_args[-1] = approval_sha256
    manifest, verified_sha, record = watchdog._production_preflight(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt_directory=receipt_directory,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        actual_watchdog_args=actual_args,
    )
    assert verified_sha == manifest_sha256
    assert record["reviewed_git_head"] == head
    assert manifest["protocol_id"] == watchdog.PROTOCOL_ID

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

    valid_sha = write_record()
    monkeypatch.setattr(watchdog, "_current_git_head", lambda: "f" * 40)
    with pytest.raises(watchdog.WatchdogError, match="does not bind current manifest"):
        watchdog._production_preflight(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            receipt_directory=receipt_directory,
            approval_path=approval_path,
            approval_sha256=valid_sha,
            actual_watchdog_args=[*actual_args[:-1], valid_sha],
        )


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

    assert first["receipt"]["status"] == "completed"
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


def test_postfit_gate_validates_all_48_durable_status_links(tmp_path: Path) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    from core.real_data.cascaded_tanks_abc6_cases import CASE_ROSTER

    campaign_claim: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": "c" * 64,
        "receipt_directory": str(receipt_directory.resolve()),
        "claimed_at_utc": "2026-09-28T00:00:00Z",
        "claim_semantics": "consumed_once_no_resume",
    }
    claim_bytes = watchdog._canonical_json(campaign_claim)
    (receipt_directory / "campaign.claim").write_bytes(claim_bytes)
    claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
    case_summaries = []
    for case_index in range(24):
        links = {}
        for component in ("fit", "baseline"):
            payload: dict[str, object] = {
                "schema_version": 1,
                "protocol_id": watchdog.PROTOCOL_ID,
                "run_id": watchdog.RUN_ID,
                "case_index": case_index,
                "case_id": CASE_ROSTER[case_index].case_id,
                "component": component,
                "status": "complete",
            }
            payload["payload_sha256"] = hashlib.sha256(
                watchdog._canonical_json(payload)
            ).hexdigest()
            raw = watchdog._canonical_json(payload)
            (receipt_directory / f"case-{case_index:02d}.{component}-status.json").write_bytes(raw)
            links[f"{component}_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        case_summaries.append(
            {
                "case_index": case_index,
                "case_id": CASE_ROSTER[case_index].case_id,
                "fit_status": "complete",
                "baseline_status": "complete",
                **links,
            }
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": watchdog.PROTOCOL_ID,
        "run_id": watchdog.RUN_ID,
        "manifest_sha256": "c" * 64,
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
    (receipt_directory / "campaign.training-summary.json").write_bytes(
        watchdog._canonical_json(summary)
    )

    gate = watchdog._inspect_fit_phase_gate(
        receipt_directory,
        expected_manifest_sha256="c" * 64,
        sync_directory=True,
    )
    assert gate["status_receipt_count"] == 48
    assert gate["all_48_status_receipts_present_and_linked"] is True
    assert gate["all_48_fit_and_baseline_statuses_complete"] is True
    assert gate["postfit_target_gate_opened"] is False
    assert gate["problems"] == []

    summary_path = receipt_directory / "campaign.training-summary.json"
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
    bool_gate = watchdog._inspect_fit_phase_gate(
        receipt_directory,
        expected_manifest_sha256="c" * 64,
        sync_directory=False,
    )
    assert bool_gate["status_receipt_count"] == 48
    assert bool_gate["all_48_fit_and_baseline_statuses_complete"] is False
    assert "training summary is not a valid durable campaign receipt" in bool_gate["problems"]

    wrong_case_id = dict(original_summary)
    wrong_case_statuses = [dict(entry) for entry in original_summary["case_statuses"]]
    wrong_case_statuses[12]["case_id"] = "wrong-case-id"
    wrong_case_id["case_statuses"] = wrong_case_statuses
    write_tampered_summary(wrong_case_id)
    wrong_case_gate = watchdog._inspect_fit_phase_gate(
        receipt_directory,
        expected_manifest_sha256="c" * 64,
        sync_directory=False,
    )
    assert wrong_case_gate["status_receipt_count"] == 48
    assert wrong_case_gate["all_48_status_receipts_present_and_linked"] is False
    assert wrong_case_gate["all_48_fit_and_baseline_statuses_complete"] is False
    assert "summary identity/status links do not match durable case receipts" in wrong_case_gate["problems"]

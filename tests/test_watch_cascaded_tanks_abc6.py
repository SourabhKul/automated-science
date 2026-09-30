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
        "schema_version": 1,
        "record_type": "abc6_independent_approval_and_launch_v1",
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
        "actual_args": actual_args,
        "child_vector": launch_vector,
        "manifest_body": manifest,
    }


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
            "schema_version": 1,
            "record_type": "abc6_independent_approval_and_launch_v1",
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


@pytest.mark.parametrize("control_file", ["manifest", "approval"])
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
    ["campaign_claim", "watchdog_claim", "scorer_marker", "receipt_output"],
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
    assert result["receipt"]["status"] == "completed"
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

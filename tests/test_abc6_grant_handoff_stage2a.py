"""Private fake-only checks for the ABC6 watchdog-to-runner bootstrap seam."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_cascaded_tanks_abc6_synthetic as runner
from scripts import watch_cascaded_tanks_abc6 as watchdog
from core.real_data import cascaded_tanks_abc6_authority as authority


_FAKE_RUN = "private-stage2a-grant"
_FAKE_PROTOCOL = "private-stage2a-protocol"
_OVERLAY_FILES = (
    "scripts/watch_cascaded_tanks_abc6.py",
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "core/real_data/cascaded_tanks_abc6_authority.py",
    "core/real_data/cascaded_tanks_abc6_campaign_fit.py",
    "core/real_data/cascaded_tanks_abc6_cases.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
    "core/real_data/cascaded_tanks_abc6_replay.py",
    "core/abc_smc_reference.py",
    "core/real_data/cascaded_tanks_abc6_receipt_io.py",
    "core/real_data/cascaded_tanks_abc6_training.py",
    "core/real_data/cascaded_tanks_models.py",
    "core/real_data/cascaded_tanks_synthetic_abc.py",
    "core/real_data/cascaded_tanks_pattern_search.py",
)
_PINNED_BOOTSTRAP_SOURCES = (
    "scripts/watch_cascaded_tanks_abc6.py",
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "core/real_data/cascaded_tanks_abc6_authority.py",
    "core/real_data/cascaded_tanks_abc6_campaign_fit.py",
    "core/real_data/cascaded_tanks_abc6_cases.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
    "core/real_data/cascaded_tanks_abc6_replay.py",
    "core/abc_smc_reference.py",
)


def _private_overlay(tmp_path: Path) -> tuple[Path, tuple[tuple[str, str], ...]]:
    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve() / "private-overlay"
    for relative in _OVERLAY_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    for relative in (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts",
        "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    sources = tuple(
        (relative, hashlib.sha256((root / relative).read_bytes()).hexdigest())
        for relative in sorted(_PINNED_BOOTSTRAP_SOURCES)
    )
    return root, sources


def _grant_run_identity(root: Path, sources: tuple[tuple[str, str], ...]):
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    root_stat = root.stat()
    receipt_stat = (root / receipt_relative).stat()
    return {
        "protocol_id": _FAKE_PROTOCOL,
        "run_id": _FAKE_RUN,
        "manifest_sha256": "1" * 64,
        "approval_record_sha256": "2" * 64,
        "review_sha256": "3" * 64,
        "reviewed_git_head": "a" * 40,
        "repository_root_realpath": str(root),
        "receipt_root_relative": receipt_relative,
        "repository_root_device": root_stat.st_dev,
        "repository_root_inode": root_stat.st_ino,
        "receipt_root_device": receipt_stat.st_dev,
        "receipt_root_inode": receipt_stat.st_ino,
        "source_hashes": dict(sources),
        "watchdog_attestation": {"private_fixture": True},
        "watchdog_image_path": str(Path(sys.executable).resolve()),
        "watchdog_image_sha256": hashlib.sha256(
            Path(sys.executable).resolve().read_bytes()
        ).hexdigest(),
    }


def test_runner_rejects_direct_api_missing_and_non_socket_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        runner.campaign_fit,
        "run_abc6_training_campaign_with_evidence",
        lambda *args: calls.append(args),
    )
    with pytest.raises(runner.ABC6SyntheticRunnerPreflightError, match="direct runner/API"):
        runner.run_cascaded_tanks_abc6_synthetic(
            tmp_path / "private-manifest.json", "4" * 64, tmp_path / "private-receipts"
        )
    with pytest.raises(runner.ABC6SyntheticRunnerPreflightError, match="canonical decimal"):
        runner._receive_launch_authority({})

    invalid_fd = os.open("/dev/null", os.O_RDONLY)
    try:
        with pytest.raises(runner.ABC6SyntheticRunnerPreflightError, match="not a socket"):
            runner._receive_launch_authority({runner.GRANT_FD_ENV: str(invalid_fd)})
    finally:
        try:
            os.close(invalid_fd)
        except OSError:
            pass
    assert calls == []


@pytest.mark.parametrize("post_send_swap", [False, True])
def test_watchdog_waits_through_launcher_then_closed_source_pin_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_send_swap: bool,
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    original_capture = authority._capture
    fake_launcher_path = str(Path(watchdog.PINNED_INTERPRETER_PATH).resolve(strict=True))
    fake_launcher_sha256, fake_launcher_identity = authority._hash_path(
        fake_launcher_path, authority.MAX_IMAGE_BYTES, "fake launcher image"
    )
    launcher_observations: dict[int, int] = {}
    stop_marker = root / "private-runner-stop.txt"
    release_marker = root / "private-child-release.txt"
    post_gate_samples = 0
    child_processes: list[object] = []
    real_popen = watchdog.subprocess.Popen
    real_snapshot = watchdog._snapshot_process_group
    publication_snapshots = []
    real_publish = authority._PendingGrant.publish

    def capture_with_fake_launcher_phase(pid: int):
        identity = original_capture(pid)
        if pid != os.getpid():
            observations = launcher_observations.get(pid, 0)
            launcher_observations[pid] = observations + 1
            if observations < 2:
                return replace(
                    identity,
                    image_path=fake_launcher_path,
                    image_sha256=fake_launcher_sha256,
                    image_id=fake_launcher_identity,
                )
        return identity

    def track_fake_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child_processes.append(child)
        return child

    def snapshot_then_release_after_gate(process_group_id: int):
        nonlocal post_gate_samples
        snapshot = real_snapshot(process_group_id)
        members, _rss, _vanished, errors = snapshot
        if (
            stop_marker.is_file()
            and child_processes
            and process_group_id == child_processes[0].pid
            and not errors
            and any(member.get("pid") == process_group_id for member in members)
            and post_gate_samples < 5
        ):
            post_gate_samples += 1
            if post_gate_samples == 5:
                release_marker.touch()
                deadline = watchdog.time.monotonic() + 1.0
                while (
                    child_processes[0].poll() is None
                    and watchdog.time.monotonic() < deadline
                ):
                    watchdog.time.sleep(0.005)
                if child_processes[0].poll() is not None:
                    snapshot = real_snapshot(process_group_id)
        return snapshot

    def publish_then_optional_swap(channel, bindings):
        snapshot = real_publish(channel, bindings)
        publication_snapshots.append(snapshot)
        if post_send_swap:
            received_marker = (
                Path(bindings.physical_root) / "private-grant-received.txt"
            )
            deadline = watchdog.time.monotonic() + 1.0
            while not received_marker.exists() and watchdog.time.monotonic() < deadline:
                watchdog.time.sleep(0.005)
            assert received_marker.is_file(), "fake child did not receive the grant frame"
            record_path = (
                Path(bindings.physical_root)
                / bindings.receipt_root_relative
                / authority.GRANT_RECORD
            )
            record = json.loads(record_path.read_text("ascii"))
            record["grant_fd"] += 1
            record_path.write_bytes(authority._json(record))
        return snapshot

    monkeypatch.setattr(authority, "_capture", capture_with_fake_launcher_phase)
    # The wrapper below adds a fake post-send race and therefore sits between
    # publish() and its watchdog caller; bypass only the local callsite guard.
    monkeypatch.setattr(authority, "_watchdog_callsite", lambda: None)
    monkeypatch.setattr(authority._PendingGrant, "publish", publish_then_optional_swap)
    monkeypatch.setattr(watchdog.subprocess, "Popen", track_fake_child)
    monkeypatch.setattr(
        watchdog, "_snapshot_process_group", snapshot_then_release_after_gate
    )
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    dummy_manifest = root / "private-manifest.json"
    expected_vector = watchdog._build_campaign_command(
        dummy_manifest, "5" * 64, receipt_path
    )
    assert expected_vector == [
        watchdog.PINNED_PYTHON_BIN,
        str(root / "scripts/run_cascaded_tanks_abc6_synthetic.py"),
        "--manifest",
        str(dummy_manifest),
        "--manifest-sha256",
        "5" * 64,
        "--receipts",
        str(receipt_path),
    ]
    # Use an in-process bootstrap probe so the private overlay is importable
    # from its working directory. The child vector itself remains unchanged.
    # It receives the grant and reaches the unchanged source-pin gate before
    # any campaign claim or prospective work.
    probe = (
        "from pathlib import Path\n"
        "import time\n"
        "from scripts import run_cascaded_tanks_abc6_synthetic as r\n"
        "try:\n"
        "    authority = r._receive_launch_authority()\n"
            "except BaseException as error:\n"
            "    Path('private-grant-error.txt').write_text(repr(error))\n"
            "    raise\n"
            "Path('private-grant-received.txt').write_text('received')\n"
        "try:\n"
        "    r.run_cascaded_tanks_abc6_synthetic('private-manifest.json','5'*64,'private-receipts',launch_authority=authority)\n"
        "except r.ABC6SyntheticRunnerPreflightError as error:\n"
        "    Path('private-runner-stop.txt').write_text(str(error)+' | '+repr(error.__cause__))\n"
        "else:\n"
        "    raise SystemExit(9)\n"
        "release = Path('private-child-release.txt')\n"
        "deadline = time.monotonic() + 1.5\n"
        "while not release.exists() and time.monotonic() < deadline: time.sleep(0.001)\n"
        "if not release.exists(): raise SystemExit(10)\n"
        "authority.close()\n"
    )
    vector = [watchdog.PINNED_PYTHON_BIN, "-c", probe]
    receipt_info = receipt_path.stat()
    run_identity = _grant_run_identity(root, source_hashes)
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    result = watchdog._supervise_command(
        launch_vector=vector,
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=run_identity,
        wall_limit_seconds=2.0,
        sample_interval_seconds=0.01,
        terminate_grace_seconds=0.1,
        kill_reap_grace_seconds=0.1,
        require_launch_grant=True,
    )
    assert result["receipt"]["status"] == "failed"
    if post_send_swap:
        assert result["receipt"]["stop_reason"] == "watchdog_monitoring_error"
        assert publication_snapshots
        assert result["receipt"]["bootstrap_grant"]["grant_record_sha256"] == (
            publication_snapshots[0].record_sha256
        )
        assert result["receipt"]["bootstrap_grant"]["delivery_status"] == "delivered"
    else:
        assert result["receipt"]["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
    assert release_marker.is_file() and post_gate_samples >= 5
    assert "integrated source pin gate is closed" in (
        root / "private-runner-stop.txt"
    ).read_text("utf-8")
    assert (root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims").is_dir()
    assert not list(
        (root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims").glob(
            f"{_FAKE_RUN}.claim"
        )
    )
    grant_receipt = receipt_path / "watchdog-child-grant.json"
    grant_body = json.loads(grant_receipt.read_text("ascii"))
    assert grant_body["grant_fd"] >= 3
    assert grant_body["grant_sha256"] == result["receipt"]["bootstrap_grant"]["grant_sha256"]
    if post_send_swap:
        assert grant_body["grant_fd"] != result["receipt"]["bootstrap_grant"]["descriptor_fd"]
        assert hashlib.sha256(grant_receipt.read_bytes()).hexdigest() != (
            result["receipt"]["bootstrap_grant"]["grant_record_sha256"]
        )
    else:
        assert result["receipt"]["bootstrap_grant"]["descriptor_fd"] == grant_body["grant_fd"]
    assert result["receipt"]["schema_version"] == 3
    assert result["receipt"]["bootstrap_grant"]["delivery_status"] == "delivered"
    grant_child_pid = grant_body["bindings"]["child_pid"]
    assert result["receipt"]["child_launch"]["observed_process"]["pid"] == grant_child_pid
    assert result["receipt"]["kill_and_reap"]["process_group_id"] == grant_child_pid
    if not post_send_swap:
        assert result["receipt"]["bootstrap_grant"]["grant_record_sha256"] == hashlib.sha256(
            grant_receipt.read_bytes()
        ).hexdigest()
    assert launcher_observations
    assert max(launcher_observations.values()) >= 4
    assert len(child_processes) == 1
    assert grant_body["bindings"]["child_image_path"] == watchdog.PINNED_PYTHON_APP
    assert grant_body["bindings"]["child_launch_image_path"] == watchdog.PINNED_INTERPRETER_PATH
    assert receipt_info.st_ino == receipt_path.stat().st_ino


def test_watchdog_times_out_in_launcher_phase_after_consuming_claim_without_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    monkeypatch.setattr(watchdog, "GRANT_RUNTIME_ATTESTATION_TIMEOUT_SECONDS", 0.12)
    original_capture = authority._capture
    fake_launcher_path = str(Path(watchdog.PINNED_INTERPRETER_PATH).resolve(strict=True))
    fake_launcher_sha256, fake_launcher_identity = authority._hash_path(
        fake_launcher_path, authority.MAX_IMAGE_BYTES, "fake launcher image"
    )

    def capture_stuck_in_fake_launcher(pid: int):
        identity = original_capture(pid)
        if pid != os.getpid():
            return replace(
                identity,
                image_path=fake_launcher_path,
                image_sha256=fake_launcher_sha256,
                image_id=fake_launcher_identity,
            )
        return identity

    monkeypatch.setattr(authority, "_capture", capture_stuck_in_fake_launcher)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    vector = [watchdog.PINNED_PYTHON_BIN, "-c", "import time; time.sleep(5)"]
    run_identity = _grant_run_identity(root, source_hashes)
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    result = watchdog._supervise_command(
        launch_vector=vector,
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=run_identity,
        wall_limit_seconds=2.0,
        sample_interval_seconds=0.01,
        terminate_grace_seconds=0.1,
        kill_reap_grace_seconds=0.1,
        require_launch_grant=True,
    )
    assert result["receipt"]["status"] == "failed"
    assert result["receipt"]["stop_reason"] == "child_runtime_attestation_timeout"
    assert claim_path.is_file()
    assert not (receipt_path / "watchdog-child-grant.json").exists()
    campaign_claims = root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims"
    assert not list(campaign_claims.glob("*.claim"))


def test_grant_channel_setup_failure_happens_before_claim_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    launches: list[object] = []

    def fail_channel_setup():
        raise RuntimeError("private fake channel setup failure")

    monkeypatch.setattr(
        watchdog.launch_authority,
        "_new_watchdog_grant_channel",
        fail_channel_setup,
    )
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    with pytest.raises(watchdog.WatchdogError, match="before claim"):
        watchdog._supervise_command(
            launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "pass"],
            observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
            observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
            claim_path=claim_path,
            receipt_path=receipt_path / "watchdog-terminal-receipt.json",
            run_identity=_grant_run_identity(root, source_hashes),
            require_launch_grant=True,
        )

    assert not claim_path.exists()
    assert not (receipt_path / "watchdog-terminal-receipt.json").exists()
    assert launches == []


def test_failed_watchdog_claim_closes_prepared_grant_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    launches: list[object] = []

    class FakeChannel:
        child_fd = 77
        closed = False

        def close(self) -> None:
            self.closed = True

    channel = FakeChannel()
    monkeypatch.setattr(
        watchdog.launch_authority,
        "_new_watchdog_grant_channel",
        lambda: channel,
    )

    def fail_claim(*args, **kwargs):
        raise watchdog.WatchdogError("private fake one-use claim failure")

    monkeypatch.setattr(watchdog, "_claim_once", fail_claim)
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    with pytest.raises(watchdog.WatchdogError, match="fake one-use claim failure"):
        watchdog._supervise_command(
            launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "pass"],
            observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
            observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
            claim_path=claim_path,
            receipt_path=receipt_path / "watchdog-terminal-receipt.json",
            run_identity=_grant_run_identity(root, source_hashes),
            require_launch_grant=True,
        )

    assert channel.closed is True
    assert not claim_path.exists()
    assert not (receipt_path / "watchdog-terminal-receipt.json").exists()
    assert launches == []


@pytest.mark.parametrize("faulted_clock", ["utc_now", "monotonic"])
def test_postclaim_timer_clock_failure_publishes_failed_terminal_without_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    faulted_clock: str,
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    campaign_claims = root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims"
    claim_is_durable = False
    real_claim_once = watchdog._claim_once

    def claim_then_mark(*args, **kwargs):
        nonlocal claim_is_durable
        digest = real_claim_once(*args, **kwargs)
        claim_is_durable = True
        return digest

    monkeypatch.setattr(watchdog, "_claim_once", claim_then_mark)
    injected_calls = 0
    launches: list[object] = []
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )

    if faulted_clock == "utc_now":
        real_clock = watchdog._utc_now

        def fail_persistently_postclaim_utc() -> str:
            nonlocal injected_calls
            if claim_is_durable:
                injected_calls += 1
                raise RuntimeError("private fake postclaim UTC clock failure")
            return real_clock()

        monkeypatch.setattr(watchdog, "_utc_now", fail_persistently_postclaim_utc)
    else:
        real_clock = watchdog.time.monotonic

        def fail_persistently_postclaim_monotonic() -> float:
            nonlocal injected_calls
            if claim_is_durable:
                injected_calls += 1
                raise RuntimeError("private fake postclaim monotonic clock failure")
            return real_clock()

        monkeypatch.setattr(
            watchdog.time, "monotonic", fail_persistently_postclaim_monotonic
        )

    result = watchdog._supervise_command(
        launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "pass"],
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=_grant_run_identity(root, source_hashes),
        require_launch_grant=True,
    )

    assert injected_calls >= (3 if faulted_clock == "utc_now" else 1)
    assert claim_path.is_file()
    assert result["receipt"]["status"] == "failed"
    assert result["receipt"]["stop_reason"] == "watchdog_monitoring_error"
    assert "private fake postclaim" in str(result["receipt"]["monitor_error"])
    assert result["receipt"]["process_return_code"] is None
    assert result["receipt"]["timer_start"]["started_monotonic_seconds"] is None
    assert "no child timer started" in result["receipt"]["timer_start"]["rule"]
    assert result["receipt"]["elapsed_wall_seconds"] == 0.0
    if faulted_clock == "utc_now":
        claimed_at_utc = json.loads(claim_path.read_text())["claimed_at_utc"]
        assert result["receipt"]["timer_start"]["started_at_utc"] == claimed_at_utc
        assert result["receipt"]["ended_at_utc"] == claimed_at_utc
    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True
    assert result["receipt_path"].is_file()
    assert result["terminal_readback_ack_path"].is_file()
    assert not (receipt_path / "watchdog-child-grant.json").exists()
    assert result["receipt"]["bootstrap_grant"]["grant_sha256"] is None
    assert not list(campaign_claims.glob("*.claim"))
    assert launches == []


def test_postclaim_elapsed_monotonic_failure_publishes_unknown_elapsed_and_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    campaign_claims = root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims"
    claim_is_durable = False
    popen_attempted = False
    real_claim_once = watchdog._claim_once
    real_monotonic = watchdog.time.monotonic

    def claim_then_mark(*args, **kwargs):
        nonlocal claim_is_durable
        digest = real_claim_once(*args, **kwargs)
        claim_is_durable = True
        return digest

    def fail_after_popen(*args, **kwargs):
        nonlocal popen_attempted
        popen_attempted = True
        raise RuntimeError("private fake child creation failure")

    def fail_elapsed_monotonic_persistently() -> float:
        if claim_is_durable and popen_attempted:
            raise RuntimeError("private fake elapsed monotonic failure")
        return real_monotonic()

    monkeypatch.setattr(watchdog, "_claim_once", claim_then_mark)
    monkeypatch.setattr(watchdog.subprocess, "Popen", fail_after_popen)
    monkeypatch.setattr(watchdog.time, "monotonic", fail_elapsed_monotonic_persistently)

    result = watchdog._supervise_command(
        launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "pass"],
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=_grant_run_identity(root, source_hashes),
        require_launch_grant=True,
    )

    receipt = result["receipt"]
    assert claim_is_durable and claim_path.is_file()
    assert popen_attempted is True
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "watchdog_monitoring_error"
    assert "private fake child creation failure" in str(receipt["monitor_error"])
    assert "elapsed monotonic timestamp unavailable" in str(receipt["monitor_error"])
    assert receipt["process_return_code"] is None
    assert type(receipt["timer_start"]["started_monotonic_seconds"]) is float
    assert receipt["elapsed_wall_seconds"] is None
    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True
    assert result["receipt_path"].is_file()
    assert result["terminal_readback_ack_path"].is_file()
    assert not (receipt_path / "watchdog-child-grant.json").exists()
    assert receipt["bootstrap_grant"]["grant_sha256"] is None
    assert not list(campaign_claims.glob("*.claim"))


def test_persistent_monotonic_failure_after_child_start_kills_and_reaps_before_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    campaign_claims = root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims"
    child_marker = tmp_path / "private-fake-child-started.txt"
    child_code = (
        "import os,time; "
        f"open({str(child_marker)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    real_popen = watchdog.subprocess.Popen
    real_monotonic = watchdog.time.monotonic
    child_processes: list[object] = []
    child_started = False

    def start_child_and_wait_for_private_marker(*args, **kwargs):
        nonlocal child_started
        child = real_popen(*args, **kwargs)
        child_processes.append(child)
        deadline = real_monotonic() + 2.0
        while not child_marker.exists() and real_monotonic() < deadline:
            watchdog.time.sleep(0.005)
        if not child_marker.exists():
            child.kill()
            child.wait()
            raise AssertionError("private fake child did not start before clock fault")
        child_started = True
        return child

    def fail_persistently_after_child_start() -> float:
        if child_started:
            raise RuntimeError("private fake persistent post-Popen monotonic failure")
        return real_monotonic()

    monkeypatch.setattr(watchdog.subprocess, "Popen", start_child_and_wait_for_private_marker)
    monkeypatch.setattr(watchdog.time, "monotonic", fail_persistently_after_child_start)

    result = watchdog._supervise_command(
        launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", child_code],
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=_grant_run_identity(root, source_hashes),
        wall_limit_seconds=2.0,
        sample_interval_seconds=0.01,
        terminate_grace_seconds=0.1,
        kill_reap_grace_seconds=0.2,
        require_launch_grant=True,
    )

    receipt = result["receipt"]
    kill_reap = receipt["kill_and_reap"]
    child_pid = int(child_marker.read_text())
    assert claim_path.is_file()
    assert child_started is True and len(child_processes) == 1
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "watchdog_monitoring_error"
    assert "private fake persistent post-Popen monotonic failure" in str(
        receipt["monitor_error"]
    )
    assert receipt["elapsed_wall_seconds"] is None
    assert receipt["process_return_code"] is not None
    assert receipt["watchdog_intervention"]["stage"] == "watchdog_monitoring_error"
    assert type(receipt["watchdog_intervention"]["monotonic_ns"]) is int
    assert receipt["intervention_capture"] == {
        "status": "recorded",
        "attempted_type": "watchdog_stop",
        "attempted_stage": "watchdog_monitoring_error",
        "failure_kind": None,
    }
    assert kill_reap["child_reaped"] is True
    assert kill_reap["membership_verification"] == "verified_empty"
    assert kill_reap["tracked_process_group_reaped"] is True
    assert kill_reap["term_sent_at_elapsed_seconds"] is None
    assert kill_reap["kill_sent_at_elapsed_seconds"] is None
    assert kill_reap["elapsed_seconds"] is None
    assert child_processes[0].poll() is not None
    assert not watchdog.psutil.pid_exists(child_pid)
    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True
    assert result["receipt_path"].is_file()
    assert result["terminal_readback_ack_path"].is_file()
    assert not (receipt_path / "watchdog-child-grant.json").exists()
    assert receipt["bootstrap_grant"]["grant_sha256"] is None
    assert not list(campaign_claims.glob("*.claim"))


def test_persistent_utc_failure_after_grant_keeps_monotonic_intervention_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    grant_published = False
    real_grant_record = authority._write_grant_record
    real_utc_now = watchdog._utc_now
    utc_failures = 0
    original_capture = authority._capture
    fake_launcher_path = str(Path(watchdog.PINNED_INTERPRETER_PATH).resolve(strict=True))
    fake_launcher_sha256, fake_launcher_identity = authority._hash_path(
        fake_launcher_path, authority.MAX_IMAGE_BYTES, "fake launcher image"
    )
    launcher_observations: dict[int, int] = {}

    def capture_with_fake_launcher_phase(pid: int):
        identity = original_capture(pid)
        if pid != os.getpid():
            observations = launcher_observations.get(pid, 0)
            launcher_observations[pid] = observations + 1
            if observations < 2:
                return replace(
                    identity,
                    image_path=fake_launcher_path,
                    image_sha256=fake_launcher_sha256,
                    image_id=fake_launcher_identity,
                )
        return identity

    def write_grant_record_then_mark(bindings, digest, grant_fd):
        nonlocal grant_published
        snapshot = real_grant_record(bindings, digest, grant_fd)
        grant_published = True
        return snapshot

    def fail_utc_after_grant() -> str:
        nonlocal utc_failures
        if grant_published:
            utc_failures += 1
            raise RuntimeError("private fake persistent post-grant UTC failure")
        return real_utc_now()

    monkeypatch.setattr(authority, "_write_grant_record", write_grant_record_then_mark)
    monkeypatch.setattr(authority, "_capture", capture_with_fake_launcher_phase)
    monkeypatch.setattr(watchdog, "_utc_now", fail_utc_after_grant)
    child_code = "import time; time.sleep(30)"
    result = watchdog._supervise_command(
        launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", child_code],
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=_grant_run_identity(root, source_hashes),
        wall_limit_seconds=2.0,
        sample_interval_seconds=0.01,
        terminate_grace_seconds=0.1,
        kill_reap_grace_seconds=0.2,
        require_launch_grant=True,
    )

    receipt = result["receipt"]
    assert grant_published is True, {
        key: receipt.get(key)
        for key in ("stop_reason", "monitor_error", "child_launch", "process_return_code")
    }
    assert utc_failures >= 2
    assert launcher_observations
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "watchdog_monitoring_error"
    assert receipt["bootstrap_grant"]["grant_sha256"] is not None
    assert receipt["bootstrap_grant"]["delivery_status"] == "delivered"
    assert receipt["bootstrap_grant"]["grant_record_sha256"] == hashlib.sha256(
        (receipt_path / authority.GRANT_RECORD).read_bytes()
    ).hexdigest()
    assert receipt["watchdog_intervention"] == {
        "type": "watchdog_stop",
        "stage": "watchdog_monitoring_error",
        "monotonic_ns": receipt["watchdog_intervention"]["monotonic_ns"],
        "occurred_at_utc": None,
        "clock": "host-local-monotonic-ns",
    }
    assert type(receipt["watchdog_intervention"]["monotonic_ns"]) is int
    assert receipt["intervention_capture"] == {
        "status": "recorded",
        "attempted_type": "watchdog_stop",
        "attempted_stage": "watchdog_monitoring_error",
        "failure_kind": None,
    }
    assert receipt["kill_and_reap"]["child_reaped"] is True
    assert receipt["kill_and_reap"]["membership_verification"] == "verified_empty"
    assert receipt["kill_and_reap"]["tracked_process_group_reaped"] is True
    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True


def test_unavailable_monotonic_event_after_child_start_is_retained_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
    receipt_relative = (
        "artifacts/cascaded_tanks_abc6_synthetic/" + _FAKE_RUN + "/receipts"
    )
    receipt_path = root / receipt_relative
    claim_path = root / "private-watchdog-claims" / f"{_FAKE_RUN}.claim"
    grant_record_written = False
    real_grant_record = authority._write_grant_record
    real_monotonic_ns = watchdog.time.monotonic_ns
    original_capture = authority._capture
    fake_launcher_path = str(Path(watchdog.PINNED_INTERPRETER_PATH).resolve(strict=True))
    fake_launcher_sha256, fake_launcher_identity = authority._hash_path(
        fake_launcher_path, authority.MAX_IMAGE_BYTES, "fake launcher image"
    )
    launcher_observations: dict[int, int] = {}

    def capture_with_fake_launcher_phase(pid: int):
        identity = original_capture(pid)
        if pid != os.getpid():
            observations = launcher_observations.get(pid, 0)
            launcher_observations[pid] = observations + 1
            if observations < 2:
                return replace(
                    identity,
                    image_path=fake_launcher_path,
                    image_sha256=fake_launcher_sha256,
                    image_id=fake_launcher_identity,
                )
        return identity

    def write_grant_then_fault_monotonic_ns(bindings, digest, grant_fd):
        nonlocal grant_record_written
        snapshot = real_grant_record(bindings, digest, grant_fd)
        grant_record_written = True
        return snapshot

    def fail_after_grant_record() -> int:
        if grant_record_written:
            raise RuntimeError("private fake persistent monotonic_ns failure")
        return real_monotonic_ns()

    monkeypatch.setattr(authority, "_capture", capture_with_fake_launcher_phase)
    monkeypatch.setattr(
        authority, "_write_grant_record", write_grant_then_fault_monotonic_ns
    )
    monkeypatch.setattr(watchdog.time, "monotonic_ns", fail_after_grant_record)
    result = watchdog._supervise_command(
        launch_vector=[watchdog.PINNED_PYTHON_BIN, "-c", "import time; time.sleep(30)"],
        observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
        observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
        claim_path=claim_path,
        receipt_path=receipt_path / "watchdog-terminal-receipt.json",
        run_identity=_grant_run_identity(root, source_hashes),
        wall_limit_seconds=2.0,
        sample_interval_seconds=0.01,
        terminate_grace_seconds=0.1,
        kill_reap_grace_seconds=0.2,
        require_launch_grant=True,
    )

    receipt = result["receipt"]
    assert grant_record_written is True
    assert receipt["status"] == "failed"
    assert receipt["intervention_capture"] == {
        "status": "unavailable",
        "attempted_type": "watchdog_stop",
        "attempted_stage": "watchdog_monitoring_error",
        "failure_kind": "monotonic_unavailable",
    }
    assert receipt["watchdog_intervention"] is None
    assert receipt["bootstrap_grant"]["grant_sha256"] is not None
    assert receipt["bootstrap_grant"]["delivery_status"] == "failed"
    assert receipt["bootstrap_grant"]["grant_record_sha256"] == hashlib.sha256(
        (receipt_path / authority.GRANT_RECORD).read_bytes()
    ).hexdigest()
    assert receipt["kill_and_reap"]["child_reaped"] is True
    assert receipt["kill_and_reap"]["membership_verification"] == "verified_empty"
    assert receipt["kill_and_reap"]["tracked_process_group_reaped"] is True
    assert result["terminal_receipt_readback_verified"] is True
    assert result["terminal_readback_ack_published"] is True


def test_incomplete_grant_sources_fail_before_watchdog_claim_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipts"
    receipt_path.mkdir()
    claim_path = tmp_path / "claims" / "private.claim"
    launches: list[object] = []
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: launches.append(args))
    with pytest.raises(watchdog.WatchdogError, match="source map is absent before claim"):
        watchdog._supervise_command(
            launch_vector=[str(Path(sys.executable).resolve()), "-c", "raise SystemExit(0)"],
            observed_executable_path=Path(watchdog.PINNED_PYTHON_APP),
            observed_executable_sha256=watchdog.PINNED_PYTHON_APP_SHA256,
            claim_path=claim_path,
            receipt_path=receipt_path / "watchdog-terminal-receipt.json",
            run_identity={
                "protocol_id": _FAKE_PROTOCOL,
                "run_id": _FAKE_RUN,
                "manifest_sha256": "1" * 64,
                "approval_record_sha256": "2" * 64,
                "review_sha256": "3" * 64,
                "reviewed_git_head": "a" * 40,
                "repository_root_realpath": str(tmp_path.resolve()),
                "receipt_root_relative": "receipts",
                "source_hashes": {},
            },
            require_launch_grant=True,
        )
    assert not claim_path.exists()
    assert launches == []

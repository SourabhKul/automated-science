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


def test_watchdog_waits_through_launcher_then_closed_source_pin_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source_hashes = _private_overlay(tmp_path)
    monkeypatch.setattr(watchdog, "_REPO_ROOT", root)
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

    monkeypatch.setattr(authority, "_capture", capture_with_fake_launcher_phase)
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
        "from scripts import run_cascaded_tanks_abc6_synthetic as r\n"
        "authority = r._receive_launch_authority()\n"
        "try:\n"
        "    r.run_cascaded_tanks_abc6_synthetic('private-manifest.json','5'*64,'private-receipts',launch_authority=authority)\n"
        "except r.ABC6SyntheticRunnerPreflightError as error:\n"
        "    Path('private-runner-stop.txt').write_text(str(error)+' | '+repr(error.__cause__))\n"
        "else:\n"
        "    raise SystemExit(9)\n"
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
    assert result["receipt"]["stop_reason"] == "child_exited_with_invalid_pre_score_gate"
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
    assert result["receipt"]["bootstrap_grant"]["descriptor_fd"] == grant_body["grant_fd"]
    assert launcher_observations
    assert max(launcher_observations.values()) >= 4
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

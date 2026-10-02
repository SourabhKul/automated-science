"""Fake-only tests for the isolated ABC6 supervised-child authority primitive."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from core.real_data import cascaded_tanks_abc6_authority as authority


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_child_source() -> str:
    return r'''from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
from core.real_data import cascaded_tanks_abc6_authority as a

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "private-authority-test-v1"
PROTOCOL = "private-authority-protocol-v1"
RECEIPTS = "artifacts/cascaded_tanks_abc6_synthetic/" + RUN_ID + "/receipts"
CLAIM = "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/" + RUN_ID + ".claim"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)

def add_payload_hash(body):
    result = dict(body)
    result["payload_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return result

def fake_case(authority_value, permit, index):
    permit.consume(index)
    duplicate_rejected = False
    try:
        permit.consume(index)
    except a.ABC6AuthorityError:
        duplicate_rejected = True
    receipts = []
    for component in ("fit", "baseline"):
        body = {
            "schema_version": 1,
            "protocol_id": PROTOCOL,
            "run_id": RUN_ID,
            "case_index": index,
            "case_id": f"private-case-{index:02d}",
            "component": component,
            "status": "complete",
        }
        raw = canonical(add_payload_hash(body))
        filename = f"case-{index:02d}.{component}-status.json"
        write_bytes(ROOT / RECEIPTS / filename, raw)
        receipts.append((index, component, hashlib.sha256(raw).hexdigest()))
    return duplicate_rejected, receipts

def fake_campaign(authority_value):
    session = authority_value.begin_training()
    claim_body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "run_id": RUN_ID,
        "manifest_sha256": "1" * 64,
        "receipt_directory": str(ROOT / RECEIPTS),
        "claimed_at_utc": "2026-09-30T00:00:00Z",
        "claim_semantics": "consumed_once_no_resume",
    }
    claim_raw = canonical(claim_body)
    write_bytes(ROOT / CLAIM, claim_raw)
    claim_sha = hashlib.sha256(claim_raw).hexdigest()

    duplicate_index_rejected = False
    case_receipts = []
    duplicate_consume_rejected = []
    for index in range(24):
        permit = session.issue_case_permit(index)
        if index == 0:
            try:
                session.issue_case_permit(index)
            except a.ABC6AuthorityError:
                duplicate_index_rejected = True
        duplicate, receipts = fake_case(authority_value, permit, index)
        duplicate_consume_rejected.append(duplicate)
        case_receipts.extend(receipts)

    statuses = [{
        "case_index": index,
        "case_id": f"private-case-{index:02d}",
        "fit_status": "complete",
        "baseline_status": "complete",
        "fit_receipt_sha256": next(d for i, c, d in case_receipts if i == index and c == "fit"),
        "baseline_receipt_sha256": next(d for i, c, d in case_receipts if i == index and c == "baseline"),
    } for index in range(24)]
    summary_body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "run_id": RUN_ID,
        "manifest_sha256": "1" * 64,
        "claim_sha256": claim_sha,
        "case_statuses": statuses,
    }
    summary_raw = canonical(add_payload_hash(summary_body))
    write_bytes(ROOT / RECEIPTS / "campaign.training-summary.json", summary_raw)
    summary_sha = hashlib.sha256(summary_raw).hexdigest()

    forecast_body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "run_id": RUN_ID,
        "training_manifest_sha256": "1" * 64,
        "training_claim_sha256": claim_sha,
        "training_summary_filename": "campaign.training-summary.json",
        "training_summary_sha256": summary_sha,
        "status_receipts": [
            {"filename": f"case-{i:02d}.{component}-status.json", "sha256": digest}
            for i, component, digest in case_receipts
        ],
        "target_free": True,
        "prospective_targets_generated_by_runner": False,
        "retry_allowed": False,
    }
    forecast_raw = canonical(add_payload_hash(forecast_body))
    write_bytes(ROOT / RECEIPTS / "campaign.target-free-forecasts.json", forecast_raw)
    forecast_sha = hashlib.sha256(forecast_raw).hexdigest()
    return case_receipts, summary_sha, forecast_sha, duplicate_index_rejected, duplicate_consume_rejected

def fake_wrong_campaign(authority_value):
    return authority_value.begin_training()

def fake_scorer(authority_value, score_permit):
    metadata = score_permit.consume()
    duplicate_score_rejected = False
    try:
        score_permit.consume()
    except a.ABC6AuthorityError:
        duplicate_score_rejected = True
    return metadata, duplicate_score_rejected

def pin_role_frame(function, role):
    _, relative_path, function_name = a.ABC6_ROLE_CONTRACT[
        {"runner": 0, "campaign": 1, "case": 2, "scorer": 3}[role]
    ]
    filename = str(ROOT / relative_path)
    if role == "campaign" and Path(__file__).name == "copied_fake_child.py":
        filename = str(Path(__file__).resolve())
    function.__code__ = function.__code__.replace(
        co_name=function_name,
        co_filename=filename,
    )

def fake_runner(authority_value):
    authority_value.consume_runner_startup()
    case_receipts, summary_sha, forecast_sha, duplicate_index, duplicate_consumes = fake_campaign(authority_value)
    handoff = authority_value.create_runner_handoff(tuple(case_receipts), summary_sha, forecast_sha)
    score_permit = authority_value.activate_scoring(handoff)
    repeated_activation_rejected = False
    try:
        authority_value.activate_scoring(handoff)
    except a.ABC6AuthorityError:
        repeated_activation_rejected = True
    metadata, duplicate_score = fake_scorer(authority_value, score_permit)
    grant_record = authority_value._grant_record
    return {
        "metadata": metadata,
        "grant_record_snapshot": {
            "record_sha256": grant_record.record_sha256,
            "file_identity": [
                grant_record.file_identity.device,
                grant_record.file_identity.inode,
                grant_record.file_identity.size,
                grant_record.file_identity.mtime_ns,
                grant_record.file_identity.ctime_ns,
            ],
            "file_mode": grant_record.file_mode,
            "regular_file": grant_record.regular_file,
        },
        "duplicate_index_rejected": duplicate_index,
        "duplicate_case_consumes_rejected": all(duplicate_consumes),
        "repeated_activation_rejected": repeated_activation_rejected,
        "duplicate_score_rejected": duplicate_score,
    }

pin_role_frame(fake_runner, "runner")
pin_role_frame(fake_campaign, "campaign")
pin_role_frame(fake_case, "case")
pin_role_frame(fake_scorer, "scorer")

def main():
    fd = int(sys.argv[1])
    mode = sys.argv[2]
    authority_value = a.receive_child_grant(fd)
    if mode == "changed-process":
        a._process_vector = lambda _pid: ("changed",)
        try:
            fake_campaign(authority_value)
        except a.ABC6AuthorityError:
            print(json.dumps({"changed_process_rejected": True}, sort_keys=True))
            return 0
        print(json.dumps({"changed_process_rejected": False}, sort_keys=True))
        return 2
    if mode in ("wrong-source", "wrong-function"):
        entry = fake_campaign if mode == "wrong-source" else fake_wrong_campaign
        try:
            entry(authority_value)
        except a.ABC6AuthorityError:
            print(json.dumps({"wrong_entrypoint_rejected": True}, sort_keys=True))
            return 0
        print(json.dumps({"wrong_entrypoint_rejected": False}, sort_keys=True))
        return 2
    print(json.dumps(fake_runner(authority_value), sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _make_fake_root(tmp_path: Path, source: bytes) -> tuple[Path, str, Path]:
    root = tmp_path.resolve()
    source_path = root / authority.ABC6_ROLE_CONTRACT[0][1]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source)
    for relative_path in authority.ABC6_MANIFEST_SOURCE_PATHS:
        target = root / relative_path
        if target == source_path:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"private fake authority source: {relative_path}\n".encode("ascii"))
    run_id = "private-authority-test-v1"
    receipt_relative = f"artifacts/cascaded_tanks_abc6_synthetic/{run_id}/receipts"
    receipt_path = root / receipt_relative
    receipt_path.mkdir(parents=True)
    (root / "artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims").mkdir(parents=True)
    return root, receipt_relative, receipt_path


def _bindings(root: Path, receipt_relative: str, identity: authority.ProcessIdentity, *, child_pid: int | None = None, vector: tuple[str, ...] | None = None, root_device: int | None = None):
    root_stat = root.stat()
    receipt_stat = (root / receipt_relative).stat()
    child_vector = identity.vector if vector is None else vector
    runtime_image_path = authority._vector_image_path(child_vector)
    runtime_image_sha256, _ = authority._hash_path(runtime_image_path, authority.MAX_IMAGE_BYTES, "fake runtime image")
    return authority.ABC6AuthorityBindings(
        protocol_id="private-authority-protocol-v1",
        run_id="private-authority-test-v1",
        manifest_sha256="1" * 64,
        approval_sha256="2" * 64,
        review_sha256="3" * 64,
        physical_root=str(root),
        receipt_root_relative=receipt_relative,
        root_device=root_stat.st_dev if root_device is None else root_device,
        root_inode=root_stat.st_ino,
        receipt_device=receipt_stat.st_dev,
        receipt_inode=receipt_stat.st_ino,
        reviewed_git_head="a" * 40,
        watchdog_claim_sha256="4" * 64,
        watchdog_pid=os.getpid(),
        watchdog_start=authority._process_start(os.getpid()),
        child_pid=identity.pid if child_pid is None else child_pid,
        child_start=identity.start if child_pid is None else "darwin:wrong-child-start",
        child_vector=child_vector,
        child_launch_image_path=identity.image_path,
        child_launch_image_sha256=identity.image_sha256,
        child_image_path=runtime_image_path,
        child_image_sha256=runtime_image_sha256,
        source_hashes=tuple(
            sorted(
                (
                    path,
                    _digest((root / path).read_bytes()),
                )
                for path in authority.ABC6_MANIFEST_SOURCE_PATHS
            )
        ),
        runtime_hashes=tuple(sorted((
            ("child_image_sha256", runtime_image_sha256),
            ("child_launch_image_sha256", identity.image_sha256),
            ("python_version_sha256", _digest(sys.version.encode())),
        ))),
        campaign_claim_relative=f"artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/private-authority-test-v1.claim",
        runner_source_path=authority.ABC6_ROLE_CONTRACT[0][1],
        runner_function=authority.ABC6_ROLE_CONTRACT[0][2],
        campaign_source_path=authority.ABC6_ROLE_CONTRACT[1][1],
        campaign_function=authority.ABC6_ROLE_CONTRACT[1][2],
        case_source_path=authority.ABC6_ROLE_CONTRACT[2][1],
        case_function=authority.ABC6_ROLE_CONTRACT[2][2],
        scorer_source_path=authority.ABC6_ROLE_CONTRACT[3][1],
        scorer_function=authority.ABC6_ROLE_CONTRACT[3][2],
    )


def _capture_stable_child(pid: int, timeout: float = 3.0) -> authority.ProcessIdentity:
    deadline = time.monotonic() + timeout
    previous = None
    consecutive = 0
    while time.monotonic() < deadline:
        identity = authority._capture(pid)
        current = (identity.start, identity.vector, identity.image_path, identity.image_sha256, identity.image_id)
        if current == previous:
            consecutive += 1
            if consecutive >= 2:
                return identity
        else:
            previous = current
            consecutive = 1
        time.sleep(0.01)
    raise authority.ABC6AuthorityError("child process identity did not stabilize before fake grant")


def test_authority_bindings_require_exact_source_map_and_role_pairs(tmp_path):
    root, receipt_relative, _receipt = _make_fake_root(tmp_path, b"private fake only")
    binding = _bindings(root, receipt_relative, authority._capture(os.getpid()))

    with pytest.raises(ValueError, match="sixteen-path"):
        replace(binding, source_hashes=binding.source_hashes[:-1])
    with pytest.raises(ValueError, match="sixteen-path"):
        replace(
            binding,
            source_hashes=tuple(
                sorted((*binding.source_hashes[:-1], ("extra.py", "9" * 64)))
            ),
        )

    for role, path, function in authority.ABC6_ROLE_CONTRACT:
        path_field = f"{role}_source_path"
        function_field = f"{role}_function"
        with pytest.raises(ValueError, match="frozen ABC6 role contract"):
            replace(binding, **{path_field: path + ".changed"})
        with pytest.raises(ValueError, match="frozen ABC6 role contract"):
            replace(binding, **{function_field: function + "_changed"})


def _wire(
    bindings: authority.ABC6AuthorityBindings,
    *,
    grant_fd: int,
    secret: str = "b" * 64,
    age_ns: int = 0,
) -> bytes:
    issued = time.monotonic_ns() - age_ns
    value = {
        "schema_version": 1,
        "issued_monotonic_ns": issued,
        "expires_monotonic_ns": issued + authority.GRANT_TTL_NS,
        "secret_hex": secret,
        "grant_fd": grant_fd,
        "bindings": bindings.payload(),
    }
    raw = authority._json(value)
    return struct.pack("!I", len(raw)) + raw


def _issue_fake_grant(
    monkeypatch,
    root: Path,
    receipt_relative: str,
    source_path: Path,
    *,
    mode: str = "valid",
):
    monkeypatch.setattr(authority, "_watchdog_callsite", lambda: None)
    channel = authority._new_watchdog_grant_channel()
    child_fd = channel.child_fd
    vector = [sys.executable, "-S", str(source_path), str(child_fd), mode]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    process = subprocess.Popen(vector, pass_fds=(child_fd,), cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    channel.close_child_copy()
    try:
        child_identity = _capture_stable_child(process.pid)
    except BaseException:
        process.kill()
        process.communicate(timeout=5)
        channel.close()
        raise
    binding = _bindings(root, receipt_relative, child_identity)
    grant_snapshot = channel.publish(binding)
    return channel, process, grant_snapshot, binding


def test_fake_child_traverses_one_use_training_and_scoring_once(tmp_path, monkeypatch):
    source = _fake_child_source().encode("utf-8")
    root, receipt_relative, receipt = _make_fake_root(tmp_path, source)
    script = root / authority.ABC6_ROLE_CONTRACT[0][1]
    channel, child, grant_digest, binding = _issue_fake_grant(monkeypatch, root, receipt_relative, script)
    stdout, stderr = child.communicate(timeout=10)
    channel.close()
    assert child.returncode == 0, stderr
    result = json.loads(stdout)
    metadata = result["metadata"]
    grant_receipt_stat = grant_digest.file_identity
    assert result["grant_record_snapshot"] == {
        "record_sha256": grant_digest.record_sha256,
        "file_identity": [
            grant_receipt_stat.device,
            grant_receipt_stat.inode,
            grant_receipt_stat.size,
            grant_receipt_stat.mtime_ns,
            grant_receipt_stat.ctime_ns,
        ],
        "file_mode": 0o600,
        "regular_file": True,
    }
    assert metadata["grant_sha256"] == grant_digest.grant_sha256
    assert metadata["run_id"] == "private-authority-test-v1"
    assert metadata["child_launch_image_path"] == binding.child_launch_image_path
    assert metadata["child_launch_image_sha256"] == binding.child_launch_image_sha256
    assert metadata["child_image_path"] == binding.child_image_path
    assert metadata["child_image_sha256"] == binding.child_image_sha256
    assert metadata["issued_case_indices"] == list(range(24))
    assert metadata["consumed_case_indices"] == list(range(24))
    assert len(metadata["status_receipts"]) == 48
    assert metadata["summary_sha256"] == _digest((receipt / "campaign.training-summary.json").read_bytes())
    assert metadata["forecast_sha256"] == _digest((receipt / "campaign.target-free-forecasts.json").read_bytes())
    assert result["duplicate_index_rejected"] is True
    assert result["duplicate_case_consumes_rejected"] is True
    assert result["repeated_activation_rejected"] is True
    assert result["duplicate_score_rejected"] is True
    grant_record = (receipt / authority.GRANT_RECORD).read_text()
    assert "secret_hex" not in grant_record
    assert grant_digest.grant_sha256 in grant_record
    assert hashlib.sha256(grant_record.encode("ascii")).hexdigest() == grant_digest.record_sha256
    assert grant_digest == channel.publication_snapshot
    authority.verify_grant_record_snapshot(grant_digest)
    assert binding.watchdog_claim_sha256 in grant_record
    assert "secret_hex" not in json.dumps(metadata)


def test_changed_process_identity_fails_before_training(tmp_path, monkeypatch):
    source = _fake_child_source().encode("utf-8")
    root, receipt_relative, _receipt = _make_fake_root(tmp_path, source)
    # The valid end-to-end case already proved the process binding. Here the
    # focused negative probe mutates the process-vector provider after receipt.
    monkeypatch.setattr(authority, "_watchdog_callsite", lambda: None)
    channel = authority._new_watchdog_grant_channel()
    fd = channel.child_fd
    script = root / authority.ABC6_ROLE_CONTRACT[0][1]
    vector = [sys.executable, "-S", str(script), str(fd), "changed-process"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen(vector, pass_fds=(fd,), cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    channel.close_child_copy()
    identity = _capture_stable_child(child.pid)
    binding = _bindings(root, receipt_relative, identity)
    channel.publish(binding)
    stdout, stderr = child.communicate(timeout=10)
    channel.close()
    assert child.returncode == 0, stderr
    assert json.loads(stdout)["changed_process_rejected"] is True


@pytest.mark.parametrize("mode", ("wrong-source", "wrong-function"))
def test_training_authority_rejects_wrong_campaign_source_or_function(
    tmp_path, monkeypatch, mode
):
    source = _fake_child_source().encode("utf-8")
    root, receipt_relative, _receipt = _make_fake_root(tmp_path, source)
    script = root / authority.ABC6_ROLE_CONTRACT[0][1]
    if mode == "wrong-source":
        script = root / "scripts/copied_fake_child.py"
        script.write_bytes(source)
    channel, child, _snapshot, _binding = _issue_fake_grant(
        monkeypatch, root, receipt_relative, script, mode=mode
    )
    stdout, stderr = child.communicate(timeout=10)
    channel.close()
    assert child.returncode == 0, stderr
    assert json.loads(stdout)["wrong_entrypoint_rejected"] is True


def test_launch_and_runtime_image_bindings_are_checked_independently(tmp_path):
    root, receipt_relative, _ = _make_fake_root(tmp_path, b"fake only")
    identity = authority._capture(os.getpid())
    binding = _bindings(root, receipt_relative, identity)

    def reject_frame(candidate, message):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            right.sendall(_wire(candidate, grant_fd=left.fileno()))
            right.shutdown(socket.SHUT_WR)
            with pytest.raises(authority.ABC6AuthorityError, match=message):
                authority.receive_child_grant(left.fileno())
        finally:
            left.close(); right.close()

    # A forged runtime image must fail against the child-observed image.
    runtime_hashes = dict(binding.runtime_hashes)
    runtime_hashes["child_image_sha256"] = "f" * 64
    wrong_runtime = replace(
        binding,
        child_image_path="/private/abc6-fake-spoofed-runtime-image",
        child_image_sha256="f" * 64,
        runtime_hashes=tuple(sorted(runtime_hashes.items())),
    )
    reject_frame(wrong_runtime, "runtime image binding")

    # A forged parent launch-image digest fails independently even when the
    # child-observed runtime path and digest remain valid.
    runtime_hashes = dict(binding.runtime_hashes)
    runtime_hashes["child_launch_image_sha256"] = "e" * 64
    wrong_launch = replace(
        binding,
        child_launch_image_sha256="e" * 64,
        runtime_hashes=tuple(sorted(runtime_hashes.items())),
    )
    reject_frame(wrong_launch, "launch image digest")


def test_distinct_fake_launcher_alias_and_child_runtime_images_are_not_collapsed(tmp_path):
    root, receipt_relative, _ = _make_fake_root(tmp_path, b"fake only")
    identity = authority._capture(os.getpid())
    binding = _bindings(root, receipt_relative, identity)
    launch_alias = root / "fake-launcher-image.bin"
    launch_alias.write_bytes(b"private fake launcher image alias")
    launch_sha256 = _digest(launch_alias.read_bytes())
    assert str(launch_alias) != binding.child_image_path
    assert launch_sha256 != binding.child_image_sha256
    runtime_hashes = dict(binding.runtime_hashes)
    runtime_hashes["child_launch_image_sha256"] = launch_sha256
    split = replace(
        binding,
        child_launch_image_path=str(launch_alias),
        child_launch_image_sha256=launch_sha256,
        runtime_hashes=tuple(sorted(runtime_hashes.items())),
    )
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        right.sendall(_wire(split, grant_fd=left.fileno()))
        right.shutdown(socket.SHUT_WR)
        # The distinct fake launch alias and child-observed runtime image both
        # pass their own checks; this fake root has no one-use grant receipt
        # to complete the bootstrap, so absence is the expected final rejection.
        with pytest.raises(authority.ABC6AuthorityError, match="grant digest receipt"):
            authority.receive_child_grant(left.fileno())
    finally:
        left.close(); right.close()


def test_untrusted_objects_paths_and_booleans_cannot_mint_authority(tmp_path):
    with pytest.raises(TypeError):
        authority.ABC6LaunchAuthority(object())
    with pytest.raises(TypeError):
        authority.ABC6TrainingSession(object(), None)
    with pytest.raises(TypeError):
        authority.ABC6TrainingPermit(object(), None, 0)
    with pytest.raises(TypeError):
        authority.ABC6RunnerHandoff(object())
    with pytest.raises(TypeError):
        authority.ABC6ScoringPermit(object())
    with pytest.raises(authority.ABC6AuthorityError):
        authority.receive_child_grant(True)
    with pytest.raises(authority.ABC6AuthorityError):
        authority.receive_child_grant(str(tmp_path / "grant.fifo"))


def test_truncated_delayed_and_wrong_child_frames_fail_closed(tmp_path, monkeypatch):
    root, receipt_relative, _ = _make_fake_root(tmp_path, b"fake only")
    identity = authority._capture(os.getpid())
    wrong_child = _bindings(root, receipt_relative, identity, child_pid=os.getpid() + 10000)

    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    right.sendall(struct.pack("!I", 40) + b"{}")
    right.shutdown(socket.SHUT_WR)
    with pytest.raises(authority.ABC6AuthorityError, match="truncated"):
        authority.receive_child_grant(left.fileno())
    left.close(); right.close()

    monkeypatch.setattr(authority, "GRANT_WAIT_NS", 10_000_000)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    def send_late():
        time.sleep(0.05)
        try:
            right.sendall(b"late")
        except OSError:
            pass
        right.close()

    sender = threading.Thread(target=send_late)
    sender.start()
    with pytest.raises(authority.ABC6AuthorityError, match="delayed"):
        authority.receive_child_grant(left.fileno())
    left.close(); sender.join(timeout=1)

    monkeypatch.setattr(authority, "GRANT_WAIT_NS", 5_000_000_000)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    right.sendall(_wire(wrong_child, grant_fd=left.fileno()))
    right.shutdown(socket.SHUT_WR)
    with pytest.raises(authority.ABC6AuthorityError, match="another child PID"):
        authority.receive_child_grant(left.fileno())
    left.close(); right.close()


def test_fifo_symlink_and_named_socket_delivery_are_rejected(tmp_path):
    fifo = tmp_path / "grant.fifo"
    os.mkfifo(fifo)
    fifo_link = tmp_path / "grant-link"
    fifo_link.symlink_to(fifo)
    fifo_fd = os.open(fifo_link, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(authority.ABC6AuthorityError, match="not a socket"):
            authority.receive_child_grant(fifo_fd)
    finally:
        os.close(fifo_fd)

    # Darwin sockaddr_un has a short pathname limit; pytest's nested temp
    # directory can exceed it, so use a unique short endpoint under /tmp.
    endpoint = Path("/tmp") / f"a6-{os.getpid()}-{time.monotonic_ns()}.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(endpoint)); server.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(endpoint))
        accepted, _ = server.accept()
        try:
            with pytest.raises(authority.ABC6AuthorityError, match="path-bound"):
                authority.receive_child_grant(accepted.fileno())
        finally:
            accepted.close(); client.close()
    finally:
        server.close()
        endpoint.unlink(missing_ok=True)


def test_root_identity_mismatch_and_extra_frame_bytes_are_rejected(tmp_path):
    root, receipt_relative, _ = _make_fake_root(tmp_path, b"fake only")
    identity = authority._capture(os.getpid())
    binding = _bindings(root, receipt_relative, identity, root_device=root.stat().st_dev + 1)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    wire = _wire(binding, grant_fd=left.fileno())
    right.sendall(wire)
    right.shutdown(socket.SHUT_WR)
    with pytest.raises(authority.ABC6AuthorityError, match="physical root"):
        authority.receive_child_grant(left.fileno())
    left.close(); right.close()

    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    wire = _wire(_bindings(root, receipt_relative, identity), grant_fd=left.fileno())
    def send_trailing_frame() -> None:
        right.sendall(wire + wire)
        right.shutdown(socket.SHUT_WR)

    sender = threading.Thread(target=send_trailing_frame)
    sender.start()
    with pytest.raises(authority.ABC6AuthorityError, match="trailing data"):
        authority.receive_child_grant(left.fileno())
    sender.join(timeout=2)
    assert not sender.is_alive()
    left.close(); right.close()


def test_grant_frame_is_bound_to_the_selected_descriptor(tmp_path):
    root, receipt_relative, _ = _make_fake_root(tmp_path, b"fake only")
    identity = authority._capture(os.getpid())
    binding = _bindings(root, receipt_relative, identity)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        wrong_fd = left.fileno() + 1
        right.sendall(_wire(binding, grant_fd=wrong_fd))
        right.shutdown(socket.SHUT_WR)
        with pytest.raises(authority.ABC6AuthorityError, match="different inherited FD"):
            authority.receive_child_grant(left.fileno())
    finally:
        left.close(); right.close()


def test_wrong_peer_vector_and_expired_grants_fail_before_authority(tmp_path):
    root, receipt_relative, _ = _make_fake_root(tmp_path, b"fake only")
    identity = authority._capture(os.getpid())

    probes = (
        (
            replace(_bindings(root, receipt_relative, identity), watchdog_pid=os.getpid() + 1),
            0,
            "peer does not match watchdog PID",
        ),
        (
            _bindings(
                root,
                receipt_relative,
                identity,
                vector=identity.vector + ("private-spoof",),
            ),
            0,
            "live child process differs",
        ),
        (
            _bindings(root, receipt_relative, identity),
            authority.GRANT_TTL_NS + 1,
            "expired or has an invalid issue time",
        ),
    )
    for candidate, age_ns, message in probes:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            right.sendall(_wire(candidate, grant_fd=left.fileno(), age_ns=age_ns))
            right.shutdown(socket.SHUT_WR)
            with pytest.raises(authority.ABC6AuthorityError, match=message):
                authority.receive_child_grant(left.fileno())
        finally:
            left.close(); right.close()

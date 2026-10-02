"""Fake-only coverage for the capability-bound campaign grant-leaf gate."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.real_data import cascaded_tanks_abc6_authority as authority
from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign

MANIFEST_SHA256 = "1" * 64
GRANT_SHA256 = "2" * 64
SCAN_ORDER = (
    "root-open",
    "execution-preclaim",
    "claim-entry",
    "pre-durable-claim",
)


class _ClaimReached(Exception):
    """The fake campaign reached its global claim without starting training."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fake_authority(root: Path) -> tuple[authority.ABC6LaunchAuthority, Path]:
    root = root.resolve()
    receipt_relative = campaign.RECEIPT_ROOT_RELATIVE
    receipt = root / receipt_relative
    receipt.mkdir(parents=True)
    root_fd = authority._open_root(str(root))
    receipt_fd = authority._open_beneath(root_fd, receipt_relative, directory=True)
    identity = authority._capture(os.getpid())
    runtime_image = authority._vector_image_path(identity.vector)
    runtime_image_sha256, _ = authority._hash_path(
        runtime_image, authority.MAX_IMAGE_BYTES, "fake child image"
    )
    paths = tuple(sorted(authority.ABC6_MANIFEST_SOURCE_PATHS))
    bindings = authority.ABC6AuthorityBindings(
        protocol_id=campaign.PROTOCOL_ID,
        run_id=campaign.RUN_ID,
        manifest_sha256=MANIFEST_SHA256,
        approval_sha256="3" * 64,
        review_sha256="4" * 64,
        physical_root=str(root),
        receipt_root_relative=receipt_relative,
        root_device=os.fstat(root_fd).st_dev,
        root_inode=os.fstat(root_fd).st_ino,
        receipt_device=os.fstat(receipt_fd).st_dev,
        receipt_inode=os.fstat(receipt_fd).st_ino,
        reviewed_git_head="a" * 40,
        watchdog_claim_sha256="5" * 64,
        watchdog_pid=os.getpid(),
        watchdog_start=authority._process_start(os.getpid()),
        child_pid=identity.pid,
        child_start=identity.start,
        child_vector=identity.vector,
        child_launch_image_path=identity.image_path,
        child_launch_image_sha256=identity.image_sha256,
        child_image_path=runtime_image,
        child_image_sha256=runtime_image_sha256,
        source_hashes=tuple((path, _sha(path.encode("utf-8"))) for path in paths),
        runtime_hashes=tuple(
            sorted(
                (
                    ("child_image_sha256", runtime_image_sha256),
                    ("child_launch_image_sha256", identity.image_sha256),
                    ("python_version_sha256", _sha(sys.version.encode())),
                )
            )
        ),
        campaign_claim_relative=(
            f"{campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE}/{campaign.RUN_ID}.claim"
        ),
        runner_source_path=authority.ABC6_ROLE_CONTRACT[0][1],
        runner_function=authority.ABC6_ROLE_CONTRACT[0][2],
        campaign_source_path=authority.ABC6_ROLE_CONTRACT[1][1],
        campaign_function=authority.ABC6_ROLE_CONTRACT[1][2],
        case_source_path=authority.ABC6_ROLE_CONTRACT[2][1],
        case_function=authority.ABC6_ROLE_CONTRACT[2][2],
        scorer_source_path=authority.ABC6_ROLE_CONTRACT[3][1],
        scorer_function=authority.ABC6_ROLE_CONTRACT[3][2],
    )
    grant_fd = 91
    raw = authority._json(
        {
            "schema_version": 1,
            "grant_sha256": GRANT_SHA256,
            "grant_fd": grant_fd,
            "bindings_sha256": _sha(authority._json(bindings.payload())),
            "bindings": bindings.payload(),
        }
    )
    path = receipt / authority.GRANT_RECORD
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
        file_info = os.fstat(fd)
    finally:
        os.close(fd)
    grant_record = authority._ReceivedGrantRecord(
        record_sha256=_sha(raw),
        file_identity=authority._FileId.of(file_info),
        file_mode=stat.S_IMODE(file_info.st_mode),
        regular_file=stat.S_ISREG(file_info.st_mode),
    )
    launch_authority = authority.ABC6LaunchAuthority(
        authority._AUTHORITY_SEAL,
        bindings,
        GRANT_SHA256,
        grant_fd,
        grant_record,
        root_fd,
        receipt_fd,
        identity,
    )
    return launch_authority, path


def _configure_fake_campaign(
    monkeypatch,
    root: Path,
    launch_authority: authority.ABC6LaunchAuthority | None,
    *,
    preflight_manifest_sha256: str = MANIFEST_SHA256,
) -> Path:
    source_hashes: dict[str, str] = {}
    for relative_path in campaign._REQUIRED_SOURCE_PATHS:
        source_path = root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_path.exists():
            source_path.write_bytes(
                f"private fake preclaim source: {relative_path}\n".encode("ascii")
            )
        source_hashes[relative_path] = _sha(source_path.read_bytes())
    monkeypatch.setattr(campaign, "_REPO_ROOT", root)
    claim_path = (
        root
        / campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{campaign.RUN_ID}.claim"
    )
    monkeypatch.setattr(campaign, "_PROJECT_RUN_CLAIM_PATH", claim_path)
    if launch_authority is not None:
        monkeypatch.setattr(authority, "_require_role", lambda *_args: None)
        launch_authority.consume_runner_startup()

        def fake_preflight(*_args, **_kwargs):
            assert launch_authority._state.training_started
            return (
                preflight_manifest_sha256,
                {
                    "repository_root_realpath": str(root),
                    "receipt_root_relative": campaign.RECEIPT_ROOT_RELATIVE,
                    "source_hashes": source_hashes,
                },
                authority._open_root(str(root)),
            )

    else:

        def fake_preflight(*_args, **_kwargs):
            return (
                preflight_manifest_sha256,
                {
                    "repository_root_realpath": str(root),
                    "receipt_root_relative": campaign.RECEIPT_ROOT_RELATIVE,
                    "source_hashes": source_hashes,
                },
                authority._open_root(str(root)),
            )

    monkeypatch.setattr(
        campaign, "_preflight_manifest_and_open_checkout_root", fake_preflight
    )
    monkeypatch.setattr(
        campaign,
        "_execute_campaign_after_claim",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_ClaimReached()),
    )
    monkeypatch.setattr(
        campaign,
        "build_synthetic_training_bundle",
        lambda: pytest.fail("fake preclaim gate reached simulation setup"),
    )
    monkeypatch.setattr(
        campaign.training,
        "run_abc6_training_case",
        lambda *_args, **_kwargs: pytest.fail(
            "fake preclaim gate reached a training fit"
        ),
    )
    monkeypatch.setattr(
        campaign.cases,
        "_materialize_prospective_targets",
        lambda *_args, **_kwargs: pytest.fail(
            "fake preclaim gate reached prospective target access"
        ),
    )
    return claim_path


def _run_public_campaign(
    root: Path,
    launch_authority: authority.ABC6LaunchAuthority | None,
) -> None:
    campaign.run_abc6_training_campaign_with_evidence(
        root / "fake-manifest.json",
        MANIFEST_SHA256,
        root / campaign.RECEIPT_ROOT_RELATIVE,
        launch_authority=launch_authority,
    )


def test_direct_campaign_remains_empty_only_and_rejects_the_grant_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    empty_root = tmp_path / "empty-checkout"
    empty_root.mkdir()
    empty_receipts = empty_root / campaign.RECEIPT_ROOT_RELATIVE
    empty_receipts.mkdir(parents=True)
    direct_claim = _configure_fake_campaign(monkeypatch, empty_root, None)
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="launch authority"):
        _run_public_campaign(empty_root, None)
    assert not direct_claim.exists()

    monkeypatch.undo()
    grant_root = tmp_path / "grant-checkout"
    grant_root.mkdir()
    receipt = grant_root / campaign.RECEIPT_ROOT_RELATIVE
    receipt.mkdir(parents=True)
    (receipt / authority.GRANT_RECORD).write_bytes(b"fake grant leaf")
    direct_claim = _configure_fake_campaign(monkeypatch, grant_root, None)
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="launch authority"):
        _run_public_campaign(grant_root, None)
    assert not direct_claim.exists()


def test_supervised_public_entry_checks_all_four_scans_before_claim(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    launch_authority, grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root
    observed_scans = []

    def record_scan(session, repository_root_fd, receipt_root_fd, *, scan, **context):
        observed_scans.append(scan)
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan=scan,
            **context,
        )

    monkeypatch.setattr(
        authority.ABC6TrainingSession, "verify_preclaim_root", record_scan
    )
    with pytest.raises(_ClaimReached):
        _run_public_campaign(root, launch_authority)
    assert observed_scans == list(SCAN_ORDER)
    assert launch_authority._state.training_started
    assert launch_authority._state.issued == set(range(campaign.CASE_COUNT))
    assert launch_authority._grant_record.file_identity == authority._FileId.of(
        os.stat(grant_path, follow_symlinks=False)
    )
    assert claim_path.is_file()
    launch_authority.close()


@pytest.mark.parametrize("scan", SCAN_ORDER)
@pytest.mark.parametrize(
    "snapshot_field",
    ("record_sha256", "device", "inode", "size", "mtime_ns", "ctime_ns", "mode"),
)
def test_each_preclaim_scan_rejects_stale_snapshot_fields(
    tmp_path: Path, monkeypatch, scan: str, snapshot_field: str
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root
    did_corrupt = False

    def corrupt_before_target(
        session,
        repository_root_fd,
        receipt_root_fd,
        *,
        scan: str,
        **context,
    ):
        nonlocal did_corrupt
        if scan == corrupt_before_target.target_scan:
            snapshot = launch_authority._grant_record
            if snapshot_field == "record_sha256":
                launch_authority._grant_record = replace(
                    snapshot, record_sha256="f" * 64
                )
            elif snapshot_field == "mode":
                launch_authority._grant_record = replace(
                    snapshot, file_mode=snapshot.file_mode ^ 0o001
                )
            else:
                identity = snapshot.file_identity
                field_name = snapshot_field
                launch_authority._grant_record = replace(
                    snapshot,
                    file_identity=replace(
                        identity,
                        **{field_name: getattr(identity, field_name) + 1},
                    ),
                )
            did_corrupt = True
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan=scan,
            **context,
        )

    corrupt_before_target.target_scan = scan
    monkeypatch.setattr(
        authority.ABC6TrainingSession,
        "verify_preclaim_root",
        corrupt_before_target,
    )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="preclaim"):
        _run_public_campaign(root, launch_authority)
    assert did_corrupt
    assert not claim_path.exists()
    assert launch_authority._state.issued == set(range(campaign.CASE_COUNT))
    launch_authority.close()


@pytest.mark.parametrize("scan", SCAN_ORDER)
@pytest.mark.parametrize(
    "mutation", ("bytes", "mode", "symlink", "fifo", "missing", "extra", "copy")
)
def test_each_preclaim_scan_rejects_changed_or_unexpected_leaf(
    tmp_path: Path, monkeypatch, scan: str, mutation: str
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    launch_authority, grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root
    did_mutate = False

    def mutate_before_target(
        session,
        repository_root_fd,
        receipt_root_fd,
        *,
        scan: str,
        **context,
    ):
        nonlocal did_mutate
        if scan == mutate_before_target.target_scan:
            if mutation == "bytes":
                with grant_path.open("r+b") as stream:
                    stream.write(b"x" * grant_path.stat().st_size)
                    stream.flush()
                    os.fsync(stream.fileno())
            elif mutation == "mode":
                grant_path.chmod(0o640)
            elif mutation == "symlink":
                copy = grant_path.with_name("grant-copy")
                copy.write_bytes(grant_path.read_bytes())
                copy.chmod(0o600)
                grant_path.unlink()
                grant_path.symlink_to(copy.name)
            elif mutation == "fifo":
                grant_path.unlink()
                os.mkfifo(grant_path)
            elif mutation == "missing":
                grant_path.unlink()
            elif mutation == "extra":
                grant_path.with_name("unexpected-leaf").write_bytes(b"extra")
            elif mutation == "copy":
                raw = grant_path.read_bytes()
                grant_path.unlink()
                grant_path.write_bytes(raw)
                grant_path.chmod(0o600)
            did_mutate = True
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan=scan,
            **context,
        )

    mutate_before_target.target_scan = scan
    monkeypatch.setattr(
        authority.ABC6TrainingSession,
        "verify_preclaim_root",
        mutate_before_target,
    )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="preclaim"):
        _run_public_campaign(root, launch_authority)
    assert did_mutate
    assert not claim_path.exists()
    assert launch_authority._state.issued == set(range(campaign.CASE_COUNT))
    launch_authority.close()


def test_final_scan_observes_a_bounded_concurrent_extra_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    launch_authority, grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root

    def add_extra_before_final(
        session,
        repository_root_fd,
        receipt_root_fd,
        *,
        scan: str,
        **context,
    ):
        if scan == "pre-durable-claim":
            grant_path.with_name("concurrent-extra").write_bytes(b"bounded fake")
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan=scan,
            **context,
        )

    monkeypatch.setattr(
        authority.ABC6TrainingSession,
        "verify_preclaim_root",
        add_extra_before_final,
    )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="only the pinned grant"):
        _run_public_campaign(root, launch_authority)
    assert not claim_path.exists()
    launch_authority.close()


def test_supervised_entry_rejects_forged_and_stale_authority(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)

    with pytest.raises(campaign.ABC6CampaignPreflightError, match="received launch authority"):
        _run_public_campaign(root, object())
    assert not claim_path.exists()

    # A closed authority is stale and fails before the first receipt scan.
    launch_authority.close()
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="authority was rejected"):
        _run_public_campaign(root, launch_authority)
    assert not claim_path.exists()


def test_replayed_training_authority_is_rejected_after_failed_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root

    def fail_first_scan(
        session,
        repository_root_fd,
        receipt_root_fd,
        *,
        scan: str,
        **context,
    ):
        if scan == "root-open":
            launch_authority._grant_record = replace(
                launch_authority._grant_record, record_sha256="f" * 64
            )
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan=scan,
            **context,
        )

    monkeypatch.setattr(
        authority.ABC6TrainingSession,
        "verify_preclaim_root",
        fail_first_scan,
    )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="preclaim"):
        _run_public_campaign(root, launch_authority)
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="one-use"):
        _run_public_campaign(root, launch_authority)
    assert launch_authority._state.training_started
    assert not claim_path.exists()
    launch_authority.close()


def test_campaign_preclaim_rejects_a_different_root_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    other_root = (tmp_path / "other-root").resolve()
    other_root.mkdir()
    wrong_root_fd = authority._open_root(str(other_root))
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root

    def substitute_root(
        session,
        repository_root_fd,
        receipt_root_fd,
        *,
        scan: str,
        **context,
    ):
        if scan == "root-open":
            return original_verify(
                session,
                wrong_root_fd,
                receipt_root_fd,
                scan=scan,
                **context,
            )
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan=scan,
            **context,
        )

    monkeypatch.setattr(
        authority.ABC6TrainingSession,
        "verify_preclaim_root",
        substitute_root,
    )
    try:
        with pytest.raises(campaign.ABC6CampaignPreflightError, match="descriptors"):
            _run_public_campaign(root, launch_authority)
    finally:
        os.close(wrong_root_fd)
    assert not claim_path.exists()
    launch_authority.close()


@pytest.mark.parametrize("binding", ("protocol", "run", "manifest"))
def test_preclaim_session_is_bound_to_campaign_identity(
    tmp_path: Path, monkeypatch, binding: str
) -> None:
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    wrong_manifest_sha256 = "f" * 64 if binding == "manifest" else MANIFEST_SHA256
    claim_path = _configure_fake_campaign(
        monkeypatch,
        root,
        launch_authority,
        preflight_manifest_sha256=wrong_manifest_sha256,
    )
    if binding == "protocol":
        monkeypatch.setattr(campaign, "PROTOCOL_ID", "private-wrong-protocol")
    elif binding == "run":
        monkeypatch.setattr(campaign, "RUN_ID", "private-wrong-run")
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="identity differs"):
        _run_public_campaign(root, launch_authority)
    assert not claim_path.exists()
    launch_authority.close()


def test_training_session_rejects_out_of_order_preclaim_scan(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    original_verify = authority.ABC6TrainingSession.verify_preclaim_root

    def skip_to_claim_entry(
        session,
        repository_root_fd,
        receipt_root_fd,
        *,
        scan: str,
        **context,
    ):
        return original_verify(
            session,
            repository_root_fd,
            receipt_root_fd,
            scan="claim-entry" if scan == "root-open" else scan,
            **context,
        )

    monkeypatch.setattr(
        authority.ABC6TrainingSession,
        "verify_preclaim_root",
        skip_to_claim_entry,
    )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="preclaim"):
        _run_public_campaign(root, launch_authority)
    assert not claim_path.exists()
    launch_authority.close()


def test_campaign_preclaim_rejects_a_replaced_receipt_root_ancestor(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    launch_authority, _grant_path = _fake_authority(root)
    claim_path = _configure_fake_campaign(monkeypatch, root, launch_authority)
    moved_root = tmp_path / "moved-checkout"
    original_check = campaign._check_preclaim_receipt_root

    def replace_ancestor(identity, receipt_fd, session, *, scan, **context):
        if scan == "execution-preclaim":
            root.rename(moved_root)
            root.symlink_to(moved_root, target_is_directory=True)
        return original_check(
            identity,
            receipt_fd,
            session,
            scan=scan,
            **context,
        )

    monkeypatch.setattr(campaign, "_check_preclaim_receipt_root", replace_ancestor)
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="symlinked"):
        _run_public_campaign(root, launch_authority)
    assert not claim_path.exists()
    launch_authority.close()

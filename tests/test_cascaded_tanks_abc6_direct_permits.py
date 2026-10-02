"""Fake-only Stage A checks for the ABC6 direct-entrypoint permit gate."""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from core.real_data import cascaded_tanks_abc6_authority as authority
from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_training as training
from core.real_data.cascaded_tanks_models import TankState

_MANIFEST_SHA256 = "1" * 64
_GRANT_SHA256 = "2" * 64


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _copy_manifest_sources(root: Path) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    result: dict[str, str] = {}
    for relative in authority.ABC6_MANIFEST_SOURCE_PATHS:
        source = repository / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        destination.write_bytes(raw)
        result[relative] = _sha(raw)
    return result


def _fake_authority(root: Path) -> tuple[authority.ABC6LaunchAuthority, dict[str, str]]:
    root = root.resolve()
    source_hashes = _copy_manifest_sources(root)
    receipt_relative = campaign.RECEIPT_ROOT_RELATIVE
    receipt_path = root / receipt_relative
    receipt_path.mkdir(parents=True)
    root_fd = authority._open_root(str(root))
    receipt_fd = authority._open_beneath(root_fd, receipt_relative, directory=True)
    identity = authority._capture(os.getpid())
    runtime_image = authority._vector_image_path(identity.vector)
    runtime_image_sha256, _ = authority._hash_path(
        runtime_image, authority.MAX_IMAGE_BYTES, "fake child image"
    )
    root_stat = os.fstat(root_fd)
    receipt_stat = os.fstat(receipt_fd)
    bindings = authority.ABC6AuthorityBindings(
        protocol_id=campaign.PROTOCOL_ID,
        run_id=campaign.RUN_ID,
        manifest_sha256=_MANIFEST_SHA256,
        approval_sha256="3" * 64,
        review_sha256="4" * 64,
        physical_root=str(root),
        receipt_root_relative=receipt_relative,
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        receipt_device=receipt_stat.st_dev,
        receipt_inode=receipt_stat.st_ino,
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
        source_hashes=tuple(sorted(source_hashes.items())),
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
    grant_bytes = authority._json(
        {
            "schema_version": 1,
            "grant_sha256": _GRANT_SHA256,
            "grant_fd": grant_fd,
            "bindings_sha256": _sha(authority._json(bindings.payload())),
            "bindings": bindings.payload(),
        }
    )
    grant_path = receipt_path / authority.GRANT_RECORD
    fd = os.open(grant_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, grant_bytes)
        os.fsync(fd)
        grant_stat = os.fstat(fd)
    finally:
        os.close(fd)
    grant_record = authority._ReceivedGrantRecord(
        record_sha256=_sha(grant_bytes),
        file_identity=authority._FileId.of(grant_stat),
        file_mode=stat.S_IMODE(grant_stat.st_mode),
        regular_file=stat.S_ISREG(grant_stat.st_mode),
    )
    return (
        authority.ABC6LaunchAuthority(
            authority._AUTHORITY_SEAL,
            bindings,
            _GRANT_SHA256,
            grant_fd,
            grant_record,
            root_fd,
            receipt_fd,
            identity,
        ),
        source_hashes,
    )


def _pin_role_frames(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        campaign.run_abc6_training_campaign_with_evidence,
        "__code__",
        campaign.run_abc6_training_campaign_with_evidence.__code__.replace(
            co_filename=str(root / authority.ABC6_ROLE_CONTRACT[1][1])
        ),
    )
    monkeypatch.setattr(
        cases.get_training_case_data,
        "__code__",
        cases.get_training_case_data.__code__.replace(
            co_filename=str(root / authority.ABC6_ROLE_CONTRACT[2][1])
        ),
    )


def _runner_start(authority_value: authority.ABC6LaunchAuthority) -> None:
    authority_value.consume_runner_startup()


def _start_runner(monkeypatch, root: Path, authority_value) -> None:
    monkeypatch.setattr(
        _runner_start,
        "__code__",
        _runner_start.__code__.replace(
            co_name=authority.ABC6_ROLE_CONTRACT[0][2],
            co_filename=str(root / authority.ABC6_ROLE_CONTRACT[0][1]),
        ),
    )
    _runner_start(authority_value)


def _configure_preflight(
    monkeypatch,
    root: Path,
    source_hashes: dict[str, str],
    *,
    manifest_sha256: str = _MANIFEST_SHA256,
) -> Path:
    root = root.resolve()
    monkeypatch.setattr(campaign, "_REPO_ROOT", root)
    claim_path = (
        root
        / campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{campaign.RUN_ID}.claim"
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(campaign, "_PROJECT_RUN_CLAIM_PATH", claim_path)

    def fake_preflight(*_args, **_kwargs):
        return (
            manifest_sha256,
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
    return claim_path


def _fake_bundle() -> cases.ABC6TrainingBundle:
    rows = []
    for case in cases.CASE_ROSTER:
        inputs = cases.TRAINING_INPUT_S if case.input_window == "S" else cases.TRAINING_INPUT_L
        rows.append(
            cases.ABC6TrainingCaseData(
                case=case,
                inputs=tuple(inputs),
                observed_outputs=(0.0,) * case.input_length,
            )
        )
    states = tuple(
        (truth, TankState(0.5, 0.5)) for truth in ("A", "B", "N")
    )
    return cases.ABC6TrainingBundle(tuple(rows), states)


@dataclass(frozen=True)
class _FakeFitResult:
    case_index: int
    case_id: str
    model: str
    abc_status: str = "complete"
    baseline_status: str = "complete"


def _install_fake_numerics(monkeypatch) -> list[int]:
    fit_indices: list[int] = []

    def fake_one_case(data, *, controls):
        fit_indices.append(data.case.case_index)
        return _FakeFitResult(
            case_index=data.case.case_index,
            case_id=data.case.case_id,
            model=data.case.fit_model.value,
        )

    monkeypatch.setattr(campaign, "build_synthetic_training_bundle", _fake_bundle)
    monkeypatch.setattr(training, "_run_one_case", fake_one_case)
    return fit_indices


def _configure_fake_campaign(monkeypatch, root: Path, *, preflight_sha=_MANIFEST_SHA256):
    launch_authority, source_hashes = _fake_authority(root)
    _pin_role_frames(monkeypatch, root)
    _start_runner(monkeypatch, root, launch_authority)
    claim_path = _configure_preflight(
        monkeypatch, root, source_hashes, manifest_sha256=preflight_sha
    )
    return launch_authority, source_hashes, claim_path


def test_missing_wrong_stale_and_premature_authority_stop_before_claim_or_simulation(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "checkout"
    launch_authority, source_hashes = _fake_authority(root)
    _pin_role_frames(monkeypatch, root)
    _configure_preflight(monkeypatch, root, source_hashes)
    fit_indices = _install_fake_numerics(monkeypatch)
    monkeypatch.setattr(
        campaign,
        "_preflight_manifest_and_open_checkout_root",
        lambda *_args, **_kwargs: pytest.fail("rejected authority reached preflight"),
    )

    with pytest.raises(TypeError):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused", _MANIFEST_SHA256, root / campaign.RECEIPT_ROOT_RELATIVE
        )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="launch authority"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused", _MANIFEST_SHA256, root / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=None,
        )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="launch authority"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused", _MANIFEST_SHA256, root / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=object(),
        )

    # A correctly typed but not-yet-started grant cannot begin training.
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="runner startup"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused", _MANIFEST_SHA256, root / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=launch_authority,
        )
    assert not (root / campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE / f"{campaign.RUN_ID}.claim").exists()
    assert fit_indices == []
    assert launch_authority._state.training_started is False
    launch_authority.close()

    stale, _stale_sources = _fake_authority(tmp_path / "stale")
    _pin_role_frames(monkeypatch, tmp_path / "stale")
    _start_runner(monkeypatch, tmp_path / "stale", stale)
    stale.close()
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="authority"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused", _MANIFEST_SHA256,
            tmp_path / "stale" / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=stale,
        )
    assert not (tmp_path / "stale" / campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE / f"{campaign.RUN_ID}.claim").exists()
    assert fit_indices == []


@pytest.mark.parametrize("mismatch", ("root", "manifest"))
def test_wrong_root_or_manifest_is_rejected_before_claim_or_simulation(
    tmp_path, monkeypatch, mismatch
) -> None:
    grant_root = tmp_path / "grant-root"
    launch_authority, source_hashes = _fake_authority(grant_root)
    _pin_role_frames(monkeypatch, grant_root)
    _start_runner(monkeypatch, grant_root, launch_authority)
    fit_indices = _install_fake_numerics(monkeypatch)
    if mismatch == "root":
        preflight_root = tmp_path / "different-root"
        preflight_sources = _copy_manifest_sources(preflight_root)
        (preflight_root / campaign.RECEIPT_ROOT_RELATIVE).mkdir(parents=True)
        _configure_preflight(monkeypatch, preflight_root, preflight_sources)
    else:
        preflight_root = grant_root
        _configure_preflight(
            monkeypatch, grant_root, source_hashes, manifest_sha256="6" * 64
        )
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="preclaim"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused",
            _MANIFEST_SHA256,
            preflight_root / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=launch_authority,
        )
    assert not (
        preflight_root
        / campaign.CAMPAIGN_CLAIM_PARENT_RELATIVE
        / f"{campaign.RUN_ID}.claim"
    ).exists()
    assert fit_indices == []
    launch_authority.close()


def test_case_permits_bind_order_bundle_and_single_fit_before_numerics(
    tmp_path, monkeypatch
) -> None:
    root_a = tmp_path / "authority-a"
    root_b = tmp_path / "authority-b"
    authority_a, _ = _fake_authority(root_a)
    authority_b, _ = _fake_authority(root_b)
    _pin_role_frames(monkeypatch, root_a)
    _start_runner(monkeypatch, root_a, authority_a)
    _pin_role_frames(monkeypatch, root_b)
    _start_runner(monkeypatch, root_b, authority_b)

    # These small fake role frames preserve the frozen immediate caller checks.
    def campaign_issue(launch, count):
        session = launch.begin_training()
        issued = []
        for index in range(count):
            issued.append(session.issue_case_permit(index))
        return tuple(issued)

    original_code = campaign_issue.__code__

    def permits_for(root, launch, count):
        campaign_issue.__code__ = original_code.replace(
            co_name=authority.ABC6_ROLE_CONTRACT[1][2],
            co_filename=str(root / authority.ABC6_ROLE_CONTRACT[1][1]),
        )
        return campaign_issue(launch, count)

    permits_a = permits_for(root_a, authority_a, cases.CASE_COUNT)
    permits_for(root_b, authority_b, 0)
    bundle_a, bundle_b = _fake_bundle(), _fake_bundle()
    cases._bind_training_bundle_to_authority(bundle_a, authority_a)
    cases._bind_training_bundle_to_authority(bundle_b, authority_b)
    fit_indices: list[int] = []
    monkeypatch.setattr(
        training,
        "_run_one_case",
        lambda data, *, controls: (fit_indices.append(data.case.case_index), "ok")[1],
    )

    _pin_role_frames(monkeypatch, root_a)
    with pytest.raises(authority.ABC6AuthorityError, match="wrong case index"):
        cases.get_training_case_data(bundle_a, 0, permits_a[1])
    with pytest.raises(authority.ABC6AuthorityError, match="exact bundle"):
        cases.get_training_case_data(bundle_b, 0, permits_a[0])

    authorized = cases.get_training_case_data(bundle_a, 0, permits_a[0])
    with pytest.raises(authority.ABC6AuthorityError, match="already issued|one-use"):
        cases.get_training_case_data(bundle_a, 0, permits_a[0])
    with pytest.raises(cases.ABC6StatusReceiptError, match="copied|unregistered"):
        training.run_abc6_training_case(copy.copy(authorized))
    assert training.run_abc6_training_case(authorized) == "ok"
    with pytest.raises(cases.ABC6StatusReceiptError, match="already consumed"):
        training.run_abc6_training_case(authorized)
    assert fit_indices == [0]

    # A reconstructed or cross-authority value cannot use the real identity registry.
    reconstructed = cases.ABC6AuthorizedCaseData(
        cases._AUTHORIZED_CASE_DATA_SEAL,
        authority_a,
        1,
        bundle_a,
        bundle_a.case_data[1],
    )
    with pytest.raises(cases.ABC6StatusReceiptError, match="copied|unregistered"):
        training.run_abc6_training_case(reconstructed)
    cross_authority = cases.get_training_case_data(bundle_a, 2, permits_a[2])
    object.__setattr__(cross_authority, "_authority", authority_b)
    with pytest.raises(cases.ABC6StatusReceiptError, match="grant's training bundle"):
        training.run_abc6_training_case(cross_authority)
    swapped_bundle = cases.get_training_case_data(bundle_a, 3, permits_a[3])
    object.__setattr__(swapped_bundle, "_bundle", bundle_b)
    with pytest.raises(cases.ABC6StatusReceiptError, match="grant's training bundle"):
        training.run_abc6_training_case(swapped_bundle)
    assert fit_indices == [0]
    authority_a.close()
    authority_b.close()


def test_full_fake_authorized_campaign_registers_exact_execution_object(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "authorized"
    launch_authority, source_hashes = _fake_authority(root)
    _pin_role_frames(monkeypatch, root)
    _start_runner(monkeypatch, root, launch_authority)
    claim_path = _configure_preflight(monkeypatch, root, source_hashes)
    fit_indices = _install_fake_numerics(monkeypatch)

    execution = campaign.run_abc6_training_campaign_with_evidence(
        "unused",
        _MANIFEST_SHA256,
        root / campaign.RECEIPT_ROOT_RELATIVE,
        launch_authority=launch_authority,
    )
    assert fit_indices == list(range(cases.CASE_COUNT))
    assert claim_path.is_file()
    assert launch_authority._state.issued == set(range(cases.CASE_COUNT))
    assert launch_authority._state.consumed == set(range(cases.CASE_COUNT))
    assert launch_authority._state.training_execution is execution
    assert execution._grant_sha256 == launch_authority.grant_digest
    assert execution._run_id == launch_authority._bindings.run_id
    assert execution._manifest_sha256 == launch_authority._bindings.manifest_sha256
    assert execution.receipt_root_identity.repository_root_realpath == str(root.resolve())

    for candidate in (copy.copy(execution), replace(execution)):
        with pytest.raises(authority.ABC6AuthorityError, match="exact registered object"):
            launch_authority._assert_registered_training_execution(candidate)
    assert execution.load_verified_training_evidence().training_bundle is not execution._training_bundle
    assert all(value is None for value in campaign._REVIEWED_SOURCE_SHA256.values())
    execution.close()
    launch_authority.close()


def test_failed_fit_consumes_case_and_makes_same_authority_terminal(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "failed-fit"
    launch_authority, source_hashes = _fake_authority(root)
    _pin_role_frames(monkeypatch, root)
    _start_runner(monkeypatch, root, launch_authority)
    claim_path = _configure_preflight(monkeypatch, root, source_hashes)
    _install_fake_numerics(monkeypatch)

    def fail_fit(data, *, controls):
        raise RuntimeError(f"fake fit failure for case {data.case.case_index}")

    monkeypatch.setattr(training, "_run_one_case", fail_fit)
    with pytest.raises(campaign.ABC6CampaignExecutionError, match="case 0 training failed"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused",
            _MANIFEST_SHA256,
            root / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=launch_authority,
    )
    assert claim_path.is_file()
    assert launch_authority._state.consumed == {0}
    failure = root / campaign.RECEIPT_ROOT_RELATIVE / campaign.FAILURE_FILENAME
    assert failure.is_file()
    with pytest.raises(campaign.ABC6CampaignPreflightError, match="one-use"):
        campaign.run_abc6_training_campaign_with_evidence(
            "unused",
            _MANIFEST_SHA256,
            root / campaign.RECEIPT_ROOT_RELATIVE,
            launch_authority=launch_authority,
        )
    assert launch_authority._state.consumed == {0}
    launch_authority.close()

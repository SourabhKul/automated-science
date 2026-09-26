from __future__ import annotations

import copy

import pytest

from core.real_data import silverbox_v2_manifest as protocol


def _fixture():
    code_hashes = {
        name: format(index + 1, "064x")
        for index, name in enumerate(protocol.CODE_FILES)
    }
    actual = {
        "reviewed_git_commit": "a" * 40,
        "reviewed_tree_clean": True,
        "runtime": dict(protocol.PINNED_RUNTIME),
        "code_sha256": code_hashes,
    }
    manifest = {
        "protocol_id": protocol.PROTOCOL_ID,
        "abc_attempts_per_population": protocol.ABC_ATTEMPTS_PER_POPULATION,
        "reviewed_git_commit": actual["reviewed_git_commit"],
        "reviewed_tree_clean": True,
        "source": {"archive_sha256": "b" * 64, "archive_bytes": 987654},
        "runtime": dict(protocol.PINNED_RUNTIME),
        "code_sha256": code_hashes,
    }
    return manifest, actual


def test_manifest_accepts_exact_caller_frozen_provenance():
    manifest, actual = _fixture()
    protocol.verify_manifest(
        manifest,
        actual=actual,
        archive_sha256="b" * 64,
        archive_bytes=987654,
    )


@pytest.mark.parametrize(
    ("mutate_manifest", "mutate_actual", "archive_sha256", "archive_bytes"),
    [
        (lambda body: body.update({"abc_attempts_per_population": 512}), lambda state: None, "b" * 64, 987654),
        (lambda body: body["source"].update({"archive_sha256": "c" * 64}), lambda state: None, "b" * 64, 987654),
        (lambda body: body["source"].update({"archive_bytes": 1}), lambda state: None, "b" * 64, 987654),
        (lambda body: body["runtime"].update({"numpy": "0.0.0"}), lambda state: None, "b" * 64, 987654),
        (lambda body: body.update({"reviewed_tree_clean": False}), lambda state: None, "b" * 64, 987654),
        (lambda body: None, lambda state: state.update({"reviewed_git_commit": "c" * 40}), "b" * 64, 987654),
        (lambda body: None, lambda state: state.update({"reviewed_tree_clean": False}), "b" * 64, 987654),
        (lambda body: None, lambda state: state["runtime"].update({"psutil": "0.0.0"}), "b" * 64, 987654),
        (lambda body: None, lambda state: state["code_sha256"].update({"fit": "0" * 64}), "b" * 64, 987654),
        (lambda body: None, lambda state: None, "c" * 64, 987654),
        (lambda body: None, lambda state: None, "b" * 64, 987653),
    ],
)
def test_manifest_rejects_any_protocol_or_provenance_mismatch(
    mutate_manifest, mutate_actual, archive_sha256, archive_bytes
):
    manifest, actual = _fixture()
    manifest = copy.deepcopy(manifest)
    actual = copy.deepcopy(actual)
    mutate_manifest(manifest)
    mutate_actual(actual)
    with pytest.raises(ValueError):
        protocol.verify_manifest(
            manifest,
            actual=actual,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
        )


def test_manifest_requires_exact_code_hash_inventory():
    manifest, actual = _fixture()
    del manifest["code_sha256"]["final_scorer"]
    with pytest.raises(ValueError, match="code hash set"):
        protocol.verify_manifest(
            manifest,
            actual=actual,
            archive_sha256="b" * 64,
            archive_bytes=987654,
        )

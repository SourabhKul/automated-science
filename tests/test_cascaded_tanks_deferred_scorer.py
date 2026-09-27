from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.real_data import cascaded_tanks_deferred_scorer as scorer
from core.real_data import cascaded_tanks_deferred_targets as target_parser
from core.real_data import cascaded_tanks_development_forecast as forecast
from core.real_data import cascaded_tanks_forecast_gate as gate
from core.real_data import cascaded_tanks_training_fit as fit
from core.real_data.cascaded_tanks_controlled import ArchiveExpectation
from tests.test_cascaded_tanks_controlled import (
    _archive_bytes,
    _csv_bytes,
    _numeric_field_width,
)
from tests.test_cascaded_tanks_forecast_gate import _make_case, _run_case, _write_json


@dataclass(frozen=True)
class _Case:
    root: Path
    archive: Path
    expectation: ArchiveExpectation
    manifest: Path
    fit_receipt: Path
    comparison: Path
    gate_receipt: Path
    candidate_bundle: Path
    baseline_bundle: Path
    baseline_provenance: Path
    scoring_declaration: Path
    artifact_dir: Path

    @property
    def json_paths(self) -> dict[str, Path]:
        return {
            "fit_manifest": self.manifest,
            "fit_receipt": self.fit_receipt,
            "comparison_declaration": self.comparison,
            "gate_receipt": self.gate_receipt,
            "candidate_bundle": self.candidate_bundle,
            "baseline_bundle": self.baseline_bundle,
            "baseline_provenance": self.baseline_provenance,
            "scoring_declaration": self.scoring_declaration,
        }


def _csv_with_excluded_sentinels() -> bytes:
    sentinels = {
        (row, name): token * _numeric_field_width(row, slot)
        for row in range(1_024)
        for slot, name, token in ((1, "uVal", b"u"), (3, "yVal", b"v"))
    }
    return _csv_bytes(mutate=sentinels)


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = _canonical_sha256(value)
    return value


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def base_case(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    root = tmp_path_factory.mktemp("deferred-score-base")
    source_case = _make_case(
        root,
        archive_data=_archive_bytes(_csv_with_excluded_sentinels()),
    )
    outcome = _run_case(source_case)
    assert outcome.status == "complete"
    assert outcome.forecasts is not None
    archive, expectation, manifest, fit_receipt, comparison, *_ = source_case
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    fit_receipt_raw = fit_receipt.read_bytes()
    manifest_raw = manifest.read_bytes()
    comparison_raw = comparison.read_bytes()

    gate_receipt = root / "gate-receipt.json"
    gate_raw = _write_json(gate_receipt, outcome.receipt)
    candidate_bundle = {
        "schema": scorer.FORECAST_BUNDLE_SCHEMA,
        "scope": "synthetic-fixture-only",
        "run_id": manifest_doc["run_id"],
        "protocol_id": manifest_doc["protocol_id"],
        "archive_sha256": expectation.archive_sha256,
        "fit_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "fit_receipt_sha256": hashlib.sha256(fit_receipt_raw).hexdigest(),
        "comparison_declaration_sha256": hashlib.sha256(comparison_raw).hexdigest(),
        "gate_receipt_sha256": hashlib.sha256(gate_raw).hexdigest(),
        "training_indices": [0, 768],
        "forecast_indices": [768, 1024],
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "model": manifest_doc["candidates"][index]["model"],
                "values": list(candidate.values),
                "forecast_sha256": forecast._sequence_sha256(
                    candidate.values, start_index=768
                ),
            }
            for index, candidate in enumerate(outcome.forecasts)
        ],
    }
    _sealed(candidate_bundle, "bundle_sha256")
    candidate_path = root / "candidate-bundle.json"
    candidate_raw = _write_json(candidate_path, candidate_bundle)

    baseline_values = [0.0] * 256
    baseline_bundle = {
        "schema": scorer.BASELINE_BUNDLE_SCHEMA,
        "scope": "synthetic-fixture-only",
        "run_id": manifest_doc["run_id"],
        "protocol_id": manifest_doc["protocol_id"],
        "archive_sha256": expectation.archive_sha256,
        "gate_receipt_sha256": hashlib.sha256(gate_raw).hexdigest(),
        "forecast_indices": [768, 1024],
        "baseline_id": "synthetic-zero-baseline",
        "method": "frozen-zero-valued-synthetic-forecast",
        "forecast_sha256": forecast._sequence_sha256(
            baseline_values, start_index=768
        ),
        "values": baseline_values,
    }
    _sealed(baseline_bundle, "bundle_sha256")
    baseline_path = root / "baseline-bundle.json"
    baseline_raw = _write_json(baseline_path, baseline_bundle)

    source_stages = manifest_doc["source"]["stage_sha256"]
    baseline_provenance = {
        "schema": scorer.BASELINE_PROVENANCE_SCHEMA,
        "scope": "synthetic-fixture-only",
        "run_id": manifest_doc["run_id"],
        "protocol_id": manifest_doc["protocol_id"],
        "archive_sha256": expectation.archive_sha256,
        "fit_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "fit_receipt_sha256": hashlib.sha256(fit_receipt_raw).hexdigest(),
        "gate_receipt_sha256": hashlib.sha256(gate_raw).hexdigest(),
        "baseline_bundle_sha256": hashlib.sha256(baseline_raw).hexdigest(),
        "baseline_id": baseline_bundle["baseline_id"],
        "training_indices": [0, 768],
        "forecast_indices": [768, 1024],
        "training_stage_sha256": source_stages["training_sha256"],
        "forecast_inputs_stage_sha256": source_stages["forecast_inputs_sha256"],
        "development_y_est_materialized": 0,
        "u_val_materialized": 0,
        "y_val_materialized": 0,
    }
    _sealed(baseline_provenance, "provenance_sha256")
    provenance_path = root / "baseline-provenance.json"
    provenance_raw = _write_json(provenance_path, baseline_provenance)

    scoring_declaration = {
        "schema": scorer.SCORING_DECLARATION_SCHEMA,
        "scope": "synthetic-fixture-only",
        "run_id": manifest_doc["run_id"],
        "protocol_id": manifest_doc["protocol_id"],
        "archive_sha256": expectation.archive_sha256,
        "fit_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "fit_receipt_sha256": hashlib.sha256(fit_receipt_raw).hexdigest(),
        "comparison_declaration_sha256": hashlib.sha256(comparison_raw).hexdigest(),
        "gate_receipt_sha256": hashlib.sha256(gate_raw).hexdigest(),
        "candidate_bundle_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "baseline_bundle_sha256": hashlib.sha256(baseline_raw).hexdigest(),
        "baseline_provenance_sha256": hashlib.sha256(provenance_raw).hexdigest(),
        "attempt_id": "synthetic-attempt-1",
        "metric": "root-mean-square-error",
        "decision_rule": {
            "rule": "select-lowest-candidate-if-rmse-improves-baseline-by-more-than-minimum",
            "minimum_absolute_improvement": 0.0,
            "tie_policy": "retain-baseline",
        },
        "structural_separation": {
            "decision": "eligible",
            "reason": "synthetic structural check passed before scoring",
        },
        "qwen_disposition": "synthetic-abstain-no-proposal",
    }
    _sealed(scoring_declaration, "declaration_sha256")
    scoring_path = root / "scoring-declaration.json"
    _write_json(scoring_path, scoring_declaration)
    artifact_dir = root / "run-artifacts"
    artifact_dir.mkdir()
    return _Case(
        root=root,
        archive=archive,
        expectation=expectation,
        manifest=manifest,
        fit_receipt=fit_receipt,
        comparison=comparison,
        gate_receipt=gate_receipt,
        candidate_bundle=candidate_path,
        baseline_bundle=baseline_path,
        baseline_provenance=provenance_path,
        scoring_declaration=scoring_path,
        artifact_dir=artifact_dir,
    )


def _copy_case(case: _Case, destination: Path) -> _Case:
    destination.mkdir(parents=True)
    copies: dict[str, Path] = {}
    originals = {
        "archive": case.archive,
        "manifest": case.manifest,
        "fit_receipt": case.fit_receipt,
        "comparison": case.comparison,
        "gate_receipt": case.gate_receipt,
        "candidate_bundle": case.candidate_bundle,
        "baseline_bundle": case.baseline_bundle,
        "baseline_provenance": case.baseline_provenance,
        "scoring_declaration": case.scoring_declaration,
    }
    for name, original in originals.items():
        target = destination / original.name
        shutil.copyfile(original, target)
        copies[name] = target
    artifact_dir = destination / "run-artifacts"
    artifact_dir.mkdir()
    return _Case(
        root=destination,
        archive=copies["archive"],
        expectation=case.expectation,
        manifest=copies["manifest"],
        fit_receipt=copies["fit_receipt"],
        comparison=copies["comparison"],
        gate_receipt=copies["gate_receipt"],
        candidate_bundle=copies["candidate_bundle"],
        baseline_bundle=copies["baseline_bundle"],
        baseline_provenance=copies["baseline_provenance"],
        scoring_declaration=copies["scoring_declaration"],
        artifact_dir=artifact_dir,
    )


def _score(case: _Case, monkeypatch: pytest.MonkeyPatch):
    # Every fixture copy in one test shares this trusted marker root. The score
    # API receives no root argument; only tests can replace the module constant.
    marker_root = case.root.parent / "trusted-marker-root"
    marker_root.mkdir(exist_ok=True)
    monkeypatch.setattr(scorer, "_MARKER_ROOT", marker_root)
    return _call_score(case)


def _call_score(case: _Case):
    return scorer._score_synthetic_fixture_development(
        case.archive,
        case.manifest,
        case.fit_receipt,
        case.comparison,
        case.gate_receipt,
        case.candidate_bundle,
        case.baseline_bundle,
        case.baseline_provenance,
        case.scoring_declaration,
        expected_archive=case.expectation,
    )


def _rewrite_json(path: Path, payload: dict[str, Any]) -> bytes:
    return _write_json(path, payload)


def test_success_scores_exact_suffix_and_never_converts_excluded_sentinels(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "success")
    calls: list[bytes] = []
    parse = target_parser._parse_synthetic_development_targets

    def instrumented_parse(wire: bytes):
        calls.append(wire)
        return parse(wire)

    monkeypatch.setattr(
        target_parser, "_parse_synthetic_development_targets", instrumented_parse
    )
    result = _score(case, monkeypatch)

    assert result.status == "complete", result.receipt.get("failure")
    assert result.receipt["consumed"] is True
    assert result.receipt["target_sha256"]
    assert len(calls) == 1
    assert result.receipt["score"]["forecast_indices"] == [768, 1024]
    expected_targets = np.asarray(
        [
            (row + 2) % 10 + (0.1234 if row * 4 + 2 < 3_352 else 0.123)
            for row in range(768, 1_024)
        ],
        dtype=np.float64,
    )
    expected_target_hash = fit._float_array_sha256(expected_targets)
    assert result.receipt["target_sha256"] == expected_target_hash
    expected_baseline_rmse = math.sqrt(float(np.mean(expected_targets**2)))
    assert result.receipt["score"]["baseline_rmse"] == pytest.approx(
        expected_baseline_rmse
    )
    candidate_values = json.loads(case.candidate_bundle.read_text(encoding="utf-8"))[
        "candidates"
    ][0]["values"]
    expected_candidate_rmse = math.sqrt(
        float(np.mean((np.asarray(candidate_values) - expected_targets) ** 2))
    )
    assert result.receipt["score"]["candidate_rmse"][0]["rmse"] == pytest.approx(
        expected_candidate_rmse
    )
    marker_root = case.root.parent / "trusted-marker-root"
    assert list(marker_root.iterdir()) == [
        marker_root
        / f"{result.receipt['run_id']}.{scorer.SCORE_CONDITION}.started.json"
    ]


def test_bad_preflight_calls_no_target_parser_and_creates_no_marker(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "bad-preflight")
    receipt = json.loads(case.gate_receipt.read_text(encoding="utf-8"))
    receipt["status"] = "terminal_failure"
    _write_json(case.gate_receipt, receipt)
    calls: list[str] = []
    monkeypatch.setattr(
        target_parser,
        "_parse_synthetic_development_targets",
        lambda wire: calls.append("called"),
    )

    result = _score(case, monkeypatch)

    assert result.status == "terminal_failure"
    assert result.receipt["consumed"] is False
    assert result.receipt["failure"]["code"] == "preflight_failure"
    assert calls == []
    assert list((case.root.parent / "trusted-marker-root").iterdir()) == []


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants_before_marker(
    raw: bytes,
    base_case: _Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _copy_case(base_case, tmp_path / hashlib.sha256(raw).hexdigest()[:8])
    case.scoring_declaration.write_bytes(raw)
    calls: list[str] = []
    monkeypatch.setattr(
        target_parser,
        "_parse_synthetic_development_targets",
        lambda wire: calls.append("called"),
    )

    result = _score(case, monkeypatch)

    assert result.status == "terminal_failure"
    assert result.receipt["consumed"] is False
    assert calls == []
    assert list((case.root.parent / "trusted-marker-root").iterdir()) == []


def test_changed_candidate_forecast_hash_fails_before_marker(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "forecast-hash")
    bundle = json.loads(case.candidate_bundle.read_text(encoding="utf-8"))
    bundle.pop("bundle_sha256")
    bundle["candidates"][0]["values"][0] += 1.0
    _sealed(bundle, "bundle_sha256")
    _write_json(case.candidate_bundle, bundle)
    calls: list[str] = []
    monkeypatch.setattr(
        target_parser,
        "_parse_synthetic_development_targets",
        lambda wire: calls.append("called"),
    )

    result = _score(case, monkeypatch)

    assert result.status == "terminal_failure"
    assert result.receipt["consumed"] is False
    assert calls == []
    assert list((case.root.parent / "trusted-marker-root").iterdir()) == []


def test_marker_is_consumed_after_post_lock_failure_and_new_attempt_is_blocked(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "retry")
    calls: list[str] = []

    def fail_parser(wire: bytes):
        calls.append("called")
        raise RuntimeError("synthetic forced parser interruption")

    monkeypatch.setattr(target_parser, "_parse_synthetic_development_targets", fail_parser)
    first = _score(case, monkeypatch)
    assert first.status == "terminal_failure"
    assert first.receipt["consumed"] is True
    assert first.receipt["failure"]["code"] == "post_marker_failure"
    marker_root = case.root.parent / "trusted-marker-root"
    marker_path = next(marker_root.iterdir())
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["attempt_id"] == "synthetic-attempt-1"
    assert marker["archive_sha256"] == case.expectation.archive_sha256
    assert marker["scoring_declaration_sha256"] == first.receipt[
        "scoring_declaration_sha256"
    ]

    retry_case = _copy_case(base_case, tmp_path / "retry-other-dir")
    declaration = json.loads(retry_case.scoring_declaration.read_text(encoding="utf-8"))
    declaration.pop("declaration_sha256")
    declaration["attempt_id"] = "synthetic-attempt-2"
    _sealed(declaration, "declaration_sha256")
    _rewrite_json(retry_case.scoring_declaration, declaration)
    second = _score(retry_case, monkeypatch)

    assert second.status == "terminal_failure"
    assert second.receipt["consumed"] is True
    assert second.receipt["failure"]["code"] == "run_condition_already_consumed"
    assert calls == ["called"]
    assert len(list(marker_root.iterdir())) == 1


def test_fixed_marker_root_blocks_same_run_copied_to_another_fixture_directory(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_case = _copy_case(base_case, tmp_path / "copy-one")
    second_case = _copy_case(base_case, tmp_path / "copy-two")
    calls: list[str] = []
    parse = target_parser._parse_synthetic_development_targets

    def instrumented_parse(wire: bytes):
        calls.append("called")
        return parse(wire)

    monkeypatch.setattr(
        target_parser, "_parse_synthetic_development_targets", instrumented_parse
    )
    first = _score(first_case, monkeypatch)
    second = _score(second_case, monkeypatch)

    assert first_case.root != second_case.root
    assert first_case.artifact_dir != second_case.artifact_dir
    assert (
        first_case.scoring_declaration.read_bytes()
        == second_case.scoring_declaration.read_bytes()
    )
    assert first.status == "complete"
    assert second.status == "terminal_failure"
    assert second.receipt["consumed"] is True
    assert second.receipt["failure"]["code"] == "run_condition_already_consumed"
    assert calls == ["called"]
    marker_root = tmp_path / "trusted-marker-root"
    assert len(list(marker_root.iterdir())) == 1
    assert list(first_case.artifact_dir.iterdir()) == []
    assert list(second_case.artifact_dir.iterdir()) == []


def test_marker_root_swap_cannot_redirect_claim_away_from_pinned_directory(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_case = _copy_case(base_case, tmp_path / "swap-one")
    second_case = _copy_case(base_case, tmp_path / "swap-two")
    marker_root_path = tmp_path / "trusted-marker-root"
    moved_root_path = tmp_path / "pinned-marker-root"
    replacement_root = tmp_path / "replacement-marker-root"
    original_claim = scorer._claim_one_use_marker
    parse = target_parser._parse_synthetic_development_targets
    target_calls: list[str] = []
    swapped = False

    def swap_then_claim(marker_root, **kwargs):
        nonlocal swapped
        if swapped:
            return original_claim(marker_root, **kwargs)
        assert marker_root_path.exists()
        pinned_identity = os.fstat(marker_root.descriptor)
        marker_root_path.rename(moved_root_path)
        replacement_root.mkdir()
        marker_root_path.symlink_to(replacement_root, target_is_directory=True)
        try:
            assert marker_root_path.is_symlink()
            assert (os.fstat(marker_root.descriptor).st_dev, os.fstat(
                marker_root.descriptor
            ).st_ino) == (pinned_identity.st_dev, pinned_identity.st_ino)
            result = original_claim(marker_root, **kwargs)
            swapped = True
            return result
        finally:
            if marker_root_path.is_symlink():
                marker_root_path.unlink()
            moved_root_path.rename(marker_root_path)

    def count_parse(wire: bytes):
        target_calls.append("called")
        return parse(wire)

    monkeypatch.setattr(scorer, "_claim_one_use_marker", swap_then_claim)
    monkeypatch.setattr(
        target_parser, "_parse_synthetic_development_targets", count_parse
    )
    first = _score(first_case, monkeypatch)
    second = _score(second_case, monkeypatch)

    assert swapped
    assert first.status == "complete"
    assert second.status == "terminal_failure"
    assert second.receipt["failure"]["code"] == "run_condition_already_consumed"
    assert target_calls == ["called"]
    assert len(list(marker_root_path.iterdir())) == 1
    assert list(replacement_root.iterdir()) == []


def test_marker_file_and_directory_are_fsynced_before_target_parser(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "fsync-order")
    fsync = os.fsync
    events: list[str] = []

    def record_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        events.append("directory" if stat.S_ISDIR(mode) else "file")
        fsync(descriptor)

    parse = target_parser._parse_synthetic_development_targets

    def assert_synced_then_parse(wire: bytes):
        assert events == ["file", "directory"]
        return parse(wire)

    monkeypatch.setattr(scorer.os, "fsync", record_fsync)
    monkeypatch.setattr(
        target_parser, "_parse_synthetic_development_targets", assert_synced_then_parse
    )

    result = _score(case, monkeypatch)

    assert result.status == "complete"
    assert events == ["file", "directory"]


@pytest.mark.parametrize(
    ("artifact", "label"),
    [
        ("fit_manifest", "fit manifest"),
        ("fit_receipt", "fit receipt"),
        ("comparison_declaration", "comparison declaration"),
        ("gate_receipt", "gate receipt"),
        ("candidate_bundle", "candidate forecast bundle"),
        ("baseline_bundle", "baseline forecast bundle"),
        ("baseline_provenance", "baseline provenance"),
        ("scoring_declaration", "scoring declaration"),
    ],
)
def test_json_inputs_parse_the_same_inode_snapshots(
    artifact: str,
    label: str,
    base_case: _Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _copy_case(base_case, tmp_path / artifact)
    target_path = case.json_paths[artifact]
    original = target_path.read_bytes()
    original_inode = target_path.stat().st_ino
    parse = gate._parse_json_document
    mutated = False

    def mutate_then_parse(raw: bytes, document_label: str):
        nonlocal mutated
        if document_label == label and not mutated:
            with target_path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"!" * len(original))
                stream.flush()
            mutated = True
        return parse(raw, document_label)

    monkeypatch.setattr(gate, "_parse_json_document", mutate_then_parse)
    result = _score(case, monkeypatch)

    assert mutated
    assert target_path.stat().st_ino == original_inode
    assert target_path.read_bytes() != original
    assert result.status == "complete"


def test_archive_hash_and_target_parse_use_the_same_post_lock_snapshot(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "archive-snapshot")
    inode = case.archive.stat().st_ino
    other_csv = _csv_bytes(
        mutate={
            (row, "yEst"): (
                b"9.9999" if _numeric_field_width(row, 2) == 6 else b"9.999"
            )
            for row in range(768, 1_024)
        }
    )
    changed_archive = _archive_bytes(other_csv)
    parse = target_parser._parse_synthetic_development_targets
    changed = False

    def mutate_after_snapshot(wire: bytes):
        nonlocal changed
        if not changed:
            with case.archive.open("r+b") as stream:
                stream.seek(0)
                stream.write(changed_archive)
                stream.truncate()
                stream.flush()
            changed = True
        return parse(wire)

    monkeypatch.setattr(
        target_parser, "_parse_synthetic_development_targets", mutate_after_snapshot
    )
    result = _score(case, monkeypatch)

    expected_targets = np.asarray(
        [
            (row + 2) % 10 + (0.1234 if row * 4 + 2 < 3_352 else 0.123)
            for row in range(768, 1_024)
        ],
        dtype=np.float64,
    )
    assert changed
    assert case.archive.stat().st_ino == inode
    assert result.status == "complete"
    assert result.receipt["archive_sha256"] == case.expectation.archive_sha256
    assert result.receipt["target_sha256"] == fit._float_array_sha256(expected_targets)


def test_structural_abstention_cannot_be_reversed_by_a_large_rmse_gain(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "structural-abstention")
    baseline = json.loads(case.baseline_bundle.read_text(encoding="utf-8"))
    baseline.pop("bundle_sha256")
    baseline["values"] = [1_000_000.0] * 256
    baseline["forecast_sha256"] = forecast._sequence_sha256(
        baseline["values"], start_index=768
    )
    _sealed(baseline, "bundle_sha256")
    baseline_raw = _write_json(case.baseline_bundle, baseline)

    provenance = json.loads(case.baseline_provenance.read_text(encoding="utf-8"))
    provenance.pop("provenance_sha256")
    provenance["baseline_bundle_sha256"] = hashlib.sha256(baseline_raw).hexdigest()
    _sealed(provenance, "provenance_sha256")
    provenance_raw = _write_json(case.baseline_provenance, provenance)

    declaration = json.loads(case.scoring_declaration.read_text(encoding="utf-8"))
    declaration.pop("declaration_sha256")
    declaration["baseline_bundle_sha256"] = hashlib.sha256(baseline_raw).hexdigest()
    declaration["baseline_provenance_sha256"] = hashlib.sha256(
        provenance_raw
    ).hexdigest()
    declaration["structural_separation"] = {
        "decision": "abstain",
        "reason": "synthetic structures remain separated before target access",
    }
    _sealed(declaration, "declaration_sha256")
    _write_json(case.scoring_declaration, declaration)

    result = _score(case, monkeypatch)

    assert result.status == "complete"
    score = result.receipt["score"]
    assert score["baseline_rmse"] > score["candidate_rmse"][0]["rmse"]
    assert score["selection_status"] == "structural_abstention"
    assert score["selected_candidate_id"] is None


def test_public_real_source_entrypoint_fails_closed_without_opening_paths(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must-not-be-opened"
    with pytest.raises(scorer.CascadedTanksDeferredScorerError, match="disabled"):
        scorer.run_cascaded_tanks_deferred_development_score(
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
        )
    assert not missing.exists()


def test_missing_or_symlink_marker_root_fails_closed(
    base_case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(base_case, tmp_path / "bad-marker-root")
    calls: list[str] = []
    monkeypatch.setattr(
        target_parser,
        "_parse_synthetic_development_targets",
        lambda wire: calls.append("called"),
    )
    missing_root = tmp_path / "missing-marker-root"
    monkeypatch.setattr(scorer, "_MARKER_ROOT", missing_root)
    missing_result = _call_score(case)
    real_root = tmp_path / "real-marker-root"
    real_root.mkdir()
    symlink_root = tmp_path / "marker-root-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(scorer, "_MARKER_ROOT", symlink_root)
    symlink_result = _call_score(case)

    assert missing_result.status == "terminal_failure"
    assert symlink_result.status == "terminal_failure"
    assert missing_result.receipt["consumed"] is False
    assert symlink_result.receipt["consumed"] is False
    assert calls == []
    assert list(real_root.iterdir()) == []

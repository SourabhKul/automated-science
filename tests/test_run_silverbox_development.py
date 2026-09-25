from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from core.real_data.silverbox_controlled import (
    SAMPLE_TIME_SECONDS,
    TRAIN_SOURCE_START,
    TRAIN_WINDOW_LENGTH,
    TRAIN_WINDOW_RELATIVE_STARTS,
    STATE_INITIALIZATION_LENGTH,
    VALIDATION_SOURCE_START,
    VALIDATION_SOURCE_STOP,
    SilverboxControlledDevelopment,
    SilverboxControlledSeries,
)
from core.real_data.silverbox import (
    ARROW_FULL_START,
    ARROW_FULL_STOP,
    MULTISINE_START,
    MULTISINE_STOP,
    MULTISINE_TRAIN_STOP,
    REQUIRED_ARCHIVE_MEMBERS,
    SAMPLE_COUNT,
    SNLS_CSV_MEMBER,
)
from core.real_data import silverbox
from core.real_data.silverbox_proposer import (
    ENDPOINT,
    FROZEN_REQUEST_PAYLOAD_SHA256,
    MODEL_ID,
    MODELS_ENDPOINT,
    SilverboxProposalResult,
    _build_request_payload,
)
from scripts import run_silverbox_development as runner


def _development() -> SilverboxControlledDevelopment:
    windows = []
    for index, relative_start in enumerate(TRAIN_WINDOW_RELATIVE_STARTS):
        start = TRAIN_SOURCE_START + relative_start
        input_u = np.full(TRAIN_WINDOW_LENGTH, 0.01 * (index + 1), dtype=float)
        observed_y = np.linspace(0.01, 0.02 * (index + 1), TRAIN_WINDOW_LENGTH)
        windows.append(
            SilverboxControlledSeries(
                role="train",
                source_start=start,
                source_stop=start + TRAIN_WINDOW_LENGTH,
                input_u=input_u,
                initialization_y=observed_y[:50],
                target_y=observed_y[50:],
                sampling_time=SAMPLE_TIME_SECONDS,
            )
        )
    validation_count = VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START
    validation_y = np.full(validation_count, 0.03, dtype=float)
    validation = SilverboxControlledSeries(
        role="validation",
        source_start=VALIDATION_SOURCE_START,
        source_stop=VALIDATION_SOURCE_STOP,
        input_u=np.full(validation_count, 0.01, dtype=float),
        initialization_y=validation_y[:50],
        target_y=validation_y[50:],
        sampling_time=SAMPLE_TIME_SECONDS,
    )
    return SilverboxControlledDevelopment(tuple(windows), validation, SAMPLE_TIME_SECONDS)


def _archive_and_manifest(tmp_path: Path) -> tuple[Path, Path, str]:
    archive = tmp_path / "synthetic-SilverboxFiles.zip"
    archive_bytes = b"synthetic archive bytes used only by offline bridge tests"
    archive.write_bytes(archive_bytes)
    digest = hashlib.sha256(archive_bytes).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {
                    "archive_name": "SilverboxFiles.zip",
                    "raw_archive_path": str(archive),
                    "raw_archive_bytes": len(archive_bytes),
                    "raw_archive_sha256": digest,
                }
            }
        ),
        encoding="utf-8",
    )
    return archive, manifest, digest


def _sentinel_archive_and_manifest(
    tmp_path: Path, *, poison_validation_targets: bool = False
) -> tuple[Path, Path]:
    archive = tmp_path / "synthetic-SilverboxFiles.zip"
    sealed_ranges = (
        (ARROW_FULL_START, ARROW_FULL_STOP),
        (MULTISINE_TRAIN_STOP, MULTISINE_STOP),
    )
    rows = ["V1,V2,\n"]
    for index in range(SAMPLE_COUNT):
        if any(start <= index < stop for start, stop in sealed_ranges):
            # An unterminated quote makes csv.reader consume following physical
            # lines as one record. The safe development reader must skip this
            # early sealed prefix as raw lines without tokenizing it.
            if ARROW_FULL_START <= index < ARROW_FULL_STOP:
                rows.append(f'"SEALED_INPUT_{index},SEALED_OUTPUT_{index}\n')
            else:
                rows.append(f"SEALED_INPUT_{index},SEALED_OUTPUT_{index},\n")
        elif (
            VALIDATION_SOURCE_START + STATE_INITIALIZATION_LENGTH <= index < VALIDATION_SOURCE_STOP
            and poison_validation_targets
        ):
            rows.append(f"{index / 10:.8f},POISON_TARGET_{index},\n")
        elif MULTISINE_START <= index < MULTISINE_TRAIN_STOP:
            rows.append(f"{index / 10:.8f},{index / 20:.8f},\n")
        else:
            rows.append("0.0,0.0,\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for member in REQUIRED_ARCHIVE_MEMBERS:
            if member == SNLS_CSV_MEMBER:
                zip_file.writestr(member, "".join(rows))
            else:
                zip_file.writestr(member, "placeholder")
    archive_bytes = archive.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {
                    "archive_name": "SilverboxFiles.zip",
                    "raw_archive_path": str(archive),
                    "raw_archive_bytes": len(archive_bytes),
                    "raw_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return archive, manifest


def _fake_proposal(receipt_dir: Path, *, term_id: str = "u_cubed") -> SilverboxProposalResult:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "synthetic-proposal.json"
    request_payload = _build_request_payload()
    preflight_body = json.dumps(
        {"object": "list", "data": [{"id": MODEL_ID}, {"id": "another-installed-model"}]},
        separators=(",", ":"),
    ).encode("utf-8")
    assistant_content = json.dumps(
        {
            "term_id": term_id,
            "falsifying_prediction": "The selected term should matter at a larger input amplitude.",
        },
        separators=(",", ":"),
    )
    response_body = json.dumps(
        {
            "model": MODEL_ID,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": assistant_content},
                }
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    receipt = {
        "endpoint": ENDPOINT,
        "timeout_seconds": 120.0,
        "model_preflight_endpoint": MODELS_ENDPOINT,
        "model_preflight_timeout_seconds": 120.0,
        "model_preflight_status": "success",
        "model_preflight_http_status": 200,
        "model_preflight_model_ids": [MODEL_ID, "another-installed-model"],
        "model_preflight_raw_response_base64": base64.b64encode(preflight_body).decode("ascii"),
        "model_preflight_response_sha256": hashlib.sha256(preflight_body).hexdigest(),
        "status": "success",
        "parser_outcome": "success",
        "returned_model": MODEL_ID,
        "request_payload": request_payload,
        "request_payload_sha256": FROZEN_REQUEST_PAYLOAD_SHA256,
        "raw_response_base64": base64.b64encode(response_body).decode("ascii"),
        "response_sha256": hashlib.sha256(response_body).hexdigest(),
        "http_status": 200,
        "finish_reason": "stop",
        "chat_dispatched": True,
        "term_id": term_id,
        "falsifying_prediction": "The selected term should matter at a larger input amplitude.",
        "falsifying_prediction_status": "unscored_selection_independent",
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt_path.write_bytes(encoded)
    return SilverboxProposalResult(
        term_id=term_id,
        falsifying_prediction=receipt["falsifying_prediction"],
        receipt_path=receipt_path,
        receipt_sha256=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True)
class _FakeFit:
    status: str
    receipt_path: Path
    receipt_sha256: str


@dataclass(frozen=True)
class _FakeSelection:
    status: str
    selected_hypothesis: str | None
    receipt_path: Path
    receipt_sha256: str
    reason: str


def _fake_fit_result(tmp_path: Path, status: str = "complete") -> _FakeFit:
    receipt = tmp_path / "fake-fit.json"
    receipt.write_text("synthetic fit receipt", encoding="utf-8")
    return _FakeFit(status, receipt, hashlib.sha256(receipt.read_bytes()).hexdigest())


def test_successful_bridge_verifies_bytes_and_calls_stages_in_order(tmp_path):
    archive, manifest, source_sha256 = _archive_and_manifest(tmp_path)
    development = _development()
    events: list[str] = []
    fit_calls: list[tuple[tuple, str, dict]] = []
    selection_calls: list[tuple[object, object]] = []

    def loader(path):
        events.append("loader")
        assert path == archive
        return development

    def proposal_request(*, receipt_dir):
        events.append("proposal")
        return _fake_proposal(receipt_dir)

    def fit(train_windows, term_id, **kwargs):
        events.append("fit")
        fit_calls.append((train_windows, term_id, kwargs))
        return _fake_fit_result(tmp_path)

    def select(fit_handle, validation):
        events.append("select")
        selection_calls.append((fit_handle, validation))
        receipt = tmp_path / "fake-selection.json"
        receipt.write_text("synthetic selection receipt", encoding="utf-8")
        return _FakeSelection(
            "selected",
            "linear",
            receipt,
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "synthetic-only",
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=loader,
        proposal_request=proposal_request,
        fit=fit,
        select=select,
        monotonic=lambda: 123.0,
        run_id_factory=lambda: "synthetic-success",
    )

    assert result.status == "selected"
    assert events == ["loader", "proposal", "fit", "select"]
    assert len(fit_calls) == 1
    received_windows, received_term, fit_kwargs = fit_calls[0]
    assert received_windows is development.train_windows
    assert received_term == "u_cubed"
    assert fit_kwargs["source_sha256"] == source_sha256
    assert fit_kwargs["proposal_receipt_sha256"] == result.proposal_receipt_sha256
    assert fit_kwargs["run_id"] == "synthetic-success"
    assert fit_kwargs["pilot_started_monotonic"] == 123.0
    assert fit_kwargs["receipt_dir"] == tmp_path / "artifacts"
    assert selection_calls == [(result.fit, development.validation)]
    for name in (
        "source_verified.json",
        "data_loaded.json",
        "proposal_succeeded.json",
        "fit_completed.json",
        "selection_completed.json",
        "run_finished.json",
    ):
        receipt = json.loads((result.stage_directory / name).read_text(encoding="utf-8"))
        assert receipt["run_id"] == "synthetic-success"
    proposal_stage = json.loads(
        (result.stage_directory / "proposal_succeeded.json").read_text(encoding="utf-8")
    )
    assert proposal_stage["proposal_receipt_sha256"] == result.proposal_receipt_sha256
    assert proposal_stage["model_id"] == MODEL_ID


def test_runner_default_loader_never_parses_or_exposes_sealed_rows(monkeypatch, tmp_path):
    archive, manifest = _sentinel_archive_and_manifest(tmp_path)
    original_safe_loader = runner.load_silverbox_controlled_development
    loaded_developments = []

    def record_safe_loader(path, **kwargs):
        assert path == archive
        development = original_safe_loader(path, **kwargs)
        loaded_developments.append(development)
        return development

    def forbidden_broad_loader(*args, **kwargs):
        raise AssertionError("runner must not call the broad all-rows loader")

    monkeypatch.setattr(runner, "load_silverbox_controlled_development", record_safe_loader)
    monkeypatch.setattr(silverbox, "load_silverbox_archive", forbidden_broad_loader)

    def fake_select(fit_handle, validation):
        assert validation is loaded_developments[0].validation
        assert validation.target_loaded is False
        receipt = tmp_path / "fake-selection.json"
        receipt.write_text("synthetic selection receipt", encoding="utf-8")
        return _FakeSelection(
            "selected",
            "linear",
            receipt,
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "synthetic-only",
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        proposal_request=lambda **kwargs: _fake_proposal(kwargs["receipt_dir"]),
        fit=lambda *args, **kwargs: _fake_fit_result(tmp_path),
        select=fake_select,
        monotonic=lambda: 321.0,
        run_id_factory=lambda: "synthetic-runner-sentinel",
    )

    assert result.status == "selected"
    assert len(loaded_developments) == 1
    development = loaded_developments[0]
    assert "SEALED_INPUT_" not in repr(development)
    assert "SEALED_OUTPUT_" not in repr(development)
    assert all(np.all(np.isfinite(series.target_y)) for series in development.train_windows)
    assert development.validation.target_loaded is False


def test_archive_sha_mismatch_blocks_loader_and_proposal(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    archive.write_bytes(b"mutated bytes")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("a failed source gate must stop before this stage")

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=forbidden,
        proposal_request=forbidden,
        fit=forbidden,
        select=forbidden,
        run_id_factory=lambda: "synthetic-bad-source",
    )
    assert result.status == "unresolved"
    assert result.reason == "source_manifest_or_archive_verification_failed"
    assert calls == []
    assert json.loads((result.stage_directory / "source_failed.json").read_text())["validation_accessed"] is False


def test_terminal_proposal_failure_never_fits_or_touches_validation(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)

    class NoValidationAccess:
        train_windows = _development().train_windows

        @property
        def validation(self):
            raise AssertionError("validation must remain untouched")

    development = NoValidationAccess()
    calls: list[str] = []

    def proposal_failure(*, receipt_dir):
        calls.append("proposal")
        raise RuntimeError("synthetic terminal proposal failure")

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        raise AssertionError("terminal proposal failure must block fit and selection")

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: development,
        proposal_request=proposal_failure,
        fit=forbidden,
        select=forbidden,
        monotonic=lambda: 200.0,
        run_id_factory=lambda: "synthetic-proposal-failure",
    )
    assert result.status == "unresolved"
    assert result.reason == "terminal_qwen_proposal_failure"
    assert calls == ["proposal"]
    failure = json.loads((result.stage_directory / "proposal_failed.json").read_text())
    assert failure["validation_accessed"] is False


def test_incomplete_fit_skips_selection_and_does_not_access_validation(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)

    class NoValidationAccess:
        def __init__(self):
            self.train_windows = _development().train_windows

        @property
        def validation(self):
            raise AssertionError("incomplete fit must not access validation")

    development = NoValidationAccess()
    fit_result = _fake_fit_result(tmp_path, status="incomplete")
    select_calls: list[str] = []

    def forbidden_selection(*args, **kwargs):
        select_calls.append("selection")
        raise AssertionError("incomplete fit must skip selection")

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: development,
        proposal_request=lambda **kwargs: _fake_proposal(kwargs["receipt_dir"]),
        fit=lambda *args, **kwargs: fit_result,
        select=forbidden_selection,
        monotonic=lambda: 400.0,
        run_id_factory=lambda: "synthetic-incomplete-fit",
    )
    assert result.status == "unresolved"
    assert result.reason == "fit_incomplete"
    assert select_calls == []
    skipped = json.loads((result.stage_directory / "selection_skipped.json").read_text())
    assert skipped["validation_accessed"] is False


def test_default_loader_does_not_materialize_poisoned_validation_targets_for_incomplete_fit(tmp_path):
    # The synthetic source keeps its validation input and initializer valid,
    # but poisons every output in the target suffix. Loading development must
    # succeed; an incomplete fit must never ask for those target values.
    archive, manifest = _sentinel_archive_and_manifest(tmp_path, poison_validation_targets=True)

    development_seen = []
    original_loader = runner.load_silverbox_controlled_development

    def capture_loader(path, **kwargs):
        development = original_loader(path, **kwargs)
        development_seen.append(development)
        return development

    fit_result = _fake_fit_result(tmp_path, status="incomplete")
    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=capture_loader,
        proposal_request=lambda **kwargs: _fake_proposal(kwargs["receipt_dir"]),
        fit=lambda *args, **kwargs: fit_result,
        select=lambda *args, **kwargs: pytest.fail("incomplete fit must skip selection"),
        monotonic=lambda: 420.0,
        run_id_factory=lambda: "synthetic-poisoned-target-incomplete-fit",
    )

    assert result.status == "unresolved"
    assert result.reason == "fit_incomplete"
    assert len(development_seen) == 1
    assert development_seen[0].validation.target_loaded is False
    assert not hasattr(development_seen[0].validation, "target_y")


def test_fit_completion_after_global_deadline_skips_validation(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)

    class NoValidationAccess:
        train_windows = _development().train_windows

        @property
        def validation(self):
            raise AssertionError("expired fit must not access validation")

    ticks = iter([100.0] * 5 + [700.0])
    selection_calls: list[str] = []

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: NoValidationAccess(),
        proposal_request=lambda **kwargs: _fake_proposal(kwargs["receipt_dir"]),
        fit=lambda *args, **kwargs: _fake_fit_result(tmp_path),
        select=lambda *args, **kwargs: selection_calls.append("select"),
        monotonic=lambda: next(ticks, 700.0),
        run_id_factory=lambda: "synthetic-fit-completion-deadline",
    )

    assert result.status == "unresolved"
    assert result.reason == "wall_time_limit_after_fit"
    assert selection_calls == []
    fit_stage = json.loads((result.stage_directory / "fit_completed.json").read_text())
    assert fit_stage["status"] == "unresolved"
    assert fit_stage["reason"] == "wall_time_limit"
    skipped = json.loads((result.stage_directory / "selection_skipped.json").read_text())
    assert skipped["validation_accessed"] is False


def test_deadline_after_selection_result_does_not_publish_selected_outcome(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    ticks = iter([100.0] * 7 + [700.0])

    def selected_result(*args, **kwargs):
        receipt = tmp_path / "fake-selection.json"
        receipt.write_text("synthetic selection receipt", encoding="utf-8")
        return _FakeSelection(
            "selected",
            "linear",
            receipt,
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "synthetic-only",
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=lambda **kwargs: _fake_proposal(kwargs["receipt_dir"]),
        fit=lambda *args, **kwargs: _fake_fit_result(tmp_path),
        select=selected_result,
        monotonic=lambda: next(ticks, 700.0),
        run_id_factory=lambda: "synthetic-selection-completion-deadline",
    )

    assert result.status == "unresolved"
    assert result.reason == "wall_time_limit"
    assert result.selection is None
    stage = json.loads((result.stage_directory / "selection_completed.json").read_text())
    assert stage["status"] == "unresolved"
    assert stage["reason"] == "wall_time_limit"
    assert stage["selected_hypothesis"] is None
    finished = json.loads((result.stage_directory / "run_finished.json").read_text())
    assert finished["status"] == "unresolved"
    assert finished["reason"] == "wall_time_limit"


def test_run_finished_write_that_crosses_deadline_is_downgraded_atomically(monkeypatch, tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    now = [100.0]
    original_write = runner._write_stage_receipt

    def cross_budget_during_run_finished_write(path, payload):
        digest = original_write(path, payload)
        if path.name == "run_finished.json" and payload["status"] == "selected":
            now[0] = 700.0
        return digest

    monkeypatch.setattr(runner, "_write_stage_receipt", cross_budget_during_run_finished_write)

    def selected_result(*args, **kwargs):
        receipt = tmp_path / "fake-selection.json"
        receipt.write_text("synthetic selection receipt", encoding="utf-8")
        return _FakeSelection(
            "selected",
            "linear",
            receipt,
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "synthetic-only",
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=lambda **kwargs: _fake_proposal(kwargs["receipt_dir"]),
        fit=lambda *args, **kwargs: _fake_fit_result(tmp_path),
        select=selected_result,
        monotonic=lambda: now[0],
        run_id_factory=lambda: "synthetic-terminal-write-deadline",
    )

    assert result.status == "unresolved"
    assert result.reason == "wall_time_limit"
    assert result.selection is None
    selection_stage = json.loads((result.stage_directory / "selection_completed.json").read_text())
    assert selection_stage["status"] == "unresolved"
    assert selection_stage["selected_hypothesis"] is None
    finished = json.loads((result.stage_directory / "run_finished.json").read_text())
    assert finished["status"] == "unresolved"
    assert finished["reason"] == "wall_time_limit"


def test_qwen_elapsed_time_counts_against_global_deadline(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    development = _development()
    ticks = iter([50.0, 50.0, 50.0, 50.0, 650.0])
    proposal_calls: list[str] = []
    fit_calls: list[str] = []
    select_calls: list[str] = []

    def proposal_request(*, receipt_dir):
        proposal_calls.append("proposal")
        return _fake_proposal(receipt_dir)

    def forbidden_fit(*args, **kwargs):
        fit_calls.append("fit")
        raise AssertionError("600-second global deadline must prevent fit")

    def forbidden_select(*args, **kwargs):
        select_calls.append("select")
        raise AssertionError("no fit means no validation selection")

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: development,
        proposal_request=proposal_request,
        fit=forbidden_fit,
        select=forbidden_select,
        monotonic=lambda: next(ticks),
        run_id_factory=lambda: "synthetic-global-deadline",
    )
    assert result.status == "unresolved"
    assert result.reason == "wall_time_limit_before_fit"
    assert proposal_calls == ["proposal"]
    assert fit_calls == []
    assert select_calls == []
    skipped = json.loads((result.stage_directory / "fit_skipped.json").read_text())
    assert skipped["validation_accessed"] is False


def test_changed_proposal_receipt_bytes_are_terminal(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    fit_calls: list[str] = []

    def tampered_proposal(*, receipt_dir):
        proposal = _fake_proposal(receipt_dir)
        proposal.receipt_path.write_bytes(proposal.receipt_path.read_bytes() + b" ")
        return proposal

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=tampered_proposal,
        fit=lambda *args, **kwargs: fit_calls.append("fit"),
        select=lambda *args, **kwargs: pytest.fail("proposal receipt failure must block selection"),
        monotonic=lambda: 500.0,
        run_id_factory=lambda: "synthetic-tampered-receipt",
    )
    assert result.status == "unresolved"
    assert result.reason == "terminal_qwen_proposal_failure"
    assert fit_calls == []
    failed = json.loads((result.stage_directory / "proposal_failed.json").read_text())
    assert failed["failure_code"] == "proposal_or_receipt_failure"


def test_proposal_receipt_outside_fresh_run_directory_blocks_fit(tmp_path):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    fit_calls: list[str] = []

    def stale_proposal(*, receipt_dir):
        outside_directory = tmp_path / "old-run" / "proposal_receipts"
        return _fake_proposal(outside_directory)

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=stale_proposal,
        fit=lambda *args, **kwargs: fit_calls.append("fit"),
        select=lambda *args, **kwargs: pytest.fail("stale proposal receipt must block selection"),
        monotonic=lambda: 560.0,
        run_id_factory=lambda: "synthetic-stale-proposal-receipt",
    )

    assert result.status == "unresolved"
    assert result.reason == "terminal_qwen_proposal_failure"
    assert fit_calls == []


@pytest.mark.parametrize("field", ["timeout_seconds", "model_preflight_timeout_seconds"])
def test_noncontract_proposal_timeout_blocks_fit(tmp_path, field):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    fit_calls: list[str] = []

    def changed_timeout(*, receipt_dir):
        result = _fake_proposal(receipt_dir)
        receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
        receipt[field] = 119.0
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result.receipt_path.write_bytes(encoded)
        return SilverboxProposalResult(
            term_id=result.term_id,
            falsifying_prediction=result.falsifying_prediction,
            receipt_path=result.receipt_path,
            receipt_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=changed_timeout,
        fit=lambda *args, **kwargs: fit_calls.append("fit"),
        select=lambda *args, **kwargs: pytest.fail("invalid timeout must block selection"),
        monotonic=lambda: 575.0,
        run_id_factory=lambda: f"synthetic-timeout-{field}",
    )

    assert result.status == "unresolved"
    assert result.reason == "terminal_qwen_proposal_failure"
    assert fit_calls == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("returned_model", "other-model"), ("term_id", "y_cubed")],
)
def test_model_or_term_mismatch_in_exact_receipt_blocks_fit(tmp_path, field, replacement):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    fit_calls: list[str] = []

    def mismatched_proposal(*, receipt_dir):
        result = _fake_proposal(receipt_dir, term_id="u_cubed")
        receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
        receipt[field] = replacement
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result.receipt_path.write_bytes(encoded)
        return SilverboxProposalResult(
            term_id=result.term_id,
            falsifying_prediction=result.falsifying_prediction,
            receipt_path=result.receipt_path,
            receipt_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=mismatched_proposal,
        fit=lambda *args, **kwargs: fit_calls.append("fit"),
        select=lambda *args, **kwargs: pytest.fail("invalid proposal receipt must block selection"),
        monotonic=lambda: 550.0,
        run_id_factory=lambda: f"synthetic-mismatch-{field}",
    )
    assert result.status == "unresolved"
    assert result.reason == "terminal_qwen_proposal_failure"
    assert fit_calls == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_preflight_status", "failed"),
        ("model_preflight_http_status", 503),
        ("model_preflight_model_ids", ["another-installed-model"]),
        ("http_status", 503),
        ("finish_reason", "length"),
        ("response_sha256", "0" * 64),
        ("model_preflight_response_sha256", "f" * 64),
    ],
)
def test_receipt_preflight_and_raw_response_integrity_gate_blocks_fit(tmp_path, field, replacement):
    archive, manifest, _ = _archive_and_manifest(tmp_path)
    fit_calls: list[str] = []

    def tampered_proposal(*, receipt_dir):
        result = _fake_proposal(receipt_dir)
        receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
        receipt[field] = replacement
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result.receipt_path.write_bytes(encoded)
        return SilverboxProposalResult(
            term_id=result.term_id,
            falsifying_prediction=result.falsifying_prediction,
            receipt_path=result.receipt_path,
            receipt_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    result = runner.run_silverbox_development(
        archive_path=archive,
        manifest_path=manifest,
        receipt_root=tmp_path / "artifacts",
        loader=lambda _: _development(),
        proposal_request=tampered_proposal,
        fit=lambda *args, **kwargs: fit_calls.append("fit"),
        select=lambda *args, **kwargs: pytest.fail("invalid proposal receipt must block selection"),
        monotonic=lambda: 600.0,
        run_id_factory=lambda: f"synthetic-receipt-integrity-{field}",
    )

    assert result.status == "unresolved"
    assert result.reason == "terminal_qwen_proposal_failure"
    assert fit_calls == []

"""One-child source-free integration for the declared synthetic ABC6 control.

This runner never constructs prospective truth. It verifies the source pins,
executes the one-use training entry point once, reloads verified training
evidence, forecasts the complete ordered roster, freezes that target-free
roster durably, and delegates the reveal/score phase to the reviewed scorer.
The scorer remains the only owner of the one-use reveal marker and target gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit
from core.real_data import cascaded_tanks_abc6_cases as cases
from core.real_data import cascaded_tanks_abc6_forecast as forecast_module
from core.real_data import cascaded_tanks_abc6_scoring as scoring
from core.real_data import cascaded_tanks_abc6_authority as authority_module

_REPO_ROOT: Final = campaign_fit._REPO_ROOT
_RUNNER_SOURCE_PATH: Final = "scripts/run_cascaded_tanks_abc6_synthetic.py"
_FORECAST_SOURCE_PATH: Final = "core/real_data/cascaded_tanks_abc6_forecast.py"
_SCORER_SOURCE_PATH: Final = "core/real_data/cascaded_tanks_abc6_scoring.py"
_INTEGRATED_SOURCE_PATHS: Final = (
    _RUNNER_SOURCE_PATH,
    _FORECAST_SOURCE_PATH,
    _SCORER_SOURCE_PATH,
)
FORECAST_ARTIFACT_FILENAME: Final = "campaign.target-free-forecasts.json"
GRANT_FD_ENV: Final = "ABC6_GRANT_FD"


class ABC6SyntheticRunnerPreflightError(RuntimeError):
    """The integrated source pin gate is incomplete or inconsistent."""


class ABC6SyntheticRunnerIntegrityError(RuntimeError):
    """A status, forecast, or durable forecast artifact failed verification."""


@dataclass(frozen=True, slots=True)
class ABC6SyntheticRunResult:
    """Durable target-free forecasts and the scorer's terminal diagnostics."""

    training_campaign: campaign_fit.ABC6CampaignResult
    verified_evidence_manifest_sha256: str
    forecast_artifact_path: Path
    forecast_artifact_sha256: str
    frozen_forecasts: scoring.ABC6FrozenForecastRoster
    deferred_score: scoring.ABC6DeferredScoreResult


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _source_pin_check() -> tuple[tuple[str, str], ...]:
    """Require a separately reviewed manifest expansion before the claim call."""

    if _REPO_ROOT != campaign_fit._REPO_ROOT:
        raise ABC6SyntheticRunnerPreflightError(
            "runner and campaign source roots do not match"
        )
    allowlisted = set(campaign_fit._REQUIRED_SOURCE_PATHS)
    reviewed = campaign_fit._REVIEWED_SOURCE_SHA256
    missing_allowlist = tuple(
        path for path in _INTEGRATED_SOURCE_PATHS if path not in allowlisted
    )
    missing_reviewed_hashes = tuple(
        path for path in _INTEGRATED_SOURCE_PATHS if path not in reviewed
    )
    if missing_allowlist or missing_reviewed_hashes:
        details = []
        if missing_allowlist:
            details.append(
                "not in campaign source allowlist: " + ", ".join(missing_allowlist)
            )
        if missing_reviewed_hashes:
            details.append(
                "not in reviewed source hashes: " + ", ".join(missing_reviewed_hashes)
            )
        raise ABC6SyntheticRunnerPreflightError(
            "integrated source pin gate is closed; " + "; ".join(details)
        )
    try:
        current = campaign_fit._current_source_hashes()
    except Exception as error:
        raise ABC6SyntheticRunnerPreflightError(
            "cannot hash integrated source files before the training claim"
        ) from error
    checked: list[tuple[str, str]] = []
    for path in _INTEGRATED_SOURCE_PATHS:
        digest = current.get(path)
        pinned = reviewed.get(path)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(pinned, str)
            or digest != pinned
        ):
            raise ABC6SyntheticRunnerPreflightError(
                f"integrated source hash is not independently pinned: {path}"
            )
        checked.append((path, digest))
    return tuple(checked)


def _receive_launch_authority(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> authority_module.ABC6LaunchAuthority:
    """Read one canonical inherited-FD locator and consume its sealed grant."""

    values = os.environ if environ is None else environ
    raw = values.get(GRANT_FD_ENV)
    if type(raw) is not str or re.fullmatch(r"[1-9][0-9]*", raw) is None:
        raise ABC6SyntheticRunnerPreflightError(
            "runner requires one canonical decimal ABC6_GRANT_FD locator"
        )
    fd = int(raw)
    if fd < 3:
        raise ABC6SyntheticRunnerPreflightError(
            "runner grant descriptor must be an inherited non-stdio FD"
        )
    try:
        return authority_module.receive_child_grant(fd)
    except authority_module.ABC6AuthorityError as error:
        raise ABC6SyntheticRunnerPreflightError(
            f"supervised launch grant failed verification: {error}"
        ) from error
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _expected_status_receipts(
    campaign_result: campaign_fit.ABC6CampaignResult,
) -> tuple[tuple[str, str], ...]:
    if len(campaign_result.case_statuses) != cases.CASE_COUNT:
        raise ABC6SyntheticRunnerIntegrityError(
            "training campaign result must contain all 24 ordered statuses"
        )
    expected: list[tuple[str, str]] = []
    for index, (case, status) in enumerate(
        zip(cases.CASE_ROSTER, campaign_result.case_statuses, strict=True)
    ):
        if status.case_index != index or status.case_id != case.case_id:
            raise ABC6SyntheticRunnerIntegrityError(
                "training campaign statuses differ from the frozen roster"
            )
        expected.extend(
            (
                (f"case-{index:02d}.fit-status.json", status.fit_receipt_sha256),
                (
                    f"case-{index:02d}.baseline-status.json",
                    status.baseline_receipt_sha256,
                ),
            )
        )
    return tuple(expected)


def _execution_receipt_root_identity(
    execution: campaign_fit.ABC6TrainingCampaignExecution,
) -> campaign_fit.ABC6ReceiptRootIdentity:
    identity = getattr(execution, "receipt_root_identity", None)
    if not isinstance(identity, campaign_fit.ABC6ReceiptRootIdentity):
        raise ABC6SyntheticRunnerIntegrityError(
            "training execution has no typed physical receipt-root identity"
        )
    if (
        identity.repository_root_realpath != str(campaign_fit._REPO_ROOT)
        or identity.receipt_root_relative != campaign_fit.RECEIPT_ROOT_RELATIVE
        or identity.receipt_root_relative != cases.RECEIPT_ROOT_RELATIVE
    ):
        raise ABC6SyntheticRunnerIntegrityError(
            "training execution receipt root differs from the fixed checkout path"
        )
    try:
        identity.verify()
    except Exception as error:
        raise ABC6SyntheticRunnerIntegrityError(
            "training execution receipt-root identity is not live"
        ) from error
    receipt_root = identity.receipt_root_path
    result = execution.campaign_result
    if (
        result.summary_path != receipt_root / campaign_fit.SUMMARY_FILENAME
        or execution.evidence_manifest_path
        != receipt_root / campaign_fit.EVIDENCE_MANIFEST_FILENAME
    ):
        raise ABC6SyntheticRunnerIntegrityError(
            "training summary or evidence manifest is outside the fixed receipt root"
        )
    return identity


def _verify_status_gate(
    receipt_root_identity: campaign_fit.ABC6ReceiptRootIdentity,
    campaign_result: campaign_fit.ABC6CampaignResult,
    results: Sequence[object],
) -> tuple[tuple[tuple[str, str], ...], str]:
    if (
        campaign_result.protocol_id != scoring.PROTOCOL_ID
        or campaign_result.run_id != scoring.RUN_ID
        or len(results) != cases.CASE_COUNT
    ):
        raise ABC6SyntheticRunnerIntegrityError(
            "training campaign identity does not match the scoring protocol"
        )
    gate = None
    try:
        gate, statuses, receipt_hashes, summary_sha256 = (
            scoring._read_verified_statuses(receipt_root_identity)
        )
    except Exception as error:
        raise ABC6SyntheticRunnerIntegrityError(
            "all 48 durable fit/baseline receipts and summary must verify "
            "before forecast"
        ) from error
    try:
        if gate is None:
            raise ABC6SyntheticRunnerIntegrityError(
                "typed execution did not open the 48-status receipt gate"
            )
        expected_receipts = _expected_status_receipts(campaign_result)
        if tuple(receipt_hashes) != expected_receipts:
            raise ABC6SyntheticRunnerIntegrityError(
                "verified receipt hashes do not match the training campaign result"
            )
        if summary_sha256 != campaign_result.summary_sha256:
            raise ABC6SyntheticRunnerIntegrityError(
                "verified training summary hash differs from campaign result"
            )
        expected_statuses = tuple(
            (item.fit_status, item.baseline_status)
            for item in campaign_result.case_statuses
        )
        if tuple(statuses) != expected_statuses:
            raise ABC6SyntheticRunnerIntegrityError(
                "verified receipt statuses differ from the training campaign result"
            )
        for index, result in enumerate(results):
            try:
                scoring._validate_result_identity(result, index)
            except Exception as error:
                raise ABC6SyntheticRunnerIntegrityError(
                    f"training result {index} does not match its verified status/roster"
                ) from error
            if statuses[index] != (result.abc_status, result.baseline_status):
                raise ABC6SyntheticRunnerIntegrityError(
                    f"training result {index} disagrees with durable status receipts"
                )
        receipt_root_identity.verify()
        return tuple(receipt_hashes), summary_sha256
    finally:
        if gate is not None:
            gate._receipt_anchor.close()
            gate._root_anchor.close()


def _verify_forecast_roster(
    frozen: scoring.ABC6FrozenForecastRoster,
    results: Sequence[object],
) -> None:
    if (
        len(frozen.forecasts) != cases.CASE_COUNT
        or len(frozen.case_sha256) != cases.CASE_COUNT
        or len(results) != cases.CASE_COUNT
    ):
        raise ABC6SyntheticRunnerIntegrityError(
            "forecast and result rosters must each contain all 24 cases"
        )
    for index, (forecast, result, case) in enumerate(
        zip(frozen.forecasts, results, cases.CASE_ROSTER, strict=True)
    ):
        try:
            scoring._validate_result_identity(result, index)
            posterior = scoring._validated_posterior(result, index)
            scoring._validated_forecast(forecast, result, index, posterior)
            case_digest = scoring._forecast_case_hash(index, forecast)
        except Exception as error:
            raise ABC6SyntheticRunnerIntegrityError(
                f"target-free forecast {index} failed full scorer validation"
            ) from error
        if case_digest != frozen.case_sha256[index]:
            raise ABC6SyntheticRunnerIntegrityError(
                f"target-free forecast {index} hash differs from frozen roster"
            )
        if case.truth_id == "N":
            if (
                forecast.status != "abstained_n"
                or forecast.prospective_inputs is not None
                or forecast.particles
                or forecast.particle_trajectories
            ):
                raise ABC6SyntheticRunnerIntegrityError(
                    "N must abstain without any prospective target forecast"
                )
        elif result.abc_status != "complete":
            if (
                forecast.status != "incomplete_abc_fit"
                or forecast.particles
                or forecast.weights
                or forecast.particle_trajectories
                or forecast.pointwise_weighted_mean is not None
                or forecast.pointwise_weighted_median is not None
                or forecast.pointwise_q05 is not None
                or forecast.pointwise_q95 is not None
            ):
                raise ABC6SyntheticRunnerIntegrityError(
                    "incomplete fit must remain a null forecast record"
                )

    expected_roster_hash = scoring._sha256(
        scoring._canonical_json(
            {
                "protocol_id": scoring.PROTOCOL_ID,
                "run_id": scoring.RUN_ID,
                "case_sha256": frozen.case_sha256,
            }
        )
    )
    if frozen.roster_sha256 != expected_roster_hash:
        raise ABC6SyntheticRunnerIntegrityError(
            "target-free forecast roster hash is invalid"
        )


def _forecast_artifact_payload(
    *,
    training_result: campaign_fit.ABC6CampaignResult,
    evidence_manifest_sha256: str,
    source_hashes: tuple[tuple[str, str], ...],
    receipt_hashes: tuple[tuple[str, str], ...],
    frozen: scoring.ABC6FrozenForecastRoster,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": training_result.protocol_id,
        "run_id": training_result.run_id,
        "training_manifest_sha256": training_result.manifest_sha256,
        "training_claim_sha256": training_result.claim_sha256,
        "training_summary_filename": training_result.summary_path.name,
        "training_summary_sha256": training_result.summary_sha256,
        "training_evidence_manifest_sha256": evidence_manifest_sha256,
        "integrated_source_hashes": [
            {"path": path, "sha256": digest} for path, digest in source_hashes
        ],
        "ordered_case_identities": [
            {
                "case_index": case.case_index,
                "case_id": case.case_id,
                "truth_id": case.truth_id,
                "input_window": case.input_window,
                "replicate": case.replicate,
                "fit_model": case.fit_model.value,
            }
            for case in cases.CASE_ROSTER
        ],
        "status_receipts": [
            {"filename": filename, "sha256": digest}
            for filename, digest in receipt_hashes
        ],
        "forecast_case_sha256": list(frozen.case_sha256),
        "forecast_roster_sha256": frozen.roster_sha256,
        "forecasts": [scoring._json_ready(item) for item in frozen.forecasts],
        "target_free": True,
        "prospective_targets_generated_by_runner": False,
        "retry_allowed": False,
    }
    body["payload_sha256"] = scoring._sha256(scoring._canonical_json(body))
    return scoring._canonical_json(body)


def _publish_forecasts(
    receipt_root_identity: campaign_fit.ABC6ReceiptRootIdentity,
    payload: bytes,
) -> None:
    try:
        receipt_root_identity.verify()
        receipt_fd = receipt_root_identity.duplicate_receipt_root_fd()
        try:
            campaign_fit._write_exclusive_durable_at(
                receipt_fd, FORECAST_ARTIFACT_FILENAME, payload
            )
        finally:
            os.close(receipt_fd)
        receipt_root_identity.verify()
    except Exception as error:
        raise ABC6SyntheticRunnerIntegrityError(
            "could not durably publish the target-free forecast through the "
            "pinned receipt root"
        ) from error


def _verify_published_forecasts(
    receipt_root_identity: campaign_fit.ABC6ReceiptRootIdentity,
    expected_payload: bytes,
) -> str:
    try:
        receipt_root_identity.verify()
        receipt_fd = receipt_root_identity.duplicate_receipt_root_fd()
        try:
            raw = cases._read_regular_file_at(
                receipt_fd,
                FORECAST_ARTIFACT_FILENAME,
                maximum_bytes=cases._MAX_HANDOFF_ARTIFACT_BYTES,
                label="durable target-free forecast artifact",
            )
        finally:
            os.close(receipt_fd)
        receipt_root_identity.verify()
    except Exception as error:
        raise ABC6SyntheticRunnerIntegrityError(
            "durable target-free forecast artifact is unreadable through its "
            "pinned receipt root"
        ) from error
    if raw != expected_payload:
        raise ABC6SyntheticRunnerIntegrityError(
            "durable target-free forecast artifact differs from the frozen roster"
        )
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ABC6SyntheticRunnerIntegrityError(
            "durable target-free forecast artifact is not valid ASCII JSON"
        ) from error
    if not isinstance(decoded, dict) or scoring._canonical_json(decoded) != raw:
        raise ABC6SyntheticRunnerIntegrityError(
            "durable target-free forecast artifact is not canonical JSON"
        )
    body = {key: value for key, value in decoded.items() if key != "payload_sha256"}
    if decoded.get("payload_sha256") != scoring._sha256(scoring._canonical_json(body)):
        raise ABC6SyntheticRunnerIntegrityError(
            "durable target-free forecast payload hash is invalid"
        )
    return scoring._sha256(raw)


def run_cascaded_tanks_abc6_synthetic(
    manifest_path: str | os.PathLike[str],
    manifest_sha256: str,
    receipt_directory: str | os.PathLike[str],
    *,
    launch_authority: authority_module.ABC6LaunchAuthority | None = None,
) -> ABC6SyntheticRunResult:
    """Run the source-pinned training/forecast/frozen-score integration once.

    The integrated source gate is deliberately first. At this revision the
    reviewed campaign manifest does not yet pin this runner, the forecast
    module, and the scorer together, so the normal code path fails before the
    campaign claim until a separately reviewed manifest expansion lands.
    """

    if not isinstance(launch_authority, authority_module.ABC6LaunchAuthority):
        raise ABC6SyntheticRunnerPreflightError(
            "direct runner/API invocation requires a watchdog-issued launch grant"
        )
    try:
        launch_authority.consume_runner_startup()
    except authority_module.ABC6AuthorityError as error:
        raise ABC6SyntheticRunnerPreflightError(
            f"runner startup grant is invalid, expired, or already consumed: {error}"
        ) from error
    integrated_source_hashes = _source_pin_check()
    try:
        receipt_text = os.fspath(receipt_directory)
    except TypeError as error:
        raise ABC6SyntheticRunnerPreflightError(
            "receipt directory must equal the fixed schema-v2 checkout path"
        ) from error
    expected_receipt_path = (
        campaign_fit._REPO_ROOT / campaign_fit.RECEIPT_ROOT_RELATIVE
    )
    if (
        type(receipt_text) is not str
        or receipt_text != str(expected_receipt_path)
        or cases.RECEIPT_ROOT_RELATIVE != campaign_fit.RECEIPT_ROOT_RELATIVE
    ):
        raise ABC6SyntheticRunnerPreflightError(
            "receipt directory must equal the fixed schema-v2 checkout path"
        )
    receipt_path = expected_receipt_path

    # Exactly one public claimed campaign call. It performs strict manifest
    # preflight before consuming its O_EXCL run claim.
    execution = campaign_fit.run_abc6_training_campaign_with_evidence(
        manifest_path,
        manifest_sha256,
        receipt_path,
        launch_authority=launch_authority,
    )
    if not isinstance(execution, campaign_fit.ABC6TrainingCampaignExecution):
        raise ABC6SyntheticRunnerIntegrityError(
            "training entry point returned no typed evidence execution"
        )
    receipt_root_identity = _execution_receipt_root_identity(execution)

    # This is the sole advertised path from the execution to mutable numerical
    # evidence. The accessor rehashes its durable artifacts before any forecast.
    verified = execution.load_verified_training_evidence()
    if not isinstance(verified, campaign_fit.ABC6VerifiedTrainingEvidence):
        raise ABC6SyntheticRunnerIntegrityError(
            "training execution did not return verified evidence"
        )
    results = verified.training_results
    if len(results) != cases.CASE_COUNT:
        raise ABC6SyntheticRunnerIntegrityError(
            "verified training evidence is not a complete 24-case roster"
        )

    receipt_hashes, _summary_sha256 = _verify_status_gate(
        receipt_root_identity, execution.campaign_result, results
    )
    forecasts = tuple(
        forecast_module.forecast_abc6_posterior_and_baseline(
            verified.training_bundle.data_for_case(index), result
        )
        for index, result in enumerate(results)
    )
    # Freeze only after every case (including null incomplete records and N's
    # explicit abstention) has been produced. The scorer's validators enforce
    # all-particle retention and suppress aggregates on any particle failure.
    frozen = scoring.freeze_abc6_forecasts(forecasts)
    _verify_forecast_roster(frozen, results)

    payload = _forecast_artifact_payload(
        training_result=execution.campaign_result,
        evidence_manifest_sha256=execution.evidence_manifest_sha256,
        source_hashes=integrated_source_hashes,
        receipt_hashes=receipt_hashes,
        frozen=frozen,
    )
    artifact_path = receipt_root_identity.receipt_root_path / FORECAST_ARTIFACT_FILENAME
    _publish_forecasts(receipt_root_identity, payload)
    artifact_sha256 = _verify_published_forecasts(receipt_root_identity, payload)

    # Recheck all durable receipt and forecast links at the handoff boundary.
    receipt_hashes_now, _summary_sha256_now = _verify_status_gate(
        receipt_root_identity, execution.campaign_result, results
    )
    if receipt_hashes_now != receipt_hashes:
        raise ABC6SyntheticRunnerIntegrityError(
            "status receipts changed after target-free forecasts were frozen"
        )
    _verify_forecast_roster(frozen, results)
    if (
        _verify_published_forecasts(receipt_root_identity, payload)
        != artifact_sha256
    ):
        raise ABC6SyntheticRunnerIntegrityError(
            "target-free forecast artifact digest changed before scoring"
        )

    # No target generator or reveal marker is owned here. The reviewed scorer
    # receives the typed execution and durable forecast artifact, verifies its
    # own evidence/source/receipt chain, claims its one-use marker, hashes target
    # truth, and only then invokes the deferred target generator.
    score = scoring.score_deferred_abc6_synthetic(
        execution,
        frozen,
        artifact_path,
        artifact_sha256,
    )
    return ABC6SyntheticRunResult(
        training_campaign=execution.campaign_result,
        verified_evidence_manifest_sha256=execution.evidence_manifest_sha256,
        forecast_artifact_path=artifact_path,
        forecast_artifact_sha256=artifact_sha256,
        frozen_forecasts=frozen,
        deferred_score=score,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipts", required=True, type=Path)
    args = parser.parse_args(argv)
    launch_authority = _receive_launch_authority()
    try:
        result = run_cascaded_tanks_abc6_synthetic(
            args.manifest,
            args.manifest_sha256,
            args.receipts,
            launch_authority=launch_authority,
        )
    finally:
        launch_authority.close()
    summary = {
        "protocol_id": result.deferred_score.protocol_id,
        "run_id": result.deferred_score.run_id,
        "forecast_artifact": str(result.forecast_artifact_path),
        "forecast_artifact_sha256": result.forecast_artifact_sha256,
        "forecast_array_sha256": result.deferred_score.forecast_array_sha256,
        "target_arrays_artifact": (
            None
            if result.deferred_score.target_arrays_artifact_path is None
            else str(result.deferred_score.target_arrays_artifact_path)
        ),
        "target_arrays_artifact_sha256": (
            result.deferred_score.target_arrays_artifact_sha256
        ),
        "score_result_sha256": result.deferred_score.score_result_sha256,
        "score_event_monotonic_ns": (
            result.deferred_score.score_event_monotonic_ns
        ),
        "score_event_utc": result.deferred_score.score_event_utc,
        "score_receipt": (
            None
            if result.deferred_score.score_receipt_path is None
            else str(result.deferred_score.score_receipt_path)
        ),
        "score_receipt_sha256": result.deferred_score.score_receipt_sha256,
        "target_sha256_by_truth": result.deferred_score.target_sha256_by_truth,
        "case_count": len(result.deferred_score.case_scores),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - explicit command-line execution.
    raise SystemExit(main())

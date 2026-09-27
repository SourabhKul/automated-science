"""All-candidate target-free Cascaded Tanks forecast-completeness gate.

The public entrypoint accepts only source, fit-manifest, fit-receipt, and
comparison-declaration paths. It checks the independently declared ordered
roster against the exact fit manifest and reconciles every terminal ABC
posterior with its final recorded reference population before forecasting.
No development target or official test value is parsed or returned.

The private fixture entrypoint is exclusively for synthetic archive tests. It
requires an explicit synthetic archive expectation and can never issue an
official-source identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core.real_data import cascaded_tanks_controlled as _source
from core.real_data import cascaded_tanks_development_forecast as _forecast
from core.real_data import cascaded_tanks_training_fit as _fit
from core.real_data.cascaded_tanks_controlled import (
    CSV_MEMBER_BYTES,
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_DOI,
    ArchiveExpectation,
    CascadedTanksDevelopmentData,
    SyntheticCascadedTanksDevelopmentData,
)
from core.real_data.cascaded_tanks_development_forecast import (
    CascadedTanksForecastEnsemble,
    CascadedTanksForecastParticle,
)
from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankState,
)

FORECAST_GATE_SCHEMA = "cascaded-tanks-forecast-completeness-receipt-v1"
COMPARISON_SCHEMA = "cascaded-tanks-forecast-comparison-declaration-v1"
TRAIN_STOP = 768
FORECAST_STOP = 1_024
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_DECLARATION_BYTES = 1_000_000
_MAX_RECEIPT_BYTES = 64_000_000
_ALIGNMENT = (
    "y[k] reads state[k] before u[k] transitions to state[k+1]; "
    "training u[767] produces boundary state[768]; development u[768:1024] "
    "produces forecasts y[768:1024] from that carried state"
)
_MANIFEST_KEYS = {
    "schema",
    "protocol_id",
    "run_id",
    "scope",
    "source",
    "runtime",
    "code_sha256",
    "abc_kernel",
    "candidates",
}
_RECEIPT_KEYS = {
    "schema",
    "status",
    "complete",
    "all_declared_candidate_fits_complete",
    "candidate_roster_scope",
    "development_score_eligible",
    "protocol_id",
    "run_id",
    "scope",
    "manifest_sha256",
    "source_receipt_sha256",
    "source",
    "training_fit_payload",
    "training_fit_payload_sha256",
    "candidate_specifications",
    "candidate_specifications_sha256",
    "abc_kernel",
    "abc_kernel_sha256",
    "runtime",
    "runtime_sha256",
    "code_sha256",
    "abc_reference_path",
    "target_access",
    "candidate_results",
    "receipt_sha256",
}
_CANDIDATE_RESULT_KEYS = {
    "candidate_id",
    "model",
    "status",
    "termination_reason",
    "posterior",
    "failure_categories",
    "discrepancy_failure_categories",
    "runner_error",
    "abc_diagnostics",
}
_POSTERIOR_KEYS = {
    "unit_parameters",
    "free_parameter_values",
    "parameter_order",
    "weights",
    "effective_sample_size",
    "distances",
}


class CascadedTanksForecastGateError(ValueError):
    """Raised only by lower-level helpers; the entrypoint returns a failure receipt."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CandidateForecast:
    """Weighted-median forecast for one declared candidate."""

    candidate_id: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ForecastCompletenessOutcome:
    """Gate receipt and promotable forecasts, present only after full success."""

    status: Literal["complete", "terminal_failure"]
    receipt: dict[str, Any]
    forecasts: tuple[CandidateForecast, ...] | None


def run_cascaded_tanks_forecast_gate(
    archive_path: str | os.PathLike[str],
    fit_manifest_path: str | os.PathLike[str],
    fit_receipt_path: str | os.PathLike[str],
    comparison_declaration_path: str | os.PathLike[str],
) -> ForecastCompletenessOutcome:
    """Verify a production fit and forecast every ABC particle target-free.

    Inputs are filesystem paths only. The production path rejects synthetic
    manifests and uses the pinned official archive loader. Any evidence or
    particle failure returns a content-hashed terminal receipt with
    ``forecasts=None``.
    """

    return _run_gate(
        archive_path,
        fit_manifest_path,
        fit_receipt_path,
        comparison_declaration_path,
        expected_archive=None,
        required_scope="official-source",
    )


def _run_synthetic_fixture_forecast_gate(
    archive_path: str | os.PathLike[str],
    fit_manifest_path: str | os.PathLike[str],
    fit_receipt_path: str | os.PathLike[str],
    comparison_declaration_path: str | os.PathLike[str],
    *,
    expected_archive: ArchiveExpectation,
) -> ForecastCompletenessOutcome:
    """Synthetic-only gate seam; never accepts arrays or official identities."""

    if type(expected_archive) is not ArchiveExpectation:
        raise TypeError("expected_archive must be an explicit ArchiveExpectation")
    return _run_gate(
        archive_path,
        fit_manifest_path,
        fit_receipt_path,
        comparison_declaration_path,
        expected_archive=expected_archive,
        required_scope="synthetic-fixture-only",
    )


def _run_gate(
    archive_path: str | os.PathLike[str],
    fit_manifest_path: str | os.PathLike[str],
    fit_receipt_path: str | os.PathLike[str],
    comparison_declaration_path: str | os.PathLike[str],
    *,
    expected_archive: ArchiveExpectation | None,
    required_scope: Literal["official-source", "synthetic-fixture-only"],
) -> ForecastCompletenessOutcome:
    input_hashes: dict[str, str | None] = {
        "fit_manifest_sha256": None,
        "fit_receipt_sha256": None,
        "comparison_declaration_sha256": None,
    }
    context: dict[str, Any] = {
        "scope": required_scope,
        "protocol_id": None,
        "run_id": None,
        "source": None,
        "archive_sha256": None,
        "fit_manifest_sha256": None,
        "fit_receipt_sha256": None,
        "comparison_declaration_sha256": None,
        "code_sha256": _current_code_sha256(),
        "runtime": _fit._current_runtime(),
    }
    partial_candidates: list[dict[str, Any]] = []
    try:
        _require_path(archive_path, "archive_path")
        _require_path(fit_manifest_path, "fit_manifest_path")
        _require_path(fit_receipt_path, "fit_receipt_path")
        _require_path(comparison_declaration_path, "comparison_declaration_path")
        if (required_scope == "official-source") != (expected_archive is None):
            raise CascadedTanksForecastGateError(
                "invalid_source_scope", "source scope and loader do not match"
            )

        manifest_raw, manifest_hash = _read_snapshot(
            fit_manifest_path, _MAX_MANIFEST_BYTES, "fit manifest"
        )
        input_hashes["fit_manifest_sha256"] = manifest_hash
        context["fit_manifest_sha256"] = manifest_hash
        manifest = _parse_fit_manifest(manifest_raw)
        context["protocol_id"] = manifest.protocol_id
        context["run_id"] = manifest.run_id
        context["source"] = manifest.source
        if manifest.scope != required_scope:
            raise CascadedTanksForecastGateError(
                "manifest_scope_mismatch", "fit manifest source scope is not eligible"
            )

        receipt_raw, receipt_hash = _read_snapshot(
            fit_receipt_path, _MAX_RECEIPT_BYTES, "fit receipt"
        )
        input_hashes["fit_receipt_sha256"] = receipt_hash
        context["fit_receipt_sha256"] = receipt_hash
        fit_receipt = _parse_json_document(receipt_raw, "fit receipt")

        declaration_raw, declaration_hash = _read_snapshot(
            comparison_declaration_path,
            _MAX_DECLARATION_BYTES,
            "comparison declaration",
        )
        input_hashes["comparison_declaration_sha256"] = declaration_hash
        context["comparison_declaration_sha256"] = declaration_hash
        declaration = _parse_json_document(declaration_raw, "comparison declaration")

        declaration_roster = _validate_comparison_declaration(
            declaration,
            manifest,
            manifest_hash,
            required_scope=required_scope,
        )
        _validate_fit_receipt_binding(fit_receipt, manifest, manifest_hash)

        data, archive_binding = _load_source_view(
            archive_path,
            expected_archive=expected_archive,
            required_scope=required_scope,
        )
        context["archive_sha256"] = archive_binding["archive_sha256"]
        _fit._validate_source_binding(
            manifest,
            source_kind=required_scope,
            data=data,
            archive=archive_binding,
        )
        _validate_training_payload(fit_receipt, data)
        _validate_fit_populations(fit_receipt, manifest)

        source_kind: Literal["official", "synthetic-fixture-only"] = (
            "official"
            if required_scope == "official-source"
            else "synthetic-fixture-only"
        )
        successful_forecasts: list[CandidateForecast] = []
        for candidate, candidate_result, roster_entry in zip(
            manifest.candidates,
            fit_receipt["candidate_results"],
            declaration_roster,
            strict=True,
        ):
            candidate_result_record, median = _forecast_candidate(
                data,
                candidate,
                candidate_result,
                roster_entry,
                source_kind=source_kind,
            )
            partial_candidates.append(candidate_result_record)
            if median is not None:
                successful_forecasts.append(
                    CandidateForecast(candidate.candidate_id, median)
                )

        all_complete = len(partial_candidates) == len(manifest.candidates) and all(
            item["status"] == "complete" for item in partial_candidates
        )
        receipt = _build_receipt(
            status="complete" if all_complete else "terminal_failure",
            input_hashes=input_hashes,
            context=context,
            candidate_records=partial_candidates,
            failures=[]
            if all_complete
            else [
                {
                    "code": "particle_forecast_incomplete",
                    "message": "at least one declared particle did not produce a complete forecast",
                }
            ],
            comparison=declaration,
        )
        return ForecastCompletenessOutcome(
            status="complete" if all_complete else "terminal_failure",
            receipt=receipt,
            forecasts=tuple(successful_forecasts) if all_complete else None,
        )
    except CascadedTanksForecastGateError as error:
        return _failed_outcome(
            error.code,
            str(error),
            input_hashes=input_hashes,
            context=context,
            candidate_records=partial_candidates,
        )
    except Exception as error:  # noqa: BLE001 - gate failure must never promote a partial result
        return _failed_outcome(
            "gate_validation_error",
            f"{type(error).__name__}: {error}",
            input_hashes=input_hashes,
            context=context,
            candidate_records=partial_candidates,
        )


def _require_path(value: object, label: str) -> None:
    if not isinstance(value, (str, os.PathLike)):
        raise CascadedTanksForecastGateError(
            "path_required", f"{label} must be a filesystem path"
        )


def _read_snapshot(
    path: str | os.PathLike[str], maximum: int, label: str
) -> tuple[bytes, str]:
    """Read once into immutable bytes; the returned digest covers those bytes."""

    try:
        with Path(path).open("rb") as stream:
            initial_size = os.fstat(stream.fileno()).st_size
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise CascadedTanksForecastGateError(
            "artifact_read_failure", f"{label} could not be read"
        ) from error
    if initial_size > maximum or len(raw) > maximum:
        raise CascadedTanksForecastGateError(
            "artifact_size_limit", f"{label} exceeds its bounded size limit"
        )
    return raw, hashlib.sha256(raw).hexdigest()


def _parse_json_document(raw: bytes, label: str) -> Any:
    def parse_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def reject_constant(token: str) -> None:
        raise ValueError(f"invalid JSON constant {token}")

    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CascadedTanksForecastGateError(
            "strict_json_failure",
            f"{label} is not strict UTF-8 JSON without duplicate keys",
        ) from error
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_fit_manifest(raw: bytes) -> _fit._FitManifest:
    decoded = _parse_json_document(raw, "fit manifest")
    if not isinstance(decoded, dict) or set(decoded) != _MANIFEST_KEYS:
        raise CascadedTanksForecastGateError(
            "manifest_schema_failure",
            "fit manifest keys do not match the frozen schema",
        )
    if decoded.get("schema") != _fit.FIT_MANIFEST_SCHEMA:
        raise CascadedTanksForecastGateError(
            "manifest_schema_failure", "fit manifest schema is unsupported"
        )
    try:
        protocol_id = _fit._nonempty_text(decoded.get("protocol_id"), "protocol_id")
        run_id = _fit._nonempty_text(decoded.get("run_id"), "run_id")
        scope = decoded.get("scope")
        if scope not in {"official-source", "synthetic-fixture-only"}:
            raise _fit.CascadedTanksTrainingFitError("fit manifest scope is invalid")
        source = _fit._validate_source_manifest(decoded.get("source"), scope=scope)
        runtime = _fit._validate_runtime_manifest(decoded.get("runtime"))
        code_hashes = _fit._validate_code_manifest(decoded.get("code_sha256"))
        kernel = _fit._validate_kernel(decoded.get("abc_kernel"))
        raw_candidates = decoded.get("candidates")
        if (
            not isinstance(raw_candidates, list)
            or not raw_candidates
            or len(raw_candidates) > 8
        ):
            raise _fit.CascadedTanksTrainingFitError(
                "manifest must declare between 1 and 8 candidates"
            )
        candidates = tuple(_fit._parse_candidate(item) for item in raw_candidates)
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise _fit.CascadedTanksTrainingFitError(
                "candidate_id values must be unique"
            )
        return _fit._FitManifest(
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            protocol_id=protocol_id,
            run_id=run_id,
            scope=scope,
            source=source,
            runtime=runtime,
            code_sha256=code_hashes,
            kernel=kernel,
            candidates=candidates,
        )
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, CascadedTanksForecastGateError):
            raise
        raise CascadedTanksForecastGateError(
            "manifest_validation_failure", str(error)
        ) from error


def _current_code_sha256() -> dict[str, str]:
    root = Path(_fit.__file__).resolve().parents[2]
    paths = {
        "fit_runner": root / "core/real_data/cascaded_tanks_training_fit.py",
        "source_adapter": root / "core/real_data/cascaded_tanks_controlled.py",
        "tank_simulator": root / "core/real_data/cascaded_tanks_models.py",
        "abc_reference": root / "core/abc_smc_reference.py",
        "forecast_runner": root
        / "core/real_data/cascaded_tanks_development_forecast.py",
        "gate_runner": Path(__file__).resolve(),
    }
    result = {name: _sha256_file(path) for name, path in sorted(paths.items())}
    result["forecast_bundle"] = _forecast._code_sha256()
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_comparison_declaration(
    declaration: object,
    manifest: _fit._FitManifest,
    manifest_hash: str,
    *,
    required_scope: str,
) -> list[dict[str, Any]]:
    expected_keys = {
        "schema",
        "protocol_id",
        "run_id",
        "scope",
        "fit_manifest_sha256",
        "source",
        "candidate_roster",
        "forecast",
        "code_sha256",
    }
    if not isinstance(declaration, dict) or set(declaration) != expected_keys:
        raise CascadedTanksForecastGateError(
            "comparison_schema_failure", "comparison declaration keys are invalid"
        )
    if declaration.get("schema") != COMPARISON_SCHEMA:
        raise CascadedTanksForecastGateError(
            "comparison_schema_failure", "comparison declaration schema is unsupported"
        )
    if (
        declaration.get("protocol_id") != manifest.protocol_id
        or declaration.get("run_id") != manifest.run_id
        or declaration.get("scope") != required_scope
        or declaration.get("fit_manifest_sha256") != manifest_hash
        or declaration.get("source") != manifest.source
    ):
        raise CascadedTanksForecastGateError(
            "comparison_binding_mismatch",
            "comparison declaration does not bind this exact manifest and source",
        )
    declared_codes = declaration.get("code_sha256")
    current_codes = _current_code_sha256()
    if declared_codes != current_codes:
        raise CascadedTanksForecastGateError(
            "comparison_code_mismatch",
            "comparison code pins do not match the current fit, source, simulator, ABC, forecast, and gate code",
        )
    expected_forecast = {
        "training_indices": [0, TRAIN_STOP],
        "boundary_state_index": TRAIN_STOP,
        "forecast_indices": [TRAIN_STOP, FORECAST_STOP],
        "forecast_length": FORECAST_STOP - TRAIN_STOP,
        "alignment": _ALIGNMENT,
    }
    if declaration.get("forecast") != expected_forecast:
        raise CascadedTanksForecastGateError(
            "forecast_alignment_mismatch",
            "comparison forecast split or alignment is invalid",
        )
    roster = declaration.get("candidate_roster")
    if not isinstance(roster, list) or len(roster) != len(manifest.candidates):
        raise CascadedTanksForecastGateError(
            "candidate_roster_mismatch",
            "comparison roster must include every fit candidate in order",
        )
    expected_roster: list[dict[str, Any]] = []
    for candidate in manifest.candidates:
        expected_roster.append(
            {
                "candidate_id": candidate.candidate_id,
                "model": candidate.model.value,
                "expected_particle_count": candidate.target_samples,
                "simulation_limits": {
                    "max_steps": candidate.simulation_limits.max_steps,
                    "max_magnitude": candidate.simulation_limits.max_magnitude,
                },
            }
        )
    if roster != expected_roster:
        raise CascadedTanksForecastGateError(
            "candidate_roster_mismatch",
            "comparison roster IDs, families, counts, limits, or order differ from the fit manifest",
        )
    return expected_roster


def _validate_fit_receipt_binding(
    receipt: object, manifest: _fit._FitManifest, manifest_hash: str
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise CascadedTanksForecastGateError(
            "fit_receipt_schema_failure", "fit receipt keys do not match its schema"
        )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _fit._canonical_sha256(unsigned):
        raise CascadedTanksForecastGateError(
            "fit_receipt_hash_mismatch", "fit receipt content hash is invalid"
        )
    expected_scope = manifest.scope
    source_receipt = {"scope": expected_scope, "source": manifest.source}
    if (
        receipt.get("schema") != _fit.FIT_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("complete") is not True
        or receipt.get("all_declared_candidate_fits_complete") is not True
        or receipt.get("candidate_roster_scope") != "manifest_declared_candidates_only"
        or receipt.get("development_score_eligible") is not False
        or receipt.get("scope") != expected_scope
        or receipt.get("protocol_id") != manifest.protocol_id
        or receipt.get("run_id") != manifest.run_id
        or receipt.get("manifest_sha256") != manifest_hash
        or receipt.get("source") != manifest.source
        or receipt.get("source_receipt_sha256")
        != _fit._canonical_sha256(source_receipt)
        or receipt.get("runtime") != manifest.runtime
        or receipt.get("runtime_sha256") != _fit._canonical_sha256(manifest.runtime)
        or receipt.get("code_sha256") != manifest.code_sha256
        or receipt.get("abc_kernel") != manifest.kernel
        or receipt.get("abc_kernel_sha256") != _fit._canonical_sha256(manifest.kernel)
        or receipt.get("abc_reference_path") != "gaussian_abc_smc_reference_opt_in"
    ):
        raise CascadedTanksForecastGateError(
            "fit_receipt_binding_mismatch",
            "fit receipt status, source, manifest, runtime, code, or ABC pins do not match",
        )
    if receipt.get("target_access") != {
        "training_y_est_materialized": TRAIN_STOP,
        "development_y_est_materialized": 0,
        "u_val_materialized": 0,
        "y_val_materialized": 0,
    }:
        raise CascadedTanksForecastGateError(
            "fit_target_boundary_mismatch",
            "fit receipt target-access declaration is invalid",
        )
    specifications = [candidate.declaration() for candidate in manifest.candidates]
    if receipt.get("candidate_specifications") != specifications or receipt.get(
        "candidate_specifications_sha256"
    ) != _fit._canonical_sha256(specifications):
        raise CascadedTanksForecastGateError(
            "fit_candidate_specification_mismatch",
            "fit receipt candidate specifications differ from the exact manifest roster",
        )
    results = receipt.get("candidate_results")
    if not isinstance(results, list) or len(results) != len(manifest.candidates):
        raise CascadedTanksForecastGateError(
            "fit_candidate_result_mismatch", "fit receipt result roster is incomplete"
        )
    for candidate, result in zip(manifest.candidates, results, strict=True):
        if (
            not isinstance(result, dict)
            or set(result) != _CANDIDATE_RESULT_KEYS
            or result.get("candidate_id") != candidate.candidate_id
            or result.get("model") != candidate.model.value
            or result.get("status") != "complete"
            or result.get("termination_reason") != "completed"
            or result.get("posterior") is None
            or result.get("runner_error") is not None
        ):
            raise CascadedTanksForecastGateError(
                "fit_candidate_result_mismatch",
                "fit receipt has a missing, failed, or reordered candidate result",
            )


def _load_source_view(
    archive_path: str | os.PathLike[str],
    *,
    expected_archive: ArchiveExpectation | None,
    required_scope: str,
) -> tuple[
    CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    dict[str, Any],
]:
    if required_scope == "official-source":
        data = _source.load_development_data(archive_path)
        if type(data) is not CascadedTanksDevelopmentData:
            raise CascadedTanksForecastGateError(
                "source_view_type_mismatch",
                "official source loader returned an unexpected view",
            )
        if data.archive_sha256 != SOURCE_ARCHIVE_SHA256 or SOURCE_ARCHIVE_BYTES <= 0:
            raise CascadedTanksForecastGateError(
                "source_identity_mismatch", "official archive identity is not pinned"
            )
        return data, {
            "archive_bytes": SOURCE_ARCHIVE_BYTES,
            "archive_sha256": data.archive_sha256,
            "csv_member": _source.CSV_MEMBER,
            "csv_member_bytes": CSV_MEMBER_BYTES,
            "source_doi": SOURCE_DOI,
            "source_version": "1",
        }
    if type(expected_archive) is not ArchiveExpectation:
        raise CascadedTanksForecastGateError(
            "synthetic_archive_expectation_required",
            "synthetic fixture path requires an explicit archive expectation",
        )
    data = _source._load_fixture_development_data(
        archive_path, expected_archive=expected_archive
    )
    if type(data) is not SyntheticCascadedTanksDevelopmentData:
        raise CascadedTanksForecastGateError(
            "source_view_type_mismatch", "fixture loader returned an unexpected view"
        )
    return data, {
        "archive_bytes": expected_archive.archive_bytes,
        "archive_sha256": expected_archive.archive_sha256,
        "csv_member": expected_archive.csv_member,
        "csv_member_bytes": CSV_MEMBER_BYTES,
        "source_doi": "synthetic-fixture-only",
        "source_version": "synthetic-fixture-only",
    }


def _validate_training_payload(
    fit_receipt: dict[str, Any],
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
) -> None:
    inputs = np.asarray(data.training_u_est, dtype=np.float64)
    observed = np.asarray(data.training_y_est, dtype=np.float64)
    expected = {
        "training_indices": [0, TRAIN_STOP],
        "input_sha256": _fit._float_array_sha256(inputs),
        "observed_output_sha256": _fit._float_array_sha256(observed),
        "length": TRAIN_STOP,
        "discrepancy": "all_trajectory_rmse",
    }
    if fit_receipt.get("training_fit_payload") != expected or fit_receipt.get(
        "training_fit_payload_sha256"
    ) != _fit._canonical_sha256(expected):
        raise CascadedTanksForecastGateError(
            "training_payload_mismatch",
            "fit receipt training payload does not bind the loaded [0,768) arrays",
        )


def _validate_fit_populations(
    receipt: dict[str, Any], manifest: _fit._FitManifest
) -> None:
    for candidate, candidate_result in zip(
        manifest.candidates, receipt["candidate_results"], strict=True
    ):
        diagnostics = candidate_result.get("abc_diagnostics")
        if not isinstance(diagnostics, dict) or not _fit._verify_complete_reference(
            diagnostics, candidate
        ):
            raise CascadedTanksForecastGateError(
                "abc_population_invalid",
                f"candidate {candidate.candidate_id!r} lacks a verified final ABC population",
            )
        posterior = candidate_result.get("posterior")
        if not isinstance(posterior, dict) or set(posterior) != _POSTERIOR_KEYS:
            raise CascadedTanksForecastGateError(
                "posterior_schema_failure",
                f"candidate {candidate.candidate_id!r} posterior schema is invalid",
            )
        if posterior.get("parameter_order") != list(candidate.parameter_order):
            raise CascadedTanksForecastGateError(
                "posterior_decoder_mismatch",
                f"candidate {candidate.candidate_id!r} parameter order differs from its manifest decoder",
            )
        final_population = diagnostics["populations"][-1]
        units = _numeric_array(
            posterior.get("unit_parameters"),
            (candidate.target_samples, len(candidate.parameter_order)),
            "unit_parameters",
        )
        diagnostic_units = _numeric_array(
            final_population.get("accepted_params"),
            (candidate.target_samples, len(candidate.parameter_order)),
            "final accepted_params",
        )
        weights = _numeric_array(
            posterior.get("weights"), (candidate.target_samples,), "posterior weights"
        )
        diagnostic_weights = _numeric_array(
            final_population.get("weights"),
            (candidate.target_samples,),
            "final weights",
        )
        distances = _numeric_array(
            posterior.get("distances"),
            (candidate.target_samples,),
            "posterior distances",
        )
        diagnostic_distances = _numeric_array(
            final_population.get("distances"),
            (candidate.target_samples,),
            "final distances",
        )
        if not (
            np.array_equal(units, diagnostic_units)
            and np.array_equal(weights, diagnostic_weights)
            and np.array_equal(distances, diagnostic_distances)
        ):
            raise CascadedTanksForecastGateError(
                "posterior_population_mismatch",
                f"candidate {candidate.candidate_id!r} posterior arrays differ from the final ABC population",
            )
        expected_free = {
            name: [low + float(point[index]) * (high - low) for point in units]
            for index, (name, (low, high)) in enumerate(
                candidate.free_parameter_bounds.items()
            )
        }
        free_values = posterior.get("free_parameter_values")
        if free_values != expected_free:
            raise CascadedTanksForecastGateError(
                "posterior_decoder_mismatch",
                f"candidate {candidate.candidate_id!r} physical values do not match ordered manifest bounds",
            )
        expected_ess = 1.0 / float(np.sum(weights * weights))
        ess = _finite_number(
            posterior.get("effective_sample_size"), "effective_sample_size"
        )
        if not math.isclose(ess, expected_ess, rel_tol=1e-10, abs_tol=1e-12):
            raise CascadedTanksForecastGateError(
                "posterior_ess_mismatch",
                f"candidate {candidate.candidate_id!r} effective sample size is inconsistent",
            )


def _numeric_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CascadedTanksForecastGateError(
            "posterior_array_invalid", f"{label} is not a numeric array"
        ) from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise CascadedTanksForecastGateError(
            "posterior_array_invalid",
            f"{label} has an invalid shape or non-finite value",
        )
    return array


def _forecast_candidate(
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    candidate: _fit._CandidateSpec,
    candidate_result: dict[str, Any],
    roster_entry: dict[str, Any],
    *,
    source_kind: Literal["official", "synthetic-fixture-only"],
) -> tuple[dict[str, Any], tuple[float, ...] | None]:
    posterior = candidate_result["posterior"]
    units = _numeric_array(
        posterior["unit_parameters"],
        (candidate.target_samples, len(candidate.parameter_order)),
        "unit_parameters",
    )
    weights = _numeric_array(
        posterior["weights"], (candidate.target_samples,), "posterior weights"
    )
    free_values = posterior["free_parameter_values"]
    particles: list[CascadedTanksForecastParticle] = []
    decoded_hashes: list[str] = []
    for index in range(candidate.target_samples):
        values = dict(candidate.fixed_parameters)
        for name in candidate.parameter_order:
            values[name] = float(free_values[name][index])
        _validate_decoded_parameters(candidate, values, index)
        particle = CascadedTanksForecastParticle(
            model=candidate.model,
            parameters=TankParameters(values["a"], values["c"], values["p"]),
            initial_state=TankState(values["x1_0"], values["x2_0"]),
            ceiling=values.get("ceiling"),
            weight=float(weights[index]),
        )
        particles.append(particle)
        decoded_hashes.append(
            _fit._canonical_sha256(
                {
                    "model": candidate.model.value,
                    "decoded_parameters": values,
                    "unit_parameters": [float(value) for value in units[index]],
                }
            )
        )
    ensemble = CascadedTanksForecastEnsemble(tuple(particles))
    outcome = _forecast._forecast_view(
        data,
        ensemble,
        candidate.simulation_limits,
        source_kind=(
            "official" if source_kind == "official" else "synthetic-fixture-only"
        ),
    )
    _validate_forecast_receipt(
        outcome, data, candidate, particles, source_kind=source_kind
    )
    forecast_receipt = asdict(outcome.receipt)
    particle_rows = []
    if len(outcome.receipt.particle_receipts) != candidate.target_samples:
        raise CascadedTanksForecastGateError(
            "forecast_receipt_incomplete",
            "forecast path omitted a declared particle receipt",
        )
    for index, particle_receipt in enumerate(outcome.receipt.particle_receipts):
        if particle_receipt.particle_index != index or particle_receipt.weight != float(
            weights[index]
        ):
            raise CascadedTanksForecastGateError(
                "forecast_particle_binding_mismatch",
                "forecast path changed a stable particle index or ABC weight",
            )
        particle_rows.append(
            {
                "identity": {
                    "candidate_id": candidate.candidate_id,
                    "final_population_index": index,
                },
                "weight": float(weights[index]),
                "decoded_parameter_sha256": decoded_hashes[index],
                "status": "complete"
                if particle_receipt.forecast_sha256 is not None
                and particle_receipt.development_status == "complete"
                else "terminal_failure",
                "forecast_sha256": particle_receipt.forecast_sha256,
                "boundary_state_sha256": particle_receipt.boundary_state_sha256,
                "training_status": particle_receipt.training_status,
                "development_status": particle_receipt.development_status,
            }
        )
    median: tuple[float, ...] | None = None
    status = "complete"
    if (
        outcome.status != "complete"
        or outcome.forecast is None
        or len(outcome.forecast) != FORECAST_STOP - TRAIN_STOP
        or not all(math.isfinite(value) for value in outcome.forecast)
        or outcome.receipt.weighted_median_forecast_sha256 is None
    ):
        status = "terminal_failure"
    else:
        median = tuple(float(value) for value in outcome.forecast)
        expected_hash = _forecast._sequence_sha256(median, start_index=TRAIN_STOP)
        if expected_hash != outcome.receipt.weighted_median_forecast_sha256:
            status = "terminal_failure"
            median = None
    record = {
        "candidate_id": candidate.candidate_id,
        "model": candidate.model.value,
        "expected_particle_count": roster_entry["expected_particle_count"],
        "status": status,
        "original_weights": [float(value) for value in weights],
        "particle_results": particle_rows,
        "weighted_median_forecast_sha256": (
            outcome.receipt.weighted_median_forecast_sha256
            if status == "complete"
            else None
        ),
        "development_forecast_receipt": forecast_receipt,
        "failures": [asdict(item) for item in outcome.receipt.failures],
    }
    return record, median


def _validate_forecast_receipt(
    outcome: Any,
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    candidate: _fit._CandidateSpec,
    particles: list[CascadedTanksForecastParticle],
    *,
    source_kind: Literal["official", "synthetic-fixture-only"],
) -> None:
    receipt = outcome.receipt
    forecast_source_kind = (
        "official" if source_kind == "official" else "synthetic-fixture-only"
    )
    expected_parameter_hash = _forecast._canonical_json_hash(
        {
            "schema": "cascaded-tanks-supplied-ensemble-v1",
            "particles": [
                {"index": index, **_forecast._particle_parameters(particle)}
                for index, particle in enumerate(particles)
            ],
        }
    )
    expected_models = tuple(
        (index, candidate.model.value) for index in range(len(particles))
    )
    expected_fields = (
        receipt.schema == _forecast.FORECAST_SCHEMA,
        receipt.status == outcome.status,
        receipt.source_kind == forecast_source_kind,
        receipt.track_id == data.track_id,
        receipt.contract_sha256 == data.contract_sha256,
        receipt.archive_sha256 == getattr(data, "archive_sha256", None),
        receipt.source_visible_sha256 == data.source_visible_sha256,
        receipt.training_stage_sha256 == data.stage_receipts.training_sha256,
        receipt.forecast_inputs_stage_sha256
        == data.stage_receipts.forecast_inputs_sha256,
        receipt.source_receipt_sha256
        == _forecast._source_receipt_sha256(data, forecast_source_kind),
        receipt.ensemble_parameter_sha256 == expected_parameter_hash,
        receipt.code_sha256 == _forecast._code_sha256(),
        receipt.sample_interval_seconds == _source.SAMPLE_INTERVAL_SECONDS,
        receipt.alignment == _ALIGNMENT,
        receipt.training_indices == (0, TRAIN_STOP),
        receipt.boundary_state_index == TRAIN_STOP,
        receipt.forecast_indices == (TRAIN_STOP, FORECAST_STOP),
        receipt.model_alignment == expected_models,
        receipt.simulation_limits
        == (
            candidate.simulation_limits.max_steps,
            float(candidate.simulation_limits.max_magnitude),
        ),
        receipt.target_sha256 is None,
        len(receipt.particle_receipts) == len(particles),
    )
    if not all(expected_fields):
        raise CascadedTanksForecastGateError(
            "forecast_receipt_binding_mismatch",
            f"candidate {candidate.candidate_id!r} forecast receipt source, code, split, limits, or ensemble binding is invalid",
        )
    expected_failed_particles: set[tuple[int, str]] = set()
    for index, (particle, particle_receipt) in enumerate(
        zip(particles, receipt.particle_receipts, strict=True)
    ):
        parameter_hash = _forecast._canonical_json_hash(
            _forecast._particle_parameters(particle)
        )
        if (
            particle_receipt.particle_index != index
            or particle_receipt.model != candidate.model.value
            or particle_receipt.weight != particle.weight
            or particle_receipt.parameter_sha256 != parameter_hash
            or particle_receipt.boundary_state_index != TRAIN_STOP
            or particle_receipt.forecast_indices != (TRAIN_STOP, FORECAST_STOP)
        ):
            raise CascadedTanksForecastGateError(
                "forecast_particle_binding_mismatch",
                f"candidate {candidate.candidate_id!r} particle {index} receipt does not preserve its ABC identity",
            )
        if (
            particle_receipt.training_status == "complete"
            and particle_receipt.development_status == "complete"
            and _is_sha256(particle_receipt.forecast_sha256)
            and _is_sha256(particle_receipt.boundary_state_sha256)
        ):
            continue
        if particle_receipt.training_status == "failed":
            expected_failed_particles.add((index, "training"))
            if (
                particle_receipt.development_status != "not_run"
                or particle_receipt.forecast_sha256 is not None
                or particle_receipt.boundary_state_sha256 is not None
            ):
                raise CascadedTanksForecastGateError(
                    "forecast_particle_status_invalid",
                    f"candidate {candidate.candidate_id!r} failed training receipt is inconsistent",
                )
        elif particle_receipt.training_status == "complete":
            expected_failed_particles.add((index, "development"))
            if (
                particle_receipt.development_status != "failed"
                or particle_receipt.forecast_sha256 is not None
                or not _is_sha256(particle_receipt.boundary_state_sha256)
            ):
                raise CascadedTanksForecastGateError(
                    "forecast_particle_status_invalid",
                    f"candidate {candidate.candidate_id!r} failed development receipt is inconsistent",
                )
        else:
            raise CascadedTanksForecastGateError(
                "forecast_particle_status_invalid",
                f"candidate {candidate.candidate_id!r} particle status is invalid",
            )
    observed_failures = {
        (failure.particle_index, failure.phase) for failure in receipt.failures
    }
    if observed_failures != expected_failed_particles or len(receipt.failures) != len(
        expected_failed_particles
    ):
        raise CascadedTanksForecastGateError(
            "forecast_failure_diagnostics_invalid",
            f"candidate {candidate.candidate_id!r} failure diagnostics do not match its particle receipts",
        )
    should_be_complete = not expected_failed_particles
    if receipt.status == "complete":
        if (
            not should_be_complete
            or outcome.forecast is None
            or len(outcome.forecast) != FORECAST_STOP - TRAIN_STOP
            or not all(math.isfinite(value) for value in outcome.forecast)
            or not _is_sha256(receipt.weighted_median_forecast_sha256)
        ):
            raise CascadedTanksForecastGateError(
                "forecast_completeness_invalid",
                f"candidate {candidate.candidate_id!r} claims completion without all finite forecasts",
            )
    elif (
        should_be_complete
        or outcome.forecast is not None
        or receipt.weighted_median_forecast_sha256 is not None
    ):
        raise CascadedTanksForecastGateError(
            "forecast_completeness_invalid",
            f"candidate {candidate.candidate_id!r} failure status and forecasts are inconsistent",
        )


def _validate_decoded_parameters(
    candidate: _fit._CandidateSpec, values: dict[str, float], index: int
) -> None:
    required = {"a", "c", "p", "x1_0", "x2_0"}
    if candidate.model is not TankModel.S0:
        required.add("ceiling")
    if set(values) != required:
        raise CascadedTanksForecastGateError(
            "posterior_decoder_mismatch",
            f"candidate {candidate.candidate_id!r} particle {index} has an incomplete physical decoder",
        )
    if any(not math.isfinite(value) for value in values.values()):
        raise CascadedTanksForecastGateError(
            "posterior_decoder_mismatch",
            f"candidate {candidate.candidate_id!r} particle {index} decodes to a non-finite parameter",
        )


def _build_receipt(
    *,
    status: Literal["complete", "terminal_failure"],
    input_hashes: dict[str, str | None],
    context: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": FORECAST_GATE_SCHEMA,
        "status": status,
        "complete": status == "complete",
        "development_score_eligible": False,
        "target_access": {
            "development_y_est_materialized": 0,
            "u_val_materialized": 0,
            "y_val_materialized": 0,
        },
        "target_sha256": None,
        "scope": context["scope"],
        "protocol_id": comparison.get("protocol_id")
        if comparison
        else context.get("protocol_id"),
        "run_id": comparison.get("run_id") if comparison else context.get("run_id"),
        "comparison_declaration_sha256": input_hashes["comparison_declaration_sha256"],
        "fit_manifest_sha256": input_hashes["fit_manifest_sha256"],
        "fit_receipt_sha256": input_hashes["fit_receipt_sha256"],
        "source": context.get("source"),
        "archive_sha256": context.get("archive_sha256"),
        "source_stage_sha256": (
            context["source"].get("stage_sha256")
            if isinstance(context.get("source"), dict)
            else None
        ),
        "forecast": comparison.get("forecast") if comparison else None,
        "candidate_roster": comparison.get("candidate_roster") if comparison else None,
        "runtime": context.get("runtime"),
        "code_sha256": context.get("code_sha256"),
        "candidate_results": candidate_records,
        "failures": failures,
    }
    receipt["receipt_sha256"] = _fit._canonical_sha256(receipt)
    return receipt


def _failed_outcome(
    code: str,
    message: str,
    *,
    input_hashes: dict[str, str | None],
    context: dict[str, Any],
    candidate_records: list[dict[str, Any]],
) -> ForecastCompletenessOutcome:
    receipt = _build_receipt(
        status="terminal_failure",
        input_hashes=input_hashes,
        context=context,
        candidate_records=candidate_records,
        failures=[{"code": code, "message": message}],
        comparison=None,
    )
    return ForecastCompletenessOutcome("terminal_failure", receipt, None)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise CascadedTanksForecastGateError(
            "numeric_value_invalid", f"{label} must be a finite real number"
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise CascadedTanksForecastGateError(
            "numeric_value_invalid", f"{label} must be a finite real number"
        ) from error
    if not math.isfinite(number):
        raise CascadedTanksForecastGateError(
            "numeric_value_invalid", f"{label} must be finite"
        )
    return number


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

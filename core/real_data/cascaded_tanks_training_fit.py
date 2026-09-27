"""Source-gated, estimation-training-only Cascaded Tanks ABC-SMC fit.

The production entrypoint accepts only archive and frozen-manifest paths. It
loads the pinned source through ``cascaded_tanks_controlled`` and sends only
``uEst/yEst[0:768]`` to the checked Gaussian ABC-SMC reference path. There is
no development-target, official-test, forecast, or caller-array interface.

The private fixture entrypoint exists solely for synthetic archive tests and
cannot produce an official-source receipt. This module deliberately chooses
no real-data priors, tolerances, seeds, or budgets: every numeric setting must
be present in the caller-frozen manifest and is included in the receipt.
Receipt completeness covers only the candidates declared in that manifest;
the receipt is never eligible for development scoring. A separately frozen
protocol/scorer must verify that its required candidate roster is present.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core.abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference
from core.real_data import cascaded_tanks_controlled as _source
from core.real_data.cascaded_tanks_controlled import (
    CONTRACT_SHA256,
    CSV_MEMBER,
    CSV_MEMBER_BYTES,
    SAMPLE_INTERVAL_SECONDS,
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_DOI,
    SYNTHETIC_FIXTURE_CONTRACT_SHA256,
    SYNTHETIC_FIXTURE_TRACK_ID,
    TRACK_ID,
    ArchiveExpectation,
    CascadedTanksDevelopmentData,
    SyntheticCascadedTanksDevelopmentData,
    _load_fixture_development_data,
    _preflight_fixture_archive,
)
from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

FIT_MANIFEST_SCHEMA = "cascaded-tanks-training-fit-manifest-v1"
FIT_RECEIPT_SCHEMA = "cascaded-tanks-training-fit-receipt-v1"
TRAIN_STOP = 768
_MAX_MANIFEST_BYTES = 1_000_000
_CODE_FILES = {
    "fit_runner": "core/real_data/cascaded_tanks_training_fit.py",
    "source_adapter": "core/real_data/cascaded_tanks_controlled.py",
    "tank_simulator": "core/real_data/cascaded_tanks_models.py",
    "abc_reference": "core/abc_smc_reference.py",
}
_BASE_PARAMETER_NAMES = ("a", "c", "p", "x1_0", "x2_0")
_CEILING_MODELS = (TankModel.O2, TankModel.C2)
_SHA256_LENGTH = 64


class CascadedTanksTrainingFitError(ValueError):
    """A manifest, source view, or fit receipt failed closed validation."""


class DuplicateManifestKeyError(CascadedTanksTrainingFitError):
    """A JSON manifest repeated an object key."""


class _TankFitSimulationFailure(RuntimeError):
    """A simulator trajectory failed in a counted ABC proposal."""


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    candidate_id: str
    model: TankModel
    fixed_parameters: dict[str, float]
    free_parameter_bounds: dict[str, tuple[float, float]]
    target_samples: int
    epsilon_schedule: tuple[float, ...]
    max_attempts_per_population: tuple[int, ...]
    seed: int
    output_units: str
    simulation_limits: TankSimulationLimits
    discrepancy: str

    @property
    def parameter_order(self) -> tuple[str, ...]:
        return tuple(self.free_parameter_bounds)

    def declaration(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model.value,
            "fixed_parameters": dict(self.fixed_parameters),
            "free_parameter_bounds": {
                name: list(bounds)
                for name, bounds in self.free_parameter_bounds.items()
            },
            "target_samples": self.target_samples,
            "epsilon_schedule": list(self.epsilon_schedule),
            "max_attempts_per_population": list(self.max_attempts_per_population),
            "seed": self.seed,
            "output_units": self.output_units,
            "simulation_limits": {
                "max_steps": self.simulation_limits.max_steps,
                "max_magnitude": self.simulation_limits.max_magnitude,
            },
            "discrepancy": self.discrepancy,
        }


@dataclass(frozen=True, slots=True)
class _FitManifest:
    raw_sha256: str
    protocol_id: str
    run_id: str
    scope: Literal["official-source", "synthetic-fixture-only"]
    source: dict[str, Any]
    runtime: dict[str, str]
    code_sha256: dict[str, str]
    kernel: dict[str, float]
    candidates: tuple[_CandidateSpec, ...]


def run_cascaded_tanks_training_fit(
    archive_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Fit declared candidates to official ``uEst/yEst[0:768]`` only.

    A caller-frozen JSON manifest is mandatory. Its source identity, stage
    receipts, runtime versions, code hashes, numeric ABC settings, and
    candidate specifications must match the current loader and runtime.
    Caller-supplied arrays and externally supplied ABC populations are not
    accepted. ``complete`` means every manifest-declared candidate completed;
    it does not certify an expected roster and never grants score eligibility.
    """

    _reject_array_arguments(archive_path, manifest_path)
    manifest = _read_manifest(manifest_path)
    if manifest.scope != "official-source":
        raise CascadedTanksTrainingFitError(
            "production fit requires an official-source manifest"
        )
    preflight = _source.preflight_archive(archive_path)
    data = _source.load_development_data(archive_path)
    if type(data) is not CascadedTanksDevelopmentData:
        raise CascadedTanksTrainingFitError(
            "production fit requires data returned by the pinned official loader"
        )
    # Both adapter operations independently verify the pinned archive digest.
    # Bind their results explicitly so a path swap between preflight and load
    # cannot splice metadata from one source identity onto another view.
    if (
        preflight.archive_sha256 != data.archive_sha256
        or preflight.archive_bytes != SOURCE_ARCHIVE_BYTES
        or preflight.archive_sha256 != SOURCE_ARCHIVE_SHA256
    ):
        raise CascadedTanksTrainingFitError(
            "preflight and training view do not bind the same pinned archive bytes"
        )
    _validate_source_binding(
        manifest,
        source_kind="official-source",
        data=data,
        archive={
            "archive_bytes": preflight.archive_bytes,
            "archive_sha256": preflight.archive_sha256,
            "csv_member": preflight.csv_member,
            "csv_member_bytes": preflight.csv_member_bytes,
            "source_doi": preflight.source_doi,
            "source_version": preflight.source_version,
        },
    )
    return _fit_training_view(manifest, data, source_kind="official-source")


def _run_synthetic_fixture_training_fit(
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_archive: ArchiveExpectation,
) -> dict[str, Any]:
    """Private synthetic-only fixture seam; never identifies as production."""

    if type(expected_archive) is not ArchiveExpectation:
        raise TypeError("expected_archive must be ArchiveExpectation")
    manifest = _read_manifest(manifest_path)
    if manifest.scope != "synthetic-fixture-only":
        raise CascadedTanksTrainingFitError(
            "synthetic fixture fit requires a synthetic-fixture-only manifest"
        )
    preflight = _preflight_fixture_archive(
        archive_path, expected_archive=expected_archive
    )
    data = _load_fixture_development_data(
        archive_path, expected_archive=expected_archive
    )
    if type(data) is not SyntheticCascadedTanksDevelopmentData:
        raise CascadedTanksTrainingFitError(
            "fixture fit requires the private synthetic loader result"
        )
    _validate_source_binding(
        manifest,
        source_kind="synthetic-fixture-only",
        data=data,
        archive={
            "archive_bytes": preflight.archive_bytes,
            "archive_sha256": preflight.archive_sha256,
            "csv_member": preflight.csv_member,
            "csv_member_bytes": preflight.csv_member_bytes,
            "source_doi": "synthetic-fixture-only",
            "source_version": "synthetic-fixture-only",
        },
    )
    return _fit_training_view(manifest, data, source_kind="synthetic-fixture-only")


def _read_manifest(path: str | Path) -> _FitManifest:
    if not isinstance(path, (str, Path)):
        raise TypeError("manifest_path must be a filesystem path")
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise CascadedTanksTrainingFitError("fit manifest exceeds the size limit")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, DuplicateManifestKeyError):
            raise
        raise CascadedTanksTrainingFitError(
            "fit manifest is not strict UTF-8 JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise CascadedTanksTrainingFitError("fit manifest must be one JSON object")
    expected = {
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
    if set(decoded) != expected:
        raise CascadedTanksTrainingFitError(
            "fit manifest keys do not match the frozen schema"
        )
    if decoded.get("schema") != FIT_MANIFEST_SCHEMA:
        raise CascadedTanksTrainingFitError("fit manifest schema is not supported")
    protocol_id = _nonempty_text(decoded.get("protocol_id"), "protocol_id")
    run_id = _nonempty_text(decoded.get("run_id"), "run_id")
    scope = decoded.get("scope")
    if scope not in {"official-source", "synthetic-fixture-only"}:
        raise CascadedTanksTrainingFitError("fit manifest scope is invalid")
    source = _validate_source_manifest(decoded.get("source"), scope=scope)
    runtime = _validate_runtime_manifest(decoded.get("runtime"))
    code_sha256 = _validate_code_manifest(decoded.get("code_sha256"))
    kernel = _validate_kernel(decoded.get("abc_kernel"))
    raw_candidates = decoded.get("candidates")
    if (
        not isinstance(raw_candidates, list)
        or not raw_candidates
        or len(raw_candidates) > 8
    ):
        raise CascadedTanksTrainingFitError(
            "manifest must declare between 1 and 8 candidates"
        )
    candidates = tuple(_parse_candidate(item) for item in raw_candidates)
    candidate_ids = [item.candidate_id for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CascadedTanksTrainingFitError("candidate_id values must be unique")
    return _FitManifest(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        protocol_id=protocol_id,
        run_id=run_id,
        scope=scope,
        source=source,
        runtime=runtime,
        code_sha256=code_sha256,
        kernel=kernel,
        candidates=candidates,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateManifestKeyError(f"fit manifest repeats key {key!r}")
        result[key] = value
    return result


def _validate_source_manifest(value: object, *, scope: str) -> dict[str, Any]:
    keys = {
        "track_id",
        "contract_sha256",
        "archive_sha256",
        "archive_bytes",
        "csv_member",
        "csv_member_bytes",
        "training_indices",
        "stage_sha256",
        "source_doi",
        "source_version",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise CascadedTanksTrainingFitError(
            "manifest source identity has an invalid schema"
        )
    expected_track = (
        TRACK_ID if scope == "official-source" else SYNTHETIC_FIXTURE_TRACK_ID
    )
    expected_contract = (
        CONTRACT_SHA256
        if scope == "official-source"
        else SYNTHETIC_FIXTURE_CONTRACT_SHA256
    )
    if (
        value.get("track_id") != expected_track
        or value.get("contract_sha256") != expected_contract
    ):
        raise CascadedTanksTrainingFitError(
            "manifest track or contract identity is invalid"
        )
    _require_sha256(value.get("archive_sha256"), "source archive SHA-256")
    _require_sha256(value.get("contract_sha256"), "source contract SHA-256")
    _positive_integer(value.get("archive_bytes"), "source archive byte count")
    _positive_integer(value.get("csv_member_bytes"), "source CSV member byte count")
    if value.get("csv_member") != CSV_MEMBER:
        raise CascadedTanksTrainingFitError(
            "manifest CSV member does not match the source contract"
        )
    indices = value.get("training_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != 2
        or any(type(index) is not int for index in indices)
        or indices != [0, TRAIN_STOP]
    ):
        raise CascadedTanksTrainingFitError(
            "manifest training split must be exactly [0, 768)"
        )
    stage = value.get("stage_sha256")
    if not isinstance(stage, dict) or set(stage) != {
        "source_visible_sha256",
        "training_sha256",
        "forecast_inputs_sha256",
    }:
        raise CascadedTanksTrainingFitError("manifest stage hash set is incomplete")
    for label, digest in stage.items():
        _require_sha256(digest, f"stage {label}")
    expected_doi = (
        SOURCE_DOI if scope == "official-source" else "synthetic-fixture-only"
    )
    expected_version = "1" if scope == "official-source" else "synthetic-fixture-only"
    if (
        value.get("source_doi") != expected_doi
        or value.get("source_version") != expected_version
    ):
        raise CascadedTanksTrainingFitError("manifest source DOI or version is invalid")
    if scope == "official-source" and (
        value["archive_sha256"] != SOURCE_ARCHIVE_SHA256
        or value["archive_bytes"] != SOURCE_ARCHIVE_BYTES
        or value["csv_member_bytes"] != CSV_MEMBER_BYTES
    ):
        raise CascadedTanksTrainingFitError(
            "manifest does not pin the official archive contract"
        )
    return {
        **value,
        "stage_sha256": dict(stage),
    }


def _validate_runtime_manifest(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"python", "numpy", "platform"}:
        raise CascadedTanksTrainingFitError(
            "manifest runtime identity has an invalid schema"
        )
    result: dict[str, str] = {}
    for name, item in value.items():
        result[name] = _nonempty_text(item, f"runtime.{name}")
    return result


def _validate_code_manifest(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(_CODE_FILES):
        raise CascadedTanksTrainingFitError(
            "manifest code hash set does not match the fit seam"
        )
    for name, digest in value.items():
        _require_sha256(digest, f"code hash {name}")
    return dict(value)


def _validate_kernel(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {
        "covariance_scale",
        "lambda_noise",
        "nugget",
    }:
        raise CascadedTanksTrainingFitError(
            "manifest ABC kernel settings are incomplete"
        )
    result = {name: _finite_real(value[name], f"abc_kernel.{name}") for name in value}
    if result["covariance_scale"] <= 0.0:
        raise CascadedTanksTrainingFitError(
            "abc_kernel.covariance_scale must be positive"
        )
    if result["lambda_noise"] < 0.0 or result["nugget"] < 0.0:
        raise CascadedTanksTrainingFitError(
            "ABC kernel noise and nugget must be non-negative"
        )
    return result


def _parse_candidate(value: object) -> _CandidateSpec:
    keys = {
        "candidate_id",
        "model",
        "fixed_parameters",
        "free_parameter_bounds",
        "target_samples",
        "epsilon_schedule",
        "max_attempts_per_population",
        "seed",
        "output_units",
        "simulation_limits",
        "discrepancy",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise CascadedTanksTrainingFitError(
            "candidate specification has an invalid schema"
        )
    candidate_id = _nonempty_text(value.get("candidate_id"), "candidate_id")
    try:
        model = TankModel(value.get("model"))
    except (TypeError, ValueError) as error:
        raise CascadedTanksTrainingFitError(
            "candidate model must be S0, O2, or C2"
        ) from error
    fixed_raw = value.get("fixed_parameters")
    free_raw = value.get("free_parameter_bounds")
    if not isinstance(fixed_raw, dict) or not isinstance(free_raw, dict):
        raise CascadedTanksTrainingFitError(
            "candidate parameter declarations must be objects"
        )
    expected_parameters = set(_BASE_PARAMETER_NAMES)
    if model in _CEILING_MODELS:
        expected_parameters.add("ceiling")
    fixed: dict[str, float] = {}
    free: dict[str, tuple[float, float]] = {}
    for name, raw in fixed_raw.items():
        if name not in expected_parameters:
            raise CascadedTanksTrainingFitError(f"unknown fixed parameter {name!r}")
        fixed[name] = _finite_real(raw, f"fixed_parameters.{name}")
    for name, raw in free_raw.items():
        if name not in expected_parameters or name in fixed:
            raise CascadedTanksTrainingFitError(f"invalid free parameter {name!r}")
        if not isinstance(raw, list) or len(raw) != 2:
            raise CascadedTanksTrainingFitError(
                f"free parameter {name!r} needs two bounds"
            )
        low, high = (
            _finite_real(raw[0], f"{name}.low"),
            _finite_real(raw[1], f"{name}.high"),
        )
        if high <= low:
            raise CascadedTanksTrainingFitError(
                f"free parameter {name!r} bounds must increase"
            )
        free[name] = (low, high)
    if set(fixed) | set(free) != expected_parameters or not free:
        raise CascadedTanksTrainingFitError(
            "candidate must declare exactly its model parameters"
        )
    for name, number in fixed.items():
        if name in {"a", "c", "p"} and number <= 0.0:
            raise CascadedTanksTrainingFitError(
                f"fixed parameter {name!r} must be positive"
            )
        if name in {"x1_0", "x2_0", "ceiling"} and number < 0.0:
            raise CascadedTanksTrainingFitError(
                f"fixed parameter {name!r} must be non-negative"
            )
    for name, (low, _) in free.items():
        if name in {"a", "c", "p"} and low <= 0.0:
            raise CascadedTanksTrainingFitError(
                f"free parameter {name!r} must be positive"
            )
        if name in {"x1_0", "x2_0", "ceiling"} and low < 0.0:
            raise CascadedTanksTrainingFitError(
                f"free parameter {name!r} must be non-negative"
            )
    if model is TankModel.O2:
        highest_x2 = fixed.get("x2_0", free.get("x2_0", (0.0, 0.0))[1])
        lowest_h = fixed.get("ceiling", free.get("ceiling", (0.0, 0.0))[0])
        if highest_x2 > lowest_h:
            raise CascadedTanksTrainingFitError(
                "all O2 prior points must satisfy x2_0 <= ceiling"
            )

    target_samples = _positive_integer(value.get("target_samples"), "target_samples")
    if target_samples > 4096:
        raise CascadedTanksTrainingFitError(
            "target_samples exceeds the guarded manifest limit"
        )
    seed = _nonnegative_integer(value.get("seed"), "seed")
    if seed > 2**32 - 1:
        raise CascadedTanksTrainingFitError("seed exceeds the guarded 32-bit range")
    eps_raw = value.get("epsilon_schedule")
    if not isinstance(eps_raw, list) or not eps_raw or len(eps_raw) > 32:
        raise CascadedTanksTrainingFitError(
            "epsilon_schedule must contain 1 to 32 values"
        )
    epsilons = tuple(_finite_real(item, "epsilon_schedule") for item in eps_raw)
    if any(item < 0.0 for item in epsilons) or any(
        epsilons[index] > epsilons[index - 1] for index in range(1, len(epsilons))
    ):
        raise CascadedTanksTrainingFitError(
            "epsilon_schedule must be finite, non-negative, and non-increasing"
        )
    budget_raw = value.get("max_attempts_per_population")
    if isinstance(budget_raw, int) and not isinstance(budget_raw, bool):
        budgets = (budget_raw,) * len(epsilons)
    elif isinstance(budget_raw, list) and len(budget_raw) == len(epsilons):
        budgets = tuple(
            _positive_integer(item, "attempt budget") for item in budget_raw
        )
    else:
        raise CascadedTanksTrainingFitError(
            "one positive attempt budget is required per epsilon"
        )
    if any(item > 1_000_000 for item in budgets):
        raise CascadedTanksTrainingFitError(
            "attempt budget exceeds the guarded manifest limit"
        )
    units = _nonempty_text(value.get("output_units"), "output_units")
    discrepancy = value.get("discrepancy")
    if discrepancy != "all_trajectory_rmse":
        raise CascadedTanksTrainingFitError(
            "only all_trajectory_rmse is supported by this fit seam"
        )
    limits_raw = value.get("simulation_limits")
    if not isinstance(limits_raw, dict) or set(limits_raw) != {
        "max_steps",
        "max_magnitude",
    }:
        raise CascadedTanksTrainingFitError("simulation_limits declaration is invalid")
    max_steps = _positive_integer(
        limits_raw["max_steps"], "simulation_limits.max_steps"
    )
    if max_steps > 100_000:
        raise CascadedTanksTrainingFitError(
            "simulation_limits.max_steps exceeds the guarded limit"
        )
    max_magnitude = _finite_real(
        limits_raw["max_magnitude"], "simulation_limits.max_magnitude"
    )
    if max_magnitude <= 0.0:
        raise CascadedTanksTrainingFitError(
            "simulation_limits.max_magnitude must be positive"
        )
    return _CandidateSpec(
        candidate_id=candidate_id,
        model=model,
        fixed_parameters=fixed,
        free_parameter_bounds=free,
        target_samples=target_samples,
        epsilon_schedule=epsilons,
        max_attempts_per_population=budgets,
        seed=seed,
        output_units=units,
        simulation_limits=TankSimulationLimits(max_steps, max_magnitude),
        discrepancy=discrepancy,
    )


def _validate_source_binding(
    manifest: _FitManifest,
    *,
    source_kind: Literal["official-source", "synthetic-fixture-only"],
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    archive: dict[str, Any],
) -> None:
    if manifest.scope != source_kind:
        raise CascadedTanksTrainingFitError(
            "manifest and loaded source scope do not match"
        )
    if type(data) is CascadedTanksDevelopmentData:
        expected_track, expected_contract = TRACK_ID, CONTRACT_SHA256
        archive_sha256 = data.archive_sha256
        if archive_sha256 != SOURCE_ARCHIVE_SHA256:
            raise CascadedTanksTrainingFitError(
                "loader view has an unexpected archive identity"
            )
    elif type(data) is SyntheticCascadedTanksDevelopmentData:
        expected_track, expected_contract = (
            SYNTHETIC_FIXTURE_TRACK_ID,
            SYNTHETIC_FIXTURE_CONTRACT_SHA256,
        )
        archive_sha256 = archive["archive_sha256"]
        if data.fixture_marker != "synthetic-fixture-only":
            raise CascadedTanksTrainingFitError("synthetic source marker is missing")
    else:
        raise CascadedTanksTrainingFitError("source view type is not recognized")
    if data.track_id != expected_track or data.contract_sha256 != expected_contract:
        raise CascadedTanksTrainingFitError(
            "loaded source track or contract identity is invalid"
        )
    if data.sample_interval_seconds != SAMPLE_INTERVAL_SECONDS:
        raise CascadedTanksTrainingFitError(
            "loaded source interval differs from the contract"
        )
    if data.training_indices != tuple(range(TRAIN_STOP)):
        raise CascadedTanksTrainingFitError(
            "loaded training indices must be exactly [0, 768)"
        )
    if len(data.training_u_est) != TRAIN_STOP or len(data.training_y_est) != TRAIN_STOP:
        raise CascadedTanksTrainingFitError(
            "loader must return exactly 768 training pairs"
        )
    if any(
        not math.isfinite(float(item))
        for item in (*data.training_u_est, *data.training_y_est)
    ):
        raise CascadedTanksTrainingFitError(
            "loader returned a non-finite training value"
        )
    if any(float(item) < 0.0 for item in data.training_u_est):
        raise CascadedTanksTrainingFitError(
            "training inputs must be non-negative for this simulator"
        )
    if not _is_sha256(data.source_visible_sha256):
        raise CascadedTanksTrainingFitError("source-visible digest is malformed")
    actual_stages = {
        "source_visible_sha256": data.source_visible_sha256,
        "training_sha256": data.stage_receipts.training_sha256,
        "forecast_inputs_sha256": data.stage_receipts.forecast_inputs_sha256,
    }
    expected_source = manifest.source
    actual_source = {
        "track_id": data.track_id,
        "contract_sha256": data.contract_sha256,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive["archive_bytes"],
        "csv_member": archive["csv_member"],
        "csv_member_bytes": archive["csv_member_bytes"],
        "training_indices": [0, TRAIN_STOP],
        "stage_sha256": actual_stages,
        "source_doi": archive["source_doi"],
        "source_version": archive["source_version"],
    }
    if expected_source != actual_source:
        raise CascadedTanksTrainingFitError(
            "manifest source identity or stage hashes do not match loader output"
        )
    actual_runtime = _current_runtime()
    if manifest.runtime != actual_runtime:
        raise CascadedTanksTrainingFitError(
            "manifest runtime versions do not match this process"
        )
    actual_code = _current_code_sha256()
    if manifest.code_sha256 != actual_code:
        raise CascadedTanksTrainingFitError(
            "manifest code hashes do not match the fit implementation"
        )


def _fit_training_view(
    manifest: _FitManifest,
    data: CascadedTanksDevelopmentData | SyntheticCascadedTanksDevelopmentData,
    *,
    source_kind: Literal["official-source", "synthetic-fixture-only"],
) -> dict[str, Any]:
    # These are the only data arrays captured by the simulator/discrepancy.
    # The source view's development inputs are deliberately not forwarded.
    inputs = np.asarray(data.training_u_est, dtype=np.float64)
    observed = np.asarray(data.training_y_est, dtype=np.float64)
    if inputs.shape != (TRAIN_STOP,) or observed.shape != (TRAIN_STOP,):
        raise CascadedTanksTrainingFitError(
            "fit payload must be exactly the training prefix"
        )
    payload = {
        "training_indices": [0, TRAIN_STOP],
        "input_sha256": _float_array_sha256(inputs),
        "observed_output_sha256": _float_array_sha256(observed),
        "length": TRAIN_STOP,
        "discrepancy": "all_trajectory_rmse",
    }
    candidate_results = [
        _fit_one_candidate(candidate, inputs, observed, manifest.kernel)
        for candidate in manifest.candidates
    ]
    all_complete = all(result["status"] == "complete" for result in candidate_results)
    if not all_complete:
        # A partial candidate set is not eligible for downstream selection.
        # Keep all full/partial ABC evidence, while removing every posterior
        # from the failed batch so none can be promoted independently.
        for result in candidate_results:
            result["posterior"] = None
    source_receipt = {
        "scope": source_kind,
        "source": manifest.source,
    }
    receipt_payload: dict[str, Any] = {
        "schema": FIT_RECEIPT_SCHEMA,
        "status": "complete" if all_complete else "terminal_failure",
        "complete": all_complete,
        "all_declared_candidate_fits_complete": all_complete,
        "candidate_roster_scope": "manifest_declared_candidates_only",
        "development_score_eligible": False,
        "protocol_id": manifest.protocol_id,
        "run_id": manifest.run_id,
        "scope": source_kind,
        "manifest_sha256": manifest.raw_sha256,
        "source_receipt_sha256": _canonical_sha256(source_receipt),
        "source": manifest.source,
        "training_fit_payload": payload,
        "training_fit_payload_sha256": _canonical_sha256(payload),
        "candidate_specifications": [
            item.declaration() for item in manifest.candidates
        ],
        "candidate_specifications_sha256": _canonical_sha256(
            [item.declaration() for item in manifest.candidates]
        ),
        "abc_kernel": manifest.kernel,
        "abc_kernel_sha256": _canonical_sha256(manifest.kernel),
        "runtime": _current_runtime(),
        "runtime_sha256": _canonical_sha256(_current_runtime()),
        "code_sha256": _current_code_sha256(),
        "abc_reference_path": "gaussian_abc_smc_reference_opt_in",
        "target_access": {
            "training_y_est_materialized": TRAIN_STOP,
            "development_y_est_materialized": 0,
            "u_val_materialized": 0,
            "y_val_materialized": 0,
        },
        "candidate_results": candidate_results,
    }
    receipt_payload["receipt_sha256"] = _canonical_sha256(receipt_payload)
    return _json_safe(receipt_payload)


def _fit_one_candidate(
    candidate: _CandidateSpec,
    inputs: np.ndarray,
    observed: np.ndarray,
    kernel: dict[str, float],
) -> dict[str, Any]:
    order = candidate.parameter_order
    unit_bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (len(order), 1))
    prior_sampler, prior_logpdf = make_uniform_prior(unit_bounds)
    failures: Counter[str] = Counter()
    discrepancy_failures: Counter[str] = Counter()

    def in_support(point: np.ndarray) -> bool:
        return (
            point.shape == (len(order),)
            and bool(np.all(np.isfinite(point)))
            and bool(np.all(point >= 0.0) and np.all(point <= 1.0))
        )

    def simulator(point: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        categorized = False
        try:
            parameters = _decode(point, candidate)
            outcome = simulate_cascaded_tanks(
                inputs,
                TankParameters(parameters["a"], parameters["c"], parameters["p"]),
                TankState(parameters["x1_0"], parameters["x2_0"]),
                model=candidate.model,
                ceiling=parameters.get("ceiling"),
                limits=candidate.simulation_limits,
            )
            if isinstance(outcome, TankSimulationFailure):
                failures[outcome.category.value] += 1
                categorized = True
                raise _TankFitSimulationFailure(
                    f"tank simulator failed: {outcome.category.value}"
                )
            if not isinstance(outcome, TankSimulationSuccess):
                failures["invalid_simulator_result"] += 1
                categorized = True
                raise _TankFitSimulationFailure(
                    "tank simulator returned an unknown result type"
                )
            generated = np.asarray(outcome.observations, dtype=np.float64)
            if generated.shape != observed.shape or not np.all(np.isfinite(generated)):
                failures["invalid_trajectory"] += 1
                categorized = True
                raise RuntimeError(
                    "tank simulator did not return a full finite trajectory"
                )
            return generated
        except Exception:
            if not categorized:
                failures["unexpected_exception"] += 1
            raise

    def discrepancy(generated: Any) -> float:
        prediction = np.asarray(generated, dtype=np.float64)
        if prediction.shape != observed.shape or not np.all(np.isfinite(prediction)):
            discrepancy_failures["invalid_trajectory"] += 1
            raise ValueError("discrepancy requires a full finite training trajectory")
        distance = float(np.sqrt(np.mean(np.square(prediction - observed))))
        if not math.isfinite(distance):
            discrepancy_failures["non_finite_rmse"] += 1
            raise ValueError("training RMSE became non-finite")
        return distance

    reference: dict[str, Any] | None = None
    runner_error: dict[str, str] | None = None
    try:
        reference = run_gaussian_abc_smc_reference(
            prior_sampler,
            simulator,
            discrepancy,
            target_samples=candidate.target_samples,
            epsilon_schedule=candidate.epsilon_schedule,
            max_attempts_per_population=candidate.max_attempts_per_population,
            bounds=unit_bounds,
            in_support=in_support,
            prior_logpdf=prior_logpdf,
            covariance_scale=kernel["covariance_scale"],
            lambda_noise=kernel["lambda_noise"],
            nugget=kernel["nugget"],
            seed=candidate.seed,
        )
    except Exception as error:  # noqa: BLE001 - any runner error is terminal evidence, never a partial posterior
        runner_error = {"type": type(error).__name__, "message": str(error)}

    try:
        verified_complete = reference is not None and _verify_complete_reference(
            reference, candidate
        )
    except Exception:  # noqa: BLE001 - malformed reference output must fail closed
        verified_complete = False
    posterior: dict[str, Any] | None = None
    if verified_complete:
        assert reference is not None
        accepted = np.asarray(reference["accepted_params"], dtype=float)
        posterior = {
            "unit_parameters": accepted,
            "free_parameter_values": {
                name: np.asarray(
                    [
                        candidate.free_parameter_bounds[name][0]
                        + float(point[index])
                        * (
                            candidate.free_parameter_bounds[name][1]
                            - candidate.free_parameter_bounds[name][0]
                        )
                        for point in accepted
                    ],
                    dtype=float,
                )
                for index, name in enumerate(order)
            },
            "parameter_order": list(order),
            "weights": np.asarray(reference["weights"], dtype=float),
            "effective_sample_size": float(reference["effective_sample_size"]),
            "distances": np.asarray(reference["distances"], dtype=float),
        }
    reason = (
        "complete"
        if verified_complete
        else (
            "reference_runner_exception"
            if runner_error is not None
            else "incomplete_or_unverified_abc_population"
        )
    )
    return {
        "candidate_id": candidate.candidate_id,
        "model": candidate.model.value,
        "status": "complete" if verified_complete else "terminal_failure",
        "termination_reason": (
            reference.get("termination_reason") if reference is not None else reason
        ),
        "posterior": posterior,
        "failure_categories": dict(sorted(failures.items())),
        "discrepancy_failure_categories": dict(sorted(discrepancy_failures.items())),
        "runner_error": runner_error,
        "abc_diagnostics": _json_safe(reference) if reference is not None else None,
    }


def _verify_complete_reference(
    reference: dict[str, Any], candidate: _CandidateSpec
) -> bool:
    """Accept only the exact complete evidence shape returned by the reference path."""

    if not isinstance(reference, dict):
        return False
    if reference.get("reference_path") != "gaussian_abc_smc_reference_opt_in":
        return False
    if reference.get("canonical_runner_integrated") is not False:
        return False
    if reference.get("status") != "complete" or reference.get("complete") is not True:
        return False
    if reference.get("termination_reason") != "completed":
        return False
    if reference.get("target_samples") != candidate.target_samples:
        return False
    if reference.get("epsilon_schedule") != list(candidate.epsilon_schedule):
        return False
    if reference.get("max_attempts_per_population") != list(
        candidate.max_attempts_per_population
    ):
        return False
    populations = reference.get("populations")
    if not isinstance(populations, list) or len(populations) != len(
        candidate.epsilon_schedule
    ):
        return False
    dimension = len(candidate.parameter_order)
    for index, population in enumerate(populations):
        if not isinstance(population, dict):
            return False
        diagnostics = population.get("diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get("complete") is not True:
            return False
        if diagnostics.get("accepted") != candidate.target_samples:
            return False
        if diagnostics.get("termination_reason") != "target_reached":
            return False
        params = np.asarray(population.get("accepted_params"), dtype=float)
        weights = np.asarray(population.get("weights"), dtype=float)
        distances = np.asarray(population.get("distances"), dtype=float)
        if params.shape != (candidate.target_samples, dimension):
            return False
        if weights.shape != (candidate.target_samples,) or distances.shape != (
            candidate.target_samples,
        ):
            return False
        if (
            not np.all(np.isfinite(params))
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(distances))
        ):
            return False
        if np.any(weights < 0.0) or not np.isclose(
            np.sum(weights), 1.0, rtol=1e-10, atol=1e-12
        ):
            return False
        if np.any(params < 0.0) or np.any(params > 1.0):
            return False
        if np.any(distances > candidate.epsilon_schedule[index]):
            return False
        if (
            population.get("generation") != index
            or population.get("epsilon") != candidate.epsilon_schedule[index]
        ):
            return False
    final_population = populations[-1]
    if reference.get("diagnostics") != final_population.get("diagnostics"):
        return False
    accepted = np.asarray(reference.get("accepted_params"), dtype=float)
    weights = np.asarray(reference.get("weights"), dtype=float)
    distances = np.asarray(reference.get("distances"), dtype=float)
    ess = _finite_real(
        reference.get("effective_sample_size"), "reference effective_sample_size"
    )
    expected_ess = (
        1.0 / float(np.sum(weights * weights))
        if weights.shape == (candidate.target_samples,)
        else float("nan")
    )
    return bool(
        accepted.shape == (candidate.target_samples, dimension)
        and weights.shape == (candidate.target_samples,)
        and distances.shape == (candidate.target_samples,)
        and np.all(np.isfinite(accepted))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(distances))
        and np.all(weights >= 0.0)
        and np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-12)
        and np.array_equal(
            accepted, np.asarray(final_population["accepted_params"], dtype=float)
        )
        and np.array_equal(
            weights, np.asarray(final_population["weights"], dtype=float)
        )
        and np.array_equal(
            distances, np.asarray(final_population["distances"], dtype=float)
        )
        and math.isfinite(ess)
        and math.isclose(ess, expected_ess, rel_tol=1e-10, abs_tol=1e-12)
        and math.isclose(
            ess,
            float(final_population.get("effective_sample_size")),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    )


def _decode(point: np.ndarray, candidate: _CandidateSpec) -> dict[str, float]:
    result = dict(candidate.fixed_parameters)
    for index, name in enumerate(candidate.parameter_order):
        low, high = candidate.free_parameter_bounds[name]
        result[name] = low + float(point[index]) * (high - low)
    return result


def _current_runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _current_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: _sha256_file(root / relative)
        for name, relative in sorted(_CODE_FILES.items())
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"cascaded-tanks-training-fit-float64-v1\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "nonfinite_float": "nan"
            if math.isnan(value)
            else ("positive_infinity" if value > 0 else "negative_infinity")
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"value of type {type(value).__name__} cannot be represented in a fit receipt"
    )


def _require_sha256(value: object, label: str) -> None:
    if not _is_sha256(value):
        raise CascadedTanksTrainingFitError(
            f"{label} must be a lowercase SHA-256 digest"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise CascadedTanksTrainingFitError(f"{label} must be a finite JSON number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise CascadedTanksTrainingFitError(
            f"{label} must be a finite JSON number"
        ) from error
    if not math.isfinite(result):
        raise CascadedTanksTrainingFitError(f"{label} must be finite")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CascadedTanksTrainingFitError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CascadedTanksTrainingFitError(f"{label} must be a non-negative integer")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadedTanksTrainingFitError(f"{label} must be non-empty text")
    return value


def _reject_array_arguments(archive_path: object, manifest_path: object) -> None:
    # Retained as a small explicit guard for callers that try to treat the
    # path-based interface as the caller-array synthetic wrapper.
    if isinstance(archive_path, (np.ndarray, list, tuple)) or isinstance(
        manifest_path, (np.ndarray, list, tuple)
    ):
        raise TypeError(
            "training fit accepts paths and a frozen manifest, never caller arrays"
        )


__all__ = [
    "FIT_MANIFEST_SCHEMA",
    "FIT_RECEIPT_SCHEMA",
    "CascadedTanksTrainingFitError",
    "run_cascaded_tanks_training_fit",
]

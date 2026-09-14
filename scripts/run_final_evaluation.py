#!/usr/bin/env python
"""Evaluate a frozen canonical trajectory selection on sealed final data.

This command has no proposal, repair, parameter-fitting, or selection path.
It loads a frozen candidate and parameter map, reads a manifest that contains
only the final role, and writes a terminal receipt on both success and
failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.domain_configs import DOMAIN_CONFIGS
from core.evaluation import masked_metrics
from core.evaluation_boundary import (
    CANONICAL_PROTOCOL_VERSION,
    build_final_evaluation_receipt,
    frozen_parameters_hash,
    load_partition_manifest,
    parameter_semantics_from_code,
    sha256_json,
    sha256_text,
)
from core.generated_code import validate_generated_math_code
from core.sandbox_eval import build_candidate_model


def _write_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _parameter_vector(model, parameter_map: dict[str, Any]) -> np.ndarray:
    metadata = model.get_parameter_metadata()
    names = [str(prior.get("name", f"param_{index}")) for index, prior in enumerate(metadata)]
    missing = [name for name in names if name not in parameter_map]
    if missing:
        raise ValueError(f"frozen selection is missing parameters: {missing}")
    return np.asarray([float(parameter_map[name]) for name in names], dtype=float)


def run_frozen_final_evaluation(
    *,
    selection_path: str | Path,
    final_manifest_path: str | Path,
    domain: str,
    output_path: str | Path,
) -> dict[str, Any]:
    selection_file = Path(selection_path)
    selection: dict[str, Any] = json.loads(selection_file.read_text())
    model_code_sha256 = str(selection.get("candidate_code_sha256") or "")
    parameters_sha256 = str(selection.get("parameters_sha256") or "")
    prediction = None
    metrics = None
    failure = None
    status = "failed"
    final_loaded = None
    development_receipt = None
    protocol_receipt: dict[str, Any] = {
        "protocol_version": selection.get("evaluation_protocol"),
        "development_receipt_sha256": selection.get("development_receipt_sha256"),
        "roles": {},
    }
    try:
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"unknown domain: {domain}")
        if selection.get("evaluation_protocol") != CANONICAL_PROTOCOL_VERSION:
            raise ValueError("selection is not a canonical trajectory freeze")
        if selection.get("domain") not in {None, domain}:
            raise ValueError("frozen selection domain does not match final evaluation domain")
        if not selection.get("development_receipt_sha256"):
            raise ValueError("frozen selection is missing the development protocol binding")
        if "development_split_metadata" not in selection:
            raise ValueError("frozen selection is missing split-design metadata")
        if selection.get("final_evaluation_required") is not True:
            raise ValueError("selection does not require frozen final evaluation")
        contract = selection.get("model_contract") or {}
        config = DOMAIN_CONFIGS[domain]
        if contract.get("initial_conditions") is not None and contract.get("initial_conditions") != list(config["y0"]):
            raise ValueError("frozen initial conditions do not match the domain contract")
        if contract.get("solver_dt0") is not None and float(contract["solver_dt0"]) != float(config["dt0"]):
            raise ValueError("frozen solver dt0 does not match the domain contract")
        if contract.get("solver_max_steps") is not None and int(contract["solver_max_steps"]) != int(config["max_steps"]):
            raise ValueError("frozen solver max_steps does not match the domain contract")
        if contract.get("time_origin_policy") not in {None, "declared_manifest_time_points"}:
            raise ValueError("unsupported frozen time-origin policy")
        time_origin = contract.get("time_origin")
        if time_origin is None or not np.isfinite(float(time_origin)):
            raise ValueError("frozen selection is missing a finite time origin")
        if not selection.get("parameter_semantics_sha256") or selection.get("parameter_semantics") is None:
            raise ValueError("frozen selection is missing parameter semantics binding")
        development_manifest_value = selection.get("protocol_manifest")
        if not development_manifest_value:
            raise ValueError("frozen selection is missing the development manifest binding")
        development_manifest = Path(development_manifest_value)
        if not development_manifest.is_absolute():
            development_manifest = selection_file.parent / development_manifest
        development_loaded = load_partition_manifest(
            development_manifest,
            required_roles=("train", "validation"),
            forbidden_roles=("final",),
        )
        development_receipt = development_loaded["receipt"]
        if development_receipt.get("development_receipt_sha256") != selection["development_receipt_sha256"]:
            raise ValueError("development manifest receipt does not match frozen selection")
        final_loaded = load_partition_manifest(
            final_manifest_path,
            required_roles=("final",),
            forbidden_roles=("train", "validation"),
        )
        final_part = final_loaded["roles"]["final"]
        protocol_receipt["protocol_version"] = final_loaded["protocol_version"]
        protocol_receipt["roles"] = {"final": final_part.receipt}
        embedded_development_receipt = final_loaded.get("development_receipt")
        if embedded_development_receipt != development_receipt:
            raise ValueError("sealed final manifest is not bound to the frozen development receipt")
        development_metadata = selection.get("development_split_metadata") or {}
        final_metadata = final_loaded.get("metadata") or {}
        for key in ("split_kind", "train_fraction", "validation_fraction", "final_fraction"):
            if key in development_metadata and final_metadata.get(key) != development_metadata[key]:
                raise ValueError(f"sealed manifest split metadata does not match frozen protocol: {key}")
        code_path_value = selection.get("candidate_code_path")
        if not code_path_value:
            raise ValueError("selection is missing candidate_code_path")
        code_path = Path(code_path_value)
        if not code_path.is_absolute():
            code_path = selection_file.parent / code_path
        code = code_path.read_text()
        actual_code_hash = sha256_text(code)
        if actual_code_hash != model_code_sha256:
            raise ValueError("frozen candidate code hash does not match selection receipt")
        validate_generated_math_code(code)
        semantics = parameter_semantics_from_code(code)
        semantics_hash = sha256_json(semantics)
        if selection.get("parameter_semantics_sha256") not in {None, semantics_hash}:
            raise ValueError("frozen parameter semantics hash does not match candidate code")
        if selection.get("parameter_semantics") is not None and selection.get("parameter_semantics") != semantics:
            raise ValueError("frozen parameter semantics do not match candidate code")
        parameter_map = selection.get("parameters")
        if not isinstance(parameter_map, dict):
            raise ValueError("selection parameters must be a frozen name/value mapping")
        if frozen_parameters_hash(parameter_map) != parameters_sha256:
            raise ValueError("frozen parameter hash does not match selection receipt")
        model = build_candidate_model(code, DOMAIN_CONFIGS[domain])
        parameter_vector = _parameter_vector(model, parameter_map)
        final_times = np.asarray(final_part.time_points, dtype=float)
        if np.any(final_times < float(time_origin)):
            raise ValueError("sealed final times precede the frozen time origin")
        # Integrate from the frozen original t0/y0. A later final timestamp is
        # never treated as a fresh initial condition, including a single-point
        # final partition.
        if final_times[0] == float(time_origin):
            integration_times = final_times
            final_offset = 0
        else:
            integration_times = np.concatenate([[float(time_origin)], final_times])
            final_offset = 1
        integrated_prediction = np.asarray(
            model.simulate(parameter_vector, integration_times, model.get_initial_conditions()),
            dtype=float,
        )
        prediction = integrated_prediction[final_offset:]
        if prediction.shape != final_part.observations.shape:
            raise ValueError(
                f"frozen model produced shape {prediction.shape}, expected {final_part.observations.shape}"
            )
        if not np.all(np.isfinite(prediction)):
            raise ValueError("frozen final prediction is non-finite")
        with np.errstate(over="ignore", invalid="ignore"):
            scored = masked_metrics(final_part.observations, prediction, final_part.observation_mask)
        metrics = {
            "mse": float(scored.mse),
            "rmse": float(scored.rmse),
            "observed_count": int(scored.observed_count),
        }
        status = "success"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = "failed"
    receipt = build_final_evaluation_receipt(
        protocol_receipt=protocol_receipt,
        model_code_sha256=model_code_sha256,
        parameters_sha256=parameters_sha256,
        prediction=prediction,
        metrics=metrics,
        status=status,
        failure=failure,
    )
    receipt["selection_path"] = str(selection_file)
    receipt["final_manifest_path"] = str(final_manifest_path)
    receipt["domain"] = domain
    if final_loaded is not None:
        receipt["final_manifest_sha256"] = final_loaded["receipt"].get("manifest_sha256")
        receipt["development_data"] = development_receipt.get("roles") if development_receipt else None
    receipt["model_contract"] = selection.get("model_contract")
    receipt["protocol_binding"] = {
        "development_receipt_sha256": selection.get("development_receipt_sha256"),
        "development_split_metadata": selection.get("development_split_metadata"),
    }
    receipt["receipt_sha256"] = sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    _write_receipt(output_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen canonical trajectory model on sealed final data.")
    parser.add_argument("--selection", required=True, help="Frozen selection JSON emitted by canonical run_domain.")
    parser.add_argument("--final-manifest", required=True, help="Sealed final manifest containing only the final role.")
    parser.add_argument("--domain", required=True, help="Domain name used to construct the frozen evaluator.")
    parser.add_argument("--output", required=True, help="Final evaluation receipt JSON path.")
    args = parser.parse_args()
    receipt = run_frozen_final_evaluation(
        selection_path=args.selection,
        final_manifest_path=args.final_manifest,
        domain=args.domain,
        output_path=args.output,
    )
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

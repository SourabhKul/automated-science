#!/usr/bin/env python3
"""Serial, bounded driver for the six-cell numerical ODE control.

This driver makes no LLM request.  It records a read-only local model probe
only for provenance, runs the six predeclared decay cells serially, and writes
an atomic per-cell receipt plus an atomic partial aggregate after every cell.
The global wall budget is deliberately capped at 600 seconds by argument
validation and defaults to 540 seconds.  A cell stopped by that deadline is
returned as an incomplete reference with no posterior-like summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ode_reference_controls import (  # noqa: E402
    ODESolverConfig,
    generate_decay_observation,
    run_ode_decay_control_cell,
    solver_accuracy_check,
)


REQUESTED_MODEL = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
DEFAULT_MODELS_ENDPOINT = "http://127.0.0.1:8000/v1/models"
DEFAULT_TARGET_SAMPLES = 400
DEFAULT_MAX_ATTEMPTS = 50_000
DEFAULT_WALL_BUDGET_SECONDS = 540.0
EPSILON_SCHEDULE = (3.0, 2.0)
OBSERVATION_TIMES = (0.5, 1.0, 2.0, 3.0)
PRIOR_BOUNDS = ((0.05, 1.5),)
SIGMA = 0.05

# These are the only stochastic streams used by the pilot.  They are explicit
# data, proposal, and simulator-noise seeds, predeclared before any cell runs.
PILOT_CELLS = (
    {"cell_id": "k035_data11", "true_k": 0.35, "data_seed": 11, "proposal_seed": 10011, "simulator_noise_seed": 20011},
    {"cell_id": "k035_data29", "true_k": 0.35, "data_seed": 29, "proposal_seed": 10029, "simulator_noise_seed": 20029},
    {"cell_id": "k035_data47", "true_k": 0.35, "data_seed": 47, "proposal_seed": 10047, "simulator_noise_seed": 20047},
    {"cell_id": "k080_data11", "true_k": 0.80, "data_seed": 11, "proposal_seed": 11011, "simulator_noise_seed": 21011},
    {"cell_id": "k080_data29", "true_k": 0.80, "data_seed": 29, "proposal_seed": 11029, "simulator_noise_seed": 21029},
    {"cell_id": "k080_data47", "true_k": 0.80, "data_seed": 47, "proposal_seed": 11047, "simulator_noise_seed": 21047},
)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resource_snapshot() -> dict[str, object]:
    disk = shutil.disk_usage(ROOT)
    try:
        load_average = [float(item) for item in os.getloadavg()]
    except (AttributeError, OSError):
        load_average = None
    try:
        physical_memory = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        physical_memory = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": load_average,
        "physical_memory_bytes": physical_memory,
        "disk_free_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "monotonic_seconds": time.monotonic(),
    }


def _probe_models(endpoint: str) -> dict[str, object]:
    """Probe ``/v1/models`` without making a generation request."""

    try:
        with urlopen(endpoint, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            advertised = [str(item.get("id")) for item in payload.get("data", [])]
            return {
                "status": "ok",
                "http_status": int(response.status),
                "endpoint": endpoint,
                "advertised_models": advertised,
                "requested_model_advertised": REQUESTED_MODEL in advertised,
                "generation_requested": False,
            }
    except Exception as exc:  # pragma: no cover - endpoint availability is external
        return {
            "status": "probe_failed",
            "http_status": None,
            "endpoint": endpoint,
            "advertised_models": [],
            "requested_model_advertised": False,
            "generation_requested": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _wall_budget_cell(spec: dict[str, object]) -> dict[str, object]:
    return {
        "cell_id": spec["cell_id"],
        "status": "incomplete",
        "termination_reason": "wall_budget_exhausted",
        "seeds": {
            "data_seed": spec["data_seed"],
            "proposal_seed": spec["proposal_seed"],
            "simulator_noise_seed": spec["simulator_noise_seed"],
        },
        "posterior_summary_available": False,
        "parameter_diagnostics": {
            "status": "unavailable",
            "reason": "wall_budget_exhausted_before_cell_start",
            "posterior_summary": False,
        },
        "particle_prediction": {
            "status": "unavailable",
            "reason": "wall_budget_exhausted_before_cell_start",
            "posterior_summary": False,
        },
    }


def _manifest(artifact_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": str(path.relative_to(artifact_dir)), "bytes": path.stat().st_size, "sha256": digest})
    return records


def _markdown(result: dict[str, object]) -> str:
    budget = result["pilot_budget"]
    lines = [
        "# Numerical ODE ABC-SMC reference control pilot",
        "",
        "This is a bounded known-model control for the opt-in Gaussian ABC-SMC reference. It does not call the LLM and does not modify the canonical SBIEngine or BDSS path.",
        "",
        f"- status: `{result['status']}`; completed cells: {result['completed_cells']}/{result['expected_cells']}",
        f"- budget: {budget['target_samples_per_population']} particles/population, {budget['max_attempts_per_population']} attempts/population, epsilon schedule {budget['epsilon_schedule']}, wall cap {budget['wall_budget_seconds']} seconds",
        f"- solver: {result['solver_preflight']['solver_config']}; independent error check passed: {result['solver_preflight']['passed']}",
        f"- target: finite-epsilon noncentral-chi-square quadrature; exact Gaussian-likelihood quadrature is separately labeled in each receipt",
        f"- oMLX probe: HTTP {result['model_preflight'].get('http_status')}, exact model advertised {result['model_preflight'].get('requested_model_advertised')}, generation requested {result['model_preflight'].get('generation_requested')}",
        "",
        "| cell | true k | data seed | status | termination | final ESS | mean error | max CDF error |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for cell in result["cells"]:
        diagnostics = cell.get("parameter_diagnostics", {})
        lines.append(
            f"| {cell['cell_id']} | {cell.get('dataset', {}).get('true_k', 'n/a')} | {cell['seeds']['data_seed']} | {cell['status']} | {cell['termination_reason']} | "
            f"{diagnostics.get('effective_sample_size', 'n/a')} | {diagnostics.get('mean_error', 'n/a')} | {diagnostics.get('cdf_max_abs_error', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "Each cell has its own observed dataset and finite-epsilon target. Particle summaries are available only for a complete reference run; incomplete or wall-capped cells retain failure evidence without posterior-like summaries. Agreement diagnostics are descriptive and do not claim coverage or scientific discovery.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pilot(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    deadline = started + float(args.wall_budget_seconds)
    artifact_dir = Path(args.artifact_dir)
    partial_path = artifact_dir / "partial_result.json"
    resource_before = _resource_snapshot()
    model_preflight = _probe_models(args.models_endpoint)
    solver_config = ODESolverConfig(max_step=0.05, max_steps=100_000, error_tolerance=1e-6)
    solver_preflight = solver_accuracy_check(solver_config=solver_config)
    result: dict[str, object] = {
        "run_id": args.run_id,
        "status": "running",
        "reference_path": "gaussian_abc_smc_reference_opt_in",
        "canonical_runner_integrated": False,
        "model_preflight": model_preflight,
        "resource_before": resource_before,
        "solver_preflight": solver_preflight,
        "pilot_definition": {
            "true_k_values": [0.35, 0.8],
            "data_seeds": [11, 29, 47],
            "time_points": list(OBSERVATION_TIMES),
            "sigma": SIGMA,
            "bounds": [list(PRIOR_BOUNDS[0])],
            "discrepancy": "|| (Y_sim - y_obs) / sigma ||_2 over all four observations",
            "epsilon_schedule": list(EPSILON_SCHEDULE),
            "seed_streams": "data generation, ABC proposals, and simulator noise use separate predeclared seeds",
        },
        "pilot_budget": {
            "target_samples_per_population": int(args.target_samples),
            "max_attempts_per_population": int(args.max_attempts),
            "epsilon_schedule": list(EPSILON_SCHEDULE),
            "wall_budget_seconds": float(args.wall_budget_seconds),
            "started_monotonic": started,
        },
        "cells": [],
        "completed_cells": 0,
        "expected_cells": len(PILOT_CELLS),
    }
    _atomic_write_json(partial_path, result)

    if not solver_preflight["passed"]:
        result["status"] = "solver_preflight_failed"
        result["termination_reason"] = "solver_accuracy_check_failed"
    else:
        for index, spec in enumerate(PILOT_CELLS):
            if time.monotonic() >= deadline:
                result["cells"].extend(_wall_budget_cell(item) for item in PILOT_CELLS[index:])
                result["status"] = "wall_budget_exhausted"
                _atomic_write_json(partial_path, result)
                break
            dataset = generate_decay_observation(
                float(spec["true_k"]),
                int(spec["data_seed"]),
                time_points=OBSERVATION_TIMES,
                sigma=SIGMA,
                solver_config=solver_config,
            )
            cell = run_ode_decay_control_cell(
                dataset,
                proposal_seed=int(spec["proposal_seed"]),
                simulator_noise_seed=int(spec["simulator_noise_seed"]),
                target_samples=int(args.target_samples),
                epsilon_schedule=EPSILON_SCHEDULE,
                max_attempts_per_population=int(args.max_attempts),
                bounds=PRIOR_BOUNDS,
                solver_config=solver_config,
                quadrature_order=int(args.quadrature_order),
                wall_deadline=deadline,
            )
            cell["cell_id"] = spec["cell_id"]
            cell["posterior_summary_available"] = bool(
                cell["status"] == "complete" and cell.get("parameter_diagnostics", {}).get("posterior_summary", True)
            )
            result["cells"].append(cell)
            _atomic_write_json(artifact_dir / f"cell_{index:02d}_{spec['cell_id']}.json", cell)
            result["completed_cells"] = sum(item.get("status") == "complete" for item in result["cells"])
            if cell["termination_reason"] == "wall_budget_exhausted" or time.monotonic() >= deadline:
                result["cells"].extend(_wall_budget_cell(item) for item in PILOT_CELLS[index + 1 :])
                result["status"] = "wall_budget_exhausted"
                _atomic_write_json(partial_path, result)
                break
            _atomic_write_json(partial_path, result)
        else:
            result["status"] = "completed" if result["completed_cells"] == len(PILOT_CELLS) else "incomplete"

    result["completed_cells"] = sum(item.get("status") == "complete" for item in result["cells"])
    result["elapsed_seconds"] = float(time.monotonic() - started)
    result["resource_after"] = _resource_snapshot()
    if result["status"] == "running":
        result["status"] = "incomplete"
    _atomic_write_json(partial_path, result)
    _atomic_write_json(artifact_dir / "artifact_manifest.json", _manifest(artifact_dir))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "ode-reference-control-pilot-2026-09-17.json")
    parser.add_argument("--output-markdown", type=Path, default=ROOT / "reports" / "ode-reference-control-pilot-2026-09-17.md")
    parser.add_argument("--models-endpoint", default=DEFAULT_MODELS_ENDPOINT)
    parser.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--quadrature-order", type=int, default=256)
    parser.add_argument("--wall-budget-seconds", type=float, default=DEFAULT_WALL_BUDGET_SECONDS)
    args = parser.parse_args()
    if args.target_samples <= 0 or args.max_attempts <= 0 or args.quadrature_order < 16:
        parser.error("particle, attempt, and quadrature budgets must be positive")
    if not 0.0 < args.wall_budget_seconds <= 600.0:
        parser.error("wall-budget-seconds must be positive and no greater than 600")
    if args.artifact_dir is None:
        args.artifact_dir = ROOT / "artifacts" / "ode_reference_control" / args.run_id
    result = run_pilot(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "completed_cells": result["completed_cells"], "expected_cells": result["expected_cells"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

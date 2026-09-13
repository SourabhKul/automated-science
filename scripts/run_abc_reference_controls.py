#!/usr/bin/env python3
"""Run bounded finite-epsilon controls for the opt-in ABC reference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.abc_reference_controls import run_scalar_finite_epsilon_control  # noqa: E402


REQUESTED_MODEL = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
DEFAULT_JSON = ROOT / "reports" / "abc-reference-sequential-control-pilot-2026-09-13.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "abc-reference-sequential-control-pilot-2026-09-13.md"


def _probe_models(endpoint: str) -> dict[str, object]:
    """Make one read-only /v1/models probe; no generation request is made."""

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
    except Exception as exc:  # pragma: no cover - only when oMLX is absent
        return {
            "status": "probe_failed",
            "http_status": None,
            "endpoint": endpoint,
            "advertised_models": [],
            "requested_model_advertised": False,
            "generation_requested": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _json_number(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _population_receipts(reference: dict[str, object]) -> list[dict[str, object]]:
    receipts = []
    for population in reference["populations"]:
        diagnostics = dict(population["diagnostics"])
        # Ancestor indices are retained in the in-memory result and focused
        # tests. The persisted receipt keeps all population counters while
        # avoiding a bulky per-attempt trace.
        diagnostics.pop("ancestor_indices", None)
        weights = np.asarray(population["weights"], dtype=float)
        covariance = population["proposal_covariance"]
        receipts.append(
            {
                "generation": int(population["generation"]),
                "epsilon": float(population["epsilon"]),
                "accepted": int(len(population["accepted_params"])),
                "weights": [_json_number(value) for value in weights],
                "weights_sum": _json_number(np.sum(weights)) if len(weights) else 0.0,
                "weights_normalized": bool(len(weights) and np.all(np.isfinite(weights)) and np.isclose(np.sum(weights), 1.0)),
                "effective_sample_size": _json_number(population["effective_sample_size"]),
                "proposal_covariance": None if covariance is None else np.asarray(covariance).tolist(),
                "diagnostics": diagnostics,
            }
        )
    return receipts


def _run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    model_probe = _probe_models(args.models_endpoint)
    cases: list[dict[str, object]] = []
    wall_budget_exhausted = False
    for y in (0.1, 0.5, 0.9):
        for epsilon in (0.1, 0.025):
            for seed in args.seeds:
                if time.monotonic() - started >= args.wall_budget_seconds:
                    wall_budget_exhausted = True
                    break
                control = run_scalar_finite_epsilon_control(
                    y,
                    epsilon,
                    sigma=0.1,
                    target_samples=args.target_samples,
                    max_attempts=args.max_attempts,
                    seed=seed,
                    epsilon_schedule=[0.2, epsilon],
                )
                reference = control["reference"]
                target = control["target"]
                diagnostics = control["diagnostics"]
                final_diagnostics = reference["populations"][-1]["diagnostics"]
                cases.append(
                    {
                        "y": y,
                        "epsilon": epsilon,
                        "seed": seed,
                        "status": reference["status"],
                        "termination_reason": reference["termination_reason"],
                        "epsilon_schedule": [0.2, epsilon],
                        "accepted": int(len(reference["accepted_params"])),
                        "target_samples": args.target_samples,
                        "attempts": int(final_diagnostics["proposed"]),
                        "out_of_support": int(final_diagnostics["out_of_support"]),
                        "failed_simulations": int(final_diagnostics["failed_simulations"]),
                        "normalizer": _json_number(target.normalizer),
                        "target_mean": _json_number(target.mean),
                        "effective_sample_size": _json_number(diagnostics.get("effective_sample_size")),
                        "weighted_mean": _json_number(diagnostics.get("weighted_mean")),
                        "mean_error": _json_number(diagnostics.get("mean_error")),
                        "mean_mc_se": _json_number(diagnostics.get("mean_mc_se")),
                        "mean_error_over_mc_se": _json_number(diagnostics.get("mean_error_over_mc_se")),
                        "quantile_errors": [
                            _json_number(item) for item in np.asarray(diagnostics.get("quantile_errors", []))
                        ],
                        "quantile_error_over_mc_se": [
                            _json_number(item)
                            for item in np.asarray(diagnostics.get("quantile_error_over_mc_se", []))
                        ],
                        "cdf_max_abs_error": _json_number(diagnostics.get("cdf_max_abs_error")),
                        "cdf_max_error_over_mc_se": _json_number(diagnostics.get("cdf_max_error_over_mc_se")),
                        "populations": _population_receipts(reference),
                    }
                )
            if wall_budget_exhausted:
                break
        if wall_budget_exhausted:
            break
    elapsed = time.monotonic() - started
    complete_cells = sum(item["status"] == "complete" for item in cases)
    return {
        "status": "completed" if not wall_budget_exhausted else "wall_budget_exhausted",
        "reference_path": "gaussian_abc_smc_reference_opt_in",
        "canonical_runner_integrated": False,
        "control_definition": {
            "theta_support": [0.0, 1.0],
            "sigma": 0.1,
            "observations": [0.1, 0.5, 0.9],
            "epsilons": [0.1, 0.025],
            "quadrature": "independent fixed Gauss-Legendre quadrature on [0, 1]",
            "diagnostic_note": "ESS-scaled errors are descriptive pilot diagnostics; no formal weighted-SMC confidence claim is made.",
        },
        "pilot_budget": {
            "seeds": list(args.seeds),
            "target_samples_per_cell": args.target_samples,
            "max_attempts_per_cell": args.max_attempts,
            "wall_budget_seconds": args.wall_budget_seconds,
            "elapsed_seconds": elapsed,
        },
        "model_preflight": model_probe,
        "cells": cases,
        "completed_cells": complete_cells,
        "expected_cells": 3 * 2 * len(args.seeds),
    }


def _markdown(result: dict[str, object]) -> str:
    budget = result["pilot_budget"]
    probe = result["model_preflight"]
    lines = [
        "# Opt-in Gaussian ABC-SMC scalar control pilot",
        "",
        "This bounded pilot exercises the isolated reference path. The historical core.sbi_engine.SBIEngine runner is unchanged and is not called.",
        "",
        f"- status: {result['status']}; completed cells: {result['completed_cells']}/{result['expected_cells']}",
        f"- budget: {budget['target_samples_per_cell']} accepted particles/cell, {budget['max_attempts_per_cell']} attempts/cell, {budget['elapsed_seconds']:.3f} seconds",
        f"- scalar target: theta ~ Uniform(0,1), sigma={result['control_definition']['sigma']}, observations {result['control_definition']['observations']}, epsilons {result['control_definition']['epsilons']}",
        "- sequential schedule: each cell uses the explicit two-population epsilon schedule [0.2, target_epsilon]; generation-1 receipts persist weights, covariance, and counters",
        f"- independent target evaluation: {result['control_definition']['quadrature']}",
        f"- diagnostics: {result['control_definition']['diagnostic_note']}",
        f"- oMLX preflight: HTTP {probe.get('http_status')}, requested model advertised {probe.get('requested_model_advertised')}, generation requested {probe.get('generation_requested')}",
        "",
        "| y | epsilon | seed | status | ESS | mean error / MC SE | max CDF error / MC SE | out-of-support |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for cell in result["cells"]:
        lines.append(
            f"| {cell['y']:.3f} | {cell['epsilon']:.3f} | {cell['seed']} | {cell['status']} | "
            f"{cell.get('effective_sample_size', 'n/a')} | {cell.get('mean_error_over_mc_se', 'n/a')} | "
            f"{cell.get('cdf_max_error_over_mc_se', 'n/a')} | {cell['out_of_support']} |"
        )
    lines.extend(
        [
            "",
            "The table records finite-epsilon ABC agreement diagnostics against the independent quadrature target. It is a scalar implementation control and does not establish calibration of the canonical discovery runner, formal confidence coverage for weighted SMC particles, or scientific model recovery.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--target-samples", type=int, default=400)
    parser.add_argument("--max-attempts", type=int, default=50_000)
    parser.add_argument("--wall-budget-seconds", type=float, default=480.0)
    parser.add_argument("--models-endpoint", default="http://127.0.0.1:8000/v1/models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    args = parser.parse_args()
    if args.target_samples <= 0 or args.max_attempts <= 0 or args.wall_budget_seconds <= 0:
        parser.error("budgets must be positive")
    result = _run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_cells": result["completed_cells"],
                "expected_cells": result["expected_cells"],
                "elapsed_seconds": result["pilot_budget"]["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" and result["completed_cells"] == result["expected_cells"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

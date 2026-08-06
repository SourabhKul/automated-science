#!/usr/bin/env python3
"""Resumable non-LLM leave-one-cell-out critique runner for NASA battery data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.battery_nasa import (
    build_cell_holdout_bundles,
    build_leave_one_cell_out_splits,
    evaluate_cell_holdout_baselines,
)


DEFAULT_SEEDS = [20261860, 20261861, 20261862]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def compact_row(fold: dict[str, Any], seed: int, output: Path, result: dict[str, Any], returncode: int, duration: float) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") or {}
    cell = diagnostics.get("cell_holdout") or {}
    posterior = diagnostics.get("posterior_summary") or {}
    return {
        "fold_id": fold["fold_id"],
        "train_cells": fold["train_cells"],
        "test_cell": fold["test_cells"][0],
        "seed": seed,
        "status": result.get("status", "missing"),
        "returncode": returncode,
        "duration_seconds": duration,
        "median_distance": finite(result.get("median_distance")),
        "train_mse": finite((result.get("train_metrics") or {}).get("mse")),
        "external_mse": finite((result.get("test_metrics") or {}).get("mse")),
        "external_late_cycle_mse": finite(((cell.get("test") or {}).get("late_cycle") or {}).get("mse")),
        "posterior_parameters": posterior.get("parameters") or [],
        "bound_pressure": diagnostics.get("parameter_bound_pressure") or [],
        "warnings": diagnostics.get("warnings") or [],
        "failure_modes": diagnostics.get("failure_modes") or [],
        "error": result.get("error"),
        "output": str(output),
    }


def summarize_fold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["status"] == "success"]
    metric_values = [row["external_mse"] for row in successes if row["external_mse"] is not None]
    late_values = [row["external_late_cycle_mse"] for row in successes if row["external_late_cycle_mse"] is not None]
    k_medians = []
    for row in successes:
        for parameter in row["posterior_parameters"]:
            if parameter.get("name") == "k_lin" and finite(parameter.get("median")) is not None:
                k_medians.append(float(parameter["median"]))
    bound_pressure = any(
        float(item.get("lower_fraction", 0.0)) > 0.0 or float(item.get("upper_fraction", 0.0)) > 0.0
        for row in successes
        for item in row["bound_pressure"]
    )
    return {
        "completed_seed_count": len(rows),
        "success_seed_count": len(successes),
        "all_metrics_finite": len(successes) == len(rows) and len(metric_values) == len(rows) and len(late_values) == len(rows),
        "external_mse_median": float(np.median(metric_values)) if metric_values else None,
        "external_late_cycle_mse_median": float(np.median(late_values)) if late_values else None,
        "k_lin_medians": k_medians,
        "k_lin_relative_range_percent": (
            float((max(k_medians) - min(k_medians)) / np.median(k_medians) * 100.0)
            if k_medians and np.median(k_medians) != 0.0
            else None
        ),
        "any_bound_pressure": bound_pressure,
        "warnings": sorted({str(warning) for row in rows for warning in row["warnings"]}),
        "failure_modes": sorted({str(mode) for row in rows for mode in row["failure_modes"]}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--normalized-data", type=Path, default=Path("data/real/battery_nasa/cycle_level.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--target-samples", type=int, default=250)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--initial-particles", type=int, default=50000)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalized = pd.read_csv(args.normalized_data)
    folds = build_leave_one_cell_out_splits(normalized)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = args.results_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_baselines = {}
    for fold in folds:
        fold_path = fold_dir / f"{fold['fold_id']}.json"
        write_json(fold_path, fold)
        bundles = build_cell_holdout_bundles(normalized, fold)
        fold_baselines[fold["fold_id"]] = evaluate_cell_holdout_baselines(bundles["train"], bundles["test"])

    summary = {
        "phase": 18,
        "domain": "real_battery_nasa_capacity",
        "started_at": utc_now(),
        "completed_at": None,
        "run_spec": {
            "model": "reduced_exponential_loss",
            "seeds": args.seeds,
            "target_samples": args.target_samples,
            "generations": args.generations,
            "initial_particles": args.initial_particles,
            "timeout_seconds": args.timeout_seconds,
            "strategy": "gaussian_weighted",
            "prediction_contract": "condition_on_observed_test_cell_initial_soh",
        },
        "folds": [{**fold, "baselines": fold_baselines[fold["fold_id"]]} for fold in folds],
        "rows": [],
        "fold_summaries": {},
    }
    for fold in folds:
        fold_path = fold_dir / f"{fold['fold_id']}.json"
        for seed in args.seeds:
            output = args.results_dir / f"{fold['fold_id']}_seed_{seed}.json"
            if args.resume and output.exists():
                result = json.loads(output.read_text())
                row = compact_row(fold, seed, output, result, 0, 0.0)
                summary["rows"].append(row)
                summary["fold_summaries"][fold["fold_id"]] = summarize_fold(
                    [candidate for candidate in summary["rows"] if candidate["fold_id"] == fold["fold_id"]]
                )
                write_json(args.summary, summary)
                print(f"RESUME fold={fold['fold_id']} seed={seed} status={row['status']}", flush=True)
                continue

            command = [
                sys.executable, "-m", "core.sandbox_eval",
                "--code-file", str(args.code_file),
                "--domain", "real_battery_nasa_capacity",
                "--held-out",
                "--cell-holdout-path", str(fold_path),
                "--output", str(output),
                "--target-samples", str(args.target_samples),
                "--generations", str(args.generations),
                "--initial-particles", str(args.initial_particles),
                "--inference-strategy", "gaussian_weighted",
                "--seed", str(seed),
            ]
            print(f"START fold={fold['fold_id']} seed={seed} particles={args.initial_particles} generations={args.generations}", flush=True)
            started = time.monotonic()
            try:
                completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=args.timeout_seconds, check=False)
                result = json.loads(output.read_text()) if output.exists() else {"status": "error", "error": "sandbox produced no output"}
                row = compact_row(fold, seed, output, result, completed.returncode, time.monotonic() - started)
            except subprocess.TimeoutExpired:
                row = compact_row(fold, seed, output, {"status": "timeout", "error": f"timeout after {args.timeout_seconds}s"}, 124, time.monotonic() - started)
            summary["rows"].append(row)
            summary["fold_summaries"][fold["fold_id"]] = summarize_fold(
                [candidate for candidate in summary["rows"] if candidate["fold_id"] == fold["fold_id"]]
            )
            write_json(args.summary, summary)
            print(f"DONE fold={fold['fold_id']} seed={seed} status={row['status']} seconds={row['duration_seconds']:.1f}", flush=True)

    summary["completed_at"] = utc_now()
    write_json(args.summary, summary)
    failures = sum(row["status"] != "success" for row in summary["rows"])
    print(f"COMPLETE rows={len(summary['rows'])} failures={failures}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

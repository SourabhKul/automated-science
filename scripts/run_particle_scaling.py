from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.domain_configs import DOMAIN_CONFIGS


DEFAULT_DOMAINS = ["ecology", "battery", "climate", "synbio", "real_theophylline"]
DEFAULT_STRATEGIES = ["gaussian_weighted", "bdss"]
DEFAULT_PARTICLES = [50000, 150000, 300000, 600000]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def write_report(path: Path, summary: dict) -> None:
    rows = summary.get("rows", [])
    lines = [
        f"# Phase 8 Particle Scaling: `{summary['run_id']}`",
        "",
        "Seed-model-only benchmark. No LLM proposal loop is run.",
        "",
        f"- Held-out: `{summary['held_out']}`",
        f"- Target samples: `{summary['target_samples']}`",
        f"- Generations: `{summary['generations']}`",
        f"- Base seed: `{summary['seed']}`",
        "",
        "| domain | strategy | particles | status | seconds | peak RSS MB | median distance | min distance | test MSE | accepted |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {strategy} | {initial_particles} | {status} | {duration_seconds:.2f} | {peak_rss_mb} | {median_distance} | {min_distance} | {test_mse} | {accepted_particle_count} |".format(
                domain=row["domain"],
                strategy=row["strategy"],
                initial_particles=row["initial_particles"],
                status=row.get("status"),
                duration_seconds=float(row.get("duration_seconds") or 0.0),
                peak_rss_mb=_format_float(row.get("peak_rss_mb")),
                median_distance=_format_float(row.get("median_distance")),
                min_distance=_format_float(row.get("min_distance")),
                test_mse=_format_float(row.get("test_mse")),
                accepted_particle_count=row.get("accepted_particle_count"),
            )
        )

    completed = [row for row in rows if row.get("status") == "success"]
    if completed:
        lines.extend(["", "## Current Best By Domain", ""])
        for domain in summary.get("domains", []):
            domain_rows = [row for row in completed if row["domain"] == domain and row.get("test_mse") is not None]
            if not domain_rows:
                continue
            best = min(domain_rows, key=lambda row: row["test_mse"])
            lines.append(
                f"- `{domain}`: `{best['strategy']}` at `{best['initial_particles']}` particles, test MSE `{_format_float(best['test_mse'])}`."
            )

    path.write_text("\n".join(lines) + "\n")


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def parse_peak_rss_mb(stderr: str) -> float | None:
    match = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size", stderr)
    if not match:
        return None
    # macOS /usr/bin/time -l reports bytes for maximum resident set size.
    return int(match.group(1)) / (1024 * 1024)


def run_command_with_optional_time(cmd: list[str], cwd: Path, timeout: int | None) -> tuple[subprocess.CompletedProcess, float | None]:
    time_cmd = Path("/usr/bin/time")
    if time_cmd.exists():
        timed_cmd = [str(time_cmd), "-l", *cmd]
        proc = subprocess.run(timed_cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return proc, parse_peak_rss_mb(proc.stderr)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return proc, None


def completed_keys(summary: dict) -> set[tuple[str, str, int]]:
    keys = set()
    for row in summary.get("rows", []):
        if row.get("return_code") == 0 or row.get("status") in {"error", "failed", "timeout"}:
            keys.add((row["domain"], row["strategy"], int(row["initial_particles"])))
    return keys


def extract_row_metrics(payload: dict) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics") or {}
    fit = diagnostics.get("fit") or {}
    posterior = diagnostics.get("posterior_summary") or {}
    return {
        "status": payload.get("status"),
        "requested_strategy": payload.get("requested_strategy"),
        "effective_strategy": payload.get("effective_strategy"),
        "median_distance": payload.get("median_distance"),
        "min_distance": payload.get("min_distance"),
        "train_mse": (payload.get("train_metrics") or {}).get("mse"),
        "train_rmse": (payload.get("train_metrics") or {}).get("rmse"),
        "test_mse": (payload.get("test_metrics") or {}).get("mse"),
        "test_rmse": (payload.get("test_metrics") or {}).get("rmse"),
        "accepted_particle_count": fit.get("accepted_particle_count"),
        "posterior_parameters": posterior.get("parameters"),
        "parameter_bound_pressure": diagnostics.get("parameter_bound_pressure"),
        "warnings": diagnostics.get("warnings"),
        "failure_modes": diagnostics.get("failure_modes"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed-model particle scaling across inference strategies.")
    parser.add_argument("--run-id", default=f"phase8_particle_scaling_{datetime.now().strftime('%Y%m%d_%H%M')}")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--particles", nargs="+", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--target-samples", type=int, default=500)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--held-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=int, default=None, help="Per-combination timeout in seconds.")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid = [domain for domain in args.domains if domain not in DOMAIN_CONFIGS]
    if invalid:
        print(f"Unknown domains: {', '.join(invalid)}", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.summary).parent if args.summary else Path(f"artifacts/evaluations/{args.run_id}")
    summary_path = Path(args.summary) if args.summary else out_dir / "summary.json"
    report_path = out_dir / "report.md"
    code_dir = out_dir / "seed_code"
    eval_dir = out_dir / "eval_outputs"
    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and summary_path.exists():
        summary = read_json(summary_path)
        summary["resumed_at"] = utc_now()
    else:
        summary = {
            "run_id": args.run_id,
            "created_at": utc_now(),
            "domains": args.domains,
            "strategies": args.strategies,
            "particle_counts": args.particles,
            "target_samples": args.target_samples,
            "generations": args.generations,
            "seed": args.seed,
            "held_out": args.held_out,
            "rows": [],
            "report": str(report_path),
        }
    write_json(summary_path, summary)
    write_report(report_path, summary)

    done = completed_keys(summary) if args.resume else set()
    failures = 0
    for domain_index, domain in enumerate(args.domains):
        code_path = code_dir / f"{domain}_seed.py"
        code_path.write_text(DOMAIN_CONFIGS[domain]["seed_logic"])
        for strategy in args.strategies:
            for initial_particles in args.particles:
                key = (domain, strategy, int(initial_particles))
                if key in done:
                    print(f"Skipping completed {domain} {strategy} {initial_particles}")
                    continue

                output_path = eval_dir / f"{domain}_{strategy}_{initial_particles}.json"
                combo_seed = args.seed + domain_index * 1000 + args.particles.index(initial_particles)
                cmd = [
                    sys.executable,
                    "-m",
                    "core.sandbox_eval",
                    "--code-file",
                    str(code_path),
                    "--domain",
                    domain,
                    "--output",
                    str(output_path),
                    "--target-samples",
                    str(args.target_samples),
                    "--generations",
                    str(args.generations),
                    "--initial-particles",
                    str(initial_particles),
                    "--seed",
                    str(combo_seed),
                    "--inference-strategy",
                    strategy,
                ]
                if args.held_out:
                    cmd.append("--held-out")

                print(f"Running {domain} strategy={strategy} particles={initial_particles}")
                started_at = utc_now()
                started = time.monotonic()
                status = "failed"
                peak_rss_mb = None
                try:
                    proc, peak_rss_mb = run_command_with_optional_time(cmd, project_root, args.timeout)
                    duration = time.monotonic() - started
                    payload = read_json(output_path, default={})
                    metrics = extract_row_metrics(payload)
                    status = metrics.get("status") or ("success" if proc.returncode == 0 else "failed")
                    if proc.returncode != 0:
                        failures += 1
                    row = {
                        "domain": domain,
                        "strategy": strategy,
                        "initial_particles": int(initial_particles),
                        "target_samples": args.target_samples,
                        "generations": args.generations,
                        "seed": combo_seed,
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "duration_seconds": duration,
                        "peak_rss_mb": peak_rss_mb,
                        "return_code": proc.returncode,
                        "stdout_tail": proc.stdout[-2000:],
                        "stderr_tail": proc.stderr[-2000:],
                        "output_path": str(output_path),
                        "accepted_update_quality": None,
                        "accepted_update_quality_note": "not_applicable_seed_model_only",
                        **metrics,
                    }
                except subprocess.TimeoutExpired as exc:
                    duration = time.monotonic() - started
                    failures += 1
                    row = {
                        "domain": domain,
                        "strategy": strategy,
                        "initial_particles": int(initial_particles),
                        "target_samples": args.target_samples,
                        "generations": args.generations,
                        "seed": combo_seed,
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "duration_seconds": duration,
                        "peak_rss_mb": peak_rss_mb,
                        "return_code": None,
                        "status": "timeout",
                        "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                        "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                        "output_path": str(output_path),
                        "accepted_update_quality": None,
                        "accepted_update_quality_note": "not_applicable_seed_model_only",
                    }

                summary.setdefault("rows", []).append(row)
                summary["updated_at"] = utc_now()
                summary["failures"] = failures
                write_json(summary_path, summary)
                write_report(report_path, summary)
                print(
                    f"Finished {domain} strategy={strategy} particles={initial_particles} "
                    f"status={status} seconds={row['duration_seconds']:.2f}"
                )

    summary["finished_at"] = utc_now()
    summary["failures"] = failures
    write_json(summary_path, summary)
    write_report(report_path, summary)
    print(f"Particle scaling summary written to {summary_path}")
    print(f"Particle scaling report written to {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

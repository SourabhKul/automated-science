#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.lmstudio_models import loaded_model_instances, unload_loaded_models


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable, deadline-aware daily LLM campaign on one scientific application."
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--target-samples", type=int, default=500)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--initial-particles", type=int, default=300000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--per-model-timeout", type=int, default=5400)
    parser.add_argument("--min-remaining-minutes", type=float, default=20.0)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def job_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    model_seen: set[str] = set()
    for seed in args.seeds:
        for strategy in args.strategies:
            for model in args.models:
                run_id = f"{args.campaign_id}_{slug(model)}_{slug(strategy)}_s{seed}"
                specs.append(
                    {
                        "job_id": run_id,
                        "run_id": run_id,
                        "model": model,
                        "strategy": strategy,
                        "seed": seed,
                        "preflight": model not in model_seen,
                        "summary": f"artifacts/model_matrix/{run_id}/summary.json",
                        "status": "pending",
                        "attempts": 0,
                    }
                )
                model_seen.add(model)
    return specs


def config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "domain": args.domain,
        "models": args.models,
        "strategies": args.strategies,
        "seeds": args.seeds,
        "duration_hours": args.duration_hours,
        "epochs": args.epochs,
        "held_out": True,
        "target_samples": args.target_samples,
        "generations": args.generations,
        "initial_particles": args.initial_particles,
        "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "per_model_timeout": args.per_model_timeout,
        "min_remaining_minutes": args.min_remaining_minutes,
        "base_url": args.base_url,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_or_create_state(args: argparse.Namespace, state_path: Path) -> dict[str, Any]:
    config = config_payload(args)
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f"campaign state already exists; pass --resume: {state_path}")
        state = json.loads(state_path.read_text())
        if state.get("config") != config:
            raise ValueError("resume configuration does not match the existing campaign state")
        return state
    started_epoch = time.time()
    return {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "started_epoch": started_epoch,
        "deadline_epoch": started_epoch + args.duration_hours * 3600.0,
        "config": config,
        "jobs": job_specs(args),
        "completed_job_count": 0,
        "failed_job_count": 0,
    }


def build_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_model_matrix.py",
        "--base-url",
        args.base_url,
        "--models",
        job["model"],
        "--domains",
        args.domain,
        "--epochs",
        str(args.epochs),
        "--held-out",
        "--seed",
        str(job["seed"]),
        "--run-id",
        job["run_id"],
        "--summary",
        job["summary"],
        "--resume",
        "--per-model-timeout",
        str(args.per_model_timeout),
        "--context-length",
        str(args.context_length),
        "--max-tokens",
        str(args.max_tokens),
        "--target-samples",
        str(args.target_samples),
        "--generations",
        str(args.generations),
        "--initial-particles",
        str(args.initial_particles),
        "--inference-strategy",
        job["strategy"],
        "--unload-between-models",
    ]
    if not job["preflight"]:
        command.append("--skip-preflight")
    return command


def run_command(command: list[str], timeout_seconds: int) -> tuple[int, float, bool]:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_seconds)
        return return_code, time.monotonic() - started, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
        return process.returncode if process.returncode is not None else -9, time.monotonic() - started, True


def enforce_unloaded(base_url: str) -> dict[str, Any]:
    try:
        before = loaded_model_instances(base_url, timeout=30)
        unload_results = unload_loaded_models(base_url, timeout=60) if before else []
        time.sleep(2 if before else 0)
        after = loaded_model_instances(base_url, timeout=30)
        return {
            "before": before,
            "unload_results": unload_results,
            "after": after,
            "ok": not after,
        }
    except Exception as exc:
        return {
            "before": None,
            "unload_results": [],
            "after": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def refresh_counts(state: dict[str, Any]) -> None:
    state["completed_job_count"] = sum(job["status"] == "completed" for job in state["jobs"])
    state["failed_job_count"] = sum(job["status"] in {"failed", "timeout"} for job in state["jobs"])
    state["updated_at"] = utc_now()


def main() -> int:
    args = parse_args()
    campaign_dir = args.output_root / args.campaign_id
    state_path = campaign_dir / "campaign_state.json"
    if args.dry_run:
        print(json.dumps({"config": config_payload(args), "jobs": job_specs(args)}, indent=2))
        return 0

    state = load_or_create_state(args, state_path)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    deadline = float(state["deadline_epoch"])
    minimum_remaining = args.min_remaining_minutes * 60.0

    for job in state["jobs"]:
        if job["status"] == "completed":
            continue
        if job["status"] in {"failed", "timeout"} and not args.retry_failed:
            continue
        remaining = deadline - time.time()
        if remaining < minimum_remaining:
            state["status"] = "deadline_reached"
            state["stopped_at"] = utc_now()
            refresh_counts(state)
            write_json(state_path, state)
            print(f"DEADLINE remaining={remaining:.1f}s; no new job started", flush=True)
            break

        job["status"] = "running"
        job["started_at"] = utc_now()
        job["attempts"] = int(job.get("attempts", 0)) + 1
        command = build_command(args, job)
        job["command"] = command
        refresh_counts(state)
        write_json(state_path, state)
        print(
            f"START job={job['job_id']} model={job['model']} strategy={job['strategy']} "
            f"seed={job['seed']} remaining_hours={remaining / 3600.0:.2f}",
            flush=True,
        )

        return_code, duration, timed_out = run_command(
            command,
            timeout_seconds=args.per_model_timeout + 1200,
        )
        job["return_code"] = return_code
        job["duration_seconds"] = duration
        job["finished_at"] = utc_now()
        job["status"] = "timeout" if timed_out else ("completed" if return_code == 0 else "failed")
        job["unload_hygiene"] = enforce_unloaded(args.base_url)
        refresh_counts(state)
        write_json(state_path, state)
        print(
            f"DONE job={job['job_id']} status={job['status']} duration={duration:.1f}s "
            f"unloaded={job['unload_hygiene']['ok']}",
            flush=True,
        )

    else:
        state["status"] = "completed"
        state["completed_at"] = utc_now()
        refresh_counts(state)
        write_json(state_path, state)

    if state["status"] == "running":
        state["status"] = "completed"
        state["completed_at"] = utc_now()
        refresh_counts(state)
        write_json(state_path, state)

    final_hygiene = enforce_unloaded(args.base_url)
    state["final_unload_hygiene"] = final_hygiene
    refresh_counts(state)
    write_json(state_path, state)
    return 0 if state["failed_job_count"] == 0 and final_hygiene["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

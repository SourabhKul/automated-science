#!/usr/bin/env python3
"""Bounded, recorded ecology end-to-end scaffolding smoke.

This harness keeps the repository implementation unchanged while providing the
missing synthetic input files in the timestamped artifact directory and adding
the oMLX chat-template override required by the fixed local Qwen endpoint.

Set ``SMOKE_RUN_DIR`` to choose the artifact directory. The default creates a
new local timestamped directory under ``artifacts/scaffolding_smoke``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import requests


REPO = Path("/Users/sourabh/Documents/Codex/automated_science")
_default_stamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
RUN_DIR = Path(os.environ.get("SMOKE_RUN_DIR", str(REPO / "artifacts/scaffolding_smoke" / _default_stamp)))
DATA_DIR = RUN_DIR / "data"
PYTHON = Path(sys.executable)
MODEL_ID = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
SEED = 7
TARGET_SAMPLES = 8
GENERATIONS = 2
INITIAL_PARTICLES = 32
EPOCHS = 2
MAX_TOKENS = 2048
CANDIDATE_TIMEOUT_SECONDS = 120
PARENT_WALL_SECONDS = 600


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def environment_snapshot() -> dict:
    versions = {}
    for package in ("jax", "diffrax", "numpy", "pandas", "requests"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[package] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=False
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        revision = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "captured_at": now(),
        "python": sys.version,
        "python_executable": str(PYTHON),
        "repo": str(REPO),
        "git_revision": revision,
        "packages": versions,
        "cwd_before_run": os.getcwd(),
        "endpoint": ENDPOINT,
        "model_id": MODEL_ID,
    }


def generate_synthetic_data() -> dict:
    """Generate a short deterministic trajectory from the configured seed model."""
    sys.path.insert(0, str(REPO))
    from core.domain_configs import DOMAIN_CONFIGS
    from core.sandbox_eval import build_candidate_model
    import jax.numpy as jnp

    config = DOMAIN_CONFIGS["ecology"]
    # Short horizon keeps the run bounded while retaining a nontrivial coupled
    # trajectory. The values are known to be finite with the configured solver.
    time_points = np.linspace(0.0, 10.0, 41, dtype=np.float32)
    true_params = np.array([0.5, 0.05, 0.05, 0.5], dtype=np.float32)
    model = build_candidate_model(config["seed_logic"], config)
    trajectory = np.asarray(
        model.simulate(jnp.asarray(true_params), jnp.asarray(time_points), model.get_initial_conditions())
    )
    if trajectory.shape != (len(time_points), 2) or not np.isfinite(trajectory).all():
        raise RuntimeError(f"synthetic trajectory invalid: shape={trajectory.shape}")
    ground_truth_path = DATA_DIR / "ecology_ground_truth.npy"
    time_path = DATA_DIR / "ecology_time_points.npy"
    np.save(ground_truth_path, trajectory)
    np.save(time_path, time_points)
    return {
        "domain": "ecology",
        "generation": "configured ecology seed_logic via core.sandbox_eval.build_candidate_model",
        "true_parameters": true_params.tolist(),
        "initial_conditions": config["y0"],
        "time_start": float(time_points[0]),
        "time_end": float(time_points[-1]),
        "time_points": int(len(time_points)),
        "trajectory_shape": list(trajectory.shape),
        "trajectory_min": float(np.min(trajectory)),
        "trajectory_max": float(np.max(trajectory)),
        "ground_truth_path": str(ground_truth_path),
        "time_path": str(time_path),
        "ground_truth_sha256": sha256(ground_truth_path),
        "time_sha256": sha256(time_path),
    }


def main() -> int:
    started = time.monotonic()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RUN_DIR / "environment.json", environment_snapshot())
    config_record = {
        "created_at": now(),
        "domain": "ecology",
        "model_id": MODEL_ID,
        "endpoint": ENDPOINT,
        "seed": SEED,
        "epochs": EPOCHS,
        "target_samples": TARGET_SAMPLES,
        "generations": GENERATIONS,
        "initial_particles": INITIAL_PARTICLES,
        "inference_strategy": "gaussian_weighted",
        "held_out": True,
        "max_tokens": MAX_TOKENS,
        "candidate_timeout_seconds": CANDIDATE_TIMEOUT_SECONDS,
        "parent_wall_seconds": PARENT_WALL_SECONDS,
        "qwen_chat_template_kwargs": {"enable_thinking": False},
        "data_deviation": (
            "Canonical ecology data files are absent in this checkout. Synthetic files are created only under "
            "this artifact directory and loaded by the unchanged core.sandbox_eval evaluator through cwd."
        ),
        "known_limitations": [
            "Canonical held-out suffix score is adaptive validation because the loop selects on it.",
            "Current ABC-SMC clipping and Gaussian kernel weights are not independently calibrated here.",
            "This is a deterministic synthetic integration smoke, not a scientific discovery or independent test.",
        ],
    }
    write_json(RUN_DIR / "config.json", config_record)
    try:
        data_record = generate_synthetic_data()
        write_json(RUN_DIR / "data_generation.json", data_record)

        # Import after data setup so all repository modules resolve from REPO;
        # run_domain_main itself remains the orchestration implementation.
        sys.path.insert(0, str(REPO))
        import scripts.run_domain as run_domain

        llm_records: list[dict] = []
        subprocess_records: list[dict] = []
        original_call_llm = run_domain.call_llm
        original_subprocess_run = run_domain.subprocess.run

        def logged_call_llm(prompt, model_id, endpoint, max_tokens=8192, system_prompt=None):
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            record = {
                "request_started_at": now(),
                "request": payload,
                "endpoint": endpoint,
            }
            request_started = time.monotonic()
            try:
                response = requests.post(endpoint, json=payload, timeout=CANDIDATE_TIMEOUT_SECONDS)
                record["status_code"] = response.status_code
                record["elapsed_seconds"] = time.monotonic() - request_started
                record["response_headers"] = {
                    key: value for key, value in response.headers.items() if key.lower() in {"content-type", "server"}
                }
                record["response_text"] = response.text
                response.raise_for_status()
                body = response.json()
                record["response_json"] = body
                content = body["choices"][0]["message"]["content"]
                record["content"] = content
                record["finish_reason"] = body["choices"][0].get("finish_reason")
                record["ok"] = True
                return content
            except Exception as exc:
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                return None
            finally:
                record["request_finished_at"] = now()
                llm_records.append(record)
                write_json(RUN_DIR / "llm_requests_responses.json", {"calls": llm_records})

        def bounded_logged_subprocess_run(*args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            record = {
                "started_at": now(),
                "command": [str(item) for item in command] if isinstance(command, (list, tuple)) else str(command),
                "cwd": os.getcwd(),
                "timeout_requested_seconds": kwargs.get("timeout"),
                "timeout_applied_seconds": min(float(kwargs.get("timeout", CANDIDATE_TIMEOUT_SECONDS)), CANDIDATE_TIMEOUT_SECONDS),
            }
            started_child = time.monotonic()
            kwargs["timeout"] = CANDIDATE_TIMEOUT_SECONDS
            child_env = dict(os.environ)
            child_env["PYTHONPATH"] = str(REPO) + os.pathsep + child_env.get("PYTHONPATH", "")
            kwargs["env"] = child_env
            try:
                result = original_subprocess_run(*args, **kwargs)
                record["returncode"] = result.returncode
                record["stdout"] = result.stdout
                record["stderr"] = result.stderr
                return result
            except subprocess.TimeoutExpired as exc:
                record["timeout_expired"] = True
                record["stdout"] = exc.stdout
                record["stderr"] = exc.stderr
                raise
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - started_child
                record["finished_at"] = now()
                subprocess_records.append(record)
                write_json(RUN_DIR / "candidate_subprocesses.json", {"calls": subprocess_records})

        run_domain.call_llm = logged_call_llm
        run_domain.subprocess.run = bounded_logged_subprocess_run
        previous_cwd = Path.cwd()
        os.chdir(RUN_DIR)
        try:
            result_code = run_domain.run_domain_main(
                domain="ecology",
                family="qwen36",
                model=MODEL_ID,
                endpoint=ENDPOINT,
                tag_prefix="scaffolding_smoke",
                epochs=EPOCHS,
                held_out=True,
                seed=SEED,
                max_tokens=MAX_TOKENS,
                inference_strategy="gaussian_weighted",
                target_samples_override=TARGET_SAMPLES,
                generations_override=GENERATIONS,
                initial_particles_override=INITIAL_PARTICLES,
                dry_run=False,
            )
        finally:
            os.chdir(previous_cwd)
        result = {
            "status": "completed" if result_code == 0 else "failed",
            "return_code": result_code,
            "elapsed_seconds": time.monotonic() - started,
            "llm_call_count": len(llm_records),
            "candidate_subprocess_count": len(subprocess_records),
            "completed_at": now(),
        }
        write_json(RUN_DIR / "run_status.json", result)
        return int(result_code)
    except Exception as exc:
        result = {
            "status": "harness_error",
            "elapsed_seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "completed_at": now(),
        }
        write_json(RUN_DIR / "run_status.json", result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

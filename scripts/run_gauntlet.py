from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.domain_quarantine import partition_requested_domains
from core.domain_configs import DOMAIN_CONFIGS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    with path.open("r") as f:
        return json.load(f)


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def completed_domains(summary: dict) -> set[str]:
    return {
        run["domain"]
        for run in summary.get("runs", [])
        if run.get("return_code") == 0 or run.get("status") == "quarantined"
    }


def recorded_domains(summary: dict) -> set[str]:
    return {
        run["domain"]
        for run in summary.get("runs", [])
        if run.get("domain")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the centralized domain runner across many domains.")
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_CONFIGS), help="Domains to run.")
    parser.add_argument("--family", default="qwen3_coder_next", choices=["qwen36", "qwen3_coder_next", "gemma4"])
    parser.add_argument("--model", default=None, help="LLM model ID override.")
    parser.add_argument("--tag-prefix", default=None, help="Experiment tag prefix override; domain is appended automatically.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--held-out", action="store_true", help="Use held-out train/test selection.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed; each domain gets base seed + index.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum output tokens for each domain LLM call.")
    parser.add_argument("--target-samples", type=int, default=None, help="Override domain ABC-SMC target samples.")
    parser.add_argument("--generations", type=int, default=None, help="Override domain ABC-SMC generations.")
    parser.add_argument("--initial-particles", type=int, default=None, help="Override domain ABC-SMC initial particles.")
    parser.add_argument(
        "--inference-strategy",
        default="gaussian_weighted",
        help="ABC-SMC transition strategy passed to each domain evaluation.",
    )
    parser.add_argument("--summary", default=None, help="Path to write gauntlet summary JSON.")
    parser.add_argument("--resume", action="store_true", help="Skip domains already marked successful in the summary.")
    parser.add_argument(
        "--include-quarantined-domains",
        action="store_true",
        help="Run known unstable domains instead of recording them as quarantined skips.",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid = [domain for domain in args.domains if domain not in DOMAIN_CONFIGS]
    if invalid:
        print(f"Unknown domains: {', '.join(invalid)}", file=sys.stderr)
        return 2
    requested_domains = list(args.domains)
    stable_domains, quarantined_domains = partition_requested_domains(
        requested_domains,
        include_quarantined=args.include_quarantined_domains,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(args.summary or f"artifacts/gauntlets/{args.family}_{args.epochs}epoch_{stamp}.json")
    summary = load_summary(summary_path) if args.resume else {
        "created_at": utc_now(),
        "family": args.family,
        "model": args.model,
        "tag_prefix": args.tag_prefix,
        "epochs": args.epochs,
        "held_out": args.held_out,
        "target_samples": args.target_samples,
        "generations": args.generations,
        "initial_particles": args.initial_particles,
        "inference_strategy": args.inference_strategy,
        "base_seed": args.seed,
        "domains": requested_domains,
        "stable_domains": stable_domains,
        "stable_domain_count": len(stable_domains),
        "quarantined_domains": quarantined_domains,
        "runs": [],
    }
    summary["domains"] = requested_domains
    summary["stable_domains"] = stable_domains
    summary["stable_domain_count"] = len(stable_domains)
    summary["quarantined_domains"] = quarantined_domains
    summary["target_samples"] = args.target_samples
    summary["generations"] = args.generations
    summary["initial_particles"] = args.initial_particles
    summary["inference_strategy"] = args.inference_strategy
    done = completed_domains(summary) if args.resume else set()
    seen = recorded_domains(summary)
    domain_offsets = {domain: index for index, domain in enumerate(requested_domains)}

    project_root = Path(__file__).resolve().parents[1]
    failures = 0
    for item in quarantined_domains:
        domain = item["domain"]
        if domain in seen:
            continue
        summary.setdefault("runs", []).append({
            "domain": domain,
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "duration_seconds": 0.0,
            "return_code": None,
            "seed": None,
            "status": "quarantined",
            "quarantine_reason": item["reason"],
            "cmd": None,
        })
        summary["updated_at"] = utc_now()
        write_summary(summary_path, summary)

    for run_index, domain in enumerate(stable_domains):
        if domain in done:
            print(f"\n=== Skipping {domain}: already successful in {summary_path} ===")
            continue

        domain_seed = None if args.seed is None else args.seed + domain_offsets[domain]
        cmd = [
            sys.executable,
            "scripts/run_domain.py",
            domain,
            "--family",
            args.family,
            "--epochs",
            str(args.epochs),
        ]
        if args.held_out:
            cmd.append("--held-out")
        if domain_seed is not None:
            cmd.extend(["--seed", str(domain_seed)])
        if args.endpoint:
            cmd.extend(["--endpoint", args.endpoint])
        if args.model:
            cmd.extend(["--model", args.model])
        if args.tag_prefix:
            cmd.extend(["--tag-prefix", args.tag_prefix])
        if args.max_tokens:
            cmd.extend(["--max-tokens", str(args.max_tokens)])
        if args.target_samples is not None:
            cmd.extend(["--target-samples", str(args.target_samples)])
        if args.generations is not None:
            cmd.extend(["--generations", str(args.generations)])
        if args.initial_particles is not None:
            cmd.extend(["--initial-particles", str(args.initial_particles)])
        if args.inference_strategy:
            cmd.extend(["--inference-strategy", args.inference_strategy])

        print(f"\n=== Running {domain} ({run_index + 1}/{len(stable_domains)}) ===")
        print(" ".join(cmd))
        started_at = utc_now()
        start_time = time.monotonic()
        proc = subprocess.run(cmd, cwd=project_root)
        duration = time.monotonic() - start_time

        run_record = {
            "domain": domain,
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": duration,
            "return_code": proc.returncode,
            "seed": domain_seed,
            "cmd": cmd,
        }
        summary.setdefault("runs", []).append(run_record)
        summary["updated_at"] = utc_now()
        write_summary(summary_path, summary)

        if proc.returncode != 0:
            failures += 1
            print(f"Domain {domain} failed with return code {proc.returncode}.")
            if args.stop_on_failure:
                break

    summary["finished_at"] = utc_now()
    summary["failures"] = failures
    write_summary(summary_path, summary)
    print(f"\nGauntlet summary written to {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

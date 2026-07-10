from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_FILES = [
    "ARCHITECTURE.md",
    "README.md",
    "core/domain_configs.py",
    "core/generated_code.py",
    "core/sandbox_eval.py",
    "core/sbi_engine.py",
    "core/evaluation.py",
    "core/run_config.py",
    "scripts/run_domain.py",
    "scripts/run_gauntlet.py",
    "scripts/run_model_matrix.py",
    "scripts/lmstudio_models.py",
    "scripts/repo_inventory.py",
    "run_baselines.py",
    "artifacts/evaluations/hero_run_deep_dive_20260703.md",
    "artifacts/evaluations/hero_run_deep_dive_20260703.json",
]

LEGACY_PATTERNS = [
    "*_orchestrator.py",
    "master_scheduler.py",
    "setup_run_circuit.py",
    "orchestrator.py",
    "neuro_orchestrator.py",
    "cytokine_orchestrator.py",
    "rpe1_orchestrator.py",
    "gemma431b_*_orchestrator.py",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def count_domain_configs() -> int:
    namespace: dict[str, object] = {}
    source = (ROOT / "core/domain_configs.py").read_text()
    exec(compile(source, "core/domain_configs.py", "exec"), namespace)
    return len(namespace["DOMAIN_CONFIGS"])  # type: ignore[arg-type]


def scan_legacy_files() -> list[str]:
    files: set[Path] = set()
    for pattern in LEGACY_PATTERNS:
        files.update(ROOT.glob(pattern))
    return sorted(rel(path) for path in files if path.is_file())


def summarize_qwen36_traces() -> dict[str, object]:
    runs_root = ROOT / "artifacts/runs/qwen36_27b"
    history_files = sorted(runs_root.glob("run_*/qwen36_27b_*_run_history.csv"))
    meta_files = sorted(runs_root.glob("run_*/models_qwen36_27b_*/meta_hints.txt"))
    totals = {
        "history_files": len(history_files),
        "meta_hints_files": len(meta_files),
        "nonempty_meta_hints_files": sum(1 for path in meta_files if path.read_text(errors="ignore").strip()),
        "rows": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
        "seeds": 0,
    }
    per_domain: dict[str, dict[str, int]] = {}
    for path in history_files:
        match = re.match(r"qwen36_27b_(.*)_run_history\.csv", path.name)
        domain = match.group(1) if match else path.stem
        domain_totals = per_domain.setdefault(domain, {"accepted": 0, "rejected": 0, "errors": 0, "seeds": 0})
        with path.open() as f:
            for row in csv.DictReader(f):
                totals["rows"] += 1
                action = row.get("Action", "")
                if action == "ACCEPTED":
                    totals["accepted"] += 1
                    domain_totals["accepted"] += 1
                elif action == "REJECTED":
                    totals["rejected"] += 1
                    domain_totals["rejected"] += 1
                elif action == "SEED_EVALUATED":
                    totals["seeds"] += 1
                    domain_totals["seeds"] += 1
                elif action.startswith("ERROR") or "ERROR" in action:
                    totals["errors"] += 1
                    domain_totals["errors"] += 1
    return {"totals": totals, "per_domain": per_domain}


def build_inventory() -> dict[str, object]:
    canonical = []
    for item in CANONICAL_FILES:
        path = ROOT / item
        canonical.append({
            "path": item,
            "exists": path.exists(),
            "bytes": file_size(path),
        })

    generated_model_dirs = sorted(path for path in (ROOT / "models").glob("*") if path.is_dir())
    artifact_dirs = sorted(path for path in (ROOT / "artifacts").glob("*") if path.is_dir())

    return {
        "canonical_architecture": "ARCHITECTURE.md",
        "hero_run_report": "artifacts/evaluations/hero_run_deep_dive_20260703.md",
        "domain_count": count_domain_configs(),
        "canonical_files": canonical,
        "legacy_files": scan_legacy_files(),
        "generated_model_dir_count": len(generated_model_dirs),
        "generated_model_dirs_sample": [rel(path) for path in generated_model_dirs[:25]],
        "artifact_dirs": [rel(path) for path in artifact_dirs],
        "qwen36_legacy_trace_summary": summarize_qwen36_traces(),
    }


def print_markdown(inventory: dict[str, object]) -> None:
    print("# Repository Inventory")
    print()
    print(f"Canonical architecture: `{inventory['canonical_architecture']}`")
    print(f"Active domain configs: {inventory['domain_count']}")
    print(f"Generated model directories: {inventory['generated_model_dir_count']}")
    print()
    print("## Canonical Files")
    print()
    for item in inventory["canonical_files"]:  # type: ignore[index]
        status = "ok" if item["exists"] else "missing"
        print(f"- `{item['path']}`: {status}, {item['bytes']} bytes")
    print()
    print("## Legacy Entrypoints")
    print()
    for path in inventory["legacy_files"]:  # type: ignore[index]
        print(f"- `{path}`")
    print()
    print("## Qwen36 Legacy Trace Summary")
    print()
    totals = inventory["qwen36_legacy_trace_summary"]["totals"]  # type: ignore[index]
    for key, value in totals.items():
        print(f"- {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report canonical vs legacy repository architecture inventory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    inventory = build_inventory()
    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print_markdown(inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

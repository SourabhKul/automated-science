#!/usr/bin/env python3
"""Fixed-source C-MAPSS FD002 archive gate and source-train baseline slice."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.cmapss import evaluate_cmapss_train_prefix_baselines, load_cmapss_fd002, load_cmapss_fd002_official_test_rul, load_cmapss_train_split, partition_cmapss_train


DEFAULT_ARCHIVE = Path("data/real/cmapss/raw/CMAPSSData.zip")
DEFAULT_OUT = Path("artifacts/evaluations/phase27_cmapss_fd002_archive_gate_20260715")
SPLIT_SEED = 20260715


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split() -> dict[str, list[int]]:
    units = np.arange(1, 261)
    permutation = np.random.default_rng(SPLIT_SEED).permutation(units)
    return {
        "train": [int(item) for item in np.sort(permutation[:180])],
        "selection": [int(item) for item in np.sort(permutation[180:220])],
        "holdout": [int(item) for item in np.sort(permutation[220:])],
    }


def _column_ledger(records: list) -> list[dict[str, object]]:
    settings = np.vstack([record.operating_settings for record in records])
    sensors = np.vstack([record.sensors for record in records])
    matrix = np.column_stack([settings, sensors])
    names = [f"setting_{index}" for index in range(1, 4)] + [f"sensor_{index}" for index in range(1, 22)]
    return [
        {"name": name, "min": float(np.min(matrix[:, index])), "max": float(np.max(matrix[:, index])), "std": float(np.std(matrix[:, index])), "constant": bool(np.ptp(matrix[:, index]) == 0.0), "near_constant": bool(np.std(matrix[:, index]) <= 1e-12)}
        for index, name in enumerate(names)
    ]


def run(archive: Path = DEFAULT_ARCHIVE, output_dir: Path = DEFAULT_OUT) -> dict[str, object]:
    data = load_cmapss_fd002(archive)
    split = _split()
    partitioned = partition_cmapss_train(data["source_train"], split)
    official_rul = load_cmapss_fd002_official_test_rul(archive)
    with zipfile.ZipFile(archive) as source:
        members = source.namelist()
        readme = source.read("readme.txt").decode("latin-1")
    required_members = {"train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt", "readme.txt"}
    if not required_members.issubset(members):
        raise ValueError("fixed C-MAPSS archive lacks required FD002 members")
    source_rows = {
        name: {
            "units": len(records),
            "rows": int(sum(record.cycles.size for record in records)),
            "cycle_length_min": int(min(record.cycles.size for record in records)),
            "cycle_length_max": int(max(record.cycles.size for record in records)),
            "all_finite": bool(all(np.all(np.isfinite(record.operating_settings)) and np.all(np.isfinite(record.sensors)) for record in records)),
            "strict_cycles": bool(all(np.array_equal(record.cycles, np.arange(1, record.cycles.size + 1)) for record in records)),
        }
        for name, records in data.items()
    }
    selection_baselines = evaluate_cmapss_train_prefix_baselines(partitioned["train"], partitioned["selection"])
    result = {
        "phase": 27,
        "status": "passed_source_gate_selection_baseline_only",
        "source": {
            "direct_url": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "archive_members": members,
            "readme_mentions_fd002": "Data Set: FD002" in readme,
            "units_scaling": "not documented in bundled README; retained as unknown simulator measurements",
        },
        "fd002": {"source_rows": source_rows, "train_column_ledger": _column_ledger(data["source_train"]), "official_test_rul": {"count": int(official_rul.size), "min": float(official_rul.min()), "max": float(official_rul.max()), "positive": bool(np.all(official_rul > 0))}},
        "source_train_split": {"seed": SPLIT_SEED, **split},
        "selection_baselines": selection_baselines,
        "isolation": {"official_test_rul_used_for_selection": False, "official_test_metrics_emitted": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Predeclare native-grid synthetic and negative controls before any official-test score, ABC-SMC, LLM discovery, or campaign.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    split_path = Path("data/real/cmapss/source_train_split.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.archive, args.output_dir)
    print(json.dumps({"status": result["status"], "selection": result["selection_baselines"]["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

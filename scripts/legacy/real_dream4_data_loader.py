from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.real_data.dream4 import build_recovery_split, evaluate_recovery_baselines, load_dream4_size10


DEFAULT_SOURCE_ROOT = Path("data/real/dream4_insilico/extracted/source_tables/lightlyProcessedDownloadedData")
DEFAULT_ARCHIVE_PATH = Path("data/real/dream4_insilico/raw/DREAM4_0.99.17.tar.gz")
SOURCE_URL = "https://bioconductor.org/packages/2.12/data/experiment/src/contrib/DREAM4_0.99.17.tar.gz"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def generate_data(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    archive_path: str | Path = DEFAULT_ARCHIVE_PATH,
    output_root: str | Path = "data/real/dream4_insilico",
) -> dict[str, Any]:
    source_root = Path(source_root)
    archive_path = Path(archive_path)
    output_root = Path(output_root)
    networks = load_dream4_size10(source_root, include_gold_edges=True)
    splits = {network_id: build_recovery_split(network) for network_id, network in networks.items()}
    baselines = {
        network_id: evaluate_recovery_baselines(network, splits[network_id], gold_edges=network.gold_edges)
        for network_id, network in networks.items()
    }
    manifest = {
        "status": "ok",
        "claim_boundary": "DREAM4 in-silico recovery-dynamics benchmark only; no unseen-intervention or biological claim.",
        "source_root": str(source_root),
        "source_url": SOURCE_URL,
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest() if archive_path.exists() else None,
        "network_count": len(networks),
        "trajectory_count_per_network": 5,
        "time_points_per_trajectory": 21,
        "recovery_start_time": 500.0,
        "metric_time_start": 550.0,
        "gold_edge_policy": "Evaluation-only. The loader can omit gold edges; fitting never receives them.",
        "splits_path": str(output_root / "recovery_splits.json"),
        "baselines_path": str(output_root / "baseline_results.json"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    (output_root / "recovery_splits.json").write_text(json.dumps(_json_safe(splits), indent=2, sort_keys=True) + "\n")
    (output_root / "baseline_results.json").write_text(json.dumps(_json_safe(baselines), indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare guarded non-LLM DREAM4 recovery-baseline artifacts.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--output-root", type=Path, default=Path("data/real/dream4_insilico"))
    args = parser.parse_args()
    print(json.dumps(generate_data(source_root=args.source_root, archive_path=args.archive_path, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

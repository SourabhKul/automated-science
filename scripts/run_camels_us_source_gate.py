#!/usr/bin/env python3
"""Run the fixed CAMELS-US archive/source-quality gate without fitting a flow model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.camels_us import (
    CAMELS_SPLIT_SEED,
    deterministic_camels_split,
    eligible_camels_basins,
    inspect_camels_us_archive,
    validate_camels_split,
)


RAW_DIR = Path("data/real/camels_us/raw")
DEFAULT_ARCHIVE = RAW_DIR / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
DEFAULT_OUT = Path("artifacts/evaluations/phase28_camels_us_archive_gate_20260721")
EXPECTED = {
    "archive_md5": "8e9a466710e8270b58f01d332a87184f",
    "readme_md5": "b37d64950e9d4c5c10a8b4ef82bc6219",
    "attributes_md5": "714c68bd5bb3314ca39b14f9467bd609",
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(archive: Path = DEFAULT_ARCHIVE, output_dir: Path = DEFAULT_OUT) -> dict[str, object]:
    raw_dir = archive.parent
    hashes = {
        "archive_md5": _digest(archive, "md5"),
        "archive_sha256": _digest(archive, "sha256"),
        "readme_md5": _digest(raw_dir / "readme.txt", "md5"),
        "attributes_md5": _digest(raw_dir / "camels_attributes_v2.0.xlsx", "md5"),
    }
    if any(hashes[name] != expected for name, expected in EXPECTED.items()):
        raise ValueError("CAMELS official file hash contract failed")
    inspection = inspect_camels_us_archive(archive)
    eligible = eligible_camels_basins(inspection)
    split = deterministic_camels_split(eligible)
    validate_camels_split(split, eligible)
    status = "passed_source_gate_adapter_not_authorized"
    result = {
        "phase": 28,
        "status": status,
        "source": {
            "doi": "10.5065/D6MW2F4D",
            "zenodo_record": 15529996,
            "version": "1.2",
            "license": "CC-BY-4.0",
            "source_url": "https://zenodo.org/records/15529996",
            "archive_url": "https://zenodo.org/api/records/15529996/files/basin_timeseries_v1p2_metForcing_obsFlow.zip/content",
            "archive_size_bytes": archive.stat().st_size,
            **hashes,
            "published_archive_md5": EXPECTED["archive_md5"],
            "human_impact_scope": "Official record documents 671 HCDN-2009 CONUS basins; this gate applies no additional impact filter.",
        },
        "inspection": inspection,
        "eligible_basin_count": len(eligible),
        "eligibility_rule": "exact Daymet/USGS daily date alignment, positive source drainage area, and at least eight complete Oct-Sep hydrologic years with no M/-999 streamflow day; no interpolation or replacement",
        "split": {"seed": CAMELS_SPLIT_SEED, "method": "SHA-256 order of static basin IDs after source-quality eligibility", **split},
        "isolation": {
            "observed_streamflow_normalized": False,
            "observed_streamflow_models_fitted": False,
            "abc_smc_calls": 0,
            "llm_calls": 0,
            "campaign_launched": False,
        },
        "next_gate": "Implement only a source-normalized adapter and train-only baseline slice, then run output-isolated native-grid synthetic controls before any observed-streamflow inference or LLM work.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    split_path = Path("data/real/camels_us/source_basin_split.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(result["split"], indent=2) + "\n")
    manifest = Path("data/real/camels_us/manifest.json")
    manifest.write_text(json.dumps({"source": result["source"], "raw_schema": inspection["raw_schema"], "eligible_basin_count": len(eligible)}, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.archive, args.output_dir)
    print(json.dumps({"status": result["status"], "eligible_basin_count": result["eligible_basin_count"], "split_counts": {name: len(values) for name, values in result["split"].items() if name in {"train", "selection", "external"}}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

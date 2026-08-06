from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from core.real_data.parallel_wiener_hammerstein import build_estimation_split, evaluate_parallel_wiener_hammerstein_baselines, load_parallel_wiener_hammerstein


DEFAULT_RAW_ARCHIVE = Path("data/real/parallel_wiener_hammerstein/raw/ParWHFiles.zip")
DEFAULT_OUTPUT_ROOT = Path("data/real/parallel_wiener_hammerstein")
SOURCE_RECORD = "https://data.4tu.nl/articles/dataset/Parallel_Wiener-Hammerstein_Time_Series/12950081"
SOURCE_DOI = "10.4121/12950081"
PUBLISHED_MD5 = "5836dc66b0296b43ba4b8613a032408b"


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_data(*, raw_archive: str | Path = DEFAULT_RAW_ARCHIVE, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    raw_archive, output_root = Path(raw_archive), Path(output_root)
    local_md5 = _hash(raw_archive, "md5")
    if local_md5 != PUBLISHED_MD5:
        raise ValueError(f"MD5 mismatch for {raw_archive}: {local_md5} != {PUBLISHED_MD5}")
    records = load_parallel_wiener_hammerstein(raw_archive)
    split = build_estimation_split(records)
    baselines = evaluate_parallel_wiener_hammerstein_baselines(split)
    with zipfile.ZipFile(raw_archive) as archive:
        members = [info.filename for info in archive.infolist()]
    manifest = {
        "status": "ok",
        "claim_boundary": "Controlled physical electronic-circuit identification benchmark only; u/y physical units are not specified by the CSV README.",
        "source_record": SOURCE_RECORD,
        "source_doi": SOURCE_DOI,
        "source_version": "1",
        "license": "CC-BY-SA-4.0",
        "raw_archive": {"path": str(raw_archive), "published_md5": PUBLISHED_MD5, "local_md5": local_md5, "sha256": _hash(raw_archive, "sha256"), "members": members},
        "record_split": {"train": len(split["train"]), "selection": len(split["selection"]), "official_validation": len(split["official_validation"]), "policy": "16/4 deterministic realization split per estimation amplitude; official validation is untouched for external scoring"},
        "sample_rate_hz": 78125,
        "samples_per_record": 32768,
        "baseline_results_path": str(output_root / "baseline_results.json"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_root / "split.json").write_text(json.dumps(manifest["record_split"], indent=2, sort_keys=True) + "\n")
    (output_root / "baseline_results.json").write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Parallel Wiener-Hammerstein source and baseline artifacts.")
    parser.add_argument("--raw-archive", type=Path, default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(generate_data(raw_archive=args.raw_archive, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

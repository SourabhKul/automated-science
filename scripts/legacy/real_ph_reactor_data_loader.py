from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.real_data.ph_reactor import EXPECTED_EXPERIMENTS, EXPECTED_SAMPLES, evaluate_ph_reactor_baselines, load_ph_reactor


DEFAULT_RAW_ROOT = Path("data/real/ph_reactor/raw")
DEFAULT_OUTPUT_ROOT = Path("data/real/ph_reactor")
SOURCE_RECORD = "https://zenodo.org/records/3956067"
SOURCE_DOI = "10.5281/zenodo.3956067"
EXPECTED_MD5 = {
    "PH_U_Train.csv": "f400ecf9eabdd17523c7053b18762b73",
    "PH_Y_Train.csv": "fcfd813786954a6991bf8823afaee3f5",
    "PH_U_Val.csv": "2ec5a28bd23d411741fbd84fcb7d4623",
    "PH_Y_Val.csv": "6c1ded12aabbbd4ba4857508fe8a7618",
    "PH_U_Test.csv": "6dd0e3f1caa6735aa84e05ba0f8e8f31",
    "PH_Y_Test.csv": "3844b19937eabe4d16e5b2506c485eee",
}


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_data(*, raw_root: str | Path = DEFAULT_RAW_ROOT, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    raw_root, output_root = Path(raw_root), Path(output_root)
    data = load_ph_reactor(raw_root)
    files = {}
    for filename, published_md5 in EXPECTED_MD5.items():
        path = raw_root / filename
        local_md5 = _hash(path, "md5")
        if local_md5 != published_md5:
            raise ValueError(f"MD5 mismatch for {path}: {local_md5} != {published_md5}")
        matrix = np.loadtxt(path, delimiter=",")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        files[filename] = {"path": str(path), "source_url": f"https://zenodo.org/api/records/3956067/files/{filename}/content", "published_md5": published_md5, "local_md5": local_md5, "sha256": _hash(path, "sha256"), "shape": list(matrix.shape), "finite": bool(np.all(np.isfinite(matrix)))}
    baselines = evaluate_ph_reactor_baselines(data)
    manifest = {
        "status": "ok",
        "claim_boundary": "Controlled pH-reactor simulator input-output benchmark only; physical units and sampling interval are not documented by the fixed record and are left unknown.",
        "source_record": SOURCE_RECORD,
        "source_doi": SOURCE_DOI,
        "source_version": "1",
        "license": "CC-BY-4.0",
        "raw_root": str(raw_root),
        "files": files,
        "experiment_split": {"train": EXPECTED_EXPERIMENTS["train"], "validation": EXPECTED_EXPERIMENTS["validation"], "test": EXPECTED_EXPERIMENTS["test"], "samples_per_experiment": EXPECTED_SAMPLES},
        "input_units": "unknown_in_fixed_record",
        "output_units": "pH-reactor output; numeric unit/scaling not separately documented in fixed record",
        "sampling_interval": "unknown_in_fixed_record",
        "baseline_results_path": str(output_root / "baseline_results.json"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_root / "split.json").write_text(json.dumps(manifest["experiment_split"], indent=2, sort_keys=True) + "\n")
    (output_root / "baseline_results.json").write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare non-LLM pH-reactor U/Y baseline artifacts.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(generate_data(raw_root=args.raw_root, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

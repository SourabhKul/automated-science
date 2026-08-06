from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.real_data.warfarin_pkpd import (
    write_dense_target,
    write_normalized_artifacts,
    write_subject_holdout_split,
)


DEFAULT_SOURCE_URL = "https://monolixsuite.slp-software.com/monolix/2024R1/warfarin-data-set"
DEFAULT_FIXTURE = Path("data/real/pkpd/warfarin_schema_fixture.csv")
DEFAULT_RAW_CANDIDATES = [
    Path("data/real/pkpd/raw/warfarin_data.txt"),
    Path("data/real/pkpd/raw/warfarin_data.csv"),
    Path("data/real/pkpd/raw/warfarin.csv"),
]


def resolve_raw_path(raw_path: str | Path | None = None) -> tuple[Path, bool]:
    if raw_path is not None:
        path = Path(raw_path)
        return path, path == DEFAULT_FIXTURE
    for candidate in DEFAULT_RAW_CANDIDATES:
        if candidate.exists():
            return candidate, False
    return DEFAULT_FIXTURE, True


def generate_data(
    *,
    raw_path: str | Path | None = None,
    subject_id: str | None = None,
    output_root: str | Path = "data/real/pkpd",
    data_dir: str | Path = "data",
    holdout_seed: int = 0,
    holdout_train_fraction: float = 0.8,
) -> dict:
    raw, fixture_mode = resolve_raw_path(raw_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    normalized = write_normalized_artifacts(
        raw,
        normalized_csv=output / "warfarin_normalized.csv",
        provenance_json=output / "provenance.json",
        source_url=DEFAULT_SOURCE_URL,
    )
    target = write_dense_target(
        normalized,
        data_dir=data_dir,
        domain="real_warfarin_pkpd",
        subject_id=subject_id,
    )
    subject_holdout = write_subject_holdout_split(
        normalized,
        path=output / "subject_holdout.json",
        train_fraction=holdout_train_fraction,
        seed=holdout_seed,
    )
    manifest = {
        "status": "ok",
        "raw_path": str(raw),
        "raw_path_source": "explicit" if raw_path is not None else ("auto" if not fixture_mode else "fixture"),
        "raw_candidates": [str(candidate) for candidate in DEFAULT_RAW_CANDIDATES],
        "normalized_csv": str(output / "warfarin_normalized.csv"),
        "provenance_json": str(output / "provenance.json"),
        "subject_holdout_json": str(output / "subject_holdout.json"),
        "subject_holdout": subject_holdout,
        "observation_mask_path": target.get("observation_mask_path"),
        "target": target,
        "fixture_mode": fixture_mode,
        "warning": (
            "Default fixture is schema/smoke data only, not a scientific Warfarin result. "
            "Put downloaded data at data/real/pkpd/raw/warfarin_data.txt or pass --raw-path."
        )
        if fixture_mode
        else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the real_warfarin_pkpd dense target.")
    parser.add_argument("--raw-path", help="Raw Monolix/nlmixr-style Warfarin table. Defaults to schema fixture.")
    parser.add_argument("--subject-id", help="Optional subject to use for the dense sandbox target.")
    parser.add_argument("--output-root", default="data/real/pkpd")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--holdout-train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    manifest = generate_data(
        raw_path=args.raw_path,
        subject_id=args.subject_id,
        output_root=args.output_root,
        data_dir=args.data_dir,
        holdout_seed=args.holdout_seed,
        holdout_train_fraction=args.holdout_train_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.real_data.battery_nasa import (
    write_capacity_target,
    write_cell_holdout_split,
    write_within_cell_time_splits,
    write_normalized_artifacts,
)


DEFAULT_SOURCE_URL = "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"
DEFAULT_FIXTURE = Path("data/real/battery_nasa/battery_nasa_schema_fixture.csv")
DEFAULT_RAW_CANDIDATES = [
    Path("data/real/battery_nasa/raw/B0005.mat"),
    Path("data/real/battery_nasa/raw/B0006.mat"),
    Path("data/real/battery_nasa/raw/B0007.mat"),
    Path("data/real/battery_nasa/raw/B0018.mat"),
    Path("data/real/battery_nasa/raw/cycle_level.csv"),
]
DEFAULT_MAT_CANDIDATES = DEFAULT_RAW_CANDIDATES[:4]


def resolve_raw_path(raw_path: str | Path | None = None) -> tuple[Path, bool]:
    if raw_path is not None:
        path = Path(raw_path)
        return path, path == DEFAULT_FIXTURE
    for candidate in DEFAULT_RAW_CANDIDATES:
        if candidate.exists():
            return candidate, False
    return DEFAULT_FIXTURE, True


def resolve_raw_paths(raw_path: str | Path | None = None) -> tuple[Path | list[Path], bool]:
    if raw_path is not None:
        path = Path(raw_path)
        return path, path == DEFAULT_FIXTURE
    mat_candidates = [candidate for candidate in DEFAULT_MAT_CANDIDATES if candidate.exists()]
    if mat_candidates:
        return mat_candidates, False
    resolved, fixture_mode = resolve_raw_path(None)
    return resolved, fixture_mode


def generate_data(
    *,
    raw_path: str | Path | None = None,
    cell_id: str | None = None,
    output_root: str | Path = "data/real/battery_nasa",
    data_dir: str | Path = "data",
    target_column: str = "soh",
    holdout_seed: int = 0,
    holdout_train_fraction: float = 0.8,
    within_cell_train_fraction: float = 0.8,
) -> dict:
    raw, fixture_mode = resolve_raw_paths(raw_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    normalized = write_normalized_artifacts(
        raw,
        normalized_csv=output / "cycle_level.csv",
        provenance_json=output / "provenance.json",
        source_url=DEFAULT_SOURCE_URL,
    )
    target = write_capacity_target(
        normalized,
        data_dir=data_dir,
        domain="real_battery_nasa_capacity",
        cell_id=cell_id,
        target_column=target_column,
    )
    cell_holdout = write_cell_holdout_split(
        normalized,
        path=output / "cell_holdout.json",
        train_fraction=holdout_train_fraction,
        seed=holdout_seed,
    )
    within_cell_split = write_within_cell_time_splits(
        normalized,
        path=output / "within_cell_split.json",
        train_fraction=within_cell_train_fraction,
    )
    manifest = {
        "status": "ok",
        "raw_path": [str(path) for path in raw] if isinstance(raw, list) else str(raw),
        "raw_path_source": "explicit" if raw_path is not None else ("auto" if not fixture_mode else "fixture"),
        "raw_candidates": [str(candidate) for candidate in DEFAULT_RAW_CANDIDATES],
        "normalized_csv": str(output / "cycle_level.csv"),
        "provenance_json": str(output / "provenance.json"),
        "cell_holdout_json": str(output / "cell_holdout.json"),
        "cell_holdout": cell_holdout,
        "within_cell_split_json": str(output / "within_cell_split.json"),
        "within_cell_split": within_cell_split,
        "target": target,
        "fixture_mode": fixture_mode,
        "warning": (
            "Default fixture is schema/smoke data only, not a scientific NASA battery result. "
            "Put downloaded NASA .mat files or a normalized cycle_level.csv under data/real/battery_nasa/raw/."
        )
        if fixture_mode
        else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the real_battery_nasa_capacity target.")
    parser.add_argument("--raw-path", help="NASA battery .mat file or cycle-level CSV. Defaults to schema fixture.")
    parser.add_argument("--cell-id", help="Optional cell ID to use for the dense capacity target.")
    parser.add_argument("--output-root", default="data/real/battery_nasa")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target-column", default="soh", choices=["soh", "capacity_ah"])
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--holdout-train-fraction", type=float, default=0.8)
    parser.add_argument("--within-cell-train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    manifest = generate_data(
        raw_path=args.raw_path,
        cell_id=args.cell_id,
        output_root=args.output_root,
        data_dir=args.data_dir,
        target_column=args.target_column,
        holdout_seed=args.holdout_seed,
        holdout_train_fraction=args.holdout_train_fraction,
        within_cell_train_fraction=args.within_cell_train_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

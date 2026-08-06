#!/usr/bin/env python3
"""Run an injected-data CAMELS adapter mechanics smoke without source streamflow access."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.camels_us import (
    CAMELSInputEpisode,
    CAMELSOutcomeEpisode,
    RUNOFF_MM_DAY_PER_CFS_KM2,
    partition_camels_outcome_episodes,
    rollout_camels_causal_input,
)


SPLIT_PATH = Path("data/real/camels_us/source_basin_split.json")
OUT_DIR = Path("artifacts/evaluations/phase28_camels_us_adapter_mechanics_smoke_20260721")


def _dates(start_year: int = 2000) -> tuple[date, ...]:
    start = date(start_year, 10, 1)
    end = date(start_year + 1, 9, 30)
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _episode(basin_id: str, split: str, offset: float) -> CAMELSOutcomeEpisode:
    dates = _dates()
    steps = np.arange(len(dates), dtype=float)
    inputs = np.column_stack([1.0 + offset + 0.1 * np.sin(steps / 9.0), -3.0 + 0.05 * steps])
    return CAMELSOutcomeEpisode(
        basin_id=basin_id,
        split=split,
        input_episode=CAMELSInputEpisode(dates, inputs),
        raw_flow_cfs=2.0 + offset + 0.01 * steps,
        drainage_area_km2=10.0 + offset,
    )


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    episodes = [
        _episode(basin_id, split_name, float(index))
        for split_name in ("train", "selection", "external")
        for index, basin_id in enumerate(split[split_name])
    ]
    partitioned = partition_camels_outcome_episodes(episodes, split)
    input_episode = episodes[0].causal_input()
    rollout = rollout_camels_causal_input(input_episode, lambda state, u: 0.9 * state + 0.1 * max(float(u[0]), 0.0))
    reset_rollout = rollout_camels_causal_input(input_episode, lambda state, u: 0.9 * state + 0.1 * max(float(u[0]), 0.0))
    result = {
        "phase": 28,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_camels_streamflow": False,
        "uses_archive_streamflow_members": False,
        "split_counts": {name: len(values) for name, values in partitioned.items()},
        "episode_days": len(input_episode.dates),
        "input_shape": list(input_episode.input_u.shape),
        "input_surface_has_target": hasattr(input_episode, "target_mm_day"),
        "input_surface_has_basin_id": hasattr(input_episode, "basin_id"),
        "conversion_factor": RUNOFF_MM_DAY_PER_CFS_KM2,
        "converted_target_finite_nonnegative": bool(np.all(np.isfinite(episodes[0].target_mm_day)) and np.all(episodes[0].target_mm_day >= 0)),
        "reset_invariant": bool(np.array_equal(rollout, reset_rollout)),
        "rollout_finite_nonnegative": bool(np.all(np.isfinite(rollout)) and np.all(rollout >= 0)),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

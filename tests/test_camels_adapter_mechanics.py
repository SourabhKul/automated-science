from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from core.real_data.camels_us import (
    CAMELSInputEpisode,
    CAMELSOutcomeEpisode,
    RUNOFF_MM_DAY_PER_CFS_KM2,
    convert_cfs_to_mm_day,
    partition_camels_outcome_episodes,
    rollout_camels_causal_input,
)


SPLIT = json.loads(Path("data/real/camels_us/source_basin_split.json").read_text())


def _dates() -> tuple[date, ...]:
    start = date(2004, 10, 1)
    end = date(2005, 9, 30)
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _outcome(basin_id: str, split: str) -> CAMELSOutcomeEpisode:
    dates = _dates()
    inputs = np.column_stack([np.linspace(0.0, 3.0, len(dates)), np.linspace(-4.0, 12.0, len(dates))])
    return CAMELSOutcomeEpisode(basin_id, split, CAMELSInputEpisode(dates, inputs), np.full(len(dates), 10.0), 2.0)


def main() -> None:
    episode = _outcome(SPLIT["train"][0], "train")
    expected = np.full(len(episode.input_episode.dates), 10.0 * RUNOFF_MM_DAY_PER_CFS_KM2 / 2.0)
    assert np.allclose(episode.target_mm_day, expected)
    assert not hasattr(episode.causal_input(), "target_mm_day")
    assert not hasattr(episode.causal_input(), "basin_id")

    rollout = rollout_camels_causal_input(episode.causal_input(), lambda state, u: 0.8 * state + 0.2 * max(float(u[0]), 0.0))
    assert np.all(np.isfinite(rollout)) and np.all(rollout >= 0)
    assert np.array_equal(rollout, rollout_camels_causal_input(episode.causal_input(), lambda state, u: 0.8 * state + 0.2 * max(float(u[0]), 0.0)))

    records = [_outcome(basin_id, split_name) for split_name in ("train", "selection", "external") for basin_id in SPLIT[split_name]]
    partitioned = partition_camels_outcome_episodes(records, SPLIT)
    assert {name: len(values) for name, values in partitioned.items()} == {"train": 36, "selection": 12, "external": 16}
    wrong = list(records)
    wrong[0] = _outcome(SPLIT["train"][0], "external")
    try:
        partition_camels_outcome_episodes(wrong, SPLIT)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-split CAMELS episode must fail")

    try:
        convert_cfs_to_mm_day(np.array([-1.0]), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative raw flow must fail")
    print("SUCCESS: CAMELS injected adapter is causal-input-only, reset-safe, and split-locked")


if __name__ == "__main__":
    main()

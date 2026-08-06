from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_electricity_load import (
    HISTORY_LAGS,
    SOURCE_KW_UNIT,
    WEEK_SAMPLES,
    ElectricityLoadOutcomeWeek,
    make_electricity_load_input_week,
    native_week_calendar,
    partition_electricity_load_outcome_weeks,
    rollout_electricity_load_causal_input,
)


SPLIT_PATH = Path("data/real/uci_electricity_load/source_client_split.json")


def _target(offset: float) -> np.ndarray:
    steps = np.arange(WEEK_SAMPLES, dtype=float)
    return 2.0 + offset + 0.2 * np.sin(steps / 17.0) + 0.01 * (steps % 96)


def _weeks() -> list[ElectricityLoadOutcomeWeek]:
    split = json.loads(SPLIT_PATH.read_text())
    return [
        ElectricityLoadOutcomeWeek(client_id, split_name, "synthetic-week-001", _target(float(index)))
        for split_name in ("train", "selection", "external")
        for index, client_id in enumerate(split[split_name])
    ]


def main() -> None:
    assert SOURCE_KW_UNIT == "source kW"
    assert HISTORY_LAGS == (1, 96)
    assert native_week_calendar().shape == (WEEK_SAMPLES, 2)
    weeks = _weeks()
    partitioned = partition_electricity_load_outcome_weeks(weeks, SPLIT_PATH)
    assert {name: len(values) for name, values in partitioned.items()} == {"train": 240, "selection": 65, "external": 65}

    outcome = weeks[0]
    input_week = outcome.causal_input()
    assert not hasattr(input_week, "target_kw")
    assert not hasattr(input_week, "client_id")
    assert input_week.causal_history_kw.shape == (WEEK_SAMPLES, 2)
    assert np.array_equal(input_week.causal_history_kw[0], np.zeros(2))
    assert input_week.causal_history_kw[1, 0] == outcome.target_kw[0]
    assert input_week.causal_history_kw[95, 1] == 0.0
    assert input_week.causal_history_kw[96, 1] == outcome.target_kw[0]

    def predictor(history: np.ndarray, calendar: np.ndarray) -> float:
        return 0.5 * float(history[0]) + 0.1 * float(history[1]) + 0.01 * float(calendar[0])

    first = rollout_electricity_load_causal_input(input_week, predictor, upper_bound_kw=1000.0)
    second = rollout_electricity_load_causal_input(input_week, predictor, upper_bound_kw=1000.0)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first)) and np.all(first >= 0)

    same_surface = make_electricity_load_input_week(outcome.target_kw)
    assert np.array_equal(first, rollout_electricity_load_causal_input(same_surface, predictor, upper_bound_kw=1000.0))
    wrong = list(weeks)
    wrong[0] = ElectricityLoadOutcomeWeek(wrong[0].client_id, "external", wrong[0].source_week_label, wrong[0].target_kw)
    try:
        partition_electricity_load_outcome_weeks(wrong, SPLIT_PATH)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("cross-split electricity-load week must fail")

    try:
        rollout_electricity_load_causal_input(input_week, lambda history, calendar: float("nan"), upper_bound_kw=1000.0)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite electricity-load rollout must fail")
    print("SUCCESS: electricity-load injected adapter is causal-history-only, reset-safe, target-isolated, and split-locked")


if __name__ == "__main__":
    main()

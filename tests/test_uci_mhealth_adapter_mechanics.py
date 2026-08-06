from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_mhealth import (
    ACTIVITY_LABELS,
    MOTION_CHANNELS,
    WINDOW_SAMPLES,
    MHealthInjectedWindow,
    motion_window_from_payload,
    partition_injected_windows,
    predict_independent_windows,
)


SPLIT_PATH = Path("data/real/uci_mhealth/source_subject_split.json")


def _window(offset: float) -> np.ndarray:
    return np.arange(MOTION_CHANNELS * WINDOW_SAMPLES, dtype=float).reshape(MOTION_CHANNELS, WINDOW_SAMPLES) + offset


def _windows() -> list[MHealthInjectedWindow]:
    split = json.loads(SPLIT_PATH.read_text())
    return [
        MHealthInjectedWindow(int(subject), split_name, f"artificial-{subject}", _window(float(index)), 1 + index % len(ACTIVITY_LABELS))
        for split_name in ("train", "selection", "external")
        for index, subject in enumerate(split[split_name])
    ]


def main() -> None:
    windows = _windows()
    partitioned = partition_injected_windows(windows, SPLIT_PATH)
    assert {name: len(items) for name, items in partitioned.items()} == {"train": 4, "selection": 3, "external": 3}
    record = windows[0]
    candidate = record.candidate_input()
    assert candidate.motion_window.shape == (MOTION_CHANNELS, WINDOW_SAMPLES)
    assert not hasattr(candidate, "artificial_label")
    assert not hasattr(candidate, "subject_id")
    assert np.array_equal(candidate.motion_window, record.artificial_motion_window)
    assert np.array_equal(motion_window_from_payload({"motion_window": candidate.motion_window}).motion_window, candidate.motion_window)
    for forbidden in ("artificial_label", "subject_id", "row_index", "video_flag"):
        try:
            motion_window_from_payload({"motion_window": candidate.motion_window, forbidden: 1})
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError(f"{forbidden} must be rejected")

    states: list[float] = []
    probabilities = predict_independent_windows(
        [item.candidate_input() for item in windows],
        lambda state, values: states.append(state) or np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS)),
    )
    assert states == [0.0] * len(windows)
    assert np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)
    try:
        predict_independent_windows([candidate], lambda state, values: np.full(len(ACTIVITY_LABELS), float("nan")))
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("nonfinite probabilities must fail")
    wrong = list(windows)
    first = wrong[0]
    wrong[0] = MHealthInjectedWindow(first.subject_id, "external", first.artificial_window_key, first.artificial_motion_window, first.artificial_label)
    try:
        partition_injected_windows(wrong, SPLIT_PATH)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("cross-split injected record must fail")
    print("SUCCESS: MHEALTH injected adapter is raw-window-only, reset-safe, target-isolated, and split-locked")


if __name__ == "__main__":
    main()

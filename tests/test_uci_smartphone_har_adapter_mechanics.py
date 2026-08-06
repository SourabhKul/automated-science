from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_smartphone_har import (
    CAUSAL_PREFIX_SAMPLES,
    CHANNEL_COUNT,
    WINDOW_SAMPLES,
    SmartphoneHARInjectedWindow,
    causal_prefix_from_payload,
    partition_injected_windows,
    predict_smartphone_har_prefixes,
)


SPLIT_PATH = Path("data/real/uci_smartphone_har/source_subject_split.json")


def _window(offset: float) -> np.ndarray:
    return np.arange(CHANNEL_COUNT * WINDOW_SAMPLES, dtype=float).reshape(CHANNEL_COUNT, WINDOW_SAMPLES) + offset


def _windows() -> list[SmartphoneHARInjectedWindow]:
    split = json.loads(SPLIT_PATH.read_text())
    return [
        SmartphoneHARInjectedWindow(int(subject), split_name, "artificial-row-001", _window(float(index)), 1 + index % 6)
        for split_name in ("train", "selection", "external")
        for index, subject in enumerate(split[split_name])
    ]


def main() -> None:
    windows = _windows()
    partitioned = partition_injected_windows(windows, SPLIT_PATH)
    assert {name: len(values) for name, values in partitioned.items()} == {"train": 15, "selection": 6, "external": 9}
    outcome = windows[0]
    prefix = outcome.causal_input()
    assert prefix.prefix.shape == (CHANNEL_COUNT, CAUSAL_PREFIX_SAMPLES)
    assert not hasattr(prefix, "artificial_label")
    assert not hasattr(prefix, "subject_id")
    assert np.array_equal(prefix.prefix, outcome.artificial_window[:, :CAUSAL_PREFIX_SAMPLES])

    suffix_changed = outcome.artificial_window.copy()
    suffix_changed[:, CAUSAL_PREFIX_SAMPLES:] += 50.0
    suffix_outcome = SmartphoneHARInjectedWindow(outcome.subject_id, outcome.split, outcome.source_row_key, suffix_changed, outcome.artificial_label)
    assert np.array_equal(prefix.prefix, suffix_outcome.causal_input().prefix)
    for leaked in ({"prefix": prefix.prefix, "artificial_label": 1}, {"prefix": prefix.prefix, "subject_id": outcome.subject_id}):
        try:
            causal_prefix_from_payload(leaked)
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError("target or subject ID payload must fail")

    probabilities = predict_smartphone_har_prefixes([prefix], lambda values: np.full(6, 1.0 / 6.0))
    assert probabilities.shape == (1, 6)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    try:
        predict_smartphone_har_prefixes([prefix], lambda values: np.array([float("nan")] * 6))
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("nonfinite probabilities must fail")

    wrong = list(windows)
    wrong[0] = SmartphoneHARInjectedWindow(wrong[0].subject_id, "external", wrong[0].source_row_key, wrong[0].artificial_window, wrong[0].artificial_label)
    try:
        partition_injected_windows(wrong, SPLIT_PATH)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("cross-split injected window must fail")
    print("SUCCESS: Smartphone HAR injected adapter is prefix-only, reset-safe, target-isolated, and split-locked")


if __name__ == "__main__":
    main()

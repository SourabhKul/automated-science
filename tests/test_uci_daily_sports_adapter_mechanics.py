from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_daily_sports import (
    ACTIVITY_LABELS,
    PREFIX_SAMPLES,
    SOURCE_CHANNELS,
    DailySportsInjectedRecord,
    partition_injected_records,
    predict_independent_prefixes,
    prefix_from_payload,
)


SPLIT = Path("data/real/uci_daily_sports/source_subject_split.json")


def main() -> None:
    split = json.loads(SPLIT.read_text())
    records = [
        DailySportsInjectedRecord(int(participant), name, f"artificial-{participant}", np.ones((PREFIX_SAMPLES, SOURCE_CHANNELS)), np.zeros((61, SOURCE_CHANNELS)), 1)
        for name in ("train", "selection", "external")
        for participant in split[name]
    ]
    assert {name: len(items) for name, items in partition_injected_records(records, SPLIT).items()} == {"train": 4, "selection": 2, "external": 2}
    candidate = records[0].candidate_input()
    assert candidate.prefix.shape == (PREFIX_SAMPLES, SOURCE_CHANNELS)
    assert not hasattr(candidate, "artificial_label") and not hasattr(candidate, "participant_id") and not hasattr(candidate, "artificial_suffix")
    assert np.array_equal(prefix_from_payload({"prefix": candidate.prefix}).prefix, candidate.prefix)
    for forbidden in ("artificial_label", "participant_id", "member_path", "suffix"):
        try:
            prefix_from_payload({"prefix": candidate.prefix, forbidden: 1})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{forbidden} must be rejected")
    states: list[float] = []
    probabilities = predict_independent_prefixes([candidate], lambda state, prefix: states.append(state) or np.full(len(ACTIVITY_LABELS), 1 / len(ACTIVITY_LABELS)))
    assert states == [0.0] and np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1)
    print("SUCCESS: Daily Sports injected adapter is prefix-only, reset-safe, isolated, and split-locked")


if __name__ == "__main__":
    main()

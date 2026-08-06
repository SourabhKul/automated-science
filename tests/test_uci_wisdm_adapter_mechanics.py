from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_wisdm import (
    ACTIVITY_LABELS,
    FULL_SEGMENT_SAMPLES,
    PREFIX_SAMPLES,
    SOURCE_AXES,
    WISDMInjectedRecord,
    partition_injected_records,
    predict_independent_prefixes,
    prefix_from_payload,
)


SPLIT = Path("data/real/uci_wisdm/source_subject_split.json")


def main() -> None:
    split = json.loads(SPLIT.read_text())
    records = [
        WISDMInjectedRecord(
            int(subject),
            partition,
            f"artificial/{subject}.txt",
            np.ones((PREFIX_SAMPLES, SOURCE_AXES)),
            np.zeros((FULL_SEGMENT_SAMPLES - PREFIX_SAMPLES, SOURCE_AXES)),
            ACTIVITY_LABELS[0],
            np.arange(FULL_SEGMENT_SAMPLES, dtype=np.int64),
        )
        for partition in ("train_subjects", "selection_subjects", "external_subjects")
        for subject in split[partition]
    ]
    assert {name: len(items) for name, items in partition_injected_records(records, SPLIT).items()} == {"train_subjects": 31, "selection_subjects": 10, "external_subjects": 10}
    candidate = records[0].candidate_input()
    assert candidate.prefix.shape == (PREFIX_SAMPLES, SOURCE_AXES)
    assert not any(hasattr(candidate, forbidden) for forbidden in ("artificial_label", "subject_id", "member_path", "artificial_suffix", "artificial_timestamps"))
    assert np.array_equal(prefix_from_payload({"prefix": candidate.prefix}).prefix, candidate.prefix)
    for forbidden in ("artificial_label", "subject_id", "member_path", "timestamps", "suffix"):
        try:
            prefix_from_payload({"prefix": candidate.prefix, forbidden: 1})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{forbidden} must be rejected")
    states: list[float] = []
    probabilities = predict_independent_prefixes([candidate], lambda state, prefix: states.append(state) or np.full(len(ACTIVITY_LABELS), 1 / len(ACTIVITY_LABELS)))
    assert states == [0.0] and np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1)
    print("SUCCESS: WISDM injected adapter is prefix-only, reset-safe, isolated, and split-locked")


if __name__ == "__main__":
    main()

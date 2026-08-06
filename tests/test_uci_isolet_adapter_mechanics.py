from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from core.real_data.uci_isolet import CLASS_LABELS, FEATURE_COUNT, ISOLETInjectedRecord, vector_from_payload, verify_injected_source_files, predict_independent_vectors


SPLIT_PATH = Path("data/real/uci_isolet/source_file_split.json")


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    records = [ISOLETInjectedRecord(source, f"artificial-{index}", np.arange(FEATURE_COUNT, dtype=float) + index, 1 + index) for index, source in enumerate((split["source_train"], split["external"]))]
    partitioned = verify_injected_source_files(records, SPLIT_PATH)
    assert {name: len(items) for name, items in partitioned.items()} == {"source_train": 1, "external": 1}
    candidate = records[0].candidate_input()
    assert candidate.features.shape == (FEATURE_COUNT,)
    assert not hasattr(candidate, "artificial_label") and not hasattr(candidate, "source_file")
    for forbidden in ("label", "source_file", "selection_assignment", "row_index", "speaker_id"):
        try:
            vector_from_payload({"features": candidate.features, forbidden: 1})
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError(f"{forbidden} must be rejected")
    states: list[float] = []
    probs = predict_independent_vectors([item.candidate_input() for item in records], lambda state, values: states.append(state) or np.full(len(CLASS_LABELS), 1.0 / len(CLASS_LABELS)))
    assert states == [0.0, 0.0] and np.allclose(probs.sum(axis=1), 1.0)
    print("SUCCESS: ISOLET injected adapter is raw-vector-only, reset-safe, target-isolated, and file-split-locked")


if __name__ == "__main__":
    main()

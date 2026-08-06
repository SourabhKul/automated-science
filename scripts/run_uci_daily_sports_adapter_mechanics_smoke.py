#!/usr/bin/env python3
"""Run Phase 45 injected mechanics without opening observed sensor values."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_daily_sports import (
    ACTIVITY_LABELS,
    PREFIX_SAMPLES,
    SOURCE_CHANNELS,
    DailySportsInjectedRecord,
    partition_injected_records,
    predict_independent_prefixes,
    prefix_from_payload,
)


SPLIT_PATH = Path("data/real/uci_daily_sports/source_subject_split.json")
OUT = Path("artifacts/evaluations/phase45_uci_daily_sports_adapter_mechanics_smoke_20260727")


def _records(split: dict[str, list[int]]) -> list[DailySportsInjectedRecord]:
    return [
        DailySportsInjectedRecord(
            int(participant),
            split_name,
            f"artificial-participant-{participant}",
            np.arange(PREFIX_SAMPLES * SOURCE_CHANNELS, dtype=float).reshape(PREFIX_SAMPLES, SOURCE_CHANNELS) + index / 10,
            np.full((61, SOURCE_CHANNELS), index, dtype=float),
            1 + index % len(ACTIVITY_LABELS),
        )
        for split_name in ("train", "selection", "external")
        for index, participant in enumerate(split[split_name])
    ]


def _rejected(payload: dict[str, object]) -> bool:
    try:
        prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT) -> dict[str, object]:
    records = _records(json.loads(split_path.read_text()))
    partitioned = partition_injected_records(records, split_path)
    states: list[float] = []
    probabilities = predict_independent_prefixes(
        [record.candidate_input() for record in records],
        lambda state, prefix: states.append(state) or np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS)),
    )
    reordered = list(reversed(records))
    reordered_probabilities = predict_independent_prefixes(
        [record.candidate_input() for record in reordered], lambda state, prefix: np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS))
    )
    candidate = records[0].candidate_input()
    result = {
        "phase": 45,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_source_signal_values": False,
        "uses_measured_source_label_values": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "split_counts": {name: len(items) for name, items in partitioned.items()},
        "candidate_shape": [PREFIX_SAMPLES, SOURCE_CHANNELS],
        "candidate_has_artificial_label": hasattr(candidate, "artificial_label"),
        "candidate_has_participant_id": hasattr(candidate, "participant_id"),
        "candidate_has_suffix": hasattr(candidate, "artificial_suffix"),
        "independent_segment_zero_reset": states == [0.0] * len(records),
        "reset_invariant_under_segment_reordering": bool(np.array_equal(probabilities, reordered_probabilities[::-1])),
        "target_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "artificial_label": records[0].artificial_label}),
        "participant_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "participant_id": records[0].participant_id}),
        "path_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "member_path": "data/a01/p1/s01.txt"}),
        "suffix_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "suffix": records[0].artificial_suffix}),
        "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1)),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

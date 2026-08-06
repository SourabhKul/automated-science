#!/usr/bin/env python3
"""Run Phase 50 injected mechanics without opening observed WISDM values."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


SPLIT_PATH = Path("data/real/uci_wisdm/source_subject_split.json")
OUT = Path("artifacts/evaluations/phase50_uci_wisdm_adapter_mechanics_smoke_20260728")


def _records(split: dict[str, list[int]]) -> list[WISDMInjectedRecord]:
    records: list[WISDMInjectedRecord] = []
    for partition_index, partition in enumerate(("train_subjects", "selection_subjects", "external_subjects")):
        for subject_index, subject in enumerate(split[partition]):
            state = partition_index * 100 + subject_index
            records.append(
                WISDMInjectedRecord(
                    int(subject),
                    partition,
                    f"artificial/raw/phone/accel/data_{subject}_accel_phone.txt",
                    np.full((PREFIX_SAMPLES, SOURCE_AXES), float(state), dtype=float),
                    np.full((FULL_SEGMENT_SAMPLES - PREFIX_SAMPLES, SOURCE_AXES), float(-state), dtype=float),
                    ACTIVITY_LABELS[state % len(ACTIVITY_LABELS)],
                    np.arange(FULL_SEGMENT_SAMPLES, dtype=np.int64) + state * FULL_SEGMENT_SAMPLES,
                )
            )
    return records


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
    uniform = np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS))
    probabilities = predict_independent_prefixes(
        [record.candidate_input() for record in records],
        lambda state, prefix: states.append(state) or uniform,
    )
    reordered_probabilities = predict_independent_prefixes(
        [record.candidate_input() for record in reversed(records)], lambda state, prefix: uniform
    )
    candidate, record = records[0].candidate_input(), records[0]
    result = {
        "phase": 50,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_source_acceleration_values": False,
        "uses_measured_source_activity_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "split_counts": {name: len(items) for name, items in partitioned.items()},
        "candidate_shape": [PREFIX_SAMPLES, SOURCE_AXES],
        "candidate_has_artificial_label": hasattr(candidate, "artificial_label"),
        "candidate_has_subject_id": hasattr(candidate, "subject_id"),
        "candidate_has_member_path": hasattr(candidate, "member_path"),
        "candidate_has_suffix": hasattr(candidate, "artificial_suffix"),
        "candidate_has_timestamps": hasattr(candidate, "artificial_timestamps"),
        "independent_segment_zero_reset": states == [0.0] * len(records),
        "reset_invariant_under_segment_reordering": bool(np.array_equal(probabilities, reordered_probabilities[::-1])),
        "target_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "artificial_label": record.artificial_label}),
        "subject_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "subject_id": record.subject_id}),
        "path_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "member_path": record.member_path}),
        "timestamp_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "timestamps": record.artificial_timestamps}),
        "suffix_isolation_sentinel_rejected": _rejected({"prefix": candidate.prefix, "suffix": record.artificial_suffix}),
        "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1)),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

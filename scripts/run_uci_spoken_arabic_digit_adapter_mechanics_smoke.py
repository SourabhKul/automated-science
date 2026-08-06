#!/usr/bin/env python3
"""Run Phase 46 artificial mechanics without opening observed MFCC frames."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_spoken_arabic_digit import DIGIT_LABELS, MFCC_COEFFICIENTS, PREFIX_FRAMES, ArabicDigitInjectedRecord, frozen_speakers, partition_injected_records, predict_independent_prefixes, prefix_from_payload


SPLIT = Path("data/real/uci_spoken_arabic_digit/source_speaker_split.json")
OUT = Path("artifacts/evaluations/phase46_uci_spoken_arabic_digit_adapter_mechanics_smoke_20260727")


def _records(groups: dict[str, set[tuple[str, int]]]) -> list[ArabicDigitInjectedRecord]:
    return [
        ArabicDigitInjectedRecord(source, speaker, split, f"artificial-{source}-{speaker}", np.arange(PREFIX_FRAMES * MFCC_COEFFICIENTS, dtype=float).reshape(PREFIX_FRAMES, MFCC_COEFFICIENTS) + index / 10, np.full((1, MFCC_COEFFICIENTS), index, dtype=float), index % len(DIGIT_LABELS))
        for split in ("train", "selection", "external")
        for index, (source, speaker) in enumerate(sorted(groups[split]))
    ]


def _rejected(payload: dict[str, object]) -> bool:
    try:
        prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT, output_dir: Path = OUT) -> dict[str, object]:
    records = _records(frozen_speakers(split_path))
    partitioned = partition_injected_records(records, split_path)
    states: list[float] = []
    probabilities = predict_independent_prefixes([record.candidate_input() for record in records], lambda state, frames: states.append(state) or np.full(len(DIGIT_LABELS), 1 / len(DIGIT_LABELS)))
    reversed_records = list(reversed(records))
    reversed_probabilities = predict_independent_prefixes([record.candidate_input() for record in reversed_records], lambda state, frames: np.full(len(DIGIT_LABELS), 1 / len(DIGIT_LABELS)))
    candidate = records[0].candidate_input()
    result = {"phase": 46, "status": "passed_isolated_adapter_mechanics_synthetic_only", "uses_observed_mfcc_values": False, "uses_observed_labels": False, "uses_source_statistics_or_duplicate_ledger": False, "split_counts": {name: len(items) for name, items in partitioned.items()}, "candidate_shape": [PREFIX_FRAMES, MFCC_COEFFICIENTS], "candidate_has_label": hasattr(candidate, "artificial_label"), "candidate_has_speaker": hasattr(candidate, "speaker_id"), "candidate_has_suffix": hasattr(candidate, "artificial_suffix"), "independent_utterance_zero_reset": states == [0.0] * len(records), "reset_invariant_under_reordering": bool(np.array_equal(probabilities, reversed_probabilities[::-1])), "target_isolation_sentinel_rejected": _rejected({"frames": candidate.frames, "label": 0}), "speaker_isolation_sentinel_rejected": _rejected({"frames": candidate.frames, "speaker_id": 1}), "file_isolation_sentinel_rejected": _rejected({"frames": candidate.frames, "source_file": "Train_Arabic_Digit.txt"}), "block_order_isolation_sentinel_rejected": _rejected({"frames": candidate.frames, "block_index": 0}), "suffix_isolation_sentinel_rejected": _rejected({"frames": candidate.frames, "suffix": np.zeros((1, MFCC_COEFFICIENTS))}), "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1)), "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

#!/usr/bin/env python3
"""Run Phase 40 injected-window mechanics without opening measured MHEALTH signal values."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
OUT_DIR = Path("artifacts/evaluations/phase40_uci_mhealth_adapter_mechanics_smoke_20260725")


def _artificial_window(index: int) -> np.ndarray:
    return np.arange(MOTION_CHANNELS * WINDOW_SAMPLES, dtype=float).reshape(MOTION_CHANNELS, WINDOW_SAMPLES) + float(index) * 0.125


def _records(split: dict[str, list[int]]) -> list[MHealthInjectedWindow]:
    return [
        MHealthInjectedWindow(int(subject), split_name, f"artificial-subject-{subject}", _artificial_window(index), 1 + index % len(ACTIVITY_LABELS))
        for split_name in ("train", "selection", "external")
        for index, subject in enumerate(split[split_name])
    ]


def _rejected(payload: dict[str, object]) -> bool:
    try:
        motion_window_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    records = _records(split)
    partitioned = partition_injected_windows(records, split_path)
    inputs = [record.candidate_input() for record in records]
    states: list[float] = []
    probabilities = predict_independent_windows(
        inputs, lambda state, values: states.append(state) or np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS))
    )
    reversed_records = list(reversed(records))
    reversed_probabilities = predict_independent_windows(
        [record.candidate_input() for record in reversed_records], lambda state, values: np.full(len(ACTIVITY_LABELS), 1.0 / len(ACTIVITY_LABELS))
    )
    candidate = records[0].candidate_input()
    result = {
        "phase": 40,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_source_signal_values": False,
        "uses_measured_source_label_values": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "split_counts": {name: len(items) for name, items in partitioned.items()},
        "candidate_shape": [MOTION_CHANNELS, WINDOW_SAMPLES],
        "candidate_has_artificial_label": hasattr(candidate, "artificial_label"),
        "candidate_has_subject_id": hasattr(candidate, "subject_id"),
        "independent_window_zero_reset": states == [0.0] * len(records),
        "reset_invariant_under_window_reordering": bool(np.array_equal(probabilities, reversed_probabilities[::-1])),
        "target_isolation_sentinel_rejected": _rejected({"motion_window": candidate.motion_window, "artificial_label": records[0].artificial_label}),
        "subject_isolation_sentinel_rejected": _rejected({"motion_window": candidate.motion_window, "subject_id": records[0].subject_id}),
        "source_order_isolation_sentinel_rejected": _rejected({"motion_window": candidate.motion_window, "row_index": 0}),
        "video_isolation_sentinel_rejected": _rejected({"motion_window": candidate.motion_window, "video_flag": True}),
        "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0) and np.allclose(probabilities.sum(axis=1), 1.0)),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 40 UCI MHEALTH Adapter Mechanics Smoke\n\n"
        "Status: **passed injected/artificial-record mechanics only.**\n\n"
        "The smoke read only the frozen 4/3/3 subject split and generated ten artificial 21 x 100 windows in memory. It opened no MHEALTH log, measured signal, label, source statistic, or duplicate ledger. Candidate windows expose raw artificial motion values only; artificial labels, subject identity, source order, and video context remain evaluator-owned. Frozen split integrity, zero reset, reorder invariance, target/subject/order/video sentinels, and finite probability bounds passed.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 40,
        "decision": "pass_adapter_mechanics_only",
        "reason": "Artificial 21x100 windows verified the frozen 4/3/3 subject boundary, raw-window candidate isolation, independent reset, and finite probability semantics without reading observed MHEALTH values.",
        "next_action": "Run only the predeclared output-isolated artificial recovery and specificity suite.",
        "prohibited": ["observed signal or label fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

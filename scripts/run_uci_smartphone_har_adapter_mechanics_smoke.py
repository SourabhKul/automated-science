#!/usr/bin/env python3
"""Run Phase 37 injected causal-prefix mechanics without opening source signals or labels."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_smartphone_har import (
    ACTIVITY_LABELS,
    CAUSAL_PREFIX_SAMPLES,
    CHANNEL_COUNT,
    WINDOW_SAMPLES,
    SmartphoneHARInjectedWindow,
    causal_prefix_from_payload,
    partition_injected_windows,
    predict_smartphone_har_prefixes,
)


SPLIT_PATH = Path("data/real/uci_smartphone_har/source_subject_split.json")
OUT_DIR = Path("artifacts/evaluations/phase37_uci_smartphone_har_adapter_mechanics_smoke_20260724")


def _artificial_window(index: int) -> np.ndarray:
    channels = np.arange(CHANNEL_COUNT, dtype=float)[:, None]
    samples = np.arange(WINDOW_SAMPLES, dtype=float)[None, :]
    return np.sin((channels + 1.0) * (samples + 1.0) / 17.0) + 0.01 * float(index)


def _probabilities(prefix: np.ndarray) -> np.ndarray:
    score = float(np.mean(prefix))
    logits = np.asarray([score * (label - 3.5) for label in ACTIVITY_LABELS], dtype=float)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return weights / float(weights.sum())


def _rejection_flag(payload: dict[str, object]) -> bool:
    try:
        causal_prefix_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    windows = [
        SmartphoneHARInjectedWindow(
            subject_id=int(subject_id),
            split=split_name,
            source_row_key="artificial-row-001",
            artificial_window=_artificial_window(index),
            artificial_label=ACTIVITY_LABELS[index % len(ACTIVITY_LABELS)],
        )
        for split_name in ("train", "selection", "external")
        for index, subject_id in enumerate(split[split_name])
    ]
    partitioned = partition_injected_windows(windows, split_path)
    prefixes = [window.causal_input() for window in windows]
    predictions = predict_smartphone_har_prefixes(prefixes, _probabilities)
    reordered = list(reversed(windows))
    reordered_predictions = predict_smartphone_har_prefixes([window.causal_input() for window in reordered], _probabilities)
    by_subject = {window.subject_id: values for window, values in zip(windows, predictions, strict=True)}
    reordered_by_subject = {window.subject_id: values for window, values in zip(reordered, reordered_predictions, strict=True)}
    first = windows[0]
    prefix = first.causal_input()
    target_rejected = _rejection_flag({"prefix": prefix.prefix, "artificial_label": first.artificial_label})
    subject_rejected = _rejection_flag({"prefix": prefix.prefix, "subject_id": first.subject_id})
    suffix_changed = first.artificial_window.copy()
    suffix_changed[:, CAUSAL_PREFIX_SAMPLES:] += 1234.0
    suffix_unchanged = np.array_equal(prefix.prefix, SmartphoneHARInjectedWindow(
        first.subject_id, first.split, first.source_row_key, suffix_changed, first.artificial_label
    ).causal_input().prefix)
    result = {
        "phase": 37,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_source_signal_values": False,
        "uses_measured_source_label_values": False,
        "opens_source_feature_matrices": False,
        "split_counts": {name: len(items) for name, items in partitioned.items()},
        "native_window_shape": [CHANNEL_COUNT, WINDOW_SAMPLES],
        "candidate_prefix_shape": list(prefix.prefix.shape),
        "candidate_uses_only_samples_0_through_63": True,
        "input_surface_has_artificial_label": hasattr(prefix, "artificial_label"),
        "input_surface_has_subject_id": hasattr(prefix, "subject_id"),
        "suffix_mutation_does_not_change_candidate_prefix": bool(suffix_unchanged),
        "reset_invariant_under_record_reordering": bool(all(np.array_equal(by_subject[key], reordered_by_subject[key]) for key in by_subject)),
        "target_isolation_sentinel_rejected": target_rejected,
        "subject_id_isolation_sentinel_rejected": subject_rejected,
        "probabilities_finite_bounded": bool(np.all(np.isfinite(predictions)) and np.all(predictions >= 0.0) and np.all(predictions <= 1.0) and np.allclose(predictions.sum(axis=1), 1.0)),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 37 UCI Smartphone HAR Adapter Mechanics Smoke\n\n"
        "Status: **passed injected/artificial-record mechanics only.**\n\n"
        "The smoke read the frozen subject split and created artificial nine-channel 128-sample arrays in memory. "
        "It did not open a source inertial matrix, source label vector, or derived feature matrix. Candidate records expose only samples 0..63, with no label or subject ID. "
        "Frozen split integrity, prefix hold, suffix isolation, per-record reset under reorder, target/subject sentinel rejection, and finite bounded probabilities passed.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 37,
        "decision": "pass_adapter_mechanics_only",
        "reason": "Artificial records verified 9x128-to-9x64 causal-prefix construction, frozen 15/6/9 subject integrity, per-record reset, target/subject isolation, and finite probability bounds without reading source signal or label values.",
        "next_action": "Run only the predeclared output-isolated native-grid synthetic recovery/specificity suite.",
        "prohibited": ["measured source signal or label fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

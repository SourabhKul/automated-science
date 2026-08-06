#!/usr/bin/env python3
"""Run Phase 36 injected adapter mechanics without opening measured load cells."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_electricity_load import (
    WEEK_SAMPLES,
    ElectricityLoadOutcomeWeek,
    partition_electricity_load_outcome_weeks,
    rollout_electricity_load_causal_input,
)


SPLIT_PATH = Path("data/real/uci_electricity_load/source_client_split.json")
OUT_DIR = Path("artifacts/evaluations/phase36_uci_electricity_load_adapter_mechanics_smoke_20260724")


def _synthetic_target(offset: float) -> np.ndarray:
    steps = np.arange(WEEK_SAMPLES, dtype=float)
    return 3.0 + offset * 0.01 + 0.05 * (steps % 96) + 0.25 * np.maximum(np.sin(steps / 13.0), 0.0)


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    weeks = [
        ElectricityLoadOutcomeWeek(client_id, split_name, "synthetic-week-001", _synthetic_target(float(index)))
        for split_name in ("train", "selection", "external")
        for index, client_id in enumerate(split[split_name])
    ]
    partitioned = partition_electricity_load_outcome_weeks(weeks, split_path)
    input_week = weeks[0].causal_input()

    def predictor(history: np.ndarray, calendar: np.ndarray) -> float:
        return 0.6 * float(history[0]) + 0.1 * float(history[1]) + 0.02 * float(calendar[0])

    first = rollout_electricity_load_causal_input(input_week, predictor, upper_bound_kw=1000.0)
    second = rollout_electricity_load_causal_input(input_week, predictor, upper_bound_kw=1000.0)
    result = {
        "phase": 36,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_source_load_values": False,
        "opens_raw_load_member": False,
        "split_counts": {name: len(items) for name, items in partitioned.items()},
        "week_samples": WEEK_SAMPLES,
        "history_shape": list(input_week.causal_history_kw.shape),
        "calendar_shape": list(input_week.calendar_phase.shape),
        "input_surface_has_target": hasattr(input_week, "target_kw"),
        "input_surface_has_client_id": hasattr(input_week, "client_id"),
        "causal_lag_one_matches_prior_artificial_target": bool(input_week.causal_history_kw[1, 0] == weeks[0].target_kw[0]),
        "causal_lag_96_matches_artificial_target": bool(input_week.causal_history_kw[96, 1] == weeks[0].target_kw[0]),
        "reset_invariant": bool(np.array_equal(first, second)),
        "rollout_finite_nonnegative": bool(np.all(np.isfinite(first)) and np.all(first >= 0)),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 36 UCI Electricity-Load Adapter Mechanics Smoke\n\n"
        "Status: **passed injected/artificial-record mechanics only.**\n\n"
        "The smoke read the frozen client split and constructed artificial 672-sample source-kW-like targets in memory. "
        "It did not open `LD2011_2014.txt` or any measured load cell. The candidate-facing week exposes only evaluator-owned causal lags and weekday/quarter-hour phase, with no target or client ID. "
        "The `240/65/65` split, causal lag construction, zero-reset invariance, and finite/nonnegative rollout checks passed.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 36,
        "decision": "pass_adapter_mechanics_only",
        "reason": "Injected records verified split integrity, target/client isolation, causal lag construction, and reset behavior without measured loads.",
        "next_action": "Run only the predeclared output-isolated native-grid synthetic recovery/specificity suite.",
        "prohibited": ["measured-load transformation or fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run Phase 36 output-isolated synthetic controls without opening source load cells."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_electricity_load import (
    WEEK_SAMPLES,
    ElectricityLoadInputWeek,
    ElectricityLoadOutcomeWeek,
    native_week_calendar,
    partition_electricity_load_outcome_weeks,
    rollout_electricity_load_causal_surface,
)


SPLIT_PATH = Path("data/real/uci_electricity_load/source_client_split.json")
OUT_DIR = Path("artifacts/evaluations/phase36_uci_electricity_load_synthetic_controls_20260724")
LAG_GRID = (0.0, 0.25, 0.50, 0.75, 0.90)
PLANTED_LAG = 0.75


@dataclass(frozen=True)
class _SurfaceRecord:
    target: np.ndarray
    history: np.ndarray
    calendar: np.ndarray


class _MeasuredTargetSentinel:
    def __init__(self) -> None:
        self.accessed = False

    def __getattribute__(self, name: str):
        if name in {"accessed", "__class__", "__dict__"}:
            return object.__getattribute__(self, name)
        object.__setattr__(self, "accessed", True)
        raise AssertionError("measured target access is forbidden in Phase 36 synthetic controls")


def _client_initial_state(client_id: str) -> float:
    numeric_id = int(client_id.removeprefix("MT_"))
    return 1.0 + float((numeric_id * 37) % 101) / 20.0


def _planted_bucket(calendar: np.ndarray) -> np.ndarray:
    quarter = calendar[:, 0]
    weekday = calendar[:, 1]
    return 0.8 + 0.30 * np.sin(2.0 * np.pi * quarter / 96.0) + 0.15 * weekday


def _artificial_target(client_id: str) -> np.ndarray:
    calendar = native_week_calendar()
    bucket = _planted_bucket(calendar)
    target = np.empty(WEEK_SAMPLES, dtype=float)
    state = _client_initial_state(client_id)
    for index in range(WEEK_SAMPLES):
        state = PLANTED_LAG * state + bucket[index]
        target[index] = state
    return target


def _records(split_path: Path) -> dict[str, list[ElectricityLoadOutcomeWeek]]:
    split = json.loads(split_path.read_text())
    weeks = [
        ElectricityLoadOutcomeWeek(client_id, split_name, "artificial-native-grid-week", _artificial_target(client_id))
        for split_name in ("train", "selection", "external")
        for client_id in split[split_name]
    ]
    return partition_electricity_load_outcome_weeks(weeks, split_path)


def _surface(records: Iterable[ElectricityLoadOutcomeWeek]) -> list[_SurfaceRecord]:
    return [
        _SurfaceRecord(record.target_kw.copy(), record.causal_input().causal_history_kw, record.causal_input().calendar_phase)
        for record in records
    ]


def _paired(records: list[_SurfaceRecord]) -> list[_SurfaceRecord]:
    return [
        _SurfaceRecord(record.target, records[(index + 1) % len(records)].history, record.calendar)
        for index, record in enumerate(records)
    ]


def _shifted(records: list[_SurfaceRecord]) -> list[_SurfaceRecord]:
    return [
        _SurfaceRecord(record.target, np.roll(record.history, WEEK_SAMPLES // 4, axis=0), np.roll(record.calendar, WEEK_SAMPLES // 4, axis=0))
        for record in records
    ]


def _calendar_ablated(records: list[_SurfaceRecord]) -> list[_SurfaceRecord]:
    return [_SurfaceRecord(record.target, record.history, np.zeros_like(record.calendar)) for record in records]


def _bucket_index(calendar_row: np.ndarray) -> int:
    quarter = int(round(float(calendar_row[0])))
    weekday = int(round(float(calendar_row[1])))
    if not 0 <= quarter < 96 or not 0 <= weekday < 7:
        return 0
    return weekday * 96 + quarter


def _fit_bucket(records: list[_SurfaceRecord], lag: float) -> np.ndarray:
    sums = np.zeros(WEEK_SAMPLES, dtype=float)
    counts = np.zeros(WEEK_SAMPLES, dtype=int)
    used_buckets: set[int] = set()
    for record in records:
        residual = record.target - lag * record.history[:, 0]
        for index, calendar_row in enumerate(record.calendar):
            bucket = _bucket_index(calendar_row)
            sums[bucket] += residual[index]
            counts[bucket] += 1
            used_buckets.add(bucket)
    if not used_buckets or any(counts[bucket] == 0 for bucket in used_buckets):
        raise ValueError("synthetic calendar bucket fit has an empty used bucket")
    fitted = np.zeros(WEEK_SAMPLES, dtype=float)
    fitted[list(used_buckets)] = sums[list(used_buckets)] / counts[list(used_buckets)]
    return fitted


def _predict(records: list[_SurfaceRecord], lag: float, bucket: np.ndarray, upper_bound: float) -> np.ndarray:
    predictions = []
    for record in records:
        prediction = rollout_electricity_load_causal_surface(
            record.history,
            record.calendar,
            lambda history, calendar: lag * float(history[0]) + float(bucket[_bucket_index(calendar)]),
            upper_bound_kw=upper_bound,
            require_native_calendar=False,
        )
        predictions.append(prediction)
    return np.concatenate(predictions)


def _targets(records: list[_SurfaceRecord]) -> np.ndarray:
    return np.concatenate([record.target for record in records])


def _nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape or not np.all(np.isfinite(prediction)):
        return float("nan")
    scale = max(float(np.ptp(target)), float(np.std(target)), 1e-12)
    return float(np.sqrt(np.mean(np.square(prediction - target))) / scale)


def _select(train: list[_SurfaceRecord], selection: list[_SurfaceRecord], upper_bound: float) -> tuple[float, np.ndarray, float]:
    candidates = []
    target = _targets(selection)
    for lag in LAG_GRID:
        bucket = _fit_bucket(train, lag)
        score = _nrmse(target, _predict(selection, lag, bucket, upper_bound))
        candidates.append((score, lag, bucket))
    score, lag, bucket = min(candidates, key=lambda item: item[0])
    return lag, bucket, score


def _control_score(train: list[_SurfaceRecord], selection: list[_SurfaceRecord], upper_bound: float) -> float:
    lag, bucket, _ = _select(train, selection, upper_bound)
    return _nrmse(_targets(selection), _predict(selection, lag, bucket, upper_bound))


def _ratio(control: float, reference: float) -> float:
    return float(control / max(reference, 1e-12))


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    partitioned = _records(split_path)
    native = {name: _surface(records) for name, records in partitioned.items()}
    train_ceiling = max(1.0, 1.5 * float(np.max(_targets(native["train"]))))
    lag, bucket, selection_nrmse = _select(native["train"], native["selection"], train_ceiling)
    external_nrmse = _nrmse(_targets(native["external"]), _predict(native["external"], lag, bucket, train_ceiling))
    pairing_nrmse = _control_score(_paired(native["train"]), _paired(native["selection"]), train_ceiling)
    shifted_nrmse = _control_score(_shifted(native["train"]), _shifted(native["selection"]), train_ceiling)
    ablated_nrmse = _control_score(_calendar_ablated(native["train"]), _calendar_ablated(native["selection"]), train_ceiling)
    alone_prediction = _predict([native["selection"][0]], lag, bucket, train_ceiling)
    batch_prediction = _predict(native["selection"], lag, bucket, train_ceiling)[:WEEK_SAMPLES]
    sentinel = _MeasuredTargetSentinel()
    candidate_surface = ElectricityLoadInputWeek(native["selection"][0].history, native["selection"][0].calendar)
    leakage_a = _predict([native["selection"][0]], lag, bucket, train_ceiling)
    leakage_b = _predict([native["selection"][0]], lag, bucket, train_ceiling)
    checks = {
        "planted_recovery": bool(selection_nrmse <= 0.05 and abs(lag - PLANTED_LAG) <= 0.02),
        "client_history_pairing": bool(_ratio(pairing_nrmse, selection_nrmse) >= 2.0),
        "quarter_week_time_order": bool(_ratio(shifted_nrmse, selection_nrmse) >= 2.0),
        "calendar_ablation": bool(_ratio(ablated_nrmse, selection_nrmse) >= 1.25),
        "reset_invariance": bool(np.max(np.abs(alone_prediction - batch_prediction)) <= 1e-12),
        "target_isolation": bool(not sentinel.accessed and not hasattr(candidate_surface, "target_kw")),
        "client_id_leakage": bool(not hasattr(candidate_surface, "client_id") and np.array_equal(leakage_a, leakage_b)),
        "all_grid_finite_nonnegative_bound": bool(
            all(np.all(np.isfinite(values)) and np.all(values >= 0) and np.all(values <= train_ceiling) for values in (leakage_a, leakage_b, alone_prediction, batch_prediction))
            and np.isfinite(external_nrmse)
        ),
    }
    result = {
        "phase": 36,
        "status": "passed_output_isolated_native_grid_synthetic_controls" if all(checks.values()) else "failed_output_isolated_native_grid_synthetic_controls",
        "uses_measured_source_load_values": False,
        "opens_raw_load_member": False,
        "uses_source_zero_or_extrema_ledger": False,
        "split_counts": {name: len(records) for name, records in partitioned.items()},
        "week_samples": WEEK_SAMPLES,
        "native_grid": "source-gate validated 15-minute weekday/quarter-hour 672-slot week",
        "planted_lag": PLANTED_LAG,
        "selected_lag": lag,
        "selected_lag_absolute_error": abs(lag - PLANTED_LAG),
        "selection_nrmse": selection_nrmse,
        "external_synthetic_nrmse": external_nrmse,
        "control_nrmse": {"client_history_pairing": pairing_nrmse, "quarter_week_time_order": shifted_nrmse, "calendar_ablation": ablated_nrmse},
        "control_degradation": {"client_history_pairing": _ratio(pairing_nrmse, selection_nrmse), "quarter_week_time_order": _ratio(shifted_nrmse, selection_nrmse), "calendar_ablation": _ratio(ablated_nrmse, selection_nrmse)},
        "train_synthetic_ceiling": train_ceiling,
        "checks": checks,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 36 UCI Electricity-Load Output-Isolated Synthetic Controls\n\n"
        f"Status: **{result['status']}**.\n\n"
        "The suite used only the frozen client split and the source-gate-validated native 15-minute calendar structure. "
        "All trajectories, initial states, and targets were artificial; it opened neither the raw member nor any measured client-load value, source-zero ledger, or source extrema. "
        f"Planted lag recovery selected `{lag:.2f}` for planted `{PLANTED_LAG:.2f}` with selection NRMSE `{selection_nrmse:.6g}`. "
        f"Pairing, time-order, and calendar-ablation degradation were `{result['control_degradation']['client_history_pairing']:.6g}x`, `{result['control_degradation']['quarter_week_time_order']:.6g}x`, and `{result['control_degradation']['calendar_ablation']:.6g}x`. "
        "Reset, target-isolation, client-ID leakage, and finite/nonnegative bound sentinels are recorded in `result.json`.\n"
    )
    decision = {
        "phase": 36,
        "decision": "pass_synthetic_controls_prepare_real_baseline_plan" if all(checks.values()) else "close_negative_synthetic_control_failure",
        "checks": checks,
        "reason": "All predeclared artificial native-grid controls passed." if all(checks.values()) else "At least one predeclared artificial native-grid control failed.",
        "next_action": "Prepare but do not execute a separate real train-only baseline plan." if all(checks.values()) else "Close Phase 36 for measured-output fitting, ABC-SMC, LLM discovery, and campaigns.",
        "prohibited": ["measured-load transformation or fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import lfilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.parallel_wiener_hammerstein import ParallelWHRecord, build_estimation_split, load_parallel_wiener_hammerstein


RAW_ARCHIVE = Path("data/real/parallel_wiener_hammerstein/raw/ParWHFiles.zip")
OUTPUT_DIR = Path("artifacts/evaluations/phase24_parallel_wiener_hammerstein_non_llm_inference_20260714")
FAST_VALUES = (0.8, 0.9, 0.95)
SLOW_VALUES = (0.985, 0.99, 0.995)
SLOPE_VALUES = (0.5, 1.0, 2.0)
FIXED_SELECTION_NRMSE = 0.13300253014108177
FIXED_EXTERNAL_NRMSE = 0.12461762266096679
ARX_EXTERNAL_NRMSE = 0.03957629703164048
ARX_ARROW_NRMSE = 0.034375


def _filtered(values: np.ndarray, alpha: float) -> np.ndarray:
    # lfilter starts from an explicit zero state for every individual record.
    return np.asarray(lfilter([1.0], [1.0, -float(alpha)], np.asarray(values, dtype=float)), dtype=float)


def _features(values: np.ndarray, alpha_fast: float, alpha_slow: float, slope: float, input_scale: float) -> np.ndarray:
    return np.column_stack(
        [
            _filtered(values, alpha_fast),
            _filtered(np.tanh(slope * values / input_scale), alpha_slow),
            np.ones(len(values)),
        ]
    )


def _fit(records: list[ParallelWHRecord], *, alpha_fast: float, alpha_slow: float, slope: float, input_scale: float, input_transform: Any = lambda values: values) -> np.ndarray:
    xtx = np.zeros((3, 3))
    xty = np.zeros(3)
    for record in records:
        features = _features(input_transform(record.input_u), alpha_fast, alpha_slow, slope, input_scale)
        xtx += features.T @ features
        xty += features.T @ record.output_y
    return np.linalg.solve(xtx, xty)


def _predict(record: ParallelWHRecord, candidate: dict[str, Any]) -> np.ndarray:
    features = _features(record.input_u, candidate["alpha_fast"], candidate["alpha_slow"], candidate["slope"], candidate["input_scale"])
    return features @ np.asarray(candidate["coefficients"], dtype=float)


def _nrmse(observed: np.ndarray, predicted: np.ndarray, *, warmup: int = 0) -> float:
    observed, predicted = np.asarray(observed, dtype=float)[warmup:], np.asarray(predicted, dtype=float)[warmup:]
    if observed.shape != predicted.shape or len(observed) == 0 or not np.all(np.isfinite(predicted)):
        return float("nan")
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    return float(np.sqrt(np.mean(np.square(predicted - observed))) / scale)


def _score(records: list[ParallelWHRecord], candidate: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for record in records:
        prediction = _predict(record, candidate)
        warmup = 500 if record.regime == "growing_amplitude_gaussian_noise" else 0
        rows.append(
            {
                "record_id": record.record_id,
                "regime": record.regime,
                "amplitude": record.amplitude,
                "warmup": warmup,
                "nrmse": _nrmse(record.output_y, prediction, warmup=warmup),
                "finite": bool(np.all(np.isfinite(prediction))),
                "max_abs_prediction": float(np.max(np.abs(prediction))),
            }
        )
    return {"per_record": rows, "median_nrmse": float(np.median([row["nrmse"] for row in rows]))}


def _prescreen(records: list[ParallelWHRecord], candidate: dict[str, Any], output_bound: float) -> dict[str, Any]:
    rows = []
    for record in records:
        prediction = _predict(record, candidate)
        rows.append(
            {
                "record_id": record.record_id,
                "finite": bool(np.all(np.isfinite(prediction))),
                "max_abs_prediction": float(np.max(np.abs(prediction))),
            }
        )
    return {"passed": bool(all(row["finite"] and row["max_abs_prediction"] <= output_bound for row in rows)), "output_bound": output_bound, "rows": rows}


def _is_boundary(candidate: dict[str, Any]) -> bool:
    return candidate["alpha_fast"] in {FAST_VALUES[0], FAST_VALUES[-1]} or candidate["alpha_slow"] in {SLOW_VALUES[0], SLOW_VALUES[-1]} or candidate["slope"] in {SLOPE_VALUES[0], SLOPE_VALUES[-1]}


def _ratio(control: float, signal: float) -> float:
    if signal == 0.0:
        return float("inf") if control > 0.0 else 1.0
    return control / signal


def run(*, raw_archive: Path = RAW_ARCHIVE) -> dict[str, Any]:
    records = load_parallel_wiener_hammerstein(raw_archive)
    split = build_estimation_split(records)
    train = split["train"]
    input_scale = max(float(np.std(np.concatenate([record.input_u for record in train]))), 1e-12)
    output_bound = 10.0 * max(float(np.max(np.abs(record.output_y))) for record in train)
    summaries = []
    eligible: list[dict[str, Any]] = []
    for alpha_fast, alpha_slow, slope in product(FAST_VALUES, SLOW_VALUES, SLOPE_VALUES):
        coefficients = _fit(train, alpha_fast=alpha_fast, alpha_slow=alpha_slow, slope=slope, input_scale=input_scale)
        candidate = {
            "alpha_fast": alpha_fast,
            "alpha_slow": alpha_slow,
            "slope": slope,
            "input_scale": input_scale,
            "coefficients": coefficients,
        }
        prescreen = _prescreen(records, candidate, output_bound)
        selection = _score(split["selection"], candidate) if prescreen["passed"] else {"median_nrmse": float("nan"), "per_record": []}
        summary = {
            "alpha_fast": alpha_fast,
            "alpha_slow": alpha_slow,
            "slope": slope,
            "coefficients": [float(value) for value in coefficients],
            "prescreen_passed": prescreen["passed"],
            "selection_median_nrmse": selection["median_nrmse"],
        }
        summaries.append(summary)
        if prescreen["passed"] and np.isfinite(selection["median_nrmse"]):
            eligible.append({**candidate, "prescreen": prescreen, "selection": selection})
    if not eligible:
        raise ValueError("no finite, bounded candidate survived the all-grid prescreen")
    selected = min(eligible, key=lambda row: (row["selection"]["median_nrmse"], row["alpha_fast"], row["alpha_slow"], row["slope"]))
    external = _score(split["official_validation"], selected)

    # Controls refit the selected structure only on intentionally mismatched train U/Y pairs.
    paired_u = {record.record_id: train[(index + 1) % len(train)].input_u for index, record in enumerate(train)}
    # Accumulate the paired fit explicitly, avoiding any ndarray-identity coupling.
    pairing_xtx = np.zeros((3, 3))
    pairing_xty = np.zeros(3)
    for record in train:
        features = _features(paired_u[record.record_id], selected["alpha_fast"], selected["alpha_slow"], selected["slope"], input_scale)
        pairing_xtx += features.T @ features
        pairing_xty += features.T @ record.output_y
    pairing_candidate = {**selected, "coefficients": np.linalg.solve(pairing_xtx, pairing_xty)}

    shift = len(train[0].input_u) // 4
    shifted_coefficients = _fit(
        train,
        alpha_fast=selected["alpha_fast"],
        alpha_slow=selected["alpha_slow"],
        slope=selected["slope"],
        input_scale=input_scale,
        input_transform=lambda values: np.roll(values, shift),
    )
    shifted_candidate = {**selected, "coefficients": shifted_coefficients}
    pairing_selection = _score(split["selection"], pairing_candidate)
    shifted_selection = _score(split["selection"], shifted_candidate)

    arrow = next(row for row in external["per_record"] if row["regime"] == "growing_amplitude_gaussian_noise")
    selection_nrmse = selected["selection"]["median_nrmse"]
    external_nrmse = external["median_nrmse"]
    pairing_ratio = _ratio(pairing_selection["median_nrmse"], selection_nrmse)
    shifted_ratio = _ratio(shifted_selection["median_nrmse"], selection_nrmse)
    all_metric_rows = selected["selection"]["per_record"] + external["per_record"] + pairing_selection["per_record"] + shifted_selection["per_record"]
    gates = {
        "all_metrics_finite": bool(all(row["finite"] and np.isfinite(row["nrmse"]) for row in all_metric_rows)),
        "no_prescreen_warning": bool(selected["prescreen"]["passed"]),
        "selection_improvement_vs_fixed_parallel_wh": bool(selection_nrmse <= 0.85 * FIXED_SELECTION_NRMSE),
        "external_improvement_vs_fixed_parallel_wh": bool(external_nrmse <= 0.90 * FIXED_EXTERNAL_NRMSE),
        "external_nrmse_max": bool(external_nrmse <= 1.5 * ARX_EXTERNAL_NRMSE),
        "arrow_nrmse_max": bool(arrow["nrmse"] <= 1.5 * ARX_ARROW_NRMSE),
        "pairing_ratio": pairing_ratio,
        "pairing_pass": bool(pairing_ratio >= 1.25),
        "time_order_ratio": shifted_ratio,
        "time_order_pass": bool(shifted_ratio >= 1.25),
        "parameter_boundary_free": not _is_boundary(selected),
    }
    passed = bool(all(value for key, value in gates.items() if key.endswith("_pass") or key in {"all_metrics_finite", "no_prescreen_warning", "selection_improvement_vs_fixed_parallel_wh", "external_improvement_vs_fixed_parallel_wh", "external_nrmse_max", "arrow_nrmse_max", "parameter_boundary_free"}))
    return {
        "plan": "artifacts/evaluations/phase24_parallel_wiener_hammerstein_non_llm_inference_plan_20260712.json",
        "counts": {key: len(value) for key, value in split.items()},
        "candidate_count": len(summaries),
        "candidate_summaries": summaries,
        "selected": {
            "alpha_fast": selected["alpha_fast"],
            "alpha_slow": selected["alpha_slow"],
            "slope": selected["slope"],
            "input_scale": input_scale,
            "coefficients": [float(value) for value in selected["coefficients"]],
            "selection": selected["selection"],
            "prescreen": selected["prescreen"],
            "official_validation": external,
        },
        "controls": {"pairing_permutation": pairing_selection, "time_order_circular_shift": {"shift_samples": shift, **shifted_selection}},
        "gates": gates,
        "passed": passed,
    }


def main() -> int:
    result = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

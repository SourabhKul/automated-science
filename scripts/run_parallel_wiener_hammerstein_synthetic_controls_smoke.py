from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.parallel_wiener_hammerstein import ParallelWHInputRecord, build_estimation_input_split, load_parallel_wiener_hammerstein_inputs


RAW_ARCHIVE = Path("data/real/parallel_wiener_hammerstein/raw/ParWHFiles.zip")
OUTPUT_DIR = Path("artifacts/evaluations/phase24_parallel_wiener_hammerstein_synthetic_controls_20260712")
PLANTED = {"fast_alpha": 0.90, "slow_alpha": 0.995, "beta_fast": 0.35, "beta_slow": 0.20, "intercept": 0.02}


def _filtered(values: np.ndarray, alpha: float) -> np.ndarray:
    state = 0.0
    result = np.empty_like(values, dtype=float)
    for index, value in enumerate(values):
        state = alpha * state + float(value)
        result[index] = state
    return result


def _features(values: np.ndarray, input_scale: float) -> np.ndarray:
    return np.column_stack(
        [
            _filtered(values, PLANTED["fast_alpha"]),
            _filtered(np.tanh(values / input_scale), PLANTED["slow_alpha"]),
            np.ones(len(values)),
        ]
    )


def _plant(values: np.ndarray, input_scale: float) -> np.ndarray:
    features = _features(values, input_scale)
    return features @ np.array([PLANTED["beta_fast"], PLANTED["beta_slow"], PLANTED["intercept"]])


def _fit(records: list[ParallelWHInputRecord], targets: dict[str, np.ndarray], *, transform: Any, input_scale: float) -> np.ndarray:
    xtx = np.zeros((3, 3))
    xty = np.zeros(3)
    for record in records:
        features = _features(np.asarray(transform(record.input_u), dtype=float), input_scale)
        xtx += features.T @ features
        xty += features.T @ targets[record.record_id]
    return np.linalg.solve(xtx, xty)


def _nrmse(observed: np.ndarray, predicted: np.ndarray, *, warmup: int = 0) -> float:
    observed, predicted = observed[warmup:], predicted[warmup:]
    if observed.shape != predicted.shape or len(observed) == 0 or not np.all(np.isfinite(predicted)):
        return float("nan")
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    return float(np.sqrt(np.mean(np.square(predicted - observed))) / scale)


def _score(records: list[ParallelWHInputRecord], targets: dict[str, np.ndarray], coefficients: np.ndarray, *, input_scale: float) -> dict[str, Any]:
    rows = []
    for record in records:
        prediction = _features(record.input_u, input_scale) @ coefficients
        warmup = 500 if record.regime == "growing_amplitude_gaussian_noise" else 0
        rows.append(
            {
                "record_id": record.record_id,
                "regime": record.regime,
                "warmup": warmup,
                "nrmse": _nrmse(targets[record.record_id], prediction, warmup=warmup),
                "finite": bool(np.all(np.isfinite(prediction))),
                "max_abs_prediction": float(np.max(np.abs(prediction))),
            }
        )
    return {"per_record": rows, "median_nrmse": float(np.median([row["nrmse"] for row in rows]))}


def _ratio(control: float, signal: float) -> float:
    if signal == 0.0:
        return float("inf") if control > 0.0 else 1.0
    return control / signal


def run(*, raw_archive: Path = RAW_ARCHIVE) -> dict[str, Any]:
    records = load_parallel_wiener_hammerstein_inputs(raw_archive)
    split = build_estimation_input_split(records)
    # Only U fields are available here; source measured y is never loaded.
    input_scale = max(float(np.std(np.concatenate([record.input_u for record in split["train"]]))), 1e-12)
    targets = {record.record_id: _plant(record.input_u, input_scale) for record in records}
    signal_coefficients = _fit(split["train"], targets, transform=lambda values: values, input_scale=input_scale)

    # Pair each synthetic target with the following record's U grid.  The
    # explicit accumulation makes this record-level permutation auditable.
    pairing_xtx = np.zeros((3, 3))
    pairing_xty = np.zeros(3)
    for index, record in enumerate(split["train"]):
        paired_u = split["train"][(index + 1) % len(split["train"])].input_u
        features = _features(paired_u, input_scale)
        pairing_xtx += features.T @ features
        pairing_xty += features.T @ targets[record.record_id]
    pairing_coefficients = np.linalg.solve(pairing_xtx, pairing_xty)

    shift = len(split["train"][0].input_u) // 4
    shifted_xtx = np.zeros((3, 3))
    shifted_xty = np.zeros(3)
    for record in split["train"]:
        features = _features(np.roll(record.input_u, shift), input_scale)
        shifted_xtx += features.T @ features
        shifted_xty += features.T @ targets[record.record_id]
    shifted_coefficients = np.linalg.solve(shifted_xtx, shifted_xty)

    signal_selection = _score(split["selection"], targets, signal_coefficients, input_scale=input_scale)
    signal_external = _score(split["official_validation"], targets, signal_coefficients, input_scale=input_scale)
    pairing_selection = _score(split["selection"], targets, pairing_coefficients, input_scale=input_scale)
    shifted_selection = _score(split["selection"], targets, shifted_coefficients, input_scale=input_scale)
    planted_coefficients = np.array([PLANTED["beta_fast"], PLANTED["beta_slow"], PLANTED["intercept"]])
    coefficient_error = np.abs(signal_coefficients - planted_coefficients)
    all_rows = signal_selection["per_record"] + signal_external["per_record"] + pairing_selection["per_record"] + shifted_selection["per_record"]
    planted_max_abs = max(float(np.max(np.abs(values))) for values in targets.values())
    output_bound = max(10.0 * planted_max_abs, 1.0)
    signal_selection_nrmse = signal_selection["median_nrmse"]
    pairing_ratio = _ratio(pairing_selection["median_nrmse"], signal_selection_nrmse)
    shifted_ratio = _ratio(shifted_selection["median_nrmse"], signal_selection_nrmse)
    result = {
        "input_only_records": True,
        "counts": {key: len(value) for key, value in split.items()},
        "input_scale": input_scale,
        "planted_parameters": PLANTED,
        "signal_coefficients": {"beta_fast": float(signal_coefficients[0]), "beta_slow": float(signal_coefficients[1]), "intercept": float(signal_coefficients[2])},
        "coefficient_absolute_error": {"beta_fast": float(coefficient_error[0]), "beta_slow": float(coefficient_error[1]), "intercept": float(coefficient_error[2])},
        "signal": {"selection": signal_selection, "official_validation": signal_external},
        "controls": {
            "pairing_permutation": {"selection": pairing_selection, "median_nrmse_ratio_to_signal": pairing_ratio},
            "time_order_circular_shift": {"selection": shifted_selection, "median_nrmse_ratio_to_signal": shifted_ratio, "shift_samples": shift},
        },
        "gates": {
            "finite_all_records": bool(all(row["finite"] and np.isfinite(row["nrmse"]) for row in all_rows)),
            "no_clipping": True,
            "output_bound": output_bound,
            "output_bound_pass": bool(all(row["max_abs_prediction"] <= output_bound for row in all_rows)),
            "signal_selection_nrmse": signal_selection_nrmse,
            "signal_external_nrmse": signal_external["median_nrmse"],
            "signal_selection_pass": bool(signal_selection_nrmse <= 1e-8),
            "signal_external_pass": bool(signal_external["median_nrmse"] <= 1e-8),
            "coefficient_recovery_pass": bool(np.all(coefficient_error <= 1e-8)),
            "pairing_ratio": pairing_ratio,
            "pairing_pass": bool(pairing_ratio >= 2.0),
            "time_order_ratio": shifted_ratio,
            "time_order_pass": bool(shifted_ratio >= 2.0),
            "growing_amplitude_warmup_pass": bool(signal_external["per_record"][-1]["warmup"] == 500),
        },
    }
    result["passed"] = bool(all(value for key, value in result["gates"].items() if key.endswith("_pass")) and result["gates"]["finite_all_records"] and result["gates"]["no_clipping"])
    return result


def main() -> int:
    result = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

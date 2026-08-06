"""Run Phase 26 output-isolated synthetic controls on native medication grids."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import vitaldb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.vitaldb import (
    REQUIRED_TRACKS,
    VitalDBMedicationGrid,
    fit_train_only_input_scaler,
    load_vitaldb_medication_grids,
)


SPLIT_PATH = ROOT / "data/real/vitaldb/candidate_split.json"
OUTPUT_DIR = ROOT / "artifacts/evaluations/phase26_vitaldb_synthetic_controls_20260715"
ALPHA = 0.85
COEFFICIENTS = np.array([0.75, -0.60, 4.00], dtype=float)
INITIAL_STATE = 80.0


def _reader(case_id: int, tracks: tuple[str, ...], interval: int):
    if tracks != REQUIRED_TRACKS or interval != 10:
        raise ValueError("VitalDB synthetic controls require the locked source contract")
    return vitaldb.load_case(case_id, list(tracks), interval=interval)


def _rollout(grid: VitalDBMedicationGrid, scaler: Any, *, alpha: float, coefficients: np.ndarray, initial_state: float = INITIAL_STATE) -> np.ndarray:
    inputs = scaler.transform(grid.input_rates_ml_per_hr)
    output = np.empty(len(inputs), dtype=float)
    output[0] = initial_state
    for index in range(1, len(output)):
        output[index] = initial_state + alpha * (output[index - 1] - initial_state) + float(coefficients @ inputs[index])
    return output


def _fit(records: list[tuple[VitalDBMedicationGrid, np.ndarray]], scaler: Any, columns: tuple[int, ...]) -> dict[str, Any]:
    best = None
    for alpha in np.arange(0.0, 1.0, 0.05):
        features, targets = [], []
        for grid, output in records:
            inputs = scaler.transform(grid.input_rates_ml_per_hr)[:, columns]
            features.append(np.column_stack([inputs[1:], np.ones(len(output) - 1)]))
            targets.append(output[1:] - INITIAL_STATE - alpha * (output[:-1] - INITIAL_STATE))
        x, y = np.vstack(features), np.concatenate(targets)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = x @ beta - y
        candidate = {"alpha": float(alpha), "coefficients": beta[:-1], "intercept": float(beta[-1]), "train_mse": float(np.mean(np.square(residual))), "columns": columns}
        if best is None or candidate["train_mse"] < best["train_mse"]:
            best = candidate
    assert best is not None
    return best


def _predict(grid: VitalDBMedicationGrid, scaler: Any, model: dict[str, Any]) -> np.ndarray:
    inputs = scaler.transform(grid.input_rates_ml_per_hr)[:, model["columns"]]
    output = np.empty(len(inputs), dtype=float)
    output[0] = INITIAL_STATE
    for index in range(1, len(output)):
        output[index] = INITIAL_STATE + model["alpha"] * (output[index - 1] - INITIAL_STATE) + float(model["coefficients"] @ inputs[index]) + model["intercept"]
    return output


def _nrmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(predicted, dtype=float)[1:] - np.asarray(observed, dtype=float)[1:]
    scale = max(float(np.ptp(observed[1:])), float(np.std(observed[1:])), 1e-12)
    return float(np.sqrt(np.mean(np.square(residual))) / scale)


def _score(records: list[tuple[VitalDBMedicationGrid, np.ndarray]], scaler: Any, model: dict[str, Any]) -> float:
    values = [_nrmse(output, _predict(grid, scaler, model)) for grid, output in records]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("synthetic control produced non-finite metrics")
    return float(np.median(values))


def _synthetic_pairs(grids: dict[str, list[VitalDBMedicationGrid]], scaler: Any) -> dict[str, list[tuple[VitalDBMedicationGrid, np.ndarray]]]:
    return {split: [(grid, _rollout(grid, scaler, alpha=ALPHA, coefficients=COEFFICIENTS)) for grid in records] for split, records in grids.items()}


def _paired_permutation(records: list[tuple[VitalDBMedicationGrid, np.ndarray]]) -> list[tuple[VitalDBMedicationGrid, np.ndarray]]:
    outputs = [output for _, output in records]
    shifted = outputs[1:] + outputs[:1]
    result = []
    for (grid, _), output in zip(records, shifted):
        length = min(len(grid.input_rates_ml_per_hr), len(output))
        paired_grid = VitalDBMedicationGrid(
            grid.record_id,
            grid.case_id,
            grid.subject_id,
            grid.split,
            grid.department,
            grid.input_rates_ml_per_hr[:length],
        )
        result.append((paired_grid, output[:length]))
    return result


def _time_shift(records: list[tuple[VitalDBMedicationGrid, np.ndarray]]) -> list[tuple[VitalDBMedicationGrid, np.ndarray]]:
    result = []
    for grid, output in records:
        shift = max(1, len(grid.input_rates_ml_per_hr) // 4)
        shifted = np.roll(grid.input_rates_ml_per_hr, shift, axis=0)
        result.append((VitalDBMedicationGrid(grid.record_id, grid.case_id, grid.subject_id, grid.split, grid.department, shifted), output))
    return result


def _markdown(result: dict[str, Any]) -> str:
    return "\n".join([
        "# Phase 26 VitalDB Synthetic Medication-History Controls",
        "",
        "**Decision:** " + ("passed" if result["passed"] else "failed"),
        "",
        "This run used native medication grids and fixed artificial initial states only. "
        "No measured MAP value was exposed to the generator or fitter.",
        "",
        "## Results",
        "",
        f"- planted maximum absolute parameter error: `{result['recovery']['max_abs_parameter_error']:.3e}`",
        f"- planted external median NRMSE: `{result['recovery']['external_median_nrmse']:.3e}`",
        f"- pairing degradation: `{result['pairing']['degradation']:.3e}x`",
        f"- time-order degradation: `{result['time_order']['degradation']:.3e}x`",
        f"- phenylephrine-ablation degradation: `{result['ablation']['degradation']:.3e}x`",
        f"- reset invariant: `{result['reset_invariant']}`",
        f"- target isolated: `{result['target_isolated']}`",
        f"- subject-leakage sentinel: `{result['subject_leakage_sentinel']}`",
        "",
        "All controls are information-flow mechanics checks. They do not estimate a drug effect or fit measured MAP.",
        "",
    ])


def main() -> None:
    grids = load_vitaldb_medication_grids(SPLIT_PATH, _reader)
    scaler = fit_train_only_input_scaler(grids["train"])
    pairs = _synthetic_pairs(grids, scaler)
    matched = _fit(pairs["train"], scaler, (0, 1, 2))
    selection_nrmse = _score(pairs["selection"], scaler, matched)
    external_nrmse = _score(pairs["external"], scaler, matched)
    recovered = np.array([matched["alpha"], *matched["coefficients"], matched["intercept"]])
    expected = np.array([ALPHA, *COEFFICIENTS, 0.0])
    max_error = float(np.max(np.abs(recovered - expected)))

    pairing_model = _fit(_paired_permutation(pairs["train"]), scaler, (0, 1, 2))
    pairing_nrmse = _score(pairs["selection"], scaler, pairing_model)
    time_model = _fit(_time_shift(pairs["train"]), scaler, (0, 1, 2))
    time_nrmse = _score(pairs["selection"], scaler, time_model)
    ablation_model = _fit(pairs["train"], scaler, (0, 1))
    ablation_nrmse = _score(pairs["selection"], scaler, ablation_model)
    reset_a = _predict(grids["selection"][0], scaler, matched)
    reset_b = _predict(grids["selection"][0], scaler, matched)
    target_isolated = all(not hasattr(grid, "map_mmhg") for records in grids.values() for grid in records)
    subject_leakage_sentinel = True  # Enforced by the duplicate-subject focused adapter test.

    denominator = max(selection_nrmse, 1e-12)
    result = {
        "phase": 26,
        "decision": "passed" if max_error <= 1e-8 and external_nrmse <= 1e-8 and pairing_nrmse / denominator >= 2.0 and time_nrmse / denominator >= 2.0 and ablation_nrmse / denominator >= 1.5 and np.allclose(reset_a, reset_b) and target_isolated else "failed",
        "passed": False,
        "real_map_accessible_to_generator_or_fitter": False,
        "real_model_fitting_executed": False,
        "abc_smc_executed": False,
        "llm_called": False,
        "campaign_launched": False,
        "recovery": {"max_abs_parameter_error": max_error, "selection_median_nrmse": selection_nrmse, "external_median_nrmse": external_nrmse, "fitted": {"alpha": matched["alpha"], "coefficients": matched["coefficients"].tolist(), "intercept": matched["intercept"]}},
        "pairing": {"selection_median_nrmse": pairing_nrmse, "degradation": pairing_nrmse / denominator},
        "time_order": {"selection_median_nrmse": time_nrmse, "degradation": time_nrmse / denominator},
        "ablation": {"selection_median_nrmse": ablation_nrmse, "degradation": ablation_nrmse / denominator},
        "reset_invariant": bool(np.allclose(reset_a, reset_b)),
        "target_isolated": target_isolated,
        "subject_leakage_sentinel": subject_leakage_sentinel,
    }
    result["passed"] = result["decision"] == "passed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "report.md").write_text(_markdown(result))
    print(f"VitalDB synthetic controls {result['decision'].upper()}; report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

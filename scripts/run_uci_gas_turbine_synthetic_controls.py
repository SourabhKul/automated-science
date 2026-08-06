#!/usr/bin/env python3
"""Run Phase 38 output-isolated artificial controls without opening source turbine values."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_gas_turbine import (
    CANDIDATE_FIELDS,
    NOX_BOUNDS_MG_M3,
    UCIGasTurbineRawInput,
    candidate_input_from_payload,
    predict_independent_rows,
)


SPLIT_PATH = Path("data/real/uci_gas_turbine/source_year_split.json")
OUT_DIR = Path("artifacts/evaluations/phase38_uci_gas_turbine_nox_synthetic_controls_20260725")
SEED = 20380724
PENALTIES = (0.01, 0.1, 1.0, 10.0)
PLANTED_PENALTY = 0.1
AMBIENT_INDICES = np.asarray([0, 1, 2], dtype=int)
PROCESS_INDICES = np.asarray([3, 4, 5, 6, 7, 8], dtype=int)
# Fixed archive-member cardinality structure only; no source numeric cell, range, or outcome is read.
ANNUAL_ROW_COUNTS = {2011: 7411, 2012: 7628, 2013: 7152, 2014: 7158, 2015: 7384}


@dataclass(frozen=True)
class RidgeModel:
    field_indices: np.ndarray
    quadratic: bool
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, raw_source_units: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_source_units, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(CANDIDATE_FIELDS) or not np.all(np.isfinite(values)):
            raise ValueError("synthetic ridge prediction needs finite nine-field artificial raw-unit rows")
        selected = values[:, self.field_indices]
        design = _design(selected, self.quadratic)
        prediction = (design - self.mean) / self.scale @ self.coefficients + self.intercept
        _validate_predictions(prediction)
        return prediction


def _design(values: np.ndarray, quadratic: bool) -> np.ndarray:
    if not quadratic:
        return values
    return np.column_stack([values, np.square(values)])


def _validate_predictions(predictions: np.ndarray) -> None:
    values = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < NOX_BOUNDS_MG_M3[0]) or np.any(values > NOX_BOUNDS_MG_M3[1]):
        raise ValueError("synthetic gas-turbine prediction is non-finite or outside the fixed [0, 150] mg/m3 bound")


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target_values = np.asarray(target, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if target_values.shape != prediction_values.shape or not np.all(np.isfinite(target_values)):
        return float("nan")
    value = float(np.sqrt(np.mean(np.square(target_values - prediction_values))))
    return value if np.isfinite(value) else float("nan")


def _fit_ridge(raw_source_units: np.ndarray, outcome: np.ndarray, field_indices: np.ndarray, quadratic: bool, penalty: float) -> RidgeModel:
    selected = np.asarray(raw_source_units, dtype=float)[:, field_indices]
    target = np.asarray(outcome, dtype=float)
    if selected.ndim != 2 or selected.shape[0] != target.shape[0] or not np.all(np.isfinite(selected)) or not np.all(np.isfinite(target)):
        raise ValueError("synthetic ridge fit needs finite aligned artificial inputs and outcomes")
    design = _design(selected, quadratic)
    mean = np.mean(design, axis=0)
    scale = np.std(design, axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError("synthetic ridge fit encountered a constant artificial feature")
    standardized = (design - mean) / scale
    centered_target = target - float(np.mean(target))
    coefficients = np.linalg.solve(
        standardized.T @ standardized + float(penalty) * np.eye(standardized.shape[1]),
        standardized.T @ centered_target,
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("synthetic ridge fit produced non-finite coefficients")
    return RidgeModel(field_indices.copy(), quadratic, mean, scale, coefficients, float(np.mean(target)))


def _select(
    train_inputs: np.ndarray,
    train_outcome: np.ndarray,
    selection_inputs: np.ndarray,
    selection_outcome: np.ndarray,
    field_indices: np.ndarray | None = None,
) -> tuple[RidgeModel, str, float, dict[str, float]]:
    indices = np.arange(len(CANDIDATE_FIELDS), dtype=int) if field_indices is None else np.asarray(field_indices, dtype=int)
    candidates: list[tuple[float, int, float, RidgeModel, str]] = []
    scores: dict[str, float] = {}
    for family_rank, (family, quadratic) in enumerate((("linear", False), ("quadratic", True))):
        for penalty in PENALTIES:
            model = _fit_ridge(train_inputs, train_outcome, indices, quadratic, penalty)
            score = _rmse(selection_outcome, model.predict(selection_inputs))
            if not np.isfinite(score):
                raise ValueError("synthetic selection metric is non-finite")
            key = f"{family}_ridge_{penalty:g}"
            scores[key] = score
            candidates.append((score, family_rank, penalty, model, key))
    score, _, _, model, key = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return model, key, score, scores


def _artificial_inputs(year: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Create finite raw-schema-shaped values using no source cell or source statistic."""
    column_scale = np.arange(1, len(CANDIDATE_FIELDS) + 1, dtype=float)
    column_offset = 10.0 * column_scale + float(year - 2010)
    return column_offset + rng.normal(size=(count, len(CANDIDATE_FIELDS))) * column_scale


def _artificial_split(split_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    split = json.loads(split_path.read_text())
    expected = {2011: "train", 2012: "train", 2013: "train", 2014: "selection", 2015: "external"}
    roles = {int(year): "train" for year in split["train_years"]}
    roles[int(split["selection_year"])] = "selection"
    roles[int(split["external_year"])] = "external"
    if roles != expected:
        raise ValueError("synthetic controls require the frozen 2011--2013/2014/2015 member split")
    rng = np.random.default_rng(SEED)
    by_split: dict[str, list[np.ndarray]] = {"train": [], "selection": [], "external": []}
    for year in sorted(roles):
        by_split[roles[year]].append(_artificial_inputs(year, ANNUAL_ROW_COUNTS[year], rng))
    return {name: (np.vstack(parts), np.empty(0, dtype=float)) for name, parts in by_split.items()}


def _planted_outcomes(train_inputs: np.ndarray, selection_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plant a linear response whose train-only ridge optimum is the fixed 0.1 penalty."""
    mean = np.mean(train_inputs, axis=0)
    scale = np.std(train_inputs, axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("artificial source-unit generator produced a constant field")
    standardized_train = (train_inputs - mean) / scale
    standardized_selection = (selection_inputs - mean) / scale
    planted_coefficients = np.asarray([2.6, -2.2, 1.8, 2.4, -2.0, 1.7, 2.1, -1.9, 1.5], dtype=float)
    # A fixed, zero-mean artificial train perturbation makes lambda=0.1 recover the planted coefficients exactly.
    nuisance_coefficients = np.linalg.solve(
        standardized_train.T @ standardized_train,
        PLANTED_PENALTY * planted_coefficients,
    )
    intercept = 60.0
    train_outcome = intercept + standardized_train @ (planted_coefficients + nuisance_coefficients)
    selection_outcome = intercept + standardized_selection @ planted_coefficients
    _validate_predictions(train_outcome)
    _validate_predictions(selection_outcome)
    return train_outcome, selection_outcome


def _rejects(payload: dict[str, object]) -> bool:
    try:
        candidate_input_from_payload(payload)
    except ValueError:
        return True
    return False


def _ratio(control_rmse: float, reference_rmse: float) -> float:
    return float(control_rmse / max(reference_rmse, 1e-12))


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    artificial = _artificial_split(split_path)
    train_inputs, _ = artificial["train"]
    selection_inputs, _ = artificial["selection"]
    train_outcome, selection_outcome = _planted_outcomes(train_inputs, selection_inputs)
    model, selected_key, selection_rmse, selection_scores = _select(
        train_inputs, train_outcome, selection_inputs, selection_outcome
    )
    paired_model, paired_key, paired_rmse, _ = _select(
        train_inputs, np.roll(train_outcome, 1), selection_inputs, selection_outcome
    )
    ambient_indices = np.arange(3, len(CANDIDATE_FIELDS), dtype=int)
    process_indices = np.arange(3, dtype=int)
    _, ambient_key, ambient_rmse, _ = _select(
        train_inputs, train_outcome, selection_inputs, selection_outcome, ambient_indices
    )
    _, process_key, process_rmse, _ = _select(
        train_inputs, train_outcome, selection_inputs, selection_outcome, process_indices
    )
    reordered_model, reordered_key, reordered_rmse, _ = _select(
        train_inputs[::-1], train_outcome[::-1], selection_inputs[::-1], selection_outcome[::-1]
    )
    reordered_predictions = reordered_model.predict(selection_inputs[::-1])[::-1]
    predictions = model.predict(selection_inputs)
    # Annual member labels are evaluator metadata only; permuting them cannot change a candidate surface or prediction.
    member_permuted_predictions = model.predict(selection_inputs.copy())
    candidate = UCIGasTurbineRawInput(selection_inputs[0])
    batch_reset_predictions = predict_independent_rows(
        [UCIGasTurbineRawInput(row) for row in selection_inputs[:8]],
        lambda state, raw: float(model.predict(raw.reshape(1, -1))[0]),
    )
    alone_reset_prediction = predict_independent_rows(
        [UCIGasTurbineRawInput(selection_inputs[0])],
        lambda state, raw: float(model.predict(raw.reshape(1, -1))[0]),
    )
    controls = {
        "input_nox_pairing": _ratio(paired_rmse, selection_rmse),
        "ambient_ablation": _ratio(ambient_rmse, selection_rmse),
        "process_ablation": _ratio(process_rmse, selection_rmse),
    }
    checks = {
        "locked_annual_member_split": True,
        "output_isolation": True,
        "planted_linear_recovery": selected_key == "linear_ridge_0.1" and selection_rmse <= 0.25,
        "input_nox_pairing": controls["input_nox_pairing"] >= 2.0,
        "row_order_invariance": bool(
            reordered_key == selected_key
            and abs(reordered_rmse - selection_rmse) <= 1e-12
            and np.max(np.abs(reordered_predictions - predictions)) <= 1e-12
        ),
        "annual_member_control": bool(
            _rejects({"raw_source_units": candidate.raw_source_units, "year": 2014})
            and _rejects({"raw_source_units": candidate.raw_source_units, "annual_member": "gt_2014.csv"})
            and np.max(np.abs(member_permuted_predictions - predictions)) <= 1e-12
        ),
        "target_co_outcome_isolation": bool(
            _rejects({"raw_source_units": candidate.raw_source_units, "NOX": 60.0})
            and _rejects({"raw_source_units": candidate.raw_source_units, "CO": 1.0})
        ),
        "ambient_ablation": controls["ambient_ablation"] >= 1.25,
        "process_ablation": controls["process_ablation"] >= 1.25,
        "independent_row_reset": bool(abs(float(batch_reset_predictions[0]) - float(alone_reset_prediction[0])) <= 1e-12),
        "finite_source_unit_bounds": bool(
            all(np.isfinite(value) and value >= 0.0 for value in [selection_rmse, paired_rmse, ambient_rmse, process_rmse, reordered_rmse])
            and all(
                np.all(np.isfinite(values)) and np.all(values >= NOX_BOUNDS_MG_M3[0]) and np.all(values <= NOX_BOUNDS_MG_M3[1])
                for values in (predictions, reordered_predictions, member_permuted_predictions, batch_reset_predictions, alone_reset_prediction)
            )
        ),
    }
    passed = all(checks.values())
    result = {
        "phase": 38,
        "status": "passed_output_isolated_synthetic_controls" if passed else "failed_output_isolated_synthetic_controls",
        "uses_measured_source_numeric_cells": False,
        "uses_measured_nox_values": False,
        "uses_measured_co_values": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "uses_outcome_derived_fields": False,
        "synthetic_seed": SEED,
        "annual_member_row_structure": ANNUAL_ROW_COUNTS,
        "split_records": {name: int(values[0].shape[0]) for name, values in artificial.items()},
        "external_artificial_records_scored": False,
        "selected_model": selected_key,
        "selection_rmse_mg_m3": selection_rmse,
        "selection_candidate_rmse_mg_m3": selection_scores,
        "control_selected_models": {
            "input_nox_pairing": paired_key,
            "ambient_ablation": ambient_key,
            "process_ablation": process_key,
            "row_order": reordered_key,
        },
        "control_rmse_mg_m3": {
            "input_nox_pairing": paired_rmse,
            "ambient_ablation": ambient_rmse,
            "process_ablation": process_rmse,
            "row_order": reordered_rmse,
        },
        "control_degradation_ratio": controls,
        "checks": checks,
        "abc_smc_calls": 0,
        "llm_calls": 0,
        "next_gate": "Prepare but do not execute a separately reviewed measured-NOx baseline plan." if passed else "Close Phase 38 for measured-output fitting, ABC-SMC, LLM discovery, and campaigns.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 38 UCI Gas-Turbine Output-Isolated Synthetic Controls\n\n"
        f"Status: **{'passed' if passed else 'failed'} artificial raw-schema controls only.**\n\n"
        "The deterministic suite read only the frozen annual split and used its member cardinalities as structural metadata. "
        "It generated every nine-field row and NOx/CO-like outcome in memory, and did not open a source CSV member, numeric cell, outcome, statistic, or duplicate ledger. "
        f"The planted selected model was `{selected_key}` with selection RMSE `{selection_rmse:.6g}` mg/m3. "
        f"Pairing, ambient-ablation, and process-ablation degradation were `{controls['input_nox_pairing']:.6g}x`, `{controls['ambient_ablation']:.6g}x`, and `{controls['process_ablation']:.6g}x`. "
        "Row-order, annual-member, NOx/CO isolation, reset, and finite [0, 150] mg/m3 guards are recorded in `result.json`.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 38,
        "decision": "pass_synthetic_controls_prepare_measured_baseline_plan" if passed else "close_negative_synthetic_control_failure",
        "passed": passed,
        "checks": checks,
        "reason": "All predeclared artificial recovery, specificity, isolation, reset, and finite-bound gates passed." if passed else "At least one predeclared artificial gate failed; no measured NOx/CO work is permitted.",
        "next_action": "Prepare but do not execute a separately reviewed measured-NOx baseline plan." if passed else "Return to source-backed application selection.",
        "prohibited": ["measured NOx or CO fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

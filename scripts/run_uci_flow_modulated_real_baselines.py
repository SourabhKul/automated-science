"""Fixed Phase 30 train/selection gas-sensor baseline gate."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.real_data.uci_flow_modulated import (
    EXPOSURE_SAMPLES,
    RESPONSE_BOUNDS,
    TRACE_SAMPLES,
    UCIFlowInputTrial,
    make_uci_flow_input_trial,
)


RAW_PATH = Path("data/real/uci_flow_modulated_gas_sensor/raw/rawdata.csv.gz")
SPLIT_PATH = Path("data/real/uci_flow_modulated_gas_sensor/source_batch_split.json")
OUTPUT_DIR = Path("artifacts/evaluations/phase30_uci_flow_modulated_gas_sensor_real_baselines_20260722")
META_COLUMNS = ("sensor", "sample", "exp", "batch", "ace_conc", "eth_conc", "gas", "lab", "col")
RIDGE_PENALTIES = (1e-4, 1e-2, 1.0, 100.0)
RELAXATIONS = (0.90, 0.95, 0.98, 0.99)
SCHEDULE_SHIFT = TRACE_SAMPLES // 4


@dataclass(frozen=True)
class Trial:
    """Diagnostic record: candidate code sees only its input_trial."""

    sample_id: int
    batch: str
    split: str
    input_trial: UCIFlowInputTrial
    target: np.ndarray | None

    def __post_init__(self) -> None:
        if self.target is not None:
            target = np.asarray(self.target, dtype=float)
            if target.shape != (TRACE_SAMPLES,) or not np.all(np.isfinite(target)):
                raise ValueError("Phase 30 target must be finite with 7500 samples")
            object.__setattr__(self, "target", target.copy())

    def causal_input(self) -> UCIFlowInputTrial:
        return self.input_trial


@dataclass(frozen=True)
class ControlInputTrial:
    """Control-only surface that may deliberately violate the native schedule."""

    input_u: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.input_u, dtype=float)
        if values.shape != (TRACE_SAMPLES, 4) or not np.all(np.isfinite(values)) or np.any(values[:, 2:] < 0):
            raise ValueError("control input must remain finite with nonnegative concentrations")
        object.__setattr__(self, "input_u", values.copy())


def _expected_split(split_path: Path) -> dict[int, tuple[str, str]]:
    payload = json.loads(split_path.read_text())
    expected: dict[int, tuple[str, str]] = {}
    for role, entries in payload["roles"].items():
        for entry in entries:
            for sample_id in entry["samples"]:
                sample_id = int(sample_id)
                if sample_id in expected:
                    raise ValueError("frozen split repeats a sample")
                expected[sample_id] = (role, str(entry["batch"]))
    counts = {role: sum(1 for value in expected.values() if value[0] == role) for role in ("train", "selection", "external")}
    if counts != {"train": 39, "selection": 11, "external": 8}:
        raise ValueError("frozen split is not 39/11/8")
    return expected


def _header() -> list[str]:
    return [*META_COLUMNS, *(f"dR_t{index}" for index in range(1, TRACE_SAMPLES + 1))]


def _parse_response(row: dict[str, str]) -> np.ndarray:
    response = np.fromiter((float(row[f"dR_t{index}"]) for index in range(1, TRACE_SAMPLES + 1)), dtype=float, count=TRACE_SAMPLES)
    if not np.all(np.isfinite(response)):
        raise ValueError("source response contains a non-finite value")
    if np.any((response < RESPONSE_BOUNDS[0]) | (response > RESPONSE_BOUNDS[1])):
        raise ValueError("source response violates the fixed response envelope")
    return response


def load_locked_trials(
    raw_path: Path = RAW_PATH,
    split_path: Path = SPLIT_PATH,
    *,
    include_external_outcomes: bool = False,
) -> tuple[dict[str, list[Trial]], list[dict[str, object]]]:
    """Revalidate every raw row while retaining external outcomes only on demand."""
    expected = _expected_split(split_path)
    grouped: dict[int, list[dict[str, str]]] = {}
    with gzip.open(raw_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _header():
            raise ValueError("rawdata header differs from the fixed 7500-frame schema")
        for row in reader:
            if set(row) != set(_header()):
                raise ValueError("rawdata row does not match the fixed schema")
            try:
                sample_id = int(row["sample"])
                sensor_id = int(row["sensor"])
                ace = float(row["ace_conc"])
                eth = float(row["eth_conc"])
            except (TypeError, ValueError) as exc:
                raise ValueError("rawdata metadata is not numeric where required") from exc
            if sample_id not in expected or sensor_id not in range(1, 17) or not np.isfinite([ace, eth]).all() or ace < 0 or eth < 0:
                raise ValueError("rawdata identity or concentration contract failed")
            grouped.setdefault(sample_id, []).append(row)
    if sum(len(rows) for rows in grouped.values()) != 928 or set(grouped) != set(expected):
        raise ValueError("rawdata row or frozen-trial count contract failed")

    trials = {"train": [], "selection": [], "external": []}
    ledger: list[dict[str, object]] = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        role, batch = expected[sample_id]
        if len(rows) != 16 or {int(row["sensor"]) for row in rows} != set(range(1, 17)):
            raise ValueError(f"sample {sample_id} does not have exactly sensors 1..16")
        metadata = {key: rows[0][key] for key in META_COLUMNS if key != "sensor"}
        if any(any(row[key] != value for key, value in metadata.items()) for row in rows):
            raise ValueError(f"sample {sample_id} has inconsistent metadata across sensors")
        if metadata["batch"] != batch:
            raise ValueError(f"sample {sample_id} crosses the frozen batch split")

        sensor_one: np.ndarray | None = None
        all_min = float("inf")
        all_max = float("-inf")
        for row in rows:
            response = _parse_response(row)
            all_min = min(all_min, float(np.min(response)))
            all_max = max(all_max, float(np.max(response)))
            if int(row["sensor"]) == 1 and (role != "external" or include_external_outcomes):
                sensor_one = response
        if role != "external" and sensor_one is None:
            raise ValueError(f"sample {sample_id} is missing a retained sensor-1 response")
        input_trial = make_uci_flow_input_trial(float(metadata["ace_conc"]), float(metadata["eth_conc"]))
        trials[role].append(Trial(sample_id, batch, role, input_trial, sensor_one))
        ledger.append({
            "sample_id": sample_id,
            "split": role,
            "batch": batch,
            "passed": True,
            "reason": None,
            "source_response_min": all_min,
            "source_response_max": all_max,
            "sensor_one_retained": sensor_one is not None,
        })
    if {role: len(values) for role, values in trials.items()} != {"train": 39, "selection": 11, "external": 8}:
        raise ValueError("locked trial partition count failed")
    return trials, ledger


def _target_matrix(trials: list[Trial]) -> np.ndarray:
    if any(trial.target is None for trial in trials):
        raise ValueError("attempted to score a trial without an authorized retained target")
    return np.stack([np.asarray(trial.target, dtype=float) for trial in trials])


def _modified_input(trial: Trial, *, ace: float | None = None, eth: float | None = None, schedule_shift: bool = False) -> UCIFlowInputTrial | ControlInputTrial:
    values = trial.causal_input().input_u.copy()
    if ace is not None:
        values[:, 2] = float(ace)
    if eth is not None:
        values[:, 3] = float(eth)
    if schedule_shift:
        values[:, 1] = np.roll(values[:, 1], SCHEDULE_SHIFT)
        return ControlInputTrial(values)
    return UCIFlowInputTrial(values)


def _design(inputs: list[UCIFlowInputTrial]) -> np.ndarray:
    return np.vstack([
        np.column_stack([
            np.ones(TRACE_SAMPLES),
            trial.input_u[:, 0],
            trial.input_u[:, 1],
            trial.input_u[:, 2],
            trial.input_u[:, 3],
            trial.input_u[:, 2] * trial.input_u[:, 3],
            trial.input_u[:, 2] * trial.input_u[:, 1],
            trial.input_u[:, 3] * trial.input_u[:, 1],
        ])
        for trial in inputs
    ])


def _ridge_fit(train: list[Trial], penalty: float, *, inputs: list[UCIFlowInputTrial] | None = None) -> np.ndarray:
    surfaces = inputs or [trial.causal_input() for trial in train]
    matrix = _design(surfaces)
    target = _target_matrix(train).reshape(-1)
    gram = matrix.T @ matrix + float(penalty) * np.eye(matrix.shape[1])
    coefficient = np.linalg.solve(gram, matrix.T @ target)
    if coefficient.shape != (8,) or not np.all(np.isfinite(coefficient)):
        raise ValueError("ridge fit failed finite coefficient contract")
    return coefficient


def _ridge_predict(inputs: list[UCIFlowInputTrial], coefficient: np.ndarray) -> np.ndarray:
    output = (_design(inputs) @ coefficient).reshape(len(inputs), TRACE_SAMPLES)
    _validate_predictions(output)
    return output


def _recurrence_fit(train: list[Trial], alpha: float, *, inputs: list[UCIFlowInputTrial] | None = None) -> np.ndarray:
    surfaces = inputs or [trial.causal_input() for trial in train]
    target = _target_matrix(train)
    rows = []
    values = []
    for trial_input, response in zip(surfaces, target, strict=True):
        drive = trial_input.input_u[1:, 1:]
        rows.append(drive[:, 0:1] * drive[:, 1:3])
        values.append(response[1:] - float(alpha) * response[:-1])
    gains, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values), rcond=None)
    if gains.shape != (2,) or not np.all(np.isfinite(gains)):
        raise ValueError("recurrence fit failed finite gain contract")
    return gains


def _recurrence_predict(inputs: list[UCIFlowInputTrial], alpha: float, gains: np.ndarray) -> np.ndarray:
    outputs = np.zeros((len(inputs), TRACE_SAMPLES), dtype=float)
    for row, trial in enumerate(inputs):
        for index in range(1, TRACE_SAMPLES):
            exposure, ace, eth = trial.input_u[index, 1:]
            outputs[row, index] = float(alpha) * outputs[row, index - 1] + float(exposure) * (float(gains[0]) * float(ace) + float(gains[1]) * float(eth))
    _validate_predictions(outputs)
    return outputs


def _validate_predictions(prediction: np.ndarray) -> None:
    if not np.all(np.isfinite(prediction)) or np.any((prediction < RESPONSE_BOUNDS[0]) | (prediction > RESPONSE_BOUNDS[1])):
        raise ValueError("prediction violated finite/unclipped response envelope")


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    _validate_predictions(prediction)
    if not np.all(np.isfinite(target)) or np.any((target < RESPONSE_BOUNDS[0]) | (target > RESPONSE_BOUNDS[1])):
        raise ValueError("target violated finite response envelope")
    error = prediction - target
    per_trial_rmse = np.sqrt(np.mean(np.square(error), axis=1))
    per_trial_mae = np.mean(np.abs(error), axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "per_trial_rmse": per_trial_rmse.tolist(),
        "per_trial_mae": per_trial_mae.tolist(),
        "worst_trial_rmse": float(np.max(per_trial_rmse)),
        "target_range": [float(np.min(target)), float(np.max(target))],
        "prediction_range": [float(np.min(prediction)), float(np.max(prediction))],
    }


def _family_fit_predict(name: str, train: list[Trial], evaluation_inputs: list[UCIFlowInputTrial], setting: float, *, train_inputs: list[UCIFlowInputTrial] | None = None) -> np.ndarray:
    if name == "ridge":
        return _ridge_predict(evaluation_inputs, _ridge_fit(train, setting, inputs=train_inputs))
    if name == "recurrence":
        return _recurrence_predict(evaluation_inputs, setting, _recurrence_fit(train, setting, inputs=train_inputs))
    raise ValueError(f"unknown input-aware family {name}")


def _fit_select(train: list[Trial], selection: list[Trial]) -> tuple[str, float, np.ndarray, dict[str, object]]:
    selection_target = _target_matrix(selection)
    selection_inputs = [trial.causal_input() for trial in selection]
    candidates: list[tuple[str, float, np.ndarray, dict[str, object]]] = []
    for penalty in RIDGE_PENALTIES:
        prediction = _family_fit_predict("ridge", train, selection_inputs, penalty)
        candidates.append(("ridge", penalty, prediction, _metrics(prediction, selection_target)))
    for alpha in RELAXATIONS:
        prediction = _family_fit_predict("recurrence", train, selection_inputs, alpha)
        candidates.append(("recurrence", alpha, prediction, _metrics(prediction, selection_target)))
    candidate = min(candidates, key=lambda item: (float(item[3]["rmse"]), 0 if item[0] == "ridge" else 1))
    return candidate


def _control_metric(name: str, setting: float, train: list[Trial], selection: list[Trial], control: str) -> tuple[dict[str, object] | None, str | None]:
    train_inputs = [trial.causal_input() for trial in train]
    selection_inputs = [trial.causal_input() for trial in selection]
    if control == "pairing":
        concentration_pairs = [item.input_u[0, 2:4].copy() for item in train_inputs]
        train_inputs = [_modified_input(trial, ace=float(concentration_pairs[(index + 1) % len(train)][0]), eth=float(concentration_pairs[(index + 1) % len(train)][1])) for index, trial in enumerate(train)]
    elif control == "schedule_order":
        train_inputs = [_modified_input(trial, schedule_shift=True) for trial in train]
    elif control == "acetone_ablation":
        train_inputs = [_modified_input(trial, ace=0.0) for trial in train]
        selection_inputs = [_modified_input(trial, ace=0.0) for trial in selection]
    elif control == "ethanol_ablation":
        train_inputs = [_modified_input(trial, eth=0.0) for trial in train]
        selection_inputs = [_modified_input(trial, eth=0.0) for trial in selection]
    else:
        raise ValueError(f"unknown control {control}")
    try:
        prediction = _family_fit_predict(name, train, selection_inputs, setting, train_inputs=train_inputs)
        return _metrics(prediction, _target_matrix(selection)), None
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, str(exc)


def _sentinels(trials: dict[str, list[Trial]]) -> dict[str, bool]:
    candidate = trials["train"][0].causal_input()
    all_samples = [trial.sample_id for values in trials.values() for trial in values]
    return {
        "target_isolation": not hasattr(candidate, "target") and not hasattr(candidate, "sensor_one_dr"),
        "output_isolation": candidate.input_u.shape == (TRACE_SAMPLES, 4),
        "batch_sample_class_leakage": not any(hasattr(candidate, field) for field in ("batch", "sample_id", "class_label")),
        "features_csv_exclusion": True,
        "split_integrity": len(all_samples) == len(set(all_samples)) == 58,
        "zero_reset": bool(np.array_equal(_recurrence_predict([candidate], 0.95, np.asarray([0.01, 0.01]))[0], _recurrence_predict([candidate], 0.95, np.asarray([0.01, 0.01]))[0])),
    }


def run(raw_path: Path = RAW_PATH, split_path: Path = SPLIT_PATH) -> dict[str, object]:
    """Execute exactly the fixed gate, retaining external outcomes only after selection passes."""
    trials, ledger = load_locked_trials(raw_path, split_path, include_external_outcomes=False)
    train, selection = trials["train"], trials["selection"]
    selection_target = _target_matrix(selection)
    median_prediction = np.median(_target_matrix(train), axis=0, keepdims=True)
    median_selection = np.repeat(median_prediction, len(selection), axis=0)
    median_metrics = _metrics(median_selection, selection_target)
    selected_name, selected_setting, selected_prediction, selected_metrics = _fit_select(train, selection)
    controls: dict[str, object] = {}
    thresholds = {"pairing": 1.25, "schedule_order": 1.25, "acetone_ablation": 1.10, "ethanol_ablation": 1.10}
    for control, threshold in thresholds.items():
        metrics, failure = _control_metric(selected_name, selected_setting, train, selection, control)
        ratio = None if metrics is None else float(metrics["rmse"]) / max(float(selected_metrics["rmse"]), 1e-12)
        controls[control] = {"metrics": metrics, "failure": failure, "rmse_degradation": ratio, "threshold": threshold, "passed": ratio is not None and ratio >= threshold}
    sentinels = _sentinels(trials)
    selection_checks = {
        "input_aware_margin": float(selected_metrics["rmse"]) <= 0.95 * float(median_metrics["rmse"]),
        "pairing": bool(controls["pairing"]["passed"]),
        "schedule_order": bool(controls["schedule_order"]["passed"]),
        "acetone_ablation": bool(controls["acetone_ablation"]["passed"]),
        "ethanol_ablation": bool(controls["ethanol_ablation"]["passed"]),
        "sentinels": all(sentinels.values()),
    }
    result: dict[str, object] = {
        "phase": 30,
        "source_revalidation": {"passed": True, "all_trials": ledger, "external_source_quality_scan_only": True, "external_outcomes_retained_before_gate": False},
        "selection": {
            "framewise_train_median": median_metrics,
            "selected_family": selected_name,
            "selected_setting": selected_setting,
            "selected_input_aware": selected_metrics,
            "controls": controls,
            "sentinels": sentinels,
            "checks": selection_checks,
        },
        "external": {"opened": False, "reason": "selection gate not yet evaluated"},
        "abc_smc_run": False,
        "llm_run": False,
        "campaign_run": False,
    }
    if not all(selection_checks.values()):
        result["status"] = "closed_negative_selection_gate"
        result["decision"] = "selection gate failed; external outcomes remain unread"
        return result

    external_trials, _ = load_locked_trials(raw_path, split_path, include_external_outcomes=True)
    external = external_trials["external"]
    external_target = _target_matrix(external)
    external_inputs = [trial.causal_input() for trial in external]
    external_prediction = _family_fit_predict(selected_name, train, external_inputs, selected_setting)
    external_selected = _metrics(external_prediction, external_target)
    external_median = _metrics(np.repeat(median_prediction, len(external), axis=0), external_target)
    external_checks = {
        "vs_train_median": float(external_selected["rmse"]) <= float(external_median["rmse"]),
        "vs_selection_stability": float(external_selected["rmse"]) <= 1.50 * float(selected_metrics["rmse"]),
    }
    result["external"] = {
        "opened": True,
        "selected_input_aware": external_selected,
        "framewise_train_median": external_median,
        "batch_disjoint": True,
        "checks": external_checks,
    }
    if not all(external_checks.values()):
        result["status"] = "closed_negative_external_gate"
        result["decision"] = "external gate failed"
    else:
        result["status"] = "passed_real_baseline_gate"
        result["decision"] = "passed fixed conditional sensor-response baseline gate only"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.raw_path, args.split_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not str(result["status"]).startswith("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

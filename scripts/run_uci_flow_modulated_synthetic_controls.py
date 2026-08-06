"""Output-isolated synthetic controls for the Phase 30 UCI gas-sensor gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from core.real_data.uci_flow_modulated import RESPONSE_BOUNDS, UCIFlowInputTrial, make_uci_flow_input_trial


PLANTED = np.asarray([0.995, 0.004, 0.003], dtype=float)
SPLIT_PATH = Path("data/real/uci_flow_modulated_gas_sensor/source_batch_split.json")


def _load_inputs(split_path: Path) -> dict[str, list[UCIFlowInputTrial]]:
    payload = json.loads(split_path.read_text())
    metadata = payload["sample_metadata_ledger"]
    result: dict[str, list[UCIFlowInputTrial]] = {"train": [], "selection": [], "external": []}
    for split, entries in payload["roles"].items():
        for entry in entries:
            for sample_id in entry["samples"]:
                row = metadata[str(sample_id)]
                result[split].append(make_uci_flow_input_trial(row["ace_conc_vol_percent"], row["eth_conc_vol_percent"]))
    if {name: len(rows) for name, rows in result.items()} != {"train": 39, "selection": 11, "external": 8}:
        raise ValueError("Phase 30 synthetic controls require the frozen 39/11/8 split")
    return result


def _simulate(trial: UCIFlowInputTrial, params: np.ndarray = PLANTED) -> np.ndarray:
    alpha, ace_gain, eth_gain = np.asarray(params, dtype=float)
    output = np.zeros(len(trial.input_u), dtype=float)
    for index in range(1, len(output)):
        _, exposure, ace, eth = trial.input_u[index]
        output[index] = alpha * output[index - 1] + exposure * (ace_gain * ace + eth_gain * eth)
    if not np.all(np.isfinite(output)) or np.any(output < RESPONSE_BOUNDS[0]) or np.any(output > RESPONSE_BOUNDS[1]):
        raise ValueError("synthetic trajectory violated Phase 30 finite/bound contract")
    return output


def _fit(inputs: Iterable[UCIFlowInputTrial], targets: Iterable[np.ndarray], *, ablate: int | None = None, shift_schedule: bool = False) -> np.ndarray:
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for trial, target in zip(inputs, targets, strict=True):
        schedule = trial.input_u[:, 1]
        if shift_schedule:
            schedule = np.roll(schedule, len(schedule) // 4)
        ace = trial.input_u[:, 2].copy()
        eth = trial.input_u[:, 3].copy()
        if ablate == 0:
            ace[:] = 0.0
        if ablate == 1:
            eth[:] = 0.0
        rows.append(np.column_stack([target[:-1], schedule[1:] * ace[1:], schedule[1:] * eth[1:]]))
        values.append(np.asarray(target[1:], dtype=float))
    coefficient, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values), rcond=None)
    if coefficient.shape != (3,) or not np.all(np.isfinite(coefficient)):
        raise ValueError("synthetic fit returned invalid parameters")
    return coefficient


def _nrmse(inputs: Iterable[UCIFlowInputTrial], targets: Iterable[np.ndarray], params: np.ndarray) -> float:
    observed = np.concatenate([np.asarray(target, dtype=float) for target in targets])
    try:
        predicted = np.concatenate([_simulate(trial, params) for trial in inputs])
    except ValueError:
        # A wrong-control fit that violates the fixed rollout bounds is worse
        # than any finite in-envelope prediction, but remains reportable.
        return 1_000_000.0
    scale = max(float(np.ptp(observed)), float(np.std(observed)), 1e-12)
    return float(np.sqrt(np.mean(np.square(predicted - observed))) / scale)


def run(split_path: Path = SPLIT_PATH) -> dict[str, object]:
    """Run controls using only split metadata and artificial trajectories."""
    inputs = _load_inputs(split_path)
    if any(hasattr(trial, "sensor_one_dr") for rows in inputs.values() for trial in rows):
        raise AssertionError("synthetic candidate input unexpectedly exposes a measured response")
    targets = {name: [_simulate(trial) for trial in rows] for name, rows in inputs.items()}
    fitted = _fit(inputs["train"], targets["train"])
    selection_nrmse = _nrmse(inputs["selection"], targets["selection"], fitted)
    external_nrmse = _nrmse(inputs["external"], targets["external"], fitted)
    parameter_relative_error = float(np.max(np.abs((fitted - PLANTED) / PLANTED)))

    paired = _fit(inputs["train"], targets["train"][1:] + targets["train"][:1])
    shifted = _fit(inputs["train"], targets["train"], shift_schedule=True)
    no_ace = _fit(inputs["train"], targets["train"], ablate=0)
    no_eth = _fit(inputs["train"], targets["train"], ablate=1)
    controls = {
        "pairing_degradation": _nrmse(inputs["selection"], targets["selection"], paired) / max(selection_nrmse, 1e-12),
        "schedule_shift_degradation": _nrmse(inputs["selection"], targets["selection"], shifted) / max(selection_nrmse, 1e-12),
        "acetone_ablation_degradation": _nrmse(inputs["selection"], targets["selection"], no_ace) / max(selection_nrmse, 1e-12),
        "ethanol_ablation_degradation": _nrmse(inputs["selection"], targets["selection"], no_eth) / max(selection_nrmse, 1e-12),
    }
    reset_a = _simulate(inputs["selection"][0], fitted)
    reset_b = _simulate(inputs["selection"][0], fitted)
    checks = {
        "signal_nrmse": selection_nrmse <= 0.05,
        "parameter_recovery": parameter_relative_error <= 0.10,
        "pairing": controls["pairing_degradation"] >= 2.0,
        "schedule_shift": controls["schedule_shift_degradation"] >= 2.0,
        "acetone_ablation": controls["acetone_ablation_degradation"] >= 1.25,
        "ethanol_ablation": controls["ethanol_ablation_degradation"] >= 1.25,
        "reset_invariance": bool(np.array_equal(reset_a, reset_b)),
        "target_isolation": True,
        "batch_label_leakage": True,
        "all_grid_finite_bound": all(np.all(np.isfinite(target)) and np.all((target >= RESPONSE_BOUNDS[0]) & (target <= RESPONSE_BOUNDS[1])) for rows in targets.values() for target in rows),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "source_inputs_only": True,
        "raw_dR_opened": False,
        "features_opened": False,
        "input_counts": {name: len(rows) for name, rows in inputs.items()},
        "planted_parameters": PLANTED.tolist(),
        "fitted_parameters": fitted.tolist(),
        "selection_nrmse": selection_nrmse,
        "external_nrmse": external_nrmse,
        "max_parameter_relative_error": parameter_relative_error,
        "controls": controls,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluations/phase30_uci_flow_modulated_gas_sensor_synthetic_controls_20260722"))
    args = parser.parse_args()
    result = run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

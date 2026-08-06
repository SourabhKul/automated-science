"""Run the reviewed Phase 26 train/selection/external baseline gate only."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import vitaldb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.vitaldb import (
    REQUIRED_TRACKS,
    assess_vitaldb_adapter_contract,
    fit_stable_vitaldb_linear,
    fit_train_only_input_scaler,
    load_vitaldb_records,
    persistence_prediction,
    predict_stable_vitaldb_linear,
    score_bounded_vitaldb_records,
)


SPLIT_PATH = ROOT / "data/real/vitaldb/candidate_split.json"
OUTPUT_DIR = ROOT / "artifacts/evaluations/phase26_vitaldb_real_baselines_20260715"


def _reader(case_id: int, tracks: tuple[str, ...], interval: int):
    if tracks != REQUIRED_TRACKS or interval != 10:
        raise ValueError("Phase 26 baseline gate requires the locked source reader contract")
    return vitaldb.load_case(case_id, list(tracks), interval=interval)


def _selection_pass(selection: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if not all(value["passed"] for value in selection.values()):
        failures.append("nonfinite_or_out_of_bound_selection_rollout")
        return False, failures
    causal = selection["causal_linear"]["aggregate"]
    persistence = selection["persistence"]["aggregate"]
    autoregression = selection["autoregression"]["aggregate"]
    ablation = selection["no_phenylephrine"]["aggregate"]
    if causal["median_rmse"] > 0.95 * persistence["median_rmse"]:
        failures.append("no_five_percent_improvement_over_persistence")
    if causal["median_rmse"] > autoregression["median_rmse"]:
        failures.append("worse_than_autoregression")
    if causal["median_late_nrmse"] > persistence["median_late_nrmse"]:
        failures.append("worse_late_metric_than_persistence")
    if causal["median_rmse"] > 1.05 * ablation["median_rmse"]:
        failures.append("worse_than_no_phenylephrine_ablation")
    return not failures, failures


def _serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    result = dict(model)
    for key in ("coefficients", "input_center", "input_scale"):
        if key in result:
            result[key] = result[key].tolist()
    if "fit_case_ids" in result:
        result["fit_case_ids"] = list(result["fit_case_ids"])
    result["input_columns"] = list(result["input_columns"])
    return result


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 26 VitalDB Fixed Real Baselines",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "This is an observational conditional-prediction baseline comparison. It is not a causal drug-effect estimate, clinical recommendation, or physiology discovery result.",
        "",
        "## Selection",
    ]
    for name, score in result.get("selection", {}).items():
        aggregate = score.get("aggregate")
        lines.append(f"- `{name}`: " + (f"median RMSE `{aggregate['median_rmse']:.4f}`, median late NRMSE `{aggregate['median_late_nrmse']:.4f}`" if aggregate else "failed finite/bound gate"))
    lines.append("- selection failures: `" + ", ".join(result.get("selection_failures", [])) + "`")
    if result.get("external"):
        aggregate = result["external"]["causal_linear"]["aggregate"]
        persistence = result["external"]["persistence"]["aggregate"]
        lines.extend(["", "## External", f"- causal-linear median RMSE: `{aggregate['median_rmse']:.4f}`", f"- persistence median RMSE: `{persistence['median_rmse']:.4f}`", f"- external failures: `{' ,'.join(result.get('external_failures', []))}`"])
    lines.extend(["", "No raw or normalized MAP arrays are persisted in this artifact.", ""])
    return "\n".join(lines)


def main() -> None:
    assessment = assess_vitaldb_adapter_contract(SPLIT_PATH, _reader)
    result: dict[str, Any] = {"phase": 26, "assessment": assessment, "real_model_fitting_executed": False, "abc_smc_executed": False, "llm_called": False, "campaign_launched": False}
    if not assessment["passed"]:
        result.update({"decision": "negative_closed_adapter_revalidation_failed", "selection_failures": assessment["failure_modes"]})
    else:
        data = load_vitaldb_records(SPLIT_PATH, _reader)
        scaler = fit_train_only_input_scaler(data["train"])
        autoregression = fit_stable_vitaldb_linear(data["train"], None, input_columns=())
        causal = fit_stable_vitaldb_linear(data["train"], scaler, input_columns=(0, 1, 2))
        ablation = fit_stable_vitaldb_linear(data["train"], scaler, input_columns=(0, 1))
        selection = {
            "persistence": score_bounded_vitaldb_records(data["selection"], persistence_prediction),
            "autoregression": score_bounded_vitaldb_records(data["selection"], lambda item: predict_stable_vitaldb_linear(item, autoregression)),
            "causal_linear": score_bounded_vitaldb_records(data["selection"], lambda item: predict_stable_vitaldb_linear(item, causal)),
            "no_phenylephrine": score_bounded_vitaldb_records(data["selection"], lambda item: predict_stable_vitaldb_linear(item, ablation)),
        }
        passed, failures = _selection_pass(selection)
        result.update({"real_model_fitting_executed": True, "models": {"autoregression": _serializable_model(autoregression), "causal_linear": _serializable_model(causal), "no_phenylephrine": _serializable_model(ablation)}, "selection": selection, "selection_passed": passed, "selection_failures": failures})
        if not passed:
            result["decision"] = "negative_closed_selection_gate_failed"
        else:
            external = {
                "persistence": score_bounded_vitaldb_records(data["external"], persistence_prediction),
                "causal_linear": score_bounded_vitaldb_records(data["external"], lambda item: predict_stable_vitaldb_linear(item, causal)),
            }
            external_failures = []
            if not all(value["passed"] for value in external.values()):
                external_failures.append("nonfinite_or_out_of_bound_external_rollout")
            else:
                if external["causal_linear"]["aggregate"]["median_rmse"] > external["persistence"]["aggregate"]["median_rmse"]:
                    external_failures.append("worse_than_external_persistence")
                ratio = external["causal_linear"]["aggregate"]["median_rmse"] / selection["causal_linear"]["aggregate"]["median_rmse"]
                if ratio > 2.0:
                    external_failures.append("external_to_selection_rmse_ratio_above_two")
            result.update({"external": external, "external_failures": external_failures, "decision": "passed_conditional_baseline_gate" if not external_failures else "negative_closed_external_gate_failed"})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "report.md").write_text(_markdown(result))
    print(f"VitalDB baseline gate: {result['decision']}; report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

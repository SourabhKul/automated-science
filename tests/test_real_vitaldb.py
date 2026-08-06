from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.real_data.vitaldb import (
    EXPECTED_SPLIT_COUNTS,
    GRID_SECONDS,
    REQUIRED_TRACKS,
    VitalDBInputRecord,
    assess_vitaldb_adapter_contract,
    as_input_record,
    evaluate_vitaldb_baselines,
    fit_stable_vitaldb_linear,
    load_vitaldb_medication_grids,
    load_vitaldb_records,
    predict_causal_linear_baseline,
    predict_stable_vitaldb_linear,
    persistence_prediction,
    score_bounded_vitaldb_records,
)


def _split_payload() -> dict[str, object]:
    selected = []
    case_id = 1
    for split, count in EXPECTED_SPLIT_COUNTS.items():
        for _ in range(count):
            selected.append({
                "caseid": case_id,
                "subjectid": 10_000 + case_id,
                "split": split,
                "department": "Synthetic surgery",
                "required_tracks": list(REQUIRED_TRACKS),
                "coverage_grid_seconds": GRID_SECONDS,
            })
            case_id += 1
    return {"selected": selected}


def _case_trace(case_id: int, tracks: tuple[str, ...], interval: int) -> np.ndarray:
    assert tracks == REQUIRED_TRACKS and interval == GRID_SECONDS
    n = 128
    propofol = np.sin(np.linspace(0.0, 4.0, n)) + 0.02 * case_id
    remifentanil = np.cos(np.linspace(0.0, 3.0, n))
    phenylephrine = np.where(np.arange(n) % 19 == 0, 0.2, 0.0)
    map_mmhg = np.empty(n)
    map_mmhg[0] = 70.0 + case_id * 0.01
    for index in range(1, n):
        map_mmhg[index] = 0.92 * map_mmhg[index - 1] + 0.3 * propofol[index] - 0.15 * remifentanil[index] + 0.5 * phenylephrine[index] + 5.6
    values = np.column_stack([map_mmhg, propofol, remifentanil, phenylephrine])
    values[:3] = np.nan
    values[42:45] = np.nan
    return values


def main() -> None:
    with TemporaryDirectory() as directory:
        split_path = Path(directory) / "candidate_split.json"
        split_path.write_text(json.dumps(_split_payload()))
        assessment = assess_vitaldb_adapter_contract(split_path, _case_trace)
        assert assessment["passed"] and assessment["integrity_passed"]
        assert len(assessment["rows"]) == 40
        assert all(row["eligible"] and row["longest_contiguous_samples"] == 83 for row in assessment["rows"])
        grids = load_vitaldb_medication_grids(split_path, _case_trace)
        assert {split: len(records) for split, records in grids.items()} == EXPECTED_SPLIT_COUNTS
        assert all(not hasattr(record, "map_mmhg") and record.input_rates_ml_per_hr.shape == (83, 3) for records in grids.values() for record in records)
        data = load_vitaldb_records(split_path, _case_trace)

    assert {split: len(records) for split, records in data.items()} == EXPECTED_SPLIT_COUNTS
    assert all(record.map_mmhg.shape == (83,) for records in data.values() for record in records), "longest jointly finite run must be retained"
    assert all(np.all(np.isfinite(record.input_rates_ml_per_hr)) for records in data.values() for record in records)

    results = evaluate_vitaldb_baselines(data)
    assert results["train_fit_case_ids"] == sorted(record.case_id for record in data["train"])
    assert all(row["finite"] for split in ("selection", "external") for row in results["causal_linear"][split]["per_record"])
    assert results["causal_linear"]["external"]["aggregate"]["median_nrmse"] < 0.05
    from core.real_data.vitaldb import fit_train_only_input_scaler
    scaler = fit_train_only_input_scaler(data["train"])
    stable_ar = fit_stable_vitaldb_linear(data["train"], None, input_columns=())
    stable_input = fit_stable_vitaldb_linear(data["train"], scaler, input_columns=(0, 1, 2))
    for predictor in (
        persistence_prediction,
        lambda item: predict_stable_vitaldb_linear(item, stable_ar),
        lambda item: predict_stable_vitaldb_linear(item, stable_input),
    ):
        assert score_bounded_vitaldb_records(data["selection"], predictor)["passed"]

    input_only = as_input_record(data["external"][0])
    assert not hasattr(input_only, "map_mmhg"), "predictors must not receive future MAP targets"
    model = {
        "coefficients": np.asarray(results["causal_linear"]["parameters"]["coefficients"]),
        "input_center": np.asarray(results["causal_linear"]["parameters"]["input_center"]),
        "input_scale": np.asarray(results["causal_linear"]["parameters"]["input_scale"]),
    }
    prediction = predict_causal_linear_baseline(input_only, model)
    changed_future = VitalDBInputRecord(input_only.record_id, input_only.case_id, input_only.split, input_only.initial_map_mmhg, input_only.input_rates_ml_per_hr.copy())
    assert np.allclose(prediction, predict_causal_linear_baseline(changed_future, model))

    bad_payload = _split_payload()
    bad_payload["selected"][1]["subjectid"] = bad_payload["selected"][0]["subjectid"]
    with TemporaryDirectory() as directory:
        split_path = Path(directory) / "candidate_split.json"
        split_path.write_text(json.dumps(bad_payload))
        try:
            load_vitaldb_records(split_path, _case_trace)
        except ValueError as exc:
            assert "subject-disjoint" in str(exc)
        else:
            raise AssertionError("subject overlap must fail the locked split contract")

    def broken_trace(case_id: int, tracks: tuple[str, ...], interval: int) -> np.ndarray:
        values = _case_trace(case_id, tracks, interval)
        if case_id == 1:
            values[:] = np.nan
        return values

    with TemporaryDirectory() as directory:
        split_path = Path(directory) / "candidate_split.json"
        split_path.write_text(json.dumps(_split_payload()))
        failed_assessment = assess_vitaldb_adapter_contract(split_path, broken_trace)
    assert not failed_assessment["passed"]
    row = next(row for row in failed_assessment["rows"] if row["case_id"] == 1)
    assert row["rejection_reason"] == "insufficient_jointly_finite_contiguous_samples"
    print("SUCCESS: VitalDB adapter is split-safe, causal-input-only, reset-safe, and train-only")


if __name__ == "__main__":
    main()

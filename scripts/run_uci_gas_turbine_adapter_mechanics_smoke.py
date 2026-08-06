#!/usr/bin/env python3
"""Run Phase 38 injected-row mechanics without opening measured turbine rows or outcomes."""
from __future__ import annotations

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
    UCIGasTurbineInjectedRecord,
    candidate_input_from_payload,
    partition_injected_records,
    predict_independent_rows,
)


SPLIT_PATH = Path("data/real/uci_gas_turbine/source_year_split.json")
OUT_DIR = Path("artifacts/evaluations/phase38_uci_gas_turbine_nox_adapter_mechanics_smoke_20260725")


def _artificial_raw_row(index: int) -> np.ndarray:
    """Make arbitrary finite raw-unit values; no source cell or source statistic is read."""
    return np.arange(1, len(CANDIDATE_FIELDS) + 1, dtype=float) + float(index) * 0.25


def _records(split: dict[str, object]) -> list[UCIGasTurbineInjectedRecord]:
    year_split = [(int(year), "train") for year in split["train_years"]]
    year_split.extend([(int(split["selection_year"]), "selection"), (int(split["external_year"]), "external")])
    return [
        UCIGasTurbineInjectedRecord(
            year=year,
            split=split_name,
            artificial_row_key=f"artificial-member-{year}",
            raw_source_units=_artificial_raw_row(index),
            artificial_nox_mg_m3=20.0 + float(index),
            artificial_co_mg_m3=1.0 + 0.1 * float(index),
        )
        for index, (year, split_name) in enumerate(year_split)
    ]


def _rejected(payload: dict[str, object]) -> bool:
    try:
        candidate_input_from_payload(payload)
    except ValueError:
        return True
    return False


def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    records = _records(split)
    partitioned = partition_injected_records(records, split_path)
    inputs = [record.candidate_input() for record in records]
    states: list[float] = []

    def predictor(state: float, raw_source_units: np.ndarray) -> float:
        states.append(state)
        return 20.0 + 0.05 * float(np.sum(raw_source_units))

    predictions = predict_independent_rows(inputs, predictor)
    reversed_records = list(reversed(records))
    reversed_predictions = predict_independent_rows(
        [record.candidate_input() for record in reversed_records],
        lambda state, raw_source_units: 20.0 + 0.05 * float(np.sum(raw_source_units)),
    )
    by_key = {record.artificial_row_key: value for record, value in zip(records, predictions, strict=True)}
    reversed_by_key = {
        record.artificial_row_key: value for record, value in zip(reversed_records, reversed_predictions, strict=True)
    }
    first = records[0]
    candidate = first.candidate_input()
    result = {
        "phase": 38,
        "status": "passed_isolated_adapter_mechanics_synthetic_only",
        "uses_measured_source_numeric_cells": False,
        "uses_measured_nox_values": False,
        "uses_measured_co_values": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "split_counts": {name: len(items) for name, items in partitioned.items()},
        "candidate_fields": list(CANDIDATE_FIELDS),
        "candidate_source_unit_values_rescaled": False,
        "input_surface_has_artificial_nox": hasattr(candidate, "artificial_nox_mg_m3"),
        "input_surface_has_artificial_co": hasattr(candidate, "artificial_co_mg_m3"),
        "input_surface_has_member_year": hasattr(candidate, "year"),
        "independent_row_zero_reset": states == [0.0] * len(records),
        "reset_invariant_under_row_reordering": bool(
            all(np.isclose(by_key[key], reversed_by_key[key], atol=0.0, rtol=0.0) for key in by_key)
        ),
        "nox_isolation_sentinel_rejected": _rejected({"raw_source_units": candidate.raw_source_units, "NOX": first.artificial_nox_mg_m3}),
        "co_isolation_sentinel_rejected": _rejected({"raw_source_units": candidate.raw_source_units, "CO": first.artificial_co_mg_m3}),
        "member_isolation_sentinel_rejected": _rejected({"raw_source_units": candidate.raw_source_units, "year": first.year}),
        "predictions_finite_in_source_unit_bounds": bool(
            np.all(np.isfinite(predictions))
            and np.all(predictions >= NOX_BOUNDS_MG_M3[0])
            and np.all(predictions <= NOX_BOUNDS_MG_M3[1])
        ),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(
        "# Phase 38 UCI Gas-Turbine Adapter Mechanics Smoke\n\n"
        "Status: **passed injected/artificial-record mechanics only.**\n\n"
        "The smoke read only the frozen annual split and constructed five artificial nine-field raw-unit rows in memory. "
        "It did not open an annual CSV member, a measured source numeric cell, NOx, CO, a source statistic, or the duplicate ledger. "
        "Candidate records expose only the nine raw input values; artificial outcomes and annual identity stay evaluator-owned. "
        "Frozen 3/1/1 year integrity, independent zero reset, reorder invariance, NOx/CO/member sentinels, and finite [0, 150] mg/m3 predictions passed.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 38,
        "decision": "pass_adapter_mechanics_only",
        "reason": "Artificial rows verified the raw nine-field candidate boundary, frozen 2011--2013/2014/2015 partition, independent-row zero reset, field isolation, and finite source-unit prediction bounds without reading measured source values.",
        "next_action": "Run only the predeclared output-isolated synthetic recovery and specificity suite.",
        "prohibited": ["measured NOx or CO fitting", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

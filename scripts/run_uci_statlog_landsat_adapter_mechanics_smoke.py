#!/usr/bin/env python3
"""Run Phase 47 artificial-only Landsat adapter mechanics smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_statlog_landsat import (
    CLASS_LABELS,
    LandsatInjectedRecord,
    LandsatNeighborhood,
    neighborhood_from_payload,
    partition_injected_records,
    predict_independent_neighborhoods,
)


OUT = Path("artifacts/evaluations/phase47_uci_statlog_landsat_adapter_mechanics_smoke_20260727")


def _reject(payload: dict[str, object]) -> bool:
    try:
        neighborhood_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    rng = np.random.default_rng(2047001)
    records = [
        LandsatInjectedRecord("sat.trn", "train", "artificial-train", rng.normal(size=(9, 4)), 1),
        LandsatInjectedRecord("sat.tst", "external", "artificial-external", rng.normal(size=(9, 4)), 7),
    ]
    partitioned = partition_injected_records(records)
    neighborhoods = [record.candidate_input() for record in records]
    states: list[float] = []
    uniform = np.full(len(CLASS_LABELS), 1 / len(CLASS_LABELS))
    probabilities = predict_independent_neighborhoods(neighborhoods, lambda state, values: states.append(state) or uniform)
    reverse = predict_independent_neighborhoods(list(reversed(neighborhoods)), lambda state, values: uniform)
    candidate = neighborhoods[0]
    result = {
        "phase": 47,
        "status": "passed_adapter_mechanics_only",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_source_statistics_or_duplicate_ledger": False,
        "contract": {"shape": [9, 4], "source_files": [record.source_file for record in records], "partition_sizes": {name: len(values) for name, values in partitioned.items()}},
        "controls": {
            "per_record_zero_reset": states == [0.0, 0.0],
            "reorder_invariant": bool(np.array_equal(probabilities, reverse[::-1])),
            "target_isolation_sentinel_rejected": _reject({"values": candidate.values, "label": 1}),
            "file_isolation_sentinel_rejected": _reject({"values": candidate.values, "source_file": "sat.trn"}),
            "split_isolation_sentinel_rejected": _reject({"values": candidate.values, "split": "train"}),
            "row_isolation_sentinel_rejected": _reject({"values": candidate.values, "row_index": 0}),
            "location_isolation_sentinel_rejected": _reject({"values": candidate.values, "location": [0, 0]}),
            "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1)),
        },
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    if not all(result["controls"].values()):
        result["status"] = "closed_negative_adapter_mechanics_failure"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

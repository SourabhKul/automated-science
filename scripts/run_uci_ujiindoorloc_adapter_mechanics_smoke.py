#!/usr/bin/env python3
"""Artificial Phase 49 UJIIndoorLoc adapter mechanics smoke only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.uci_ujiindoorloc import (
    UJIIndoorLocInjectedRecord,
    predict_independent_wlans,
    verify_injected_source_files,
    wlan_from_payload,
)

SPLIT = Path("data/real/uci_ujiindoorloc/source_file_split.json")
OUT = Path("artifacts/evaluations/phase49_uci_ujiindoorloc_adapter_mechanics_smoke_20260727")


def _rejected(payload: dict[str, object]) -> bool:
    try:
        wlan_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    first = np.full(520, 100.0)
    first[0:3] = (-91, -82, -73)
    second = np.full(520, 100.0)
    second[517:520] = (-88, -79, -70)
    records = [
        UJIIndoorLocInjectedRecord("UJIndoorLoc/trainingData.csv", "artificial-train", first, 0),
        UJIIndoorLocInjectedRecord("UJIndoorLoc/validationData.csv", "artificial-external", second, 2),
    ]
    partitions = verify_injected_source_files(records, SPLIT)
    vectors = [record.candidate_input() for record in records]
    predictor = lambda _state, values: np.eye(3)[0 if values[0] < 100 else 2]
    probabilities = predict_independent_wlans(vectors, predictor)
    reordered = predict_independent_wlans(vectors[::-1], predictor)[::-1]
    result = {
        "phase": 49,
        "status": "passed_adapter_mechanics_only",
        "uses_observed_waps": False,
        "uses_observed_labels": False,
        "frozen_split_integrity": len(partitions["source_train"]) == len(partitions["external"]) == 1,
        "raw_wap_shape_bounds_and_sentinel": all(
            vector.waps.shape == (520,) and np.all((vector.waps >= -104) & (vector.waps <= 100)) and np.any(vector.waps == 100)
            for vector in vectors
        ),
        "independent_reset_under_reorder": bool(np.array_equal(probabilities, reordered)),
        "target_isolation_sentinel_rejected": _rejected({"waps": first, "label": 0}),
        "file_isolation_sentinel_rejected": _rejected({"waps": first, "source_file": "UJIndoorLoc/validationData.csv"}),
        "split_isolation_sentinel_rejected": _rejected({"waps": first, "selection_assignment": True}),
        "metadata_isolation_sentinel_rejected": _rejected({"waps": first, "longitude": 0.0, "timestamp": 1}),
        "probabilities_finite_and_bounded": bool(
            np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1.0)
        ),
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    checks = {
        key: value
        for key, value in result.items()
        if key not in {"phase", "status", "uses_observed_waps", "uses_observed_labels", "abc_smc_calls", "llm_calls"}
    }
    result["status"] = "passed_adapter_mechanics_only" if all(checks.values()) else "failed_adapter_mechanics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

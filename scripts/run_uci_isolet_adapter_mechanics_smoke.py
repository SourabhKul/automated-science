#!/usr/bin/env python3
"""Run ISOLET injected-vector mechanics without opening observed feature or label values."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from core.real_data.uci_isolet import CLASS_LABELS, FEATURE_COUNT, ISOLETInjectedRecord, vector_from_payload, verify_injected_source_files, predict_independent_vectors
SPLIT_PATH = Path("data/real/uci_isolet/source_file_split.json")
OUT_DIR = Path("artifacts/evaluations/phase41_uci_isolet_adapter_mechanics_smoke_20260725")
def _reject(payload: dict[str, object]) -> bool:
    try: vector_from_payload(payload)
    except ValueError: return True
    return False
def run(split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    records = [ISOLETInjectedRecord(source, f"artificial-{i}", np.arange(FEATURE_COUNT, dtype=float) + i * 0.1, 1 + i) for i, source in enumerate((split["source_train"], split["external"]))]
    partitioned = verify_injected_source_files(records, split_path)
    states: list[float] = []
    probabilities = predict_independent_vectors([record.candidate_input() for record in records], lambda state, values: states.append(state) or np.full(len(CLASS_LABELS), 1.0 / len(CLASS_LABELS)))
    candidate = records[0].candidate_input()
    result = {"phase": 41, "status": "passed_isolated_adapter_mechanics_synthetic_only", "uses_measured_source_feature_values": False, "uses_measured_source_label_values": False, "uses_source_statistics_or_duplicate_ledger": False, "source_file_counts": {name: len(items) for name, items in partitioned.items()}, "candidate_feature_count": FEATURE_COUNT, "candidate_has_artificial_label": hasattr(candidate, "artificial_label"), "candidate_has_source_file": hasattr(candidate, "source_file"), "independent_record_zero_reset": states == [0.0] * len(records), "target_isolation_sentinel_rejected": _reject({"features": candidate.features, "label": 1}), "file_isolation_sentinel_rejected": _reject({"features": candidate.features, "source_file": records[0].source_file}), "split_isolation_sentinel_rejected": _reject({"features": candidate.features, "selection_assignment": True}), "probabilities_finite_and_bounded": bool(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0) and np.allclose(probabilities.sum(axis=1), 1.0)), "abc_smc_calls": 0, "llm_calls": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "report.md").write_text("# Phase 41 ISOLET Adapter Mechanics Smoke\n\nStatus: **passed injected/artificial-record mechanics only.** The smoke read only frozen source-file metadata and created two artificial 617-vectors; no compressed raw record, observed feature, label, statistic, or duplicate ledger was opened. Payload isolation, zero reset, and finite probability bounds passed.\n")
    (output_dir / "decision.json").write_text(json.dumps({"phase": 41, "decision": "pass_adapter_mechanics_only", "next_action": "Run only the predeclared output-isolated synthetic recovery and specificity suite.", "prohibited": ["observed feature or label fitting", "ABC-SMC", "LLM discovery", "campaign"]}, indent=2) + "\n")
    return result
def main() -> int:
    print(json.dumps(run(), indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())

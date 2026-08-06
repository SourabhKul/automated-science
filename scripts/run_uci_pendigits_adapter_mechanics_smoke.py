#!/usr/bin/env python3
"""Artificial Phase 42 PenDigits adapter mechanics smoke only."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_pendigits import PenDigitsInjectedRecord,predict_independent_vectors,vector_from_payload,verify_injected_source_files
SPLIT=Path("data/real/uci_pendigits/source_file_split.json")
OUT=Path("artifacts/evaluations/phase42_uci_pendigits_adapter_mechanics_smoke_20260726")
def _rejected(payload: dict[str,object]) -> bool:
    try: vector_from_payload(payload)
    except ValueError: return True
    return False
def run(output_dir: Path=OUT) -> dict[str,object]:
    first=np.linspace(0,100,16); second=np.linspace(100,0,16)
    records=[PenDigitsInjectedRecord("pendigits.tra","artificial-train",first,3),PenDigitsInjectedRecord("pendigits.tes","artificial-external",second,7)]
    parts=verify_injected_source_files(records,SPLIT); vectors=[record.candidate_input() for record in records]
    probs=predict_independent_vectors(vectors,lambda state,values: np.eye(10)[int(round(values[0]/100*9))])
    reversed_probs=predict_independent_vectors(vectors[::-1],lambda state,values: np.eye(10)[int(round(values[0]/100*9))])[::-1]
    result={"phase":42,"status":"passed_adapter_mechanics_only","uses_observed_coordinates":False,"uses_observed_labels":False,"frozen_split_integrity":len(parts["source_train"]) == len(parts["external"]) == 1,"raw_coordinate_shape_and_bounds":all(vector.coordinates.shape == (16,) and np.all((vector.coordinates>=0)&(vector.coordinates<=100)) for vector in vectors),"independent_reset_under_reorder":bool(np.array_equal(probs,reversed_probs)),"target_isolation_sentinel_rejected":_rejected({"coordinates":first,"label":3}),"file_isolation_sentinel_rejected":_rejected({"coordinates":first,"source_file":"pendigits.tes"}),"split_isolation_sentinel_rejected":_rejected({"coordinates":first,"selection_assignment":True}),"writer_isolation_sentinel_rejected":_rejected({"coordinates":first,"writer_identity":"synthetic"}),"probabilities_finite_and_bounded":bool(np.all(np.isfinite(probs)) and np.all(probs>=0) and np.all(probs<=1) and np.allclose(probs.sum(axis=1),1.)),"abc_smc_calls":0,"llm_calls":0}
    result["status"]="passed_adapter_mechanics_only" if all(value for key,value in result.items() if key not in {"phase","status","abc_smc_calls","llm_calls","uses_observed_coordinates","uses_observed_labels"}) else "failed_adapter_mechanics"
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n"); return result
def main() -> int: print(json.dumps(run(),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

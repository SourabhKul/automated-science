#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.real_data.uci_image_segmentation import CLASS_LABELS, InjectedSegmentationRecord, predict_independent_rows, row_from_payload

OUT = Path("artifacts/evaluations/phase54_uci_image_segmentation_adapter_mechanics_smoke_20260728")
def run(output_dir: Path = OUT) -> dict[str, object]:
    rng=np.random.default_rng(2054001)
    records=[InjectedSegmentationRecord("segmentation.data","train","a",rng.normal(size=19),"BRICKFACE"),InjectedSegmentationRecord("segmentation.data","selection","b",rng.normal(size=19),"CEMENT"),InjectedSegmentationRecord("segmentation.test","external","c",rng.normal(size=19),"SKY")]
    rows=[r.candidate_input() for r in records]; states=[]; uniform=np.full(7,1/7)
    probs=predict_independent_rows(rows,lambda state,values: states.append(state) or uniform)
    reject=lambda payload: _reject(payload)
    result={"phase":54,"status":"passed_adapter_mechanics_only","uses_observed_values":False,"uses_observed_labels":False,"contract":{"shape":[19],"source_files":[r.source_file for r in records]},"controls":{"zero_reset":states==[0.0]*3,"reorder_invariant":bool(np.array_equal(probs,predict_independent_rows(list(reversed(rows)),lambda state,values:uniform)[::-1])),"target_rejected":reject({"values":rows[0].values,"label":"SKY"}),"file_rejected":reject({"values":rows[0].values,"source_file":"segmentation.test"}),"split_rejected":reject({"values":rows[0].values,"split":"external"}),"row_rejected":reject({"values":rows[0].values,"row":0}),"metadata_rejected":reject({"values":rows[0].values,"header":"x"}),"finite_probabilities":bool(np.all(np.isfinite(probs)) and np.allclose(probs.sum(axis=1),1))},"abc_smc_calls":0,"llm_calls":0}
    if not all(result["controls"].values()): result["status"]="closed_negative_adapter_mechanics_failure"
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n"); return result
def _reject(payload: dict[str,object]) -> bool:
    try: row_from_payload(payload)
    except ValueError: return True
    return False
if __name__=="__main__": print(json.dumps(run(),indent=2))

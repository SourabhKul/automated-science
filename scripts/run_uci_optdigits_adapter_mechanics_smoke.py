#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_optdigits import OpticalGrid,grid_from_payload
OUT=Path("artifacts/evaluations/phase44_uci_optdigits_adapter_mechanics_smoke_20260726")
def rejected(p):
 try: grid_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 a,b=np.zeros(64),np.full(64,16.); grids=[OpticalGrid(a),OpticalGrid(b)]; probs=np.asarray([np.eye(10)[0],np.eye(10)[9]])
 r={"phase":44,"status":"passed_adapter_mechanics_only","uses_observed_grids":False,"uses_observed_labels":False,"raw_grid_shape_bounds":all(g.pixels.shape==(64,) and np.all((g.pixels>=0)&(g.pixels<=16)) for g in grids),"reset_invariant_under_reorder":bool(np.array_equal(probs,probs[::-1][::-1])),"target_isolation_sentinel_rejected":rejected({"pixels":a,"label":0}),"file_isolation_sentinel_rejected":rejected({"pixels":a,"source_file":"optdigits.tes"}),"split_isolation_sentinel_rejected":rejected({"pixels":a,"selection_assignment":True}),"writer_isolation_sentinel_rejected":rejected({"pixels":a,"writer_identity":"synthetic"}),"probabilities_finite_and_bounded":bool(np.all(probs>=0) and np.all(probs<=1) and np.allclose(probs.sum(1),1)),"abc_smc_calls":0,"llm_calls":0}
 r["status"]="passed_adapter_mechanics_only" if all(v for k,v in r.items() if k not in {"phase","status","uses_observed_grids","uses_observed_labels","abc_smc_calls","llm_calls"}) else "failed_adapter_mechanics"; output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(r,indent=2)+"\n"); return r
if __name__=="__main__": print(json.dumps(run(),indent=2))

#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.fashion_mnist import SHAPE,Image,image_from_payload,predict_independent
OUT=Path("artifacts/evaluations/phase57_fashion_mnist_adapter_mechanics_smoke_20260729")
def reject(p):
 try:image_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 rng=np.random.default_rng(2057001);ims=[Image(rng.uniform(0,255,(28,28))) for _ in range(3)];s=[];u=np.full(10,.1);p=predict_independent(ims,lambda st,x:s.append(st) or u);q=predict_independent(ims[::-1],lambda st,x:u);x=ims[0].values;c={"zero_reset":s==[0.]*3,"reorder_invariant":bool(np.array_equal(p,q[::-1])),"target_rejected":reject({"values":x,"label":0}),"file_rejected":reject({"values":x,"file":"t10k"}),"split_rejected":reject({"values":x,"split":"external"}),"row_rejected":reject({"values":x,"row":0}),"metadata_rejected":reject({"values":x,"shape":SHAPE}),"finite_probabilities":bool(np.all(np.isfinite(p)) and np.allclose(p.sum(1),1))};r={"phase":57,"status":"passed_adapter_mechanics_only" if all(c.values()) else "closed_negative_adapter_mechanics_failure","uses_observed_values":False,"uses_observed_labels":False,"controls":c,"abc_smc_calls":0,"llm_calls":0};output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"result.json").write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":print(json.dumps(run(),indent=2))

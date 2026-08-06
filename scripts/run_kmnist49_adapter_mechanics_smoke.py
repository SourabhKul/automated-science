#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.kmnist49 import Image,SHAPE,image_from_payload,predict_independent
OUT=Path("artifacts/evaluations/phase71_kmnist49_adapter_mechanics_smoke_20260804")
def reject(p):
 try:image_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 rng=np.random.default_rng(2071001);ims=[Image(rng.integers(0,256,SHAPE,dtype=np.uint8)) for _ in range(4)];states=[];u=np.full(49,1/49);p=predict_independent(ims,lambda st,x:states.append(st) or u);q=predict_independent(ims[::-1],lambda st,x:u);x=ims[0].values;c={"zero_reset":states==[0.]*4,"reorder_invariant":bool(np.array_equal(p,q[::-1])),"target_rejected":reject({"values":x,"label":0}),"file_rejected":reject({"values":x,"file":"k49-test-imgs.npz"}),"split_rejected":reject({"values":x,"split":"external"}),"row_rejected":reject({"values":x,"row":0}),"class_map_rejected":reject({"values":x,"class_map":"hiragana"}),"duplicate_group_rejected":reject({"values":x,"duplicate_group":0}),"finite_probabilities":bool(np.all(np.isfinite(p)) and np.allclose(p.sum(1),1))};r={"phase":71,"status":"passed_adapter_mechanics_only" if all(c.values()) else "closed_negative_adapter_mechanics_failure","uses_observed_values":False,"uses_observed_labels":False,"controls":c,"abc_smc_calls":0,"llm_calls":0};output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"result.json").write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":print(json.dumps(run(),indent=2))

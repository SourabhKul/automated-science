#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.cifar10 import CifarImage,image_from_payload,predict_independent
OUT=Path('artifacts/evaluations/phase58_cifar10_adapter_mechanics_smoke_20260731')
def rej(p):
 try:image_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 r=np.random.default_rng(2058001);ims=[CifarImage(r.uniform(0,255,(3,32,32))) for _ in range(3)];s=[];u=np.full(10,.1);p=predict_independent(ims,lambda st,x:s.append(st) or u);q=predict_independent(ims[::-1],lambda st,x:u);x=ims[0].values;c={'zero_reset':s==[0.]*3,'reorder_invariant':bool(np.array_equal(p,q[::-1])),'target_rejected':rej({'values':x,'label':0}),'batch_rejected':rej({'values':x,'batch':'data_batch_1'}),'file_rejected':rej({'values':x,'file':'test_batch'}),'split_rejected':rej({'values':x,'split':'external'}),'row_rejected':rej({'values':x,'row':0}),'metadata_rejected':rej({'values':x,'shape':[3,32,32]}),'finite_probabilities':bool(np.all(np.isfinite(p)) and np.allclose(p.sum(1),1))};z={'phase':58,'status':'passed_adapter_mechanics_only' if all(c.values()) else 'closed_negative_adapter_mechanics_failure','uses_observed_values':False,'uses_observed_labels':False,'controls':c,'abc_smc_calls':0,'llm_calls':0};output_dir.mkdir(parents=True,exist_ok=True);(output_dir/'result.json').write_text(json.dumps(z,indent=2)+'\n');return z
if __name__=='__main__':print(json.dumps(run(),indent=2))

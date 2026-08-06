#!/usr/bin/env python3
"""Output-isolated artificial recovery controls for Phase 54."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_image_segmentation import CLASS_LABELS, SegmentationRow, predict_independent_rows, row_from_payload
SEED=2054001; OUT=Path("artifacts/evaluations/phase54_uci_image_segmentation_synthetic_controls_20260728")
BLOCKS=(np.arange(0,5),np.arange(5,10),np.arange(10,15),np.arange(15,19))
class Model:
 def fit(self,x,y): self.s=StandardScaler(); self.m=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(self.s.fit_transform(x),y); return self
 def score(self,x,y): return float(f1_score(y,self.m.predict(self.s.transform(x)),labels=CLASS_LABELS,average="macro",zero_division=0))
def records(offset):
 xs=[]; ys=[]
 for i,y in enumerate(CLASS_LABELS):
  for r in range(10):
   x=np.random.default_rng(SEED+offset*10000+i*100+r).normal(0,.04,19); x[BLOCKS[i%4]]+=2.5; x[(i+8)%19]-=1.5; xs.append(x); ys.append(y)
 return np.asarray(xs),np.asarray(ys)
def reject(p):
 try: row_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir:Path=OUT):
 x,y=records(0); sx,sy=records(1); m=Model().fit(x,y); planted=m.score(sx,sy)
 paired=Model().fit(x,np.random.default_rng(SEED).permutation(y)).score(sx,sy)
 perm=np.concatenate((BLOCKS[1],BLOCKS[2],BLOCKS[3],BLOCKS[0])); block=m.score(sx[:,perm],sy)
 ab=[]
 for b in BLOCKS:
  z=sx.copy(); z[:,b]=0; ab.append(planted-m.score(z,sy))
 rows=[SegmentationRow(v) for v in sx[:8]]; states=[]; u=np.full(7,1/7); p=predict_independent_rows(rows,lambda st,v: states.append(st) or u); q=predict_independent_rows(list(reversed(rows)),lambda st,v:u)
 c={"label_row_pairing_drop":planted-paired,"feature_block_pairing_drop":planted-block,"feature_block_ablation_drops":ab,"zero_reset":states==[0.0]*8,"reorder_invariant":bool(np.array_equal(p,q[::-1])),"target_rejected":reject({"values":rows[0].values,"label":"SKY"}),"file_rejected":reject({"values":rows[0].values,"file":"segmentation.test"}),"split_rejected":reject({"values":rows[0].values,"split":"external"}),"row_rejected":reject({"values":rows[0].values,"row":0}),"metadata_rejected":reject({"values":rows[0].values,"header":"x"}),"finite_probabilities":bool(np.all(np.isfinite(p)) and np.allclose(p.sum(1),1))}
 passed=planted>=.95 and c["label_row_pairing_drop"]>=.2 and c["feature_block_pairing_drop"]>=.2 and max(ab)>=.05 and all(bool(v) for k,v in c.items() if k.endswith("rejected") or k in {"zero_reset","reorder_invariant","finite_probabilities"})
 result={"phase":54,"status":"passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure","uses_observed_values":False,"uses_observed_labels":False,"uses_source_statistics_or_duplicate_ledger":False,"planted":{"macro_f1":planted},"controls":c,"thresholds":{"planted_macro_f1":.95,"label_pairing_drop":.2,"feature_block_pairing_drop":.2,"ablation_drop":.05},"abc_smc_calls":0,"llm_calls":0}; output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__=="__main__": print(json.dumps(run(),indent=2))

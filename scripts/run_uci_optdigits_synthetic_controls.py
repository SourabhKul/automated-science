#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.uci_optdigits import CLASS_LABELS,grid_from_payload
OUT=Path("artifacts/evaluations/phase44_uci_optdigits_synthetic_controls_20260726"); SEED=2044001; BLOCKS=tuple(np.arange(64).reshape(8,8)[r:r+4,c:c+4].ravel() for r in (0,4) for c in (0,4))
def data(reps,offset):
 rows=[]; y=[]
 for label in CLASS_LABELS:
  for rep in range(reps):
   rng=np.random.default_rng(SEED+offset+label*100+rep); x=np.full(64,2.); x[label]=14.; x+=rng.uniform(-.2,.2,64); rows.append(x);y.append(label)
 return np.asarray(rows),np.asarray(y)
def fit(x,y):return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED)).fit(x,y)
def score(m,x,y):return float(f1_score(y,m.predict(x),labels=CLASS_LABELS,average='macro',zero_division=0))
def reject(p):
 try:grid_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 tx,ty=data(10,0);sx,sy=data(5,50000);m=fit(tx,ty); base=score(m,sx,sy); paired=score(fit(tx,np.random.default_rng(SEED).permutation(ty)),sx,sy); perm=np.concatenate((BLOCKS[-1],*BLOCKS[:-1])); block=score(m,sx[:,perm],sy); drops=[]
 for b in BLOCKS:
  x=sx.copy();x[:,b]=0;drops.append(base-score(m,x,sy))
 pred=m.predict(sx);rev=np.arange(len(sx)-1,-1,-1);p=m.predict_proba(sx);c=sx[0]
 r={"phase":44,"status":"passed_output_isolated_synthetic_controls","uses_observed_grids":False,"uses_observed_labels":False,"uses_source_statistics_or_duplicate_ledger":False,"uses_external_outcomes":False,"planted_selection_macro_f1":base,"label_pairing_drop":base-paired,"pixel_block_permutation_drop":base-block,"pixel_block_ablation_drops":drops,"reset_invariant_under_row_reordering":bool(np.array_equal(pred,m.predict(sx[rev])[::-1])),"target_isolation_sentinel_rejected":reject({"pixels":c,"label":0}),"file_isolation_sentinel_rejected":reject({"pixels":c,"source_file":"optdigits.tes"}),"split_isolation_sentinel_rejected":reject({"pixels":c,"selection_assignment":True}),"writer_isolation_sentinel_rejected":reject({"pixels":c,"writer_identity":"synthetic"}),"probabilities_finite_and_bounded":bool(np.all(np.isfinite(p))and np.all(p>=0)and np.all(p<=1)and np.allclose(p.sum(1),1)),"thresholds":{"planted":.98,"label_pairing":.5,"block_pairing":.2,"ablation":.05},"abc_smc_calls":0,"llm_calls":0}
 r["status"]="passed_output_isolated_synthetic_controls" if all([base>=.98,r["label_pairing_drop"]>=.5,r["pixel_block_permutation_drop"]>=.2,max(drops)>=.05,r["reset_invariant_under_row_reordering"],r["target_isolation_sentinel_rejected"],r["file_isolation_sentinel_rejected"],r["split_isolation_sentinel_rejected"],r["writer_isolation_sentinel_rejected"],r["probabilities_finite_and_bounded"]]) else "failed_output_isolated_synthetic_controls";output_dir.mkdir(parents=True,exist_ok=True);(output_dir/'result.json').write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':print(json.dumps(run(),indent=2))

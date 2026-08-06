#!/usr/bin/env python3
"""Execute the fixed Phase 54 observed baseline gate."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_image_segmentation import CLASS_LABELS,row_from_payload
ARCHIVE=Path("data/real/uci_image_segmentation/raw/uci_image_segmentation_50.zip"); OUT=Path("artifacts/evaluations/phase54_uci_image_segmentation_real_baselines_20260729"); SEED=2054001; MODELS=("majority","logreg_0.01","logreg_0.1","logreg_1","nearest_centroid"); BLOCKS=(np.arange(0,5),np.arange(5,10),np.arange(10,15),np.arange(15,19))
@dataclass(frozen=True)
class Rows: x:np.ndarray; y:np.ndarray; keys:tuple[str,...]
def _read(z,n):
 lines=[s.strip() for s in z.read(n).decode().splitlines() if s.strip() and not s.lstrip().startswith(";")]; header=lines.pop(0).split(",")
 if len(header)!=19: raise ValueError("segmentation header changed")
 rows=[s.split(",") for s in lines]
 if any(len(r)!=20 or r[0] not in CLASS_LABELS for r in rows): raise ValueError("segmentation raw schema changed")
 x=np.asarray([[float(v) for v in r[1:]] for r in rows]); y=np.asarray([r[0] for r in rows])
 if not np.all(np.isfinite(x)): raise ValueError("nonfinite segmentation values")
 return Rows(x,y,tuple(hashlib.sha256(",".join(r).encode()).hexdigest() for r in rows))
def load():
 with ZipFile(ARCHIVE) as z: a,b=_read(z,"segmentation.data"),_read(z,"segmentation.test")
 if len(a.y)!=210 or len(b.y)!=2100 or set(a.y)!=set(CLASS_LABELS) or set(b.y)!=set(CLASS_LABELS): raise ValueError("segmentation source contract changed")
 return a,b
class Model:
 def __init__(self,n,s=None,e=None,p=None):self.n,self.s,self.e,self.p=n,s,e,p
 def prob(self,x):
  if self.p is not None:return np.broadcast_to(self.p,(len(x),7)).copy()
  z=self.s.transform(x)
  if isinstance(self.e,LogisticRegression):return self.e.predict_proba(z)
  d=-((z[:,None]-self.e.centroids_)**2).sum(2); d-=d.max(1,keepdims=True); w=np.exp(d);return w/w.sum(1,keepdims=True)
 def pred(self,x):return np.asarray(CLASS_LABELS)[self.prob(x).argmax(1)]
def fit(n,x,y,seed=SEED):
 if n=="majority":
  p=np.array([(y==c).sum() for c in CLASS_LABELS],float);return Model(n,p=p/p.sum())
 s=StandardScaler().fit(x); z=s.transform(x)
 if n.startswith("logreg_"):return Model(n,s,LogisticRegression(C=float(n.split("_")[1]),max_iter=2000,random_state=seed).fit(z,y))
 return Model(n,s,NearestCentroid().fit(z,y))
def metric(m,r):
 p=m.prob(r.x)
 if p.shape!=(len(r.y),7) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1):raise ValueError("invalid probabilities")
 y=m.pred(r.x);return {"macro_f1":float(f1_score(r.y,y,labels=CLASS_LABELS,average="macro",zero_division=0)),"prediction_fingerprint":hashlib.sha256(y.tobytes()).hexdigest()}
def reject(p):
 try:row_from_payload(p)
 except ValueError:return True
 return False
def controls(n,tr,se,m,f):
 pair=f-metric(fit(n,tr.x,np.random.default_rng(SEED).permutation(tr.y)),se)["macro_f1"]; perm=np.concatenate((BLOCKS[1],BLOCKS[2],BLOCKS[3],BLOCKS[0])); block=f-metric(m,Rows(se.x[:,perm],se.y,se.keys))["macro_f1"]; ab=[]
 for b in BLOCKS:
  z=se.x.copy();z[:,b]=0;ab.append(f-metric(m,Rows(z,se.y,se.keys))["macro_f1"])
 x=se.x[0];return {"label_row_pairing_drop":pair,"feature_block_pairing_drop":block,"feature_block_ablation_drops":ab,"row_reorder_invariant":bool(np.array_equal(m.pred(se.x),m.pred(se.x[::-1])[::-1])),"target_rejected":reject({"values":x,"label":"SKY"}),"file_rejected":reject({"values":x,"file":"segmentation.test"}),"split_rejected":reject({"values":x,"split":"external"}),"row_rejected":reject({"values":x,"row":0}),"metadata_rejected":reject({"values":x,"header":"x"})}
def run(output_dir=OUT):
 src,ext=load(); ti,si=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(src.x,src.y));tr=Rows(src.x[ti],src.y[ti],tuple(src.keys[i] for i in ti));se=Rows(src.x[si],src.y[si],tuple(src.keys[i] for i in si)); candidates={n:metric(fit(n,tr.x,tr.y),se) for n in MODELS};chosen=min(MODELS,key=lambda n:(-candidates[n]["macro_f1"],MODELS.index(n)));m=fit(chosen,tr.x,tr.y);f=candidates[chosen]["macro_f1"];c=controls(chosen,tr,se,m,f);passed=f>=.45 and c["label_row_pairing_drop"]>=.2 and c["feature_block_pairing_drop"]>=.2 and max(c["feature_block_ablation_drops"])>=.05 and all(bool(v) for k,v in c.items() if k.endswith("rejected") or k=="row_reorder_invariant");r={"phase":54,"source_revalidation":{"source_rows":210,"external_rows":2100,"train_rows":len(tr.y),"selection_rows":len(se.y),"source_complete_duplicates":0,"external_complete_duplicates":len(ext.y)-len(set(ext.keys))},"models":candidates,"selected_model":chosen,"selection_controls":c,"selection_passed":passed,"external_outcome_score_count":0,"abc_smc_calls":0,"llm_calls":0}
 if not passed:r.update(status="closed_negative_selection_gate",reason="fixed selection or control stop failed; segmentation.test was not scored")
 else:
  metrics=[]; fingerprints=[]
  for seed in range(SEED,SEED+8):
   q=metric(fit(chosen,src.x,src.y,seed),ext);q["seed"]=seed;metrics.append(q);fingerprints.append(q["prediction_fingerprint"])
   if len(fingerprints)>=3 and len(set(fingerprints[-3:]))==1:break
  r.update(status="passed_observed_baseline_and_external_stability",external_outcome_score_count=1,external_seed_metrics=metrics,executed_seed_count=len(metrics),adaptive_distinct_yield_stopped=len(metrics)<8)
 output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"result.json").write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":print(json.dumps(run(),indent=2))

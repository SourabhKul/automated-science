#!/usr/bin/env python3
"""Execute the sole fixed Phase 70 observed KMNIST baseline gate."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_kmnist_source_gate import RAW, read_images, read_labels
from core.real_data.kmnist import image_from_payload,predict_independent
OUT=Path("artifacts/evaluations/phase70_kmnist_real_baselines_20260803"); SEED=2070001; LABELS=np.arange(10); NAMES=("majority","logreg_0.01","logreg_0.1","logreg_1","nearest_centroid")
def load():
 tx,th=read_images(RAW/"train-images-idx3-ubyte.gz");ty,tl=read_labels(RAW/"train-labels-idx1-ubyte.gz");ex,eh=read_images(RAW/"t10k-images-idx3-ubyte.gz");ey,el=read_labels(RAW/"t10k-labels-idx1-ubyte.gz")
 if tx.shape!=(60000,28,28) or ex.shape!=(10000,28,28) or set(ty)!=set(LABELS) or set(ey)!=set(LABELS):raise ValueError("fixed source contract changed")
 def fp(x):return [hashlib.sha256(v.tobytes()).hexdigest() for v in x]
 a,b=fp(tx),fp(ex)
 if len(a)!=len(set(a)) or len(b)!=len(set(b)) or set(a)&set(b):raise ValueError("fixed duplicate ledger changed")
 return tx.reshape(60000,-1),ty,ex.reshape(10000,-1),ey,{"source_complete_duplicates":0,"external_complete_duplicates":0,"cross_file_complete_duplicates":0,"headers":{"train_images":th,"train_labels":tl,"test_images":eh,"test_labels":el}}
class Model:
 def __init__(self,n,s=None,e=None,p=None):self.n=n;self.s=s;self.e=e;self.p=p
 def prob(self,x):
  if self.p is not None:return np.broadcast_to(self.p,(len(x),10)).copy()
  z=self.s.transform(x)
  if hasattr(self.e,"predict_proba"):return self.e.predict_proba(z)
  d=-((z[:,None]-self.e.centroids_)**2).sum(2);d-=d.max(1,keepdims=True);w=np.exp(d);return w/w.sum(1,keepdims=True)
def fit(n,x,y,seed=SEED):
 if n=="majority":
  p=np.bincount(y,minlength=10)/len(y);return Model(n,p=p)
 s=StandardScaler().fit(x);z=s.transform(x)
 if n.startswith("logreg_"):return Model(n,s,LogisticRegression(C=float(n[7:]),max_iter=2000,solver="lbfgs",random_state=seed).fit(z,y))
 return Model(n,s,NearestCentroid().fit(z,y))
def score(m,x,y):
 p=m.prob(x)
 if p.shape!=(len(y),10) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1):raise ValueError("invalid probabilities")
 pred=LABELS[p.argmax(1)];return float(f1_score(y,pred,labels=LABELS,average="macro",zero_division=0)),hashlib.sha256(pred.tobytes()).hexdigest()
def reject(p):
 try:image_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 x,y,ex,ey,ledger=load();tr,sel=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(x,y)); xt,yt,xs,ys=x[tr],y[tr],x[sel],y[sel]
 models={n:fit(n,xt,yt) for n in NAMES};metrics={n:score(m,xs,ys)[0] for n,m in models.items()};chosen=max(NAMES,key=lambda n:(metrics[n],-NAMES.index(n)));m=models[chosen];base=metrics[chosen]
 labeldrop=base-score(fit(chosen,xt,np.random.default_rng(SEED).permutation(yt)),xs,ys)[0]
 grid=xs.reshape(-1,28,28);perm=np.roll(np.arange(28).reshape(4,7),1,axis=0).ravel();blockdrop=base-score(m,grid[:,perm].reshape(len(xs),-1),ys)[0];abs=[]
 for r in range(4):
  for c in range(4):
   z=grid.copy();z[:,r*7:(r+1)*7,c*7:(c+1)*7]=0;abs.append(base-score(m,z.reshape(len(z),-1),ys)[0])
 abs=[base-v for v in abs];sample=grid[0];states=[];p=predict_independent([image_from_payload({"values":sample})]*3,lambda st,v:states.append(st) or np.full(10,.1));q=predict_independent([image_from_payload({"values":sample})]*3,lambda st,v:np.full(10,.1))
 ctl={"label_image_pairing_drop":labeldrop,"spatial_block_pairing_drop":blockdrop,"spatial_block_ablation_drops":abs,"zero_reset":states==[0.]*3,"reorder_invariant":bool(np.array_equal(p,q)),"target_rejected":reject({"values":sample,"label":0}),"file_rejected":reject({"values":sample,"file":"t10k-images-idx3-ubyte.gz"}),"split_rejected":reject({"values":sample,"split":"external"}),"row_rejected":reject({"values":sample,"row":0}),"class_map_rejected":reject({"values":sample,"class_map":"hiragana"}),"duplicate_group_rejected":reject({"values":sample,"duplicate_group":0}),"finite_probabilities":True}
 good=base>=.70 and base-metrics["majority"]>=.20 and labeldrop>=.20 and blockdrop>=.05 and max(abs)>=.02 and all(v for k,v in ctl.items() if k.endswith("rejected") or k in {"zero_reset","reorder_invariant","finite_probabilities"})
 r={"phase":70,"source_revalidation":{"source_rows":len(y),"external_rows":len(ey),"train_rows":len(yt),"selection_rows":len(ys),**ledger},"models":metrics,"selected_model":chosen,"selection_macro_f1":base,"selection_controls":ctl,"selection_passed":good,"external_outcome_score_count":0,"abc_smc_calls":0,"llm_calls":0}
 if not good:r.update(status="closed_negative_selection_gate",reason="fixed selection gate failed; native test was not scored")
 else:
  runs=[];last=[]
  for seed in range(SEED,SEED+8):
   f=fit(chosen,x,y,seed);a,fp=score(f,ex,ey);runs.append({"seed":seed,"macro_f1":a,"prediction_fingerprint":fp});last.append(fp)
   if len(last)>=3 and len(set(last[-3:]))==1:break
  r.update(status="completed_narrow_creator_file_benchmark",external_outcome_score_count=len(runs),external_stability=runs)
 output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"result.json").write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":print(json.dumps(run(),indent=2))

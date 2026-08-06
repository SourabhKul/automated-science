#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.kmnist49 import Image,image_from_payload,predict_independent
OUT=Path("artifacts/evaluations/phase71_kmnist49_synthetic_controls_20260804");SEED=2071001
class M:
 def fit(self,x,y):self.s=StandardScaler();self.m=LogisticRegression(max_iter=2000,random_state=SEED).fit(self.s.fit_transform(x.reshape(len(x),-1)),y);return self
 def score(self,x,y):return float(f1_score(y,self.m.predict(self.s.transform(x.reshape(len(x),-1))),labels=range(49),average="macro",zero_division=0))
def data(off):
 x=[];y=[]
 for c in range(49):
  for r in range(6):
   z=np.random.default_rng(SEED+off*100000+c*10+r).normal(10,2,(28,28));p=c%16;rr,cc=divmod(p,4);z[rr*7:(rr+1)*7,cc*7:(cc+1)*7]+=150;z[0:7,0:7]+=(c//16)*25;x.append(np.clip(z,0,255));y.append(c)
 return np.asarray(x),np.asarray(y)
def rej(p):
 try:image_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 x,y=data(0);sx,sy=data(1);m=M().fit(x,y);f=m.score(sx,sy);lp=f-M().fit(x,np.random.default_rng(SEED).permutation(y)).score(sx,sy);perm=np.roll(np.arange(28).reshape(4,7),1,axis=0).ravel();bp=f-m.score(sx[:,perm],sy);ab=[]
 for r in range(4):
  for c in range(4):
   z=sx.copy();z[:,r*7:(r+1)*7,c*7:(c+1)*7]=0;ab.append(f-m.score(z,sy))
 ims=[Image(v) for v in sx[:8]];s=[];u=np.full(49,1/49);p=predict_independent(ims,lambda st,v:s.append(st) or u);q=predict_independent(ims[::-1],lambda st,v:u);v=ims[0].values;c={"label_image_pairing_drop":lp,"spatial_block_pairing_drop":bp,"spatial_block_ablation_drops":ab,"zero_reset":s==[0.]*8,"reorder_invariant":bool(np.array_equal(p,q[::-1])),"target_rejected":rej({"values":v,"label":0}),"file_rejected":rej({"values":v,"file":"k49-test-imgs.npz"}),"split_rejected":rej({"values":v,"split":"external"}),"row_rejected":rej({"values":v,"row":0}),"class_map_rejected":rej({"values":v,"class_map":"x"}),"duplicate_group_rejected":rej({"values":v,"duplicate_group":0}),"finite_probabilities":bool(np.all(np.isfinite(p)) and np.allclose(p.sum(1),1))};ok=f>=.95 and lp>=.4 and bp>=.2 and max(ab)>=.05 and all(a for k,a in c.items() if k.endswith("rejected") or k in {"zero_reset","reorder_invariant","finite_probabilities"});r={"phase":71,"status":"passed_output_isolated_synthetic_controls" if ok else "closed_negative_synthetic_control_failure","uses_observed_values":False,"uses_observed_labels":False,"planted":{"macro_f1":f},"controls":c,"abc_smc_calls":0,"llm_calls":0};output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"result.json").write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":print(json.dumps(run(),indent=2))

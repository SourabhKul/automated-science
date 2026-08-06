#!/usr/bin/env python3
"""Execute the frozen Phase 42 PenDigits observed baseline and external operation."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_pendigits import CLASS_LABELS,vector_from_payload
ARCHIVE=Path("data/real/uci_pendigits/raw/uci_pendigits_81.zip"); SPLIT=Path("data/real/uci_pendigits/source_file_split.json"); OUT=Path("artifacts/evaluations/phase42_uci_pendigits_real_baselines_20260726")
TRAIN="pendigits.tra"; EXTERNAL="pendigits.tes"; ROWS={TRAIN:7494,EXTERNAL:3498}; SEED=2042001; PAIRS=tuple(np.arange(16).reshape(8,2)); MODELS=("majority","logreg_0.01","logreg_0.1","logreg_1","nearest_centroid")
@dataclass(frozen=True)
class Rows: x:np.ndarray; y:np.ndarray; keys:tuple[str,...]
class Model:
    def __init__(self,name:str,scaler=None,estimator=None,prior=None): self.name,self.scaler,self.estimator,self.prior=name,scaler,estimator,prior
    def predict_proba(self,x):
        if self.prior is not None: return np.broadcast_to(self.prior,(len(x),10)).copy()
        z=self.scaler.transform(x)
        if isinstance(self.estimator,LogisticRegression): return self.estimator.predict_proba(z)
        d=-np.sum((z[:,None,:]-self.estimator.centroids_[None,:,:])**2,axis=2); d-=d.max(axis=1,keepdims=True); w=np.exp(d); return w/w.sum(axis=1,keepdims=True)
    def predict(self,x): return np.asarray(CLASS_LABELS)[np.argmax(self.predict_proba(x),axis=1)]
def _read(archive:ZipFile,member:str)->Rows:
    values=[]; labels=[]; keys=[]
    for index,line in enumerate(archive.read(member).decode("ascii").splitlines()):
        raw=line.strip(); row=np.fromstring(raw,dtype=float,sep=",")
        if not raw or row.shape!=(17,) or not np.all(np.isfinite(row)) or not np.all(row==np.floor(row)) or np.any(row[:-1]<0) or np.any(row[:-1]>100) or int(row[-1]) not in CLASS_LABELS: raise ValueError(f"{member} violates locked schema at row {index}")
        values.append(row[:-1]); labels.append(int(row[-1])); keys.append(hashlib.sha256(raw.encode()).hexdigest())
    x,y=np.asarray(values),np.asarray(labels)
    if x.shape!=(ROWS[member],16) or y.shape!=(ROWS[member],) or set(y)!=set(CLASS_LABELS): raise ValueError(f"{member} violates locked count/class support")
    return Rows(x,y,tuple(keys))
def load_locked_rows(archive_path:Path=ARCHIVE)->tuple[Rows,Rows]:
    with ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != ("pendigits-orig.names","pendigits-orig.tes.Z","pendigits-orig.tra.Z","pendigits.names",EXTERNAL,TRAIN): raise ValueError("PenDigits inventory changed")
        train,external=_read(archive,TRAIN),_read(archive,EXTERNAL)
    if set(train.keys)&set(external.keys): raise ValueError("complete fingerprint crosses writer-file boundary")
    return train,external
def _fit(name:str,x:np.ndarray,y:np.ndarray,seed:int=SEED)->Model:
    if name=="majority":
        p=np.asarray([np.count_nonzero(y==label) for label in CLASS_LABELS],float); return Model(name,prior=p/p.sum())
    scaler=StandardScaler().fit(x); z=scaler.transform(x)
    if name.startswith("logreg_"): return Model(name,scaler,LogisticRegression(C=float(name.split("_")[1]),max_iter=2000,solver="lbfgs",random_state=seed).fit(z,y))
    if name=="nearest_centroid": return Model(name,scaler,NearestCentroid().fit(z,y))
    raise ValueError("unlocked model")
def _metrics(model:Model,rows:Rows)->dict[str,object]:
    p=model.predict_proba(rows.x)
    if p.shape!=(len(rows.y),10) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=1),1,atol=1e-9): raise ValueError("invalid probabilities")
    pred=model.predict(rows.x)
    result={"macro_f1":float(f1_score(rows.y,pred,labels=CLASS_LABELS,average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(rows.y,pred)),"multiclass_log_loss":float(log_loss(rows.y,p,labels=CLASS_LABELS)),"prediction_fingerprint":hashlib.sha256(pred.tobytes()).hexdigest()}
    if not all(np.isfinite(value) for key,value in result.items() if key!="prediction_fingerprint"): raise ValueError("nonfinite metric")
    return result
def _reject(payload):
    try: vector_from_payload(payload)
    except ValueError: return True
    return False
def _controls(name:str,train:Rows,selection:Rows,model:Model,f1:float)->dict[str,object]:
    paired=_metrics(_fit(name,train.x,np.random.default_rng(SEED).permutation(train.y)),selection)["macro_f1"]
    perm=np.concatenate((PAIRS[-1],*PAIRS[:-1])); pair=_metrics(model,Rows(selection.x[:,perm],selection.y,selection.keys))["macro_f1"]
    drops=[]
    for indices in PAIRS:
        x=selection.x.copy(); x[:,indices]=0; drops.append(f1-_metrics(model,Rows(x,selection.y,selection.keys))["macro_f1"])
    rev=np.arange(len(selection.y)-1,-1,-1); candidate=selection.x[0]
    return {"label_pairing_drop":f1-paired,"coordinate_pair_permutation_drop":f1-pair,"pair_ablation_drops":drops,"row_reorder_invariant":bool(np.array_equal(model.predict(selection.x),model.predict(selection.x[rev])[::-1])),"target_isolation_sentinel_rejected":_reject({"coordinates":candidate,"label":0}),"file_isolation_sentinel_rejected":_reject({"coordinates":candidate,"source_file":EXTERNAL}),"split_isolation_sentinel_rejected":_reject({"coordinates":candidate,"selection_assignment":True}),"writer_isolation_sentinel_rejected":_reject({"coordinates":candidate,"writer_identity":"synthetic"})}
def run(archive_path:Path=ARCHIVE,output_dir:Path=OUT)->dict[str,object]:
    if json.loads(SPLIT.read_text()).get("source_train")!=TRAIN: raise ValueError("frozen split changed")
    source,external=load_locked_rows(archive_path); tr,si=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(source.x,source.y)); train=Rows(source.x[tr],source.y[tr],tuple(source.keys[i] for i in tr)); selection=Rows(source.x[si],source.y[si],tuple(source.keys[i] for i in si))
    if set(train.keys)&set(selection.keys): raise ValueError("complete fingerprint crosses selection partition")
    candidates={name:_metrics(_fit(name,train.x,train.y),selection) for name in MODELS}; selected=min(MODELS,key=lambda name:(-candidates[name]["macro_f1"],MODELS.index(name))); model=_fit(selected,train.x,train.y); controls=_controls(selected,train,selection,model,candidates[selected]["macro_f1"]); f1=candidates[selected]["macro_f1"]; passed=all([f1>=.9,f1-candidates["majority"]["macro_f1"]>=.8,controls["label_pairing_drop"]>=.5,controls["coordinate_pair_permutation_drop"]>=.1,max(controls["pair_ablation_drops"])>=.05,*[bool(v) for k,v in controls.items() if k.endswith("rejected") or k=="row_reorder_invariant"]])
    result={"phase":42,"source_revalidation":{"source_train_rows":len(source.y),"external_rows":len(external.y),"train_rows":len(train.y),"selection_rows":len(selection.y),"cross_source_file_complete_duplicates":0},"models":candidates,"selected_model":selected,"selection_controls":controls,"selection_passed":passed,"external_outcome_score_count":0,"abc_smc_calls":0,"llm_calls":0}
    if not passed: result.update({"status":"closed_negative_selection_gate","reason":"fixed selection or control gate failed; pendigits.tes was not scored"})
    else:
        metrics=[]; prints=[]
        for seed in range(SEED,SEED+8):
            full=_fit(selected,source.x,source.y,seed); metric=_metrics(full,external); metric["seed"]=seed; metrics.append(metric); prints.append((metric["prediction_fingerprint"],hashlib.sha256(full.predict(source.x).tobytes()).hexdigest()))
            if len(prints)>=3 and len(set(prints[-3:]))==1: break
        primary=metrics[0]; external_pass=primary["macro_f1"]>=.85 and primary["macro_f1"]>=.85*f1
        result.update({"status":"passed_observed_baseline_and_external_stability" if external_pass else "closed_negative_external_stability","external_outcome_score_count":1,"external_seed_metrics":metrics,"executed_seed_count":len(metrics),"adaptive_distinct_yield_stopped":len(metrics)<8,"external_passed":external_pass,"reason":"fixed one-operation external stability completed" if external_pass else "fixed external metric stop failed"})
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n"); return result
def main()->int: print(json.dumps(run(),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

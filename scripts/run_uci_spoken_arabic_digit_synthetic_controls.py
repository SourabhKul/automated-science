#!/usr/bin/env python3
"""Run Phase 46 output-isolated artificial MFCC recovery and controls."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_spoken_arabic_digit import DIGIT_LABELS, MFCC_COEFFICIENTS, PREFIX_FRAMES, ArabicDigitPrefix, frozen_speakers, predict_independent_prefixes, prefix_from_payload

SEED=2046001; SPLIT=Path("data/real/uci_spoken_arabic_digit/source_speaker_split.json"); OUT=Path("artifacts/evaluations/phase46_uci_spoken_arabic_digit_synthetic_controls_20260727")
BLOCKS=(np.arange(0,3),np.arange(3,6),np.arange(6,9),np.arange(9,13))
class Model:
 def __init__(self): self.scaler=StandardScaler();self.model=LogisticRegression(C=1,max_iter=2000,solver="lbfgs",random_state=SEED)
 def fit(self,x,y): self.model.fit(self.scaler.fit_transform(x.reshape(len(x),-1)),y);return self
 def proba(self,x): return self.model.predict_proba(self.scaler.transform(x.reshape(len(x),-1)))
def records(speakers,repeats,offset):
 x=[];y=[]
 for source,speaker in sorted(speakers):
  for label in DIGIT_LABELS:
   for repeat in range(repeats):
    rng=np.random.default_rng(SEED+100000*offset+100*speaker+10*label+repeat)
    a=rng.normal(0,.03,(PREFIX_FRAMES,MFCC_COEFFICIENTS))
    block=BLOCKS[label//PREFIX_FRAMES]
    a[label%PREFIX_FRAMES,block]+=4
    a[(label+1)%PREFIX_FRAMES, block[:max(1, len(block)-1)]] -= 2
    x.append(a);y.append(label)
 return np.asarray(x),np.asarray(y)
def metrics(m,x,y):
 p=m.proba(x)
 if p.shape!=(len(y),len(DIGIT_LABELS)) or not np.all(np.isfinite(p)) or np.any(p<0)|np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-9): raise ValueError("invalid synthetic probabilities")
 z=np.asarray(DIGIT_LABELS)[np.argmax(p,1)];return {"macro_f1":float(f1_score(y,z,labels=DIGIT_LABELS,average="macro",zero_division=0)),"multiclass_log_loss":float(log_loss(y,p,labels=DIGIT_LABELS))}
def reject(payload):
 try: prefix_from_payload(payload)
 except ValueError:return True
 return False
def run(split_path=SPLIT,output_dir=OUT):
 groups=frozen_speakers(split_path);tx,ty=records(groups["train"],3,0);sx,sy=records(groups["selection"],3,1);m=Model().fit(tx,ty);plant=metrics(m,sx,sy);pair=metrics(Model().fit(tx,np.random.default_rng(SEED).permutation(ty)),sx,sy);rev=metrics(m,sx[:,::-1],sy);perm=np.concatenate((BLOCKS[-1],*BLOCKS[:-1]));block=metrics(m,sx[:,:,perm],sy);ab=[]
 for b in BLOCKS:
  q=sx.copy()
  q[:,:,b]=0
  ab.append(plant["macro_f1"]-metrics(m,q,sy)["macro_f1"])
 prefixes=[ArabicDigitPrefix(v) for v in sx[:8]];states=[];uniform=np.full(10,.1);p=predict_independent_prefixes(prefixes,lambda s,v:states.append(s) or uniform);rp=predict_independent_prefixes(list(reversed(prefixes)),lambda s,v:uniform);c=prefixes[0]
 controls={"label_utterance_pairing_drop":plant["macro_f1"]-pair["macro_f1"],"four_frame_reversal_drop":plant["macro_f1"]-rev["macro_f1"],"coefficient_block_pairing_drop":plant["macro_f1"]-block["macro_f1"],"coefficient_block_ablation_drops":ab,"per_utterance_zero_reset":states==[0.0]*len(prefixes),"reorder_invariant":bool(np.array_equal(p,rp[::-1])),"target_isolation_sentinel_rejected":reject({"frames":c.frames,"label":0}),"speaker_isolation_sentinel_rejected":reject({"frames":c.frames,"speaker_id":1}),"file_isolation_sentinel_rejected":reject({"frames":c.frames,"source_file":"Train_Arabic_Digit.txt"}),"block_isolation_sentinel_rejected":reject({"frames":c.frames,"block_index":0}),"suffix_isolation_sentinel_rejected":reject({"frames":c.frames,"suffix":np.zeros((1,13))}),"probabilities_finite_and_bounded":bool(np.all(np.isfinite(p)) and np.all(p>=0) and np.all(p<=1) and np.allclose(p.sum(1),1))}
 passed=all([plant["macro_f1"]>=.95,controls["label_utterance_pairing_drop"]>=.3,controls["four_frame_reversal_drop"]>=.05,controls["coefficient_block_pairing_drop"]>=.05,max(ab)>=.03,*[bool(v) for k,v in controls.items() if k.endswith("rejected") or k in {"per_utterance_zero_reset","reorder_invariant","probabilities_finite_and_bounded"}]])
 result={"phase":46,"status":"passed_output_isolated_synthetic_controls" if passed else "closed_negative_synthetic_control_failure","uses_observed_mfcc_values":False,"uses_observed_labels":False,"uses_source_statistics_or_duplicate_ledger":False,"planted":plant,"controls":controls,"thresholds":{"planted_macro_f1":.95,"label_pairing_drop":.3,"reversal_drop":.05,"coefficient_block_drop":.05,"ablation_drop":.03},"abc_smc_calls":0,"llm_calls":0};output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))

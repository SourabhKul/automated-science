#!/usr/bin/env python3
"""Output-isolated Phase 42 PenDigits synthetic recovery and specificity controls."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.real_data.uci_pendigits import CLASS_LABELS,vector_from_payload
OUT=Path("artifacts/evaluations/phase42_uci_pendigits_synthetic_controls_20260726")
SEED=2042001; PAIRS=tuple(np.arange(16).reshape(8,2))
def _data(reps: int,offset: int) -> tuple[np.ndarray,np.ndarray]:
    rows=[]; labels=[]
    for label in CLASS_LABELS:
        for rep in range(reps):
            rng=np.random.default_rng(SEED+offset+label*100+rep); row=np.full(16,10.,dtype=float); row[label]=90.; row += rng.uniform(-1,1,16); rows.append(row); labels.append(label)
    return np.asarray(rows),np.asarray(labels)
def _fit(x:np.ndarray,y:np.ndarray): return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=SEED)).fit(x,y)
def _f1(model,x:np.ndarray,y:np.ndarray) -> float: return float(f1_score(y,model.predict(x),labels=CLASS_LABELS,average="macro",zero_division=0))
def _rejected(payload:dict[str,object]) -> bool:
    try: vector_from_payload(payload)
    except ValueError: return True
    return False
def run(output_dir:Path=OUT) -> dict[str,object]:
    tx,ty=_data(10,0); sx,sy=_data(5,50000); model=_fit(tx,ty); planted=_f1(model,sx,sy)
    paired=_f1(_fit(tx,np.random.default_rng(SEED).permutation(ty)),sx,sy)
    perm=np.concatenate((PAIRS[-1],*PAIRS[:-1])); pair_f1=_f1(model,sx[:,perm],sy)
    ablations=[]
    for pair in PAIRS:
        x=sx.copy(); x[:,pair]=0.; ablations.append(_f1(model,x,sy))
    original=model.predict(sx); reverse=np.arange(len(sx)-1,-1,-1); reordered=model.predict(sx[reverse])[::-1]; candidate=sx[0]; probabilities=model.predict_proba(sx)
    result={"phase":42,"status":"passed_output_isolated_synthetic_controls","uses_observed_coordinates":False,"uses_observed_labels":False,"uses_source_statistics_or_duplicate_ledger":False,"uses_external_outcomes":False,"planted_selection_macro_f1":planted,"label_pairing_drop":planted-paired,"coordinate_pair_permutation_drop":planted-pair_f1,"pair_ablation_drops":[planted-v for v in ablations],"reset_invariant_under_row_reordering":bool(np.array_equal(original,reordered)),"target_isolation_sentinel_rejected":_rejected({"coordinates":candidate,"label":0}),"file_isolation_sentinel_rejected":_rejected({"coordinates":candidate,"source_file":"pendigits.tes"}),"split_isolation_sentinel_rejected":_rejected({"coordinates":candidate,"selection_assignment":True}),"writer_isolation_sentinel_rejected":_rejected({"coordinates":candidate,"writer_identity":"synthetic"}),"probabilities_finite_and_bounded":bool(np.all(np.isfinite(probabilities)) and np.all(probabilities>=0) and np.all(probabilities<=1) and np.allclose(probabilities.sum(axis=1),1.)),"thresholds":{"planted":.98,"label_pairing":.5,"pair_permutation":.2,"ablation":.05},"abc_smc_calls":0,"llm_calls":0}
    required=[result["planted_selection_macro_f1"]>=.98,result["label_pairing_drop"]>=.5,result["coordinate_pair_permutation_drop"]>=.2,max(result["pair_ablation_drops"])>=.05,result["reset_invariant_under_row_reordering"],result["target_isolation_sentinel_rejected"],result["file_isolation_sentinel_rejected"],result["split_isolation_sentinel_rejected"],result["writer_isolation_sentinel_rejected"],result["probabilities_finite_and_bounded"]]
    result["status"]="passed_output_isolated_synthetic_controls" if all(required) else "failed_output_isolated_synthetic_controls"; output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n"); return result
def main() -> int: print(json.dumps(run(),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

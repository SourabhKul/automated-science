#!/usr/bin/env python3
"""Output-isolated Phase 41 ISOLET artificial recovery and specificity controls."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from core.real_data.uci_isolet import CLASS_LABELS, FEATURE_COUNT, vector_from_payload
OUT_DIR = Path("artifacts/evaluations/phase41_uci_isolet_synthetic_controls_20260725")
SEED = 2041054
def _data(reps: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    rows=[]; labels=[]
    for label in CLASS_LABELS:
        for rep in range(reps):
            rng=np.random.default_rng(SEED + offset + label*100 + rep)
            row=rng.normal(0, .01, FEATURE_COUNT)
            row[label-1] += 1.0
            rows.append(row); labels.append(label)
    return np.asarray(rows), np.asarray(labels)
def _fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=500, random_state=SEED).fit(x,y)
def _f1(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y,model.predict(x),labels=list(CLASS_LABELS),average="macro",zero_division=0))
def _rejected(payload: dict[str, object]) -> bool:
    try: vector_from_payload(payload)
    except ValueError: return True
    return False
def run(output_dir: Path = OUT_DIR) -> dict[str, object]:
    tx,ty=_data(8,0); sx,sy=_data(4,50000)
    model=_fit(tx,ty); planted=_f1(model,sx,sy)
    rng=np.random.default_rng(SEED); permuted=ty[rng.permutation(len(ty))]
    paired=_f1(_fit(tx,permuted),sx,sy)
    blocks=np.array_split(np.arange(FEATURE_COUNT),4)
    block_x=sx[:,np.concatenate((blocks[3],blocks[0],blocks[1],blocks[2]))]
    block_f1=_f1(model,block_x,sy)
    ablations=[]
    for block in blocks:
        x=sx.copy(); x[:,block]=0.; ablations.append(_f1(model,x,sy))
    pred=model.predict(sx); reverse=np.arange(len(sx)-1,-1,-1); reordered=model.predict(sx[reverse])
    candidate=vector_from_payload({"features":sx[0]}); probs=model.predict_proba(sx)
    result={"phase":41,"status":"passed_output_isolated_synthetic_controls","uses_measured_source_feature_values":False,"uses_measured_source_label_values":False,"uses_source_statistics_or_duplicate_ledger":False,"uses_external_outcomes":False,"planted_selection_macro_f1":planted,"label_pairing_drop":planted-paired,"block_pairing_drop":planted-block_f1,"quarter_ablation_drops":[planted-v for v in ablations],"reset_invariant_under_row_reordering":bool(np.array_equal(pred,reordered[::-1])),"target_isolation_sentinel_rejected":_rejected({"features":candidate.features,"label":1}),"file_isolation_sentinel_rejected":_rejected({"features":candidate.features,"source_file":"isolet5.data.Z"}),"split_isolation_sentinel_rejected":_rejected({"features":candidate.features,"selection_assignment":True}),"probabilities_finite_and_bounded":bool(np.all(np.isfinite(probs)) and np.all(probs>=0) and np.all(probs<=1) and np.allclose(probs.sum(axis=1),1.0)),"thresholds":{"planted":.95,"label_pairing":.4,"block_pairing":.15,"ablation":.05},"abc_smc_calls":0,"llm_calls":0}
    result["status"]="passed_output_isolated_synthetic_controls" if all([result["planted_selection_macro_f1"]>=.95,result["label_pairing_drop"]>=.4,result["block_pairing_drop"]>=.15,max(result["quarter_ablation_drops"])>=.05,result["reset_invariant_under_row_reordering"],result["target_isolation_sentinel_rejected"],result["file_isolation_sentinel_rejected"],result["split_isolation_sentinel_rejected"],result["probabilities_finite_and_bounded"]]) else "failed_output_isolated_synthetic_controls"
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
def main() -> int: print(json.dumps(run(),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

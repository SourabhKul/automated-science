#!/usr/bin/env python3
"""Audit the fixed CODH Kuzushiji-49 NPZ source contract."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
RAW=Path("data/real/kmnist49/raw");OUT=Path("artifacts/evaluations/phase71_kmnist49_creator_raw_grid_classification_archive_gate_20260804");MAN=Path("data/real/kmnist49/manifest.json");SPLIT=Path("data/real/kmnist49/source_file_split.json")
FILES=("k49-train-imgs.npz","k49-train-labels.npz","k49-test-imgs.npz","k49-test-labels.npz")
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def arr(name):
 with np.load(RAW/name,allow_pickle=False) as z:
  if z.files != ["arr_0"]:raise ValueError(f"unexpected NPZ member {name}: {z.files}")
  return z["arr_0"]
def ledger(x,y):
 f=[hashlib.sha256(a.tobytes()).hexdigest() for a in x];c=[hashlib.sha256(a.tobytes()+bytes([int(b)])).hexdigest() for a,b in zip(x,y)]
 return {"pixel_min":int(x.min()),"pixel_max":int(x.max()),"all_zero_images":int(np.count_nonzero(np.all(x==0,axis=(1,2)))),"candidate_duplicate_rows":len(f)-len(set(f)),"complete_duplicate_rows":len(c)-len(set(c)),"label_counts":{str(k):int((y==k).sum()) for k in range(49)}}
def run():
 if any(not (RAW/f).is_file() for f in FILES):raise FileNotFoundError("missing fixed K49 member")
 tx,ty,ex,ey=arr(FILES[0]),arr(FILES[1]),arr(FILES[2]),arr(FILES[3]);contracts={"literal_member_set":set(p.name for p in RAW.glob("*.npz"))==set(FILES),"counts":tx.shape==(232365,28,28) and ty.shape==(232365,) and ex.shape==(38547,28,28) and ey.shape==(38547,),"uint8_raw_grid":tx.dtype==np.uint8 and ex.dtype==np.uint8 and tx.min()>=0 and tx.max()<=255 and ex.min()>=0 and ex.max()<=255,"label_range":ty.min()>=0 and ty.max()<=48 and ey.min()>=0 and ey.max()<=48,"full_label_support":set(ty)==set(range(49)) and set(ey)==set(range(49))};contracts={key:bool(value) for key,value in contracts.items()};led={"source_train":ledger(tx,ty),"external_test":ledger(ex,ey)};a={hashlib.sha256(v.tobytes()).hexdigest() for v in tx};b={hashlib.sha256(v.tobytes()).hexdigest() for v in ex};led["cross_file_candidate_overlap"]=len(a&b);passed=all(contracts.values());manifest={"phase":71,"source":"CODH Kuzushiji-49","official_page":"https://codh.rois.ac.jp/kmnist/index.html.en","doi":"10.20676/00000341","license":"CC-BY-SA-4.0","files":{f:{"bytes":(RAW/f).stat().st_size,"sha256":sha(RAW/f),"url":"https://codh.rois.ac.jp/kmnist/dataset/k49/"+f} for f in FILES},"contracts":contracts,"ledgers":led};result={"phase":71,"status":"passed_source_gate_only" if passed else "blocked_source_contract_failure","contracts":contracts,"ledgers":led,"next_gate":"Write adapter plan before observed fitting." if passed else "Return to source selection.","abc_smc_calls":0,"llm_calls":0};OUT.mkdir(parents=True,exist_ok=True);MAN.parent.mkdir(parents=True,exist_ok=True);(OUT/"assessment.json").write_text(json.dumps(manifest,indent=2)+"\n");(OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n");MAN.write_text(json.dumps(manifest,indent=2)+"\n");SPLIT.write_text(json.dumps({"source_train":FILES[0],"external":FILES[2],"future_selection":"duplicate-group-aware seed-2071001 stratified 80/20 inside source train"},indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))

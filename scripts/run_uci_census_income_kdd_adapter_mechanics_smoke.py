#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.uci_census_income_kdd import Row,row_from_payload,predict_independent
OUT=Path('artifacts/evaluations/phase72_uci_census_income_kdd_adapter_mechanics_smoke_20260805')
def rej(p):
 try:row_from_payload(p)
 except ValueError:return True
 return False
def run(output_dir=OUT):
 f=tuple(f'token_{i}' for i in range(41));rs=[Row(f),Row(f[:-1]+('?',))];s=[];p=predict_independent(rs,lambda st,x:s.append(st) or np.array([.5,.5]));q=predict_independent(rs[::-1],lambda st,x:np.array([.5,.5]));c={'zero_reset':s==[0.,0.],'reorder_invariant':bool(np.array_equal(p,q[::-1])),'target_rejected':rej({'fields':f,'label':'50000+'}),'file_rejected':rej({'fields':f,'file':'census-income.test'}),'split_rejected':rej({'fields':f,'split':'external'}),'row_rejected':rej({'fields':f,'row':0}),'demographic_rejected':rej({'fields':f,'demographic':'sex'}),'duplicate_group_rejected':rej({'fields':f,'duplicate_group':0}),'finite_probabilities':bool(np.all(np.isfinite(p)) and np.allclose(p.sum(1),1))};r={'phase':72,'status':'passed_adapter_mechanics_only' if all(c.values()) else 'closed_negative_adapter_mechanics_failure','uses_observed_values':False,'uses_observed_labels':False,'controls':c};output_dir.mkdir(parents=True,exist_ok=True);(output_dir/'result.json').write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':print(json.dumps(run(),indent=2))

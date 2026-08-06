import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.real_data.uci_census_income_kdd import Row,row_from_payload,predict_independent
OUT=Path('artifacts/evaluations/phase72_uci_census_income_kdd_synthetic_controls_20260805');S=2072001
def data(n,seed):
 r=np.random.default_rng(seed);x=[];y=[]
 for z in range(n):
  a,b=r.integers(0,2,2);lab=int(a and b);f=['cat0']*41;f[0]=str(b);f[1]=f'cat{a}';x.append(Row(tuple(f)));y.append(lab)
 return x,np.array(y)
def enc(rows):return np.array([[float(q.fields[0]),q.fields[1]=='cat1'] for q in rows])
def rej(p):
 try:row_from_payload(p)
 except ValueError:return True
 return False
def run():
 x,y=data(500,S);sx,sy=data(300,S+1);m=LogisticRegression().fit(enc(x),y);f=float(f1_score(sy,m.predict(enc(sx)),average='macro'));lp=f-float(f1_score(sy,LogisticRegression().fit(enc(x),np.random.default_rng(S).permutation(y)).predict(enc(sx)),average='macro'));u=np.full(2,.5);st=[];p=predict_independent(sx[:3],lambda a,b:st.append(a) or u);c={'label_pairing_drop':lp,'zero_reset':st==[0.]*3,'target_rejected':rej({'fields':sx[0].fields,'label':1}),'finite_probabilities':bool(np.allclose(p.sum(1),1))};ok=f>=.95 and lp>=.4 and all(c[k] for k in ('zero_reset','target_rejected','finite_probabilities'));d={'phase':72,'status':'passed_output_isolated_synthetic_controls' if ok else 'closed_negative_synthetic_control_failure','uses_observed_values':False,'uses_observed_labels':False,'planted':{'macro_f1':f},'controls':c};OUT.mkdir(parents=True,exist_ok=True);(OUT/'result.json').write_text(json.dumps(d,indent=2)+'\n');return d
print(json.dumps(run(),indent=2))

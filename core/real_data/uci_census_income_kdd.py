"""Artificial-only raw-token mechanics for Phase 72."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable,Mapping
import numpy as np
FIELD_COUNT=41
@dataclass(frozen=True)
class Row:
 fields:tuple[str,...]
 def __post_init__(self):
  f=tuple(map(str,self.fields))
  if len(f)!=FIELD_COUNT or any(not x for x in f):raise ValueError("Census candidate must have 41 nonempty tokens")
  object.__setattr__(self,"fields",f)
def row_from_payload(p:Mapping[str,object])->Row:
 if set(p)!={"fields"}:raise ValueError("Census payload may contain only fields")
 return Row(tuple(p["fields"]))
def predict_independent(rows:list[Row],f:Callable[[float,tuple[str,...]],np.ndarray])->np.ndarray:
 out=[]
 for r in rows:
  p=np.asarray(f(0.,r.fields),float)
  if p.shape!=(2,) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.isclose(p.sum(),1):raise ValueError("invalid probabilities")
  out.append(p)
 return np.asarray(out)

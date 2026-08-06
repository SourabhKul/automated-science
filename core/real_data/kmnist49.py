"""Artificial-only 49-class raw-grid mechanics for Phase 71."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np
SHAPE=(28,28);CLASS_COUNT=49
@dataclass(frozen=True)
class Image:
 values:np.ndarray
 def __post_init__(self):
  x=np.asarray(self.values,float)
  if x.shape!=SHAPE or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>255):raise ValueError("K49 image must be finite 28x28 raw pixels in 0..255")
  object.__setattr__(self,"values",x.copy())
def image_from_payload(p:Mapping[str,object])->Image:
 if set(p)!={"values"}:raise ValueError("K49 payload may contain only values")
 return Image(np.asarray(p["values"],float))
def predict_independent(images:list[Image],f:Callable[[float,np.ndarray],np.ndarray])->np.ndarray:
 out=[]
 for image in images:
  p=np.asarray(f(0.,image.values.copy()),float)
  if p.shape!=(CLASS_COUNT,) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.isclose(p.sum(),1):raise ValueError("invalid K49 probabilities")
  out.append(p)
 return np.asarray(out)

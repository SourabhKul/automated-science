"""Artificial-only Phase 57 Fashion-MNIST candidate mechanics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np
SHAPE=(28,28); LABELS=tuple(range(10)); FILES=("train","t10k")
@dataclass(frozen=True)
class Image:
 values:np.ndarray
 def __post_init__(self):
  x=np.asarray(self.values,dtype=float)
  if x.shape!=SHAPE or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>255):raise ValueError("image must be finite 28x28 pixels in 0..255")
  object.__setattr__(self,"values",x.copy())
def image_from_payload(p:Mapping[str,object])->Image:
 if set(p)!={"values"}:raise ValueError("Fashion-MNIST payload may contain only values")
 return Image(np.asarray(p["values"],dtype=float))
def predict_independent(images:list[Image],f:Callable[[float,np.ndarray],np.ndarray])->np.ndarray:
 out=[]
 for im in images:
  p=np.asarray(f(0.,im.values.copy()),float)
  if p.shape!=(10,) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.isclose(p.sum(),1):raise ValueError("invalid probabilities")
  out.append(p)
 return np.asarray(out)

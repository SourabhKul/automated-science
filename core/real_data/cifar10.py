"""Artificial-only candidate surface for Phase 58 CIFAR-10."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np
SHAPE=(3,32,32); LABELS=tuple(range(10))
@dataclass(frozen=True)
class CifarImage:
 values:np.ndarray
 def __post_init__(self):
  x=np.asarray(self.values,float)
  if x.shape!=SHAPE or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>255):raise ValueError("CIFAR candidate must be finite 3x32x32 in 0..255")
  object.__setattr__(self,'values',x.copy())
def image_from_payload(p:Mapping[str,object])->CifarImage:
 if set(p)!={'values'}:raise ValueError('CIFAR payload may contain only values')
 return CifarImage(np.asarray(p['values'],float))
def predict_independent(images:list[CifarImage],f:Callable[[float,np.ndarray],np.ndarray])->np.ndarray:
 out=[]
 for im in images:
  p=np.asarray(f(0.,im.values.copy()),float)
  if p.shape!=(10,) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.isclose(p.sum(),1):raise ValueError('invalid CIFAR probabilities')
  out.append(p)
 return np.asarray(out)

"""Isolated source-unit grids for Phase 44 Optical Digits."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Mapping
FEATURE_COUNT=64; CLASS_LABELS=tuple(range(10)); SOURCE_FILES=("optdigits.tra","optdigits.tes")
@dataclass(frozen=True)
class OpticalGrid:
    pixels:np.ndarray
    def __post_init__(self):
        x=np.asarray(self.pixels,dtype=float)
        if x.shape!=(64,) or not np.all(np.isfinite(x)) or np.any(x<0)|np.any(x>16): raise ValueError("Optical Digits candidate must be finite 64-grid source units")
        object.__setattr__(self,"pixels",x.copy())
def grid_from_payload(payload:Mapping[str,object])->OpticalGrid:
    if set(payload)!={"pixels"}: raise ValueError("Optical Digits candidate payload may only contain pixels")
    return OpticalGrid(np.asarray(payload["pixels"],dtype=float))

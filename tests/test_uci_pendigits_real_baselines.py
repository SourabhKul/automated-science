from __future__ import annotations
import numpy as np
from scripts.run_uci_pendigits_real_baselines import Rows,_fit,_metrics
def main()->None:
    rng=np.random.default_rng(42); x=rng.normal(50,10,(100,16)); y=np.repeat(np.arange(10),10); rows=Rows(x,y,tuple(map(str,range(100))))
    assert 0 <= _metrics(_fit("logreg_0.1",x,y),rows)["macro_f1"] <= 1; assert _metrics(_fit("majority",x,y),rows)["multiclass_log_loss"]>0
    print("SUCCESS: PenDigits real baseline helpers are train-only, finite, and fixed-grid")
if __name__=="__main__": main()

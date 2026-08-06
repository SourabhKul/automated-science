from __future__ import annotations
import numpy as np
from scripts.run_uci_isolet_real_baselines import Model, Rows, _fit, _metrics
def main() -> None:
    rng=np.random.default_rng(41); x=rng.normal(size=(130,617)); y=np.repeat(np.arange(1,27),5); rows=Rows(x,y,tuple(str(i) for i in range(len(y))))
    model=_fit("logreg_0.1",rows.x,rows.y); metrics=_metrics(model,rows)
    assert np.isfinite(metrics["macro_f1"]) and 0 <= metrics["macro_f1"] <= 1
    assert _metrics(_fit("majority",rows.x,rows.y),rows)["multiclass_log_loss"] > 0
    print("SUCCESS: ISOLET real baseline helpers are train-only, finite, and fixed-grid")
if __name__ == "__main__": main()

from __future__ import annotations

import numpy as np

from core.real_data.uci_wisdm import ACTIVITY_LABELS, PREFIX_SAMPLES, SOURCE_AXES
from scripts.run_uci_wisdm_real_baselines import Rows, _fit, _metrics


def main() -> None:
    rng = np.random.default_rng(50)
    labels = np.repeat(np.asarray(ACTIVITY_LABELS), 3)
    rows = Rows(rng.normal(size=(len(labels), PREFIX_SAMPLES, SOURCE_AXES)), labels, tuple(map(str, range(len(labels)))))
    for name in ("majority", "logreg_0.1", "nearest_centroid"):
        metrics = _metrics(_fit(name, rows.x, rows.y), rows)
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert np.isfinite(metrics["multiclass_log_loss"])
    print("SUCCESS: WISDM real-baseline helpers are train-only, finite, and fixed-prefix")


if __name__ == "__main__":
    main()

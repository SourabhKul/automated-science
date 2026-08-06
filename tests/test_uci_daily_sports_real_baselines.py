from __future__ import annotations

import numpy as np

from scripts.run_uci_daily_sports_real_baselines import Rows, _fit, _metrics


def main() -> None:
    rng = np.random.default_rng(45)
    labels = np.repeat(np.arange(1, 20), 3)
    rows = Rows(rng.normal(size=(len(labels), 64, 45)), labels, tuple(map(str, range(len(labels)))) )
    for name in ("majority", "logreg_0.1", "nearest_centroid"):
        metrics = _metrics(_fit(name, rows.x, rows.y), rows)
        assert 0 <= metrics["macro_f1"] <= 1
        assert np.isfinite(metrics["multiclass_log_loss"])
    print("SUCCESS: Daily Sports real-baseline helpers are train-only, finite, and fixed-prefix")


if __name__ == "__main__":
    main()

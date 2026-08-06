from __future__ import annotations

import numpy as np

from scripts.run_uci_statlog_landsat_real_baselines import Rows, _fit, _metrics


def main() -> None:
    rng = np.random.default_rng(47)
    labels = np.repeat(np.asarray((1, 2, 3, 4, 5, 7)), 4)
    rows = Rows(rng.normal(size=(len(labels), 9, 4)), labels, tuple(map(str, range(len(labels)))))
    for name in ("majority", "logreg_0.1", "nearest_centroid"):
        metrics = _metrics(_fit(name, rows.x, rows.y), rows)
        assert 0 <= metrics["macro_f1"] <= 1
        assert np.isfinite(metrics["multiclass_log_loss"])
    print("SUCCESS: Statlog Landsat real-baseline helpers enforce finite class probabilities")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from scripts.run_uci_optdigits_real_baseline import Rows, _fit, _metrics


def main() -> None:
    rng = np.random.default_rng(44)
    rows = Rows(rng.integers(0, 17, size=(100, 64)).astype(float), np.repeat(np.arange(10), 10), tuple(map(str, range(100))))
    for model in ("majority", "logreg_0.1", "nearest_centroid"):
        metrics = _metrics(_fit(model, rows.x, rows.y), rows)
        assert 0 <= metrics["macro_f1"] <= 1
        assert np.isfinite(metrics["multiclass_log_loss"])
    print("SUCCESS: Optical Digits real-baseline helpers enforce finite class probabilities")


if __name__ == "__main__":
    main()

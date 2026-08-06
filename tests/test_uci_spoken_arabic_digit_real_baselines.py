from __future__ import annotations

import numpy as np

from scripts.run_uci_spoken_arabic_digit_real_baselines import Rows, _fit, _metrics


def main() -> None:
    rng = np.random.default_rng(46)
    labels = np.repeat(np.arange(10), 4)
    rows = Rows(rng.normal(size=(len(labels), 4, 13)), labels, tuple(map(str, range(len(labels)))))
    for name in ("majority", "logreg_0.1", "nearest_centroid"):
        metrics = _metrics(_fit(name, rows.x, rows.y), rows)
        assert 0 <= metrics["macro_f1"] <= 1
        assert np.isfinite(metrics["multiclass_log_loss"])
    print("SUCCESS: Spoken Arabic Digit real-baseline helpers enforce finite class probabilities")


if __name__ == "__main__":
    main()

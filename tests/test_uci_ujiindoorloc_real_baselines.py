from __future__ import annotations

import numpy as np

from scripts.run_uci_ujiindoorloc_real_baselines import Rows, _fit, _group_split, _metrics


def main() -> None:
    rng = np.random.default_rng(49)
    labels = np.repeat(np.asarray((0, 1, 2)), 20)
    rows = Rows(
        rng.integers(-104, 101, size=(len(labels), 520)).astype(float),
        labels,
        tuple(f"complete-{index}" for index in range(len(labels))),
        tuple(f"candidate-{index}" for index in range(len(labels))),
        tuple(f"candidate-label-{index}" for index in range(len(labels))),
    )
    train, selection, ledger = _group_split(rows)
    assert ledger["cross_partition_candidate_label_groups"] == 0
    for name in ("majority", "logreg_0.1", "nearest_centroid"):
        metrics = _metrics(_fit(name, train.x, train.y), selection)
        assert 0 <= metrics["macro_f1"] <= 1
        assert np.isfinite(metrics["multiclass_log_loss"])
    print("SUCCESS: UJIIndoorLoc real-baseline helpers are duplicate-group-safe and finite")


if __name__ == "__main__":
    main()

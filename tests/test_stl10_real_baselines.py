import numpy as np

from scripts.run_stl10_real_baselines import Records, fit, metric


def main() -> None:
    labels = np.repeat(np.arange(10), 3)
    values = np.random.default_rng(67).integers(0, 256, size=(len(labels), 3 * 96 * 96), dtype=np.uint8)
    records = Records(values, labels, tuple(map(str, range(len(labels)))))
    assert 0.0 <= metric(fit("logreg_0.1", records.values, records.labels), records)["macro_f1"] <= 1.0
    print("SUCCESS: STL-10 observed baseline helper emits finite probabilities")


if __name__ == "__main__":
    main()

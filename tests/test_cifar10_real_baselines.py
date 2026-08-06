import numpy as np

from scripts.run_cifar10_real_baselines import Records, fit, metric


def main() -> None:
    labels = np.repeat(np.arange(10), 3)
    values = np.random.default_rng(58).integers(0, 256, size=(len(labels), 3072), dtype=np.uint8)
    records = Records(values, labels, tuple(map(str, range(len(labels)))))
    score = metric(fit("logreg_0.1", records.values, records.labels), records)["macro_f1"]
    assert 0.0 <= score <= 1.0
    print("SUCCESS: CIFAR-10 observed baseline helper emits finite probabilities")


if __name__ == "__main__":
    main()

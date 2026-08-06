import numpy as np

from scripts.run_uci_higgs_real_baselines import Records, fit, metric


def main() -> None:
    values = np.random.default_rng(62).normal(size=(240, 28)).astype(np.float32)
    targets = (values[:, 0] + values[:, 21] > 0.0).astype(np.uint8)
    records = Records(values, targets)
    outcome = metric(fit("logistic_0.1", records.values, records.targets), records)
    assert np.isfinite(outcome["macro_f1"])
    print("SUCCESS: HIGGS observed baseline helper emits finite normalized probabilities")


if __name__ == "__main__":
    main()

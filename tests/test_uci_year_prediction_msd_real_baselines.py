import numpy as np

from scripts.run_uci_year_prediction_msd_real_baselines import Records, fit, metric


def main() -> None:
    values = np.random.default_rng(61).normal(size=(240, 90)).astype(np.float32)
    targets = 1965.0 + 2.0 * values[:, 0]
    records = Records(values, targets, tuple(map(str, range(len(targets)))))
    assert np.isfinite(metric(fit("ridge_1", records.values, records.targets), records)["rmse"])
    print("SUCCESS: YearPredictionMSD observed baseline helper emits finite bounded predictions")


if __name__ == "__main__":
    main()

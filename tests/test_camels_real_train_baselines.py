from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from core.real_data.camels_us import CAMELSInputEpisode, CAMELSOutcomeEpisode
from scripts.run_camels_us_real_train_baselines import _bucket_predict, _fit_ridge, _metrics, _ridge_predict, _seasonal, _seasonal_predict


def _record(index: int) -> CAMELSOutcomeEpisode:
    start = date(2001, 10, 1)
    dates = tuple(start + timedelta(days=offset) for offset in range(365))
    inputs = np.column_stack([np.linspace(0.1, 2.0, len(dates)) + index, np.linspace(-2.0, 10.0, len(dates))])
    flow = 1.0 + 0.2 * inputs[:, 0]
    return CAMELSOutcomeEpisode(f"{index:08d}", "train", CAMELSInputEpisode(dates, inputs), flow, 2.0)


def main() -> None:
    train = [_record(index) for index in range(1, 4)]
    selection = [_record(index) for index in range(4, 6)]
    seasonal = _seasonal(train)
    seasonal_prediction = _seasonal_predict(selection, seasonal)
    model = _fit_ridge(train, 0.01)
    ridge_prediction = _ridge_predict(selection, model)
    bucket_prediction = _bucket_predict(selection, 0.5, 0.05)
    assert all(np.all(np.isfinite(item)) for item in ridge_prediction + bucket_prediction)
    assert all(np.all(item >= 0) for item in bucket_prediction)
    assert np.isfinite(_metrics(selection, seasonal_prediction, 0.1)["rmse"])
    print("SUCCESS: CAMELS train-only seasonal, ridge, and zero-reset bucket helpers are stable")


if __name__ == "__main__":
    main()

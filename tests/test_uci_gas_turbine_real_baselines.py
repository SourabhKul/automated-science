from __future__ import annotations

import numpy as np

from scripts.run_uci_gas_turbine_real_baselines import (
    AMBIENT_INDICES,
    PROCESS_INDICES,
    LockedRows,
    _fit_ridge,
    _metrics,
    _select,
)


def main() -> None:
    rng = np.random.default_rng(38)
    train_inputs = rng.normal(size=(48, 9))
    selection_inputs = rng.normal(size=(20, 9))
    coefficients = np.asarray([2.0, -1.5, 1.2, 1.0, -0.8, 0.7, 1.4, -1.1, 0.9])
    train_target = 60.0 + train_inputs @ coefficients
    selection_target = 60.0 + selection_inputs @ coefficients
    train = LockedRows(train_inputs, train_target)
    selection = LockedRows(selection_inputs, selection_target)
    model = _fit_ridge(train_inputs, train_target, np.arange(9), False, 0.1)
    prediction = model.predict(selection_inputs)
    assert np.all(np.isfinite(prediction)) and np.all((prediction >= 0.0) & (prediction <= 150.0))
    assert np.isfinite(_metrics(selection_target, prediction)["rmse"])
    selected, name, metrics, candidates = _select(train, selection)
    assert name.startswith(("linear_ridge_", "quadratic_ridge_"))
    assert selected.field_indices.shape == (9,)
    assert set(candidates) == {f"{family}_ridge_{penalty:g}" for family in ("linear", "quadratic") for penalty in (0.01, 0.1, 1.0, 10.0)}
    assert np.array_equal(AMBIENT_INDICES, np.asarray([0, 1, 2]))
    assert np.array_equal(PROCESS_INDICES, np.asarray([3, 4, 5, 6, 7, 8]))
    assert all(np.isfinite(value) and value >= 0.0 for value in metrics.values())
    print("SUCCESS: UCI gas-turbine real baseline helpers are train-only, finite, and fixed-grid")


if __name__ == "__main__":
    main()

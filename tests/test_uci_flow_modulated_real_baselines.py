from __future__ import annotations

import numpy as np

from core.real_data.uci_flow_modulated import make_uci_flow_input_trial
from scripts.run_uci_flow_modulated_real_baselines import (
    Trial,
    _metrics,
    _recurrence_predict,
    _ridge_predict,
    load_locked_trials,
)


def main() -> None:
    input_trial = make_uci_flow_input_trial(0.3, 0.1)
    target = np.zeros(7500)
    trial = Trial(1, "synthetic", "train", input_trial, target)
    assert not hasattr(trial.causal_input(), "target")
    ridge = _ridge_predict([trial.causal_input()], np.zeros(8))
    recurrence = _recurrence_predict([trial.causal_input()], 0.95, np.zeros(2))
    assert ridge.shape == recurrence.shape == (1, 7500)
    metrics = _metrics(ridge, target[None, :])
    assert metrics["rmse"] == 0.0 and metrics["mae"] == 0.0
    try:
        _metrics(np.full((1, 7500), 21.0), target[None, :])
    except ValueError as exc:
        assert "envelope" in str(exc)
    else:
        raise AssertionError("out-of-bound prediction must fail")
    trials, ledger = load_locked_trials()
    assert {role: len(values) for role, values in trials.items()} == {"train": 39, "selection": 11, "external": 8}
    assert len(ledger) == 58
    assert all(item.target is None for item in trials["external"])
    assert sum(item.target is not None for values in trials.values() for item in values) == 50
    print("SUCCESS: Phase 30 fixed baseline helpers preserve causal inputs and hard bounds")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_flow_modulated import (
    EXPOSURE_SAMPLES,
    TRACE_SAMPLES,
    UCIFlowOutcomeTrial,
    make_uci_flow_input_trial,
    partition_uci_flow_outcome_trials,
    rollout_uci_flow_causal_input,
)


SPLIT_PATH = Path("data/real/uci_flow_modulated_gas_sensor/source_batch_split.json")


def _synthetic_trials() -> list[UCIFlowOutcomeTrial]:
    payload = json.loads(SPLIT_PATH.read_text())
    metadata = payload["sample_metadata_ledger"]
    result = []
    for split, entries in payload["roles"].items():
        for entry in entries:
            for sample_id in entry["samples"]:
                source = metadata[str(sample_id)]
                trial = make_uci_flow_input_trial(source["ace_conc_vol_percent"], source["eth_conc_vol_percent"])
                synthetic_target = np.linspace(-0.1, 0.2, TRACE_SAMPLES) + float(sample_id) * 1e-5
                result.append(UCIFlowOutcomeTrial(int(sample_id), entry["batch"], split, trial, synthetic_target))
    return result


def main() -> None:
    trials = _synthetic_trials()
    partitioned = partition_uci_flow_outcome_trials(trials, SPLIT_PATH)
    assert {name: len(items) for name, items in partitioned.items()} == {"train": 39, "selection": 11, "external": 8}

    input_trial = trials[0].causal_input()
    assert not hasattr(input_trial, "sensor_one_dr")
    assert input_trial.input_u.shape == (TRACE_SAMPLES, 4)
    assert np.array_equal(input_trial.input_u[:EXPOSURE_SAMPLES, 1], np.ones(EXPOSURE_SAMPLES))
    assert np.array_equal(input_trial.input_u[EXPOSURE_SAMPLES:, 1], np.zeros(TRACE_SAMPLES - EXPOSURE_SAMPLES))

    def step(state: float, u: np.ndarray) -> float:
        return 0.995 * state + 0.01 * float(u[1]) * (float(u[2]) + 0.5 * float(u[3]))

    first = rollout_uci_flow_causal_input(input_trial, step)
    second = rollout_uci_flow_causal_input(input_trial, step)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))

    wrong = list(trials)
    wrong[0] = UCIFlowOutcomeTrial(wrong[0].sample_id, wrong[0].batch, "external", wrong[0].input_trial, wrong[0].sensor_one_dr)
    try:
        partition_uci_flow_outcome_trials(wrong, SPLIT_PATH)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("cross-split trial must fail")

    try:
        rollout_uci_flow_causal_input(input_trial, lambda state, u: float("nan"))
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite rollout must fail")
    print("SUCCESS: UCI flow injected adapter is causal-input-only, reset-safe, and split-locked")


if __name__ == "__main__":
    main()

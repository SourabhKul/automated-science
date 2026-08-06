import json
import tempfile
from pathlib import Path

import numpy as np

from core.real_data.ph_reactor import CausalInputSignal, EXPECTED_EXPERIMENTS, EXPECTED_SAMPLES, evaluate_ph_reactor_baselines, load_ph_reactor, prescreen_ph_reactor_rollouts
from scripts.legacy.real_ph_reactor_data_loader import DEFAULT_RAW_ROOT, generate_data


def main():
    signal = CausalInputSignal(np.array([1.0, 2.0, 3.0]))
    assert signal.at(0.0) == 1.0
    assert signal.at(0.99) == 1.0
    assert signal.at(1.0) == 2.0
    assert signal.at(1.99) == 2.0
    assert signal.at(2.5) == 3.0

    data = load_ph_reactor(DEFAULT_RAW_ROOT)
    for split, count in EXPECTED_EXPERIMENTS.items():
        assert len(data[split]) == count
        for experiment in data[split]:
            assert experiment.input_u.shape == (EXPECTED_SAMPLES,)
            assert experiment.output_y.shape == (EXPECTED_SAMPLES,)
            assert np.all(np.isfinite(experiment.input_u))
            assert np.all(np.isfinite(experiment.output_y))

    baselines = evaluate_ph_reactor_baselines(data)
    assert set(baselines) == {"persistence", "regularized_arx", "stable_first_order", "hammerstein_first_order"}
    for baseline in baselines.values():
        assert np.isfinite(baseline["validation"]["aggregate"]["nrmse"])
        assert np.isfinite(baseline["test"]["aggregate"]["nrmse"])

    stable = prescreen_ph_reactor_rollouts(data, lambda u: 0.1 * np.tanh(u - np.mean(u)))
    assert stable["passed"] and stable["native_input_grid_count"] == 20
    assert not prescreen_ph_reactor_rollouts(data, lambda u: np.full_like(u, np.nan))["passed"]
    assert not prescreen_ph_reactor_rollouts(data, lambda u: np.full_like(u, 1e9))["passed"]
    assert not prescreen_ph_reactor_rollouts(data, lambda u: (np.zeros_like(u), {"state_clipped": True}))["passed"]

    with tempfile.TemporaryDirectory() as tempdir:
        manifest = generate_data(raw_root=DEFAULT_RAW_ROOT, output_root=Path(tempdir) / "ph")
        assert manifest["status"] == "ok"
        assert manifest["experiment_split"]["train"] == 15
        assert Path(manifest["baseline_results_path"]).exists()
        payload = json.loads(Path(manifest["baseline_results_path"]).read_text())
        assert np.isfinite(payload["regularized_arx"]["test"]["aggregate"]["nrmse"])

    print("SUCCESS: pH-reactor source loader and train-only baseline adapter are stable")


if __name__ == "__main__":
    main()

import json
import tempfile
from pathlib import Path

import numpy as np

from core.real_data.dream4 import (
    build_recovery_split,
    evaluate_recovery_baselines,
    fit_sparse_linear_recovery,
    load_dream4_size10,
    predict_sparse_linear_recovery,
)
from scripts.legacy.real_dream4_data_loader import DEFAULT_SOURCE_ROOT, generate_data


def main():
    without_gold = load_dream4_size10(DEFAULT_SOURCE_ROOT)
    with_gold = load_dream4_size10(DEFAULT_SOURCE_ROOT, include_gold_edges=True)
    assert sorted(without_gold) == [f"insilico_size10_{index}" for index in range(1, 6)]
    for network_id, network in without_gold.items():
        assert network.gold_edges is None
        assert len(network.trajectories) == 5
        assert network.gene_names == tuple(f"G{index}" for index in range(1, 11))
        for trajectory in network.trajectories:
            recovery_time, recovery_observed = trajectory.recovery_window()
            assert recovery_time.tolist() == [float(value) for value in range(0, 501, 50)]
            assert recovery_observed.shape == (11, 10)
            assert np.all(np.isfinite(recovery_observed))

        split = build_recovery_split(network)
        assert split["train_trajectory_ids"] == [1, 2, 3]
        assert split["validation_trajectory_id"] == 4
        assert split["external_trajectory_id"] == 5
        trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in network.trajectories}
        smoke_model = fit_sparse_linear_recovery(
            [trajectory_by_id[index] for index in split["train_trajectory_ids"]],
            trajectory_by_id[split["validation_trajectory_id"]],
        )
        smoke_prediction = predict_sparse_linear_recovery(trajectory_by_id[split["external_trajectory_id"]], smoke_model)
        assert smoke_prediction.shape == (11, 10)
        assert np.all(np.isfinite(smoke_prediction))
        no_gold_results = evaluate_recovery_baselines(network, split)
        assert "topology" not in no_gold_results["sparse_linear_state_space"]
        results = evaluate_recovery_baselines(with_gold[network_id], split, gold_edges=with_gold[network_id].gold_edges)
        for name in ("persistence", "exponential_relaxation", "sparse_linear_state_space"):
            assert np.isfinite(results[name]["mse"])
            assert np.isfinite(results[name]["nrmse"])
            assert np.isfinite(results[name]["late_nrmse"])
        topology = results["sparse_linear_state_space"]["topology"]
        assert 0.0 <= topology["aupr"] <= 1.0
        assert 0.0 <= topology["auroc"] <= 1.0
        assert 0.0 < topology["edge_prevalence"] < 1.0

    with tempfile.TemporaryDirectory() as tempdir:
        manifest = generate_data(source_root=DEFAULT_SOURCE_ROOT, output_root=Path(tempdir) / "dream4")
        assert manifest["status"] == "ok"
        assert manifest["network_count"] == 5
        assert manifest["time_points_per_trajectory"] == 21
        assert manifest["archive_sha256"]
        baseline_path = Path(manifest["baselines_path"])
        split_path = Path(manifest["splits_path"])
        assert baseline_path.exists() and split_path.exists()
        baselines = json.loads(baseline_path.read_text())
        assert sorted(baselines) == sorted(with_gold)

    print("SUCCESS: DREAM4 source loader, recovery split, baseline, and gold-edge isolation checks passed")


if __name__ == "__main__":
    main()

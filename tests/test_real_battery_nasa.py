import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.domain_configs import DOMAIN_CONFIGS
from core.generated_code import validate_generated_math_code
from core.real_data.battery_nasa import (
    build_cell_holdout_bundles,
    build_cell_holdout_split,
    build_leave_one_cell_out_splits,
    build_cell_target_bundles,
    build_capacity_target,
    build_within_cell_time_splits,
    cell_summary,
    evaluate_cell_holdout_baselines,
    evaluate_cell_predictions,
    grouped_cell_distances,
    normalize_battery_table,
    predict_cell_bundles,
    run_grouped_cell_abc_smc,
)
from scripts.legacy.real_battery_nasa_data_loader import DEFAULT_FIXTURE, generate_data, resolve_raw_path


class LinearFadeModel:
    def simulate(self, params, time_points, y0):
        if float(params[0]) < 0.0:
            return np.full((len(time_points), 1), np.nan)
        return np.asarray(y0, dtype=float).reshape(1, -1) - float(params[0]) * np.asarray(time_points, dtype=float).reshape(-1, 1)

    def get_parameter_metadata(self):
        return [{"name": "fade_rate", "range": (0.0, 0.02)}]


def main():
    normalized = normalize_battery_table(DEFAULT_FIXTURE)
    assert normalized["cell_id"].nunique() == 1
    assert normalized["soh"].iloc[0] == 1.0
    assert normalized["soh"].iloc[-1] < 0.75
    assert normalized["rul_cycles"].iloc[-1] == 0
    fixture_summary = cell_summary(normalized)
    assert fixture_summary[0]["cell_id"] == "BTEST"
    assert fixture_summary[0]["eol_reached"] is False

    time_points, target = build_capacity_target(normalized, cell_id="BTEST")
    assert target.shape == (6, 1)
    assert time_points.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
    assert np.all(np.diff(target[:, 0]) <= 0.0)

    assert "real_battery_nasa_capacity" in DOMAIN_CONFIGS
    config = DOMAIN_CONFIGS["real_battery_nasa_capacity"]
    assert len(config["y0"]) == 1
    validate_generated_math_code(config["seed_logic"])

    with tempfile.TemporaryDirectory() as fixture_tmpdir:
        fixture = Path(fixture_tmpdir)
        manifest = generate_data(
            raw_path=DEFAULT_FIXTURE,
            output_root=fixture / "real",
            data_dir=fixture / "dense",
        )
        assert manifest["fixture_mode"] is True
        assert manifest["raw_path_source"] == "explicit"
        assert Path(manifest["target"]["ground_truth_path"]).exists()
        assert Path(manifest["target"]["time_points_path"]).exists()
        assert manifest["cell_holdout"]["status"] == "skipped"
        assert "cell holdout requires at least two cells" in manifest["cell_holdout"]["reason"]
        assert manifest["within_cell_split"]["status"] == "ok"
        fixture_cell_split = manifest["within_cell_split"]["cells"][0]
        assert fixture_cell_split["cell_id"] == "BTEST"
        assert fixture_cell_split["train_cycles"] == [0, 25, 50, 75]
        assert fixture_cell_split["test_cycles"] == [100, 125]

    with tempfile.TemporaryDirectory() as multi_tmpdir:
        multi = Path(multi_tmpdir)
        raw_dir = multi / "data/real/battery_nasa/raw"
        raw_dir.mkdir(parents=True)
        auto_raw = raw_dir / "cycle_level.csv"
        auto_raw.write_text(
            "\n".join(
                [
                    "cell_id,cycle_index,elapsed_time_h,ambient_temperature_c,capacity_ah",
                    "B1,0,0,24,2.00",
                    "B1,10,1,24,1.95",
                    "B1,20,2,24,1.88",
                    "B2,0,0,30,2.10",
                    "B2,10,1,30,2.00",
                    "B2,20,2,30,1.86",
                ]
            )
            + "\n"
        )
        resolved, fixture_mode = resolve_raw_path(auto_raw)
        assert resolved == auto_raw
        assert fixture_mode is False
        multi_normalized = normalize_battery_table(auto_raw)
        assert sorted(multi_normalized["cell_id"].unique().tolist()) == ["B1", "B2"]
        split = build_cell_holdout_split(multi_normalized, train_fraction=0.5, seed=3)
        assert split["status"] == "ok"
        assert set(split["train_cells"]).isdisjoint(set(split["test_cells"]))
        assert len(split["train_cells"]) == 1
        assert len(split["test_cells"]) == 1
        assert split["train_cycle_count"] == 3
        assert split["test_cycle_count"] == 3
        within_cell = build_within_cell_time_splits(multi_normalized, train_fraction=2 / 3)
        assert within_cell["status"] == "ok"
        assert len(within_cell["cells"]) == 2
        assert within_cell["cells"][0]["train_cycles"] == [0, 10]
        assert within_cell["cells"][0]["test_cycles"] == [20]
        _, b2_target = build_capacity_target(multi_normalized, cell_id="B2")
        assert b2_target.shape == (3, 1)
        bundles = build_cell_target_bundles(multi_normalized, ["B1", "B2"], late_cycle_train_fraction=2 / 3)
        assert [bundle.cell_id for bundle in bundles] == ["B1", "B2"]
        assert bundles[0].y0.tolist() == [1.0]
        assert bundles[0].late_cycle_mask[:, 0].tolist() == [False, False, True]
        model = LinearFadeModel()
        predictions = predict_cell_bundles(model, np.array([0.005]), bundles)
        grouped_metrics = evaluate_cell_predictions(bundles, predictions)
        assert grouped_metrics["cell_count"] == 2
        assert grouped_metrics["aggregate"]["observed_count"] == 6
        assert grouped_metrics["late_cycle"]["observed_count"] == 2
        distances = grouped_cell_distances(model, np.array([[0.005], [-1.0]]), bundles)
        assert np.isfinite(distances[0])
        assert np.isinf(distances[1])
        grouped_smc_a = run_grouped_cell_abc_smc(
            model,
            bundles,
            target_samples=2,
            generations=3,
            initial_particles=8,
            strategy="bdss",
            seed=11,
        )
        grouped_smc_b = run_grouped_cell_abc_smc(
            model,
            bundles,
            target_samples=2,
            generations=3,
            initial_particles=8,
            strategy="bdss",
            seed=11,
        )
        assert grouped_smc_a["effective_strategy"] == "bdss"
        assert len(grouped_smc_a["generation_history"]) == 3
        assert np.isfinite(grouped_smc_a["median_distance"])
        assert np.allclose(grouped_smc_a["accepted_params"], grouped_smc_b["accepted_params"])
        holdout_bundles = build_cell_holdout_bundles(
            multi_normalized,
            split,
            late_cycle_train_fraction=2 / 3,
        )
        assert len(holdout_bundles["train"]) == 1
        assert len(holdout_bundles["test"]) == 1
        assert {bundle.cell_id for bundle in holdout_bundles["train"]}.isdisjoint(
            {bundle.cell_id for bundle in holdout_bundles["test"]}
        )
        multi_manifest = generate_data(
            raw_path=auto_raw,
            cell_id="B2",
            output_root=multi / "out",
            data_dir=multi / "dense",
            holdout_seed=3,
            holdout_train_fraction=0.5,
            within_cell_train_fraction=2 / 3,
        )
        assert multi_manifest["fixture_mode"] is False
        assert multi_manifest["target"]["cell_id"] == "B2"
        assert multi_manifest["cell_holdout"]["status"] == "ok"
        assert multi_manifest["cell_holdout"]["train_cells"] == split["train_cells"]
        assert multi_manifest["cell_holdout"]["test_cells"] == split["test_cells"]
        assert Path(multi_manifest["cell_holdout_json"]).exists()
        assert multi_manifest["within_cell_split"]["cells"][0]["test_cycles"] == [20]
        assert Path(multi_manifest["within_cell_split_json"]).exists()
        assert Path(multi_manifest["normalized_csv"]).exists()

    real_normalized = normalize_battery_table("data/real/battery_nasa/cycle_level.csv")
    real_manifest = json.loads(Path("data/real/battery_nasa/manifest.json").read_text())
    assert real_manifest["fixture_mode"] is False
    real_holdout = build_cell_holdout_bundles(real_normalized, real_manifest["cell_holdout"])
    assert [bundle.cell_id for bundle in real_holdout["train"]] == ["B0005", "B0006", "B0007"]
    assert [bundle.cell_id for bundle in real_holdout["test"]] == ["B0018"]
    external_predictions = predict_cell_bundles(LinearFadeModel(), np.array([0.001]), real_holdout["test"])
    external_metrics = evaluate_cell_predictions(real_holdout["test"], external_predictions)
    assert external_metrics["cell_count"] == 1
    assert external_metrics["aggregate"]["observed_count"] == 132
    assert np.isfinite(external_metrics["aggregate"]["mse"])
    baselines = evaluate_cell_holdout_baselines(real_holdout["train"], real_holdout["test"])
    assert baselines["persistence"]["test"]["aggregate"]["observed_count"] == 132
    assert baselines["linear_decay"]["test"]["late_cycle"]["observed_count"] == 27
    assert baselines["linear_decay"]["slope_per_cycle"] < 0.0
    assert np.isfinite(baselines["persistence"]["test"]["aggregate"]["mse"])
    assert np.isfinite(baselines["linear_decay"]["test"]["aggregate"]["mse"])
    loo_folds = build_leave_one_cell_out_splits(real_normalized)
    assert [fold["test_cells"] for fold in loo_folds] == [["B0005"], ["B0006"], ["B0007"], ["B0018"]]
    assert all(fold["prediction_contract"] == "condition_on_observed_test_cell_initial_soh" for fold in loo_folds)
    for fold in loo_folds:
        fold_bundles = build_cell_holdout_bundles(real_normalized, fold)
        assert len(fold_bundles["train"]) == 3
        assert len(fold_bundles["test"]) == 1
        assert fold_bundles["test"][0].cell_id == fold["test_cells"][0]
        assert fold_bundles["test"][0].y0.tolist() == [1.0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "real_battery_seed.py"
        out_path = tmp / "result.json"
        code_path.write_text(config["seed_logic"])
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(code_path),
                "--domain",
                "real_battery_nasa_capacity",
                "--output",
                str(out_path),
                "--target-samples",
                "2",
                "--generations",
                "1",
                "--initial-particles",
                "8",
                "--held-out",
                "--seed",
                "23",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        payload = json.loads(out_path.read_text())
        assert payload["status"] == "success"
        assert payload["train_metrics"] is not None
        assert payload["test_metrics"] is not None
        assert payload["effective_strategy"] == "cell_holdout_prior_rejection"
        assert payload["diagnostics"]["cell_holdout"]["train_cells"] == ["B0005", "B0006", "B0007"]
        assert payload["diagnostics"]["cell_holdout"]["test_cells"] == ["B0018"]
        assert payload["diagnostics"]["cell_holdout"]["test"]["late_cycle"]["observed_count"] > 0

        smc_out_path = tmp / "smc_result.json"
        smc_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(code_path),
                "--domain",
                "real_battery_nasa_capacity",
                "--output",
                str(smc_out_path),
                "--target-samples",
                "2",
                "--generations",
                "2",
                "--initial-particles",
                "8",
                "--held-out",
                "--inference-strategy",
                "bdss",
                "--seed",
                "24",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert smc_result.returncode == 0, smc_result.stderr or smc_result.stdout
        smc_payload = json.loads(smc_out_path.read_text())
        assert smc_payload["status"] == "success"
        assert smc_payload["effective_strategy"] == "bdss"
        assert len(smc_payload["diagnostics"]["cell_holdout"]["generation_history"]) == 2

        loo_fold = build_leave_one_cell_out_splits(real_normalized, ["B0005", "B0018"])[0]
        loo_fold_path = tmp / "loo_fold.json"
        loo_fold_path.write_text(json.dumps(loo_fold))
        loo_out_path = tmp / "loo_result.json"
        loo_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(code_path),
                "--domain",
                "real_battery_nasa_capacity",
                "--output",
                str(loo_out_path),
                "--target-samples",
                "2",
                "--generations",
                "2",
                "--initial-particles",
                "8",
                "--held-out",
                "--cell-holdout-path",
                str(loo_fold_path),
                "--seed",
                "25",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert loo_result.returncode == 0, loo_result.stderr or loo_result.stdout
        loo_payload = json.loads(loo_out_path.read_text())
        assert loo_payload["status"] == "success"
        assert loo_payload["diagnostics"]["cell_holdout"]["train_cells"] == ["B0018"]
        assert loo_payload["diagnostics"]["cell_holdout"]["test_cells"] == ["B0005"]

    print("SUCCESS: real_battery_nasa_capacity adapter smoke is stable")


if __name__ == "__main__":
    main()

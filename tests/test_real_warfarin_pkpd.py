import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from core.domain_configs import DOMAIN_CONFIGS
from core.generated_code import validate_generated_math_code
from core.sandbox_eval import evaluate_subject_holdout_candidate
import numpy as np

from core.evaluation import masked_metrics, mse, train_test_metrics
from core.real_data.warfarin_pkpd import (
    build_subject_target_bundles,
    build_subject_holdout_split,
    build_dense_endpoint_target,
    build_endpoint_target_with_mask,
    evaluate_subject_predictions,
    grouped_subject_distance,
    grouped_subject_distances,
    normalize_warfarin_table,
    predict_subject_bundles,
    subject_prediction_metrics,
)
from scripts.legacy.real_warfarin_pkpd_data_loader import DEFAULT_FIXTURE, generate_data, resolve_raw_path


def main():
    normalized = normalize_warfarin_table(DEFAULT_FIXTURE)
    assert set(["concentration", "pca_response"]).issubset(set(normalized["endpoint"]))
    assert normalized["is_observation"].sum() == 12

    time_points, target = build_dense_endpoint_target(normalized, subject_id="1")
    assert target.shape == (6, 2)
    assert list(time_points) == sorted(time_points)
    _, masked_target, observation_mask = build_endpoint_target_with_mask(normalized, subject_id="1")
    assert masked_target.shape == observation_mask.shape == (6, 2)
    assert observation_mask.all()
    with tempfile.TemporaryDirectory() as sparse_tmpdir:
        sparse_path = Path(sparse_tmpdir) / "sparse_warfarin.csv"
        sparse_path.write_text(
            "\n".join(
                [
                    "id,time,amt,dv,dvid,wt,age,sex",
                    "1,0,100,,1,66.7,50,1",
                    "1,0,0,0.00,1,66.7,50,1",
                    "1,0,0,100.00,2,66.7,50,1",
                    "1,12,0,1.52,1,66.7,50,1",
                    "1,24,0,73.10,2,66.7,50,1",
                ]
            )
            + "\n"
        )
        sparse_normalized = normalize_warfarin_table(sparse_path)
        _, sparse_target, sparse_endpoint_mask = build_endpoint_target_with_mask(sparse_normalized, subject_id="1")
        assert sparse_target.shape == sparse_endpoint_mask.shape == (3, 2)
        assert sparse_endpoint_mask.tolist() == [[True, True], [True, False], [False, True]]
    sparse_observed = np.array([[1.0, 100.0], [2.0, 90.0], [3.0, 80.0]])
    sparse_predicted = np.array([[2.0, 100.0], [2.0, 100.0], [4.0, 70.0]])
    sparse_mask = np.array([[True, False], [False, True], [True, False]])
    assert np.isclose(mse(sparse_observed, sparse_predicted, sparse_mask), 34.0)
    sparse_masked_metrics = masked_metrics(sparse_observed, sparse_predicted, sparse_mask)
    assert sparse_masked_metrics.observed_count == 3
    assert np.isclose(sparse_masked_metrics.rmse, np.sqrt(34.0))
    sparse_metrics = train_test_metrics(sparse_observed, sparse_predicted, train_fraction=2 / 3, mask=sparse_mask)
    assert sparse_metrics.train_mse == 50.5
    assert sparse_metrics.test_mse == 1.0

    assert "real_warfarin_pkpd" in DOMAIN_CONFIGS
    config = DOMAIN_CONFIGS["real_warfarin_pkpd"]
    assert len(config["y0"]) == 2
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
        assert Path(manifest["target"]["observation_mask_path"]).exists()
        assert manifest["subject_holdout"]["status"] == "skipped"

    with tempfile.TemporaryDirectory() as multi_tmpdir:
        multi = Path(multi_tmpdir)
        raw_dir = multi / "data/real/pkpd/raw"
        raw_dir.mkdir(parents=True)
        auto_raw = raw_dir / "warfarin_data.txt"
        auto_raw.write_text(
            "\n".join(
                [
                    "id,time,amt,dv,dvid,wt,age,sex",
                    "1,0,100,,1,66.7,50,1",
                    "1,0,0,0.0,1,66.7,50,1",
                    "1,0,0,100.0,2,66.7,50,1",
                    "1,12,0,1.2,1,66.7,50,1",
                    "1,12,0,88.0,2,66.7,50,1",
                    "2,0,90,,1,55.0,44,0",
                    "2,0,0,0.0,1,55.0,44,0",
                    "2,0,0,100.0,2,55.0,44,0",
                    "2,12,0,1.4,1,55.0,44,0",
                    "2,12,0,91.0,2,55.0,44,0",
                    "3,0,120,,1,72.0,61,1",
                    "3,0,0,0.0,1,72.0,61,1",
                    "3,0,0,100.0,2,72.0,61,1",
                    "3,12,0,1.7,1,72.0,61,1",
                    "3,12,0,84.0,2,72.0,61,1",
                ]
            )
            + "\n"
        )
        resolved, fixture_mode = resolve_raw_path(auto_raw)
        assert resolved == auto_raw
        assert fixture_mode is False
        multi_normalized = normalize_warfarin_table(auto_raw)
        split = build_subject_holdout_split(multi_normalized, train_fraction=2 / 3, seed=7)
        assert split["status"] == "ok"
        assert set(split["train_subjects"]).isdisjoint(set(split["test_subjects"]))
        assert len(split["train_subjects"]) == 2
        assert len(split["test_subjects"]) == 1
        bundles = build_subject_target_bundles(multi_normalized, split["train_subjects"])
        assert len(bundles) == 2
        first_bundle = bundles[0]
        assert first_bundle.subject_id in split["train_subjects"]
        assert first_bundle.observed.shape == first_bundle.observation_mask.shape
        assert first_bundle.observed.shape == (2, 2)
        assert first_bundle.time_points.tolist() == [0.0, 12.0]
        assert first_bundle.observation_mask.all()
        assert first_bundle.y0.shape == (2,)
        assert first_bundle.y0[0] == 0.0
        assert first_bundle.y0[1] == 100.0
        assert first_bundle.covariates["dose_mg"] in {90.0, 100.0, 120.0}
        assert first_bundle.covariates["weight_kg"] is not None
        perfect_subject_metrics = subject_prediction_metrics(first_bundle, first_bundle.observed.copy())
        assert perfect_subject_metrics["aggregate"]["mse"] == 0.0
        assert perfect_subject_metrics["aggregate"]["observed_count"] == 4
        assert perfect_subject_metrics["endpoint_metrics"]["concentration"]["observed_count"] == 2
        subject_predictions = {bundle.subject_id: bundle.observed.copy() for bundle in bundles}
        subject_predictions[first_bundle.subject_id][1, 0] += 2.0
        grouped_metrics = evaluate_subject_predictions(bundles, subject_predictions)
        assert grouped_metrics["subject_count"] == 2
        assert grouped_metrics["aggregate"]["observed_count"] == 8
        assert grouped_metrics["aggregate"]["mse"] == 0.5
        first_grouped = next(
            item for item in grouped_metrics["per_subject"] if item["subject_id"] == first_bundle.subject_id
        )
        assert first_grouped["endpoint_metrics"]["concentration"]["mse"] == 2.0
        try:
            evaluate_subject_predictions(bundles, {first_bundle.subject_id: first_bundle.observed.copy()})
        except ValueError as exc:
            assert "missing prediction for subject" in str(exc)
        else:
            raise AssertionError("missing subject predictions should be rejected")

        class LinearSubjectModel:
            def simulate(self, params, time_points, y0):
                params = np.asarray(params, dtype=float)
                y0 = np.asarray(y0, dtype=float)
                time_points = np.asarray(time_points, dtype=float)
                predicted = np.tile(y0, (len(time_points), 1))
                predicted[:, 0] = y0[0] + params[0] * time_points
                predicted[:, 1] = y0[1] + params[1] * time_points
                return predicted

        grouped_model = LinearSubjectModel()
        params_batch = np.array([[0.1, -1.0], [0.0, 0.0]])
        distances = grouped_subject_distances(grouped_model, params_batch, bundles)
        assert distances.shape == (2,)
        assert distances[0] < distances[1]
        first_distance = grouped_subject_distance(grouped_model, params_batch[0], bundles)
        assert np.isclose(first_distance, distances[0])
        model_predictions = predict_subject_bundles(grouped_model, params_batch[0], bundles)
        model_metrics = evaluate_subject_predictions(bundles, model_predictions)
        assert np.isclose(distances[0], np.sqrt(model_metrics["aggregate"]["mse"]))
        try:
            grouped_subject_distances(grouped_model, np.zeros((1, 1, 2)), bundles)
        except ValueError as exc:
            assert "params_batch must be one- or two-dimensional" in str(exc)
        else:
            raise AssertionError("invalid parameter batch shapes should be rejected")

        try:
            build_subject_target_bundles(multi_normalized, ["missing"])
        except ValueError as exc:
            assert "unknown Warfarin subject_id" in str(exc)
        else:
            raise AssertionError("unknown subjects should be rejected")
        multi_manifest = generate_data(
            raw_path=auto_raw,
            output_root=multi / "out",
            data_dir=multi / "dense",
            holdout_seed=7,
            holdout_train_fraction=2 / 3,
        )
        assert multi_manifest["fixture_mode"] is False
        assert multi_manifest["subject_holdout"]["status"] == "ok"
        assert Path(multi_manifest["subject_holdout_json"]).exists()
        subject_config = dict(config)
        subject_config.update(
            {
                "evaluation_mode": "subject_holdout",
                "real_data_adapter": "warfarin_pkpd",
                "normalized_data_path": str(Path(multi_manifest["normalized_csv"])),
                "subject_holdout_path": str(Path(multi_manifest["subject_holdout_json"])),
            }
        )
        subject_result = evaluate_subject_holdout_candidate(
            config["seed_logic"],
            subject_config,
            SimpleNamespace(
                target_samples=2,
                generations=1,
                initial_particles=8,
                inference_strategy="gaussian_weighted",
                seed=11,
            ),
        )
        assert subject_result["status"] == "success"
        assert subject_result["effective_strategy"] == "subject_holdout_prior_rejection"
        assert subject_result["train_metrics"]["observed_count"] == 8
        assert subject_result["test_metrics"]["observed_count"] == 4
        assert subject_result["diagnostics"]["subject_holdout"]["train_subjects"] == split["train_subjects"]
        assert subject_result["diagnostics"]["subject_holdout"]["test_subjects"] == split["test_subjects"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "real_warfarin_seed.py"
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
                "real_warfarin_pkpd",
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
                "19",
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
        expected_observed = int(np.load("data/real_warfarin_pkpd_observation_mask.npy").sum())
        assert payload["diagnostics"]["residual_summary"]["global"]["observed_count"] == expected_observed

    print("SUCCESS: real_warfarin_pkpd adapter smoke is stable")


if __name__ == "__main__":
    main()

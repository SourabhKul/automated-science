"""Synthetic mechanics checks for the development-only tank workflow."""

from __future__ import annotations

import io
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.real_data import cascaded_tanks_controlled as source
from core.real_data.cascaded_tanks_models import (
    TankFailureCategory, TankModel, TankParameters, TankSimulationFailure,
    TankState, simulate_cascaded_tanks,
)
from core.real_data import cascaded_tanks_week_score as score
from scripts import run_cascaded_tanks_week as runner


def _csv_fixture(bad_interval: bool = False) -> bytes:
    rows = [source.CSV_HEADER_WIRE + b"\n"]
    for index in range(1024):
        interval = "4.0" if index == 0 else ("4.0" if bad_interval and index == 1 else "")
        y_est = str(index if index < 768 else 1000 + index)
        rows.append(f"{index},SEALED_UVAL,{y_est},SEALED_YVAL,{interval},\n".encode("ascii"))
    rows.append(b"\n")
    return b"".join(rows)


class DevelopmentWeekTests(unittest.TestCase):
  def test_phase_scanner_reads_only_requested_suffix_and_ignores_sealed_text(self):
    raw = _csv_fixture()
    result = score._scan_fixture_for_test(io.BytesIO(raw), len(raw), {2: ("yEst_suffix", 768, 1024)})
    self.assertEqual(result, {"yEst_suffix": tuple(1000 + i for i in range(768, 1024))})
    self.assertTrue(all("SEALED" not in repr(value) for value in result.values()))

  def test_phase_scanner_rejects_wrong_sample_interval(self):
    raw = _csv_fixture(bad_interval=True)
    with self.assertRaisesRegex(score.WeekStageError, "sample interval"):
        score._scan_fixture_for_test(io.BytesIO(raw), len(raw), {2: ("suffix", 768, 1024)})

  def test_weighted_mean_and_left_inverse_bands_are_exact(self):
    values = np.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    mean, q10, q90 = runner.weighted_mean_bands(values, np.asarray([0.2, 0.5, 0.3]))
    np.testing.assert_allclose(mean, [2.2, 21.0])
    np.testing.assert_array_equal(q10, [0.0, 10.0])
    np.testing.assert_array_equal(q90, [4.0, 30.0])

  def test_weighted_bands_use_stable_value_order_and_no_interpolation(self):
    values = np.asarray([[1.0], [1.0], [3.0]])
    mean, q10, q90 = runner.weighted_mean_bands(values, [0.4, 0.2, 0.4])
    np.testing.assert_allclose(mean, [1.8])
    self.assertEqual(q10.tolist(), [1.0])
    self.assertEqual(q90.tolist(), [3.0])

  def test_development_replay_preserves_observe_then_update_alignment(self):
    inputs = np.asarray([0.5, 0.0, 1.0, 0.25, 0.75], dtype=np.float64)
    parameters = TankParameters(0.7, 0.9, 0.2)
    initial = TankState(4.0, 5.205)
    whole = simulate_cascaded_tanks(inputs, parameters, initial, model=TankModel.C2,
                                    ceiling=10.0, limits=runner.TankSimulationLimits(5, 1e6))
    prefix = simulate_cascaded_tanks(inputs[:3], parameters, initial, model=TankModel.C2,
                                     ceiling=10.0, limits=runner.TankSimulationLimits(3, 1e6))
    suffix = simulate_cascaded_tanks(inputs[3:], parameters, prefix.terminal_state, model=TankModel.C2,
                                     ceiling=10.0, limits=runner.TankSimulationLimits(2, 1e6))
    self.assertNotIsInstance(whole, TankSimulationFailure)
    self.assertNotIsInstance(prefix, TankSimulationFailure)
    self.assertNotIsInstance(suffix, TankSimulationFailure)
    self.assertEqual(whole.observations[3:], suffix.observations)
    self.assertEqual(whole.observations[:3], prefix.observations)

  def test_failed_particle_fails_whole_abc_forecast(self):
    original = runner._simulate
    calls = 0

    def simulate(model, theta, inputs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return TankSimulationFailure(TankFailureCategory.NON_FINITE, "fixture", 10, (), None)
        return original(model, theta, inputs)

    particles = np.asarray([[0.7, 0.9, 0.2, 4.0], [0.8, 0.8, 0.2, 5.0]])
    record = {"kind": "abc", "model": "S0", "seed": 11, "status": "fit_complete"}
    with patch.object(runner, "_simulate", side_effect=simulate):
        completed, _ = runner._forecast_arm(record, {"population_params": particles,
                                                     "population_weights": np.asarray([0.5, 0.5])},
                                            "S0", np.ones(1024), [time.monotonic(), 0, 0])
    self.assertEqual(completed["status"], "failed")
    self.assertIsNone(completed["forecast"])
    self.assertIn("one particle failed", completed["failure"])

  def test_fit_and_forecast_checkpoints_score_as_one_frozen_bundle(self):
    from tempfile import TemporaryDirectory

    parameters = np.tile([0.7, 0.9, 0.2, 4.0], (64, 1))
    weights = np.full(64, 1.0 / 64.0)
    data, inputs, train_y, train_u = SimpleNamespace(archive_sha256="synthetic-archive"), np.zeros(1024), np.zeros(768), np.zeros(768)

    def fit_abc(model, seed, *args):
        return ({"kind": "abc", "model": model, "seed": seed, "status": "fit_complete"},
                {"population_params": parameters, "population_weights": weights}, 256)

    def fit_pattern(model, seed, *args):
        return ({"kind": "pattern", "model": model, "seed": seed, "status": "fit_complete",
                 "parameter": [0.7, 0.9, 0.2, 4.0]}, {})

    with TemporaryDirectory() as temp:
        root = Path(temp)
        run_dir = root / "run"
        forecast_path = root / "forecasts.json"
        with patch.object(runner, "RUN_DIR", run_dir), patch.object(runner, "FORECAST_PATH", forecast_path), \
             patch.object(runner.scoring, "read_manifest", return_value=({}, "synthetic-manifest")), \
             patch.object(runner, "_load_fit_inputs", return_value=(data, inputs, train_y, train_u)), \
             patch.object(runner, "_fit_abc", side_effect=fit_abc), \
             patch.object(runner, "_fit_pattern", side_effect=fit_pattern):
            runner.fit_development("synthetic.zip", "synthetic-manifest")
            self.assertEqual((run_dir / "arms" / "fit" / "pattern.S0.11.json").exists(), True)
            self.assertEqual((run_dir / "arms" / "forecast" / "pattern.S0.11.json").exists(), True)
        bundle = json.loads(forecast_path.read_bytes())
        self.assertEqual(bundle["forecast_calls"], 585)
        self.assertTrue(all(arm["status"] == "complete" for arm in bundle["arms"]))
        self.assertEqual(sum(arm.get("forecast_steps", 0) for arm in bundle["arms"]), 585 * 1024)
        with patch.object(score, "read_manifest", return_value=({}, "synthetic-manifest")), \
             patch.object(score.source, "load_development_data", return_value=data), \
             patch.object(score, "verify_source"), \
             patch.object(score, "archive_snapshot", return_value=b"synthetic archive"), \
             patch.object(score, "parse_development_targets", return_value=tuple(np.zeros(256))):
            receipt = score.score_development("fixture.zip", "fixture.json", forecast_path, run_dir)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["target_access"], {"yEst_suffix": 256})
        self.assertEqual(len(receipt["arm_scores"]), 20)

  def test_paired_allowance_counts_failed_calls_not_support_rejects(self):
    records = [
        {"diagnostics": {"simulated": 17, "failed_simulations": 3, "out_of_support": 8}},
        {"diagnostics": {"simulated": 12, "failed_simulations": 2, "out_of_support": 5}},
    ]
    self.assertEqual(runner.paired_allowance(256, records), 290)

  def test_abc_checkpoint_retains_calibration_and_both_population_evidence(self):
    params = np.tile([0.7, 0.9, 0.2, 4.0], (64, 1))
    weights = np.full(64, 1.0 / 64.0)
    populations = []
    for generation in range(2):
        populations.append({
            "generation": generation, "epsilon": 1.0, "accepted_params": params,
            "distances": np.ones(64), "weights": weights, "log_weights": np.log(weights),
            "log_prior_density": np.full(64, -1.0),
            "log_proposal_mixture_density": np.full(64, -2.0),
            "proposal_covariance": None if generation == 0 else np.eye(4),
            "effective_sample_size": 64.0,
            "diagnostics": {"simulated": 0, "failed_simulations": 0, "complete": True},
        })
    reference = {"status": "complete", "termination_reason": "completed",
                 "effective_sample_size": 64.0, "populations": populations}
    invalid = SimpleNamespace(observations=[np.nan, 1.0, 1.0], trace=(1, 2, 3))
    valid = SimpleNamespace(observations=np.ones(3), trace=(1, 2, 3))
    calibration_results = [invalid] + [valid] * 255
    with patch.object(runner, "_guard"), patch.object(
        runner, "_simulate", side_effect=calibration_results
    ), patch.object(runner, "run_gaussian_abc_smc_reference", return_value=reference):
        record, arrays, _ = runner._fit_abc("S0", 11, np.ones(3), np.zeros(3), 0.0, [0])
    self.assertEqual(arrays["calibration_theta"].shape, (256, 4))
    self.assertTrue(np.isposinf(arrays["calibration_distance"][0]) and np.isfinite(arrays["calibration_distance"][1:]).all())
    self.assertEqual((arrays["calibration_outcome"][0], record["calibration_failed"], record["epsilon"]),
                     (2, 1, [1.0, 1.0]))
    for index in range(2):
        for field in ("params", "weights", "distances", "log_weights",
                      "log_prior_density", "log_proposal_mixture_density"):
            self.assertEqual(arrays[f"population{index}_{field}"].shape[0], 64)
    self.assertEqual(arrays["population0_proposal_covariance"].shape, (0, 0))
    np.testing.assert_array_equal(arrays["population1_proposal_covariance"], np.eye(4))
    self.assertEqual(len(record["populations"]), 2)
    self.assertTrue(record["populations"][0]["diagnostics"]["complete"])

  def test_marker_is_exclusive_and_deferred_stages_are_rejected(self):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        score.claim_marker(temp, "development-score", {"manifest_sha256": "x"})
        with self.assertRaises(FileExistsError):
            score.claim_marker(temp, "development-score", {"manifest_sha256": "x"})
    for stage in ("refit", "official-score"):
        with self.assertRaisesRegex(score.WeekStageError, "deferred"):
            score.reject_deferred_stage(stage)

  def test_family_selection_stays_unresolved_with_fewer_than_two_eligible(self):
    rows = []
    for model in score.MODELS:
        for seed in score.SEEDS:
            complete = model == "C2"
            rows.append({"kind": "abc", "model": model, "seed": seed,
                         "status": "complete" if complete else "failed",
                         "rmse": 1.0 if complete else None})
    rows.extend([
        {"kind": "arx", "model": "ARX(2,2)", "seed": 0, "rmse": 2.0},
        {"kind": "persistence", "model": "persistence", "seed": 0, "rmse": 3.0},
    ])
    result = score._decisions(rows)
    self.assertEqual(result["eligible_family_count"], 1)
    self.assertEqual(result["locked_selection"], "unresolved")

  def test_fit_and_forecast_resource_stops_mark_unstarted_arms_not_run(self):
    from tempfile import TemporaryDirectory

    data = SimpleNamespace(archive_sha256="synthetic-archive")
    inputs, train_y, train_u = np.ones(1024), np.zeros(768), np.ones(768)
    original_abc = runner._fit_abc
    guard_count = 0

    def stop_during_calibration(*args):
        nonlocal guard_count
        guard_count += 1
        if guard_count == (1 if phase == "calibration-empty" else 7):
            raise runner.ResourceStop("synthetic calibration resource stop")

    def abc_failed(model, seed, *args):
        return ({"kind": "abc", "model": model, "seed": seed,
                 "status": "failed", "failure": "synthetic calibration failure"}, {}, 256)

    def abc_complete(model, seed, *args):
        return ({"kind": "abc", "model": model, "seed": seed, "status": "fit_complete"},
                {"population_params": np.asarray([[0.7, 0.9, 0.2, 4.0]]),
                 "population_weights": np.asarray([1.0])}, 1)

    def pattern_complete(model, seed, *args):
        return ({"kind": "pattern", "model": model, "seed": seed,
                 "status": "fit_complete", "parameter": [0.7, 0.9, 0.2, 4.0]}, {})

    with TemporaryDirectory() as temp:
        for phase in ("calibration-empty", "calibration", "calibration-simulator-stop", "pattern", "forecast"):
            run_dir = Path(temp) / phase
            forecast_path = run_dir / "forecasts.json"
            fit_abc = original_abc if phase.startswith("calibration") else abc_failed if phase == "pattern" else abc_complete
            fit_pattern = runner._fit_pattern if phase in ("calibration", "pattern") else pattern_complete
            guard = (stop_during_calibration if phase.startswith("calibration") else
                     runner.ResourceStop("synthetic pattern resource stop") if phase == "pattern" else None)
            stop_message = "synthetic calibration resource stop" if phase == "calibration-simulator-stop" else "synthetic forecast resource stop"
            simulation = runner.ResourceStop(stop_message) if phase in ("calibration-simulator-stop", "forecast") else None
            simulation_result = SimpleNamespace(observations=np.ones(768)) if phase.startswith("calibration") else None
            with patch.object(runner, "RUN_DIR", run_dir), \
                 patch.object(runner, "FORECAST_PATH", forecast_path), \
                 patch.object(runner.scoring, "read_manifest", return_value=({}, "synthetic-manifest")), \
                 patch.object(runner, "_load_fit_inputs", return_value=(data, inputs, train_y, train_u)), \
                 patch.object(runner, "_fit_arx", return_value=np.zeros(5)), \
                 patch.object(runner, "_forecast_arx", return_value=np.zeros(256)), \
                 patch.object(runner, "_fit_abc", side_effect=fit_abc), \
                 patch.object(runner, "_fit_pattern", side_effect=fit_pattern), \
                 patch.object(runner, "_guard", side_effect=guard), \
                 patch.object(runner, "_simulate", side_effect=simulation,
                              return_value=simulation_result) as simulator:
                runner.fit_development("synthetic.zip", "synthetic-manifest")
                if phase.startswith("calibration"):
                    self.assertEqual(simulator.call_count, 0 if phase == "calibration-empty" else 1 if phase == "calibration-simulator-stop" else 3)
            bundle = json.loads(forecast_path.read_bytes())
            arms = bundle["arms"]
            if phase.startswith("calibration"):
                stopped = next(a for a in arms if a["kind"] == "abc" and a["model"] == "S0" and a["seed"] == 11)
                draws = 0 if phase == "calibration-empty" else 1 if phase == "calibration-simulator-stop" else 3
                self.assertEqual(bundle["fit_simulator_calls"], draws)
                with np.load(run_dir / "arms" / "fit" / "abc.S0.11.npz", allow_pickle=False) as evidence:
                    self.assertEqual(evidence["calibration_theta"].shape, (draws, 4))
                    self.assertEqual(evidence["calibration_distance"].shape, (draws,))
                    np.testing.assert_array_equal(evidence["calibration_distance"], [np.inf] if phase == "calibration-simulator-stop" else [1.0] * draws)
                    np.testing.assert_array_equal(evidence["calibration_outcome"], [3] if phase == "calibration-simulator-stop" else [0] * draws)
                arm_record = json.loads((run_dir / "arms" / "fit" / "abc.S0.11.json").read_bytes())
                self.assertEqual(arm_record["calibration_calls"], draws)
                self.assertFalse(arm_record["calibration_complete"])
                self.assertEqual(arm_record["failure"], "synthetic calibration resource stop")
            elif phase == "pattern":
                stopped = next(a for a in arms if a["kind"] == "pattern" and a["model"] == "S0" and a["seed"] == 11)
            else:
                stopped = next(a for a in arms if a["kind"] == "abc" and a["model"] == "C2" and a["seed"] == 11)
            self.assertEqual(stopped["status"], "failed")
            self.assertTrue(stopped["resource_stop"])
            self.assertIn("resource stop", stopped["failure"])
            if phase.startswith("calibration") or phase == "pattern":
                current = {("abc", "S0", 11)}
                if phase == "pattern":
                    current.add(("pattern", "S0", 11))
                self.assertTrue(all(a["status"] == "not_run" for a in arms
                                    if a["kind"] in ("abc", "pattern") and
                                       (a["kind"], a["model"], a["seed"]) not in current))
            else:
                self.assertTrue(all(a["status"] == "complete" for a in arms
                                    if a["kind"] in ("arx", "persistence")))
                self.assertTrue(all(a["status"] == "not_run" for a in arms
                                    if a["kind"] in ("abc", "pattern") and
                                       (a["kind"], a["model"], a["seed"]) !=
                                       (stopped["kind"], stopped["model"], stopped["seed"])))


if __name__ == "__main__":
    unittest.main()


"""Synthetic checks for the bounded yeast pilot adapter; no real data is read."""
import hashlib
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import run_yeast_diauxic_pilot as pilot


def synthetic_member(trailing=False, selected=(b"0.1", b"0.2", b"0.3", b"0.4"), bad_shape=False):
    rows = [b"header", pilot.BG, b"Name,example", b"Channel,OD600", b"Unit,OD"]
    for r in range(8):
        cells = [b"opaque"] * (11 if bad_shape and r == 0 else 12)
        if r == 1: cells[1:5] = selected
        rows.append(b",".join(cells + ([b""] if trailing else [])))
    rows.append(b"Assay Finished: ,a,b,c,7/19/2013 5:10:58 PM")
    data = b"\r\n".join(rows) + b"\r\n"
    expected = {"declared_bytes": len(data), "declared_crc32": f"{zlib.crc32(data) & 0xffffffff:08x}",
                "background_channel_source_token": "OD600", "canonical_channel": "OD600",
                "assay_finished_timestamp": "7/19/2013 5:10:58 PM"}
    return data, expected


def install_valid_freeze(root, data_writer=None):
    train = root / pilot.TRAIN_ID
    train.mkdir(parents=True)
    names = ("observations_train.npz", "training_freeze_data.npz", "training_diagnostics.json")
    for name in names:
        if name == "training_freeze_data.npz" and data_writer is not None:
            data_writer(train / name)
        else:
            (train / name).write_bytes(name.encode())
    command = "c" * 64
    cases = [{"index": j, "well": pilot.WELLS[j // 2], "model": pilot.CASES[j][1],
              "status": "complete", "quality_gate_failures": []} for j in range(8)]
    ready = {"run_id": pilot.TRAIN_ID, "status": "complete_training_ready_for_freeze",
             "command_sha256": command, "protocol_sha256": pilot.PROTO_SHA,
             "point_source_sha256": pilot.POINT_SOURCE_SHA, "abc_source_sha256": pilot.ABC_SOURCE_SHA,
             "cases": cases}
    (train / "receipt.json").write_text(json.dumps(ready), encoding="utf-8")
    outputs = [{"path": name, "sha256": hashlib.sha256((train / name).read_bytes()).hexdigest()}
               for name in (*names, "receipt.json")]
    manifest = {"schema": 1, "run_id": pilot.TRAIN_ID, "status": "complete_training_frozen", "complete": True,
                "protocol_sha256": pilot.PROTO_SHA, "command_sha256": command,
                "point_source_sha256": pilot.POINT_SOURCE_SHA, "abc_source_sha256": pilot.ABC_SOURCE_SHA,
                "archive_sha256": pilot.ARCHIVE_SHA, "archive_size_bytes": pilot.SRC_SIZE,
                "raw_metadata_receipt_sha256": pilot.RAW_SHA, "clock_receipt_sha256": pilot.CLOCK_SHA,
                "layout_receipt_sha256": pilot.LAYOUT_SHA, "training_member_count": 271,
                "plate2_members_opened": 0, "cases": cases, "outputs": outputs,
                "training_final_receipt_sha256": outputs[-1]["sha256"]}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    (train / "freeze.json").write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest(), command


class PilotControlsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.enterContext(mock.patch.object(pilot, "OUT_ROOT", self.tmp_path))
        self.enterContext(mock.patch.object(pilot, "STATE", {}))

    def test_innovations_match_dense_gaussian_likelihood(self):
        times = np.array([0.0, 0.13, 0.8, 2.1])
        residual = np.array([0.012, -0.004, 0.009, 0.001])
        covariance = pilot.covariance(times)
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, residual)
        dense = -0.5 * (len(residual) * np.log(2 * np.pi)
                        + 2 * np.log(np.diag(chol)).sum() + whitened @ whitened)
        np.testing.assert_allclose(pilot.loglike(residual, np.zeros_like(residual), times),
                                   dense, rtol=1e-6, atol=1e-12)

    def test_ou_covariance_is_stationary_with_measurement_noise(self):
        times = np.array([0.0, 0.5, 2.0])
        expected = pilot.SIG_P**2 * np.exp(-np.abs(times[:, None] - times[None, :]) / pilot.ELL)
        expected += pilot.SIG_E**2 * np.eye(len(times))
        np.testing.assert_allclose(pilot.covariance(times), expected, rtol=0, atol=1e-15)

    def test_two_phase_curve_contains_one_phase_boundary(self):
        times = np.linspace(0.0, pilot.T, 17)
        one = np.array([0.3, 0.6, 0.4, 0.5])
        two = np.array([*one, 0.0, 0.7, 0.8])
        np.testing.assert_allclose(pilot.curve(one, times, "one_phase"),
                                   pilot.curve(two, times, "two_phase"), rtol=0, atol=1e-15)

    def test_transformed_mixture_density_and_log_weights(self):
        x = np.array([0.2, 0.7])
        mu, cov = np.array([0.1, -0.2]), np.array([[0.8, 0.1], [0.1, 0.5]])
        jac = np.log(x).sum() + np.log1p(-x).sum()
        log_ln = pilot.normal_logpdf(np.log(x) - np.log1p(-x), mu, cov) - jac
        expected = np.logaddexp(np.log(0.8) + log_ln, np.log(0.2))
        np.testing.assert_allclose(pilot.logq(x, mu, cov), expected, rtol=1e-6, atol=1e-12)
        weights, summary = pilot.norm_weights(np.array([0.0, np.log(2.0)]))
        np.testing.assert_allclose(weights, [1 / 3, 2 / 3])
        self.assertAlmostEqual(summary["ess"], 1.8)

    def test_fixed_seed_proposal_is_reproducible(self):
        mu, cov = np.zeros(3), np.eye(3)
        first = pilot.draw_q(np.random.default_rng(91), 64, mu, cov)
        second = pilot.draw_q(np.random.default_rng(91), 64, mu, cov)
        np.testing.assert_array_equal(first, second)

    def test_freeze_refuses_missing_or_incomplete_training(self):
        with self.assertRaisesRegex(pilot.Stop, "stopped_freeze"): pilot.verify_freeze("0" * 64, "a" * 64)
        run_dir = self.tmp_path / pilot.TRAIN_ID
        run_dir.mkdir()
        manifest = {"status": "incomplete_reference_quality_gate", "complete": False,
                    "protocol_sha256": pilot.PROTO_SHA, "command_sha256": "a" * 64,
                    "archive_sha256": pilot.ARCHIVE_SHA, "cases": []}
        encoded = json.dumps(manifest, sort_keys=True).encode()
        (run_dir / "freeze.json").write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        with self.assertRaisesRegex(pilot.Stop, "stopped_freeze"): pilot.verify_freeze(digest, "a" * 64)

    def test_simulated_ou_recurrence_has_stationary_covariance(self):
        rng, times = np.random.default_rng(812), np.array([0.0, 0.5, 2.0])
        paths = np.stack([pilot.simulate_ou(rng, times) for _ in range(40000)])
        expected = pilot.SIG_P**2 * np.exp(-np.abs(times[:, None] - times[None, :]) / pilot.ELL)
        np.testing.assert_allclose(np.cov(paths, rowvar=False), expected, rtol=0, atol=4e-6)

    def test_reference_smc_stops_after_latched_simulator_failure(self):
        prior, logprior = pilot.make_uniform_prior([[0.0, 1.0]])
        latched, events = None, []
        def simulator(_x, _rng):
            nonlocal latched
            latched = pilot.latch_failure(latched, "stopped_simulator", "simulator_callback_failure")
            raise RuntimeError("synthetic callback failure")
        result = pilot.run_gaussian_abc_smc_reference(prior, simulator, lambda _x: 0.0,
            target_samples=2, epsilon_schedule=[1.0], max_attempts_per_population=[5],
            bounds=[[0.0, 1.0]], prior_logpdf=logprior,
            population_event=lambda kind, generation, record: events.append((kind, generation, record)),
            stop_requested=lambda: latched is not None, seed=7)
        self.assertFalse(result["complete"])
        self.assertEqual(result["termination_reason"], "wall_budget_exhausted")
        self.assertEqual(result["diagnostics"]["proposed"], 1)
        self.assertEqual(result["diagnostics"]["failed_simulations"], 1)
        self.assertEqual([event[:2] for event in events], [("start", 0), ("end", 0)])
        self.assertEqual(latched, {"status": "stopped_simulator", "code": "simulator_callback_failure"})

    def test_reference_smc_outer_transition_covariance_failure_is_captured(self):
        from core import abc_smc_reference
        prior, logprior = pilot.make_uniform_prior([[0.0, 1.0]])
        events = []
        def fail_covariance(*_args, **_kwargs):
            raise np.linalg.LinAlgError("synthetic transition failure")
        self.enterContext(mock.patch.object(abc_smc_reference, "make_gaussian_kernel_covariance", fail_covariance))
        with self.assertRaises(np.linalg.LinAlgError):
            pilot.run_gaussian_abc_smc_reference(prior, lambda x, _rng: x, lambda _x: 0.0,
                target_samples=2, epsilon_schedule=[1.0, 1.0], max_attempts_per_population=[2, 2],
                bounds=[[0.0, 1.0]], prior_logpdf=logprior,
                population_event=lambda kind, generation, _record: events.append((kind, generation)), seed=8)
        detail = pilot.outer_failure_detail(None, [{"event": k, "generation": g} for k, g in events])
        self.assertEqual(events, [("start", 0), ("end", 0), ("start", 1)])
        self.assertTrue(detail["outer_exception_captured"])
        self.assertEqual(detail["adapter_stop_cause"]["code"], "abc_covariance_or_runner_exception")

    def test_parser_selects_row_b_columns_2_to_5_and_accepts_only_optional_empty_tail(self):
        for trailing in (False, True):
            with self.subTest(trailing=trailing):
                data, expected = synthetic_member(trailing=trailing)
                self.assertEqual(pilot.parse_member(data, expected), [0.1, 0.2, 0.3, 0.4])

    def test_parser_rejects_invalid_selected_value_and_wrong_matrix_shape(self):
        data, expected = synthetic_member(selected=(b"0.1", b"NaN", b"0.3", b"0.4"))
        with self.assertRaises(pilot.Stop) as error: pilot.parse_member(data, expected)
        self.assertEqual(error.exception.code, "selected_numeric_syntax")
        data, expected = synthetic_member(bad_shape=True)
        with self.assertRaises(pilot.Stop) as error: pilot.parse_member(data, expected)
        self.assertEqual(error.exception.code, "matrix_field_count")

    def test_complete_freeze_and_success_receipt_are_accepted(self):
        digest, command = install_valid_freeze(self.tmp_path)
        self.assertTrue(pilot.verify_freeze(digest, command)["complete"])
        self.assertEqual(pilot.STATE["freeze_sha256"], digest)

    def test_changed_final_success_receipt_stops_before_plate2_archive_open(self):
        pilot.STATE["command_sha256"] = "c" * 64
        digest, _ = install_valid_freeze(self.tmp_path)
        (self.tmp_path / pilot.TRAIN_ID / "receipt.json").write_text('{"status":"stopped"}', encoding="utf-8")
        called = []
        self.enterContext(mock.patch.object(pilot, "read_archive", side_effect=lambda: called.append(True)))
        with self.assertRaises(pilot.Stop) as error:
            pilot.evaluate(self.tmp_path / pilot.EVAL_ID, {}, {}, np.arange(271), digest)
        self.assertEqual(error.exception.code, "freeze_output_hash_mismatch")
        self.assertEqual(called, [])

    def test_changed_or_malformed_freeze_stops_before_plate2_archive_open(self):
        pilot.STATE["command_sha256"] = "c" * 64
        digest, _ = install_valid_freeze(self.tmp_path)
        called = []
        self.enterContext(mock.patch.object(pilot, "read_archive", side_effect=lambda: called.append(True)))
        freeze = self.tmp_path / pilot.TRAIN_ID / "freeze.json"
        freeze.write_bytes(b"changed")
        with self.assertRaises(pilot.Stop) as error:
            pilot.evaluate(self.tmp_path / pilot.EVAL_ID, {}, {}, np.arange(271), digest)
        self.assertEqual(error.exception.code, "freeze_sha_mismatch")
        self.assertEqual(called, [])

    def test_malformed_frozen_cloud_stops_before_plate2_archive_open(self):
        def malformed(path):
            d, n = 4, 2 * pilot.BLOCK
            np.savez_compressed(path, point_0=np.zeros(3), proposal_center_0=np.zeros(d),
                proposal_cov_0=np.eye(d), importance_draws_0=np.zeros((n, d)),
                importance_logq_0=np.zeros(n), importance_loglike_0=np.zeros(n),
                importance_logweights_0=np.zeros(n), importance_weights_0=np.full(n, 1/n),
                abc_params_0=np.zeros((64, d)), abc_weights_0=np.full(64, 1/64),
                prediction_point_0=np.zeros(271), prediction_importance_0=np.zeros(271),
                prediction_abc_0=np.zeros(271))
        pilot.STATE["command_sha256"] = "c" * 64
        digest, _ = install_valid_freeze(self.tmp_path, malformed)
        called = []
        self.enterContext(mock.patch.object(pilot, "read_archive", side_effect=lambda: called.append(True)))
        with self.assertRaises(pilot.Stop) as error:
            pilot.evaluate(self.tmp_path / pilot.EVAL_ID, {}, {}, np.arange(271), digest)
        self.assertEqual(error.exception.code, "frozen_array_shape")
        self.assertEqual(called, [])

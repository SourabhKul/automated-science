from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import core.real_data.silverbox_first_fit as first_fit
from core.real_data import silverbox_controlled
from core.real_data.silverbox import (
    REQUIRED_ARCHIVE_MEMBERS,
    SAMPLE_COUNT,
    SNLS_CSV_MEMBER,
)
from core.real_data.silverbox_controlled import (
    SAMPLE_TIME_SECONDS,
    TRAIN_SOURCE_START,
    TRAIN_WINDOW_LENGTH,
    TRAIN_WINDOW_RELATIVE_STARTS,
    VALIDATION_SOURCE_START,
    VALIDATION_SOURCE_STOP,
    SilverboxControlledSeries,
    SilverboxControlledValidation,
    load_silverbox_controlled_development,
)


def _train_windows() -> tuple[SilverboxControlledSeries, ...]:
    windows = []
    for index, relative_start in enumerate(TRAIN_WINDOW_RELATIVE_STARTS):
        start = TRAIN_SOURCE_START + relative_start
        samples = np.arange(TRAIN_WINDOW_LENGTH, dtype=float)
        input_u = 0.015 + 0.002 * index + 0.00001 * samples
        observed_y = 0.04 + 0.004 * index + 0.00002 * samples
        windows.append(
            SilverboxControlledSeries(
                role="train",
                source_start=start,
                source_stop=start + TRAIN_WINDOW_LENGTH,
                input_u=input_u,
                initialization_y=observed_y[:50],
                target_y=observed_y[50:],
                sampling_time=SAMPLE_TIME_SECONDS,
            )
        )
    return tuple(windows)


def _validation_series(target_value: float = 0.47) -> SilverboxControlledSeries:
    count = VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START
    input_u = np.full(count, 0.02, dtype=float)
    observed_y = np.concatenate((np.full(50, 0.10), np.full(count - 50, target_value)))
    return SilverboxControlledSeries(
        role="validation",
        source_start=VALIDATION_SOURCE_START,
        source_stop=VALIDATION_SOURCE_STOP,
        input_u=input_u,
        initialization_y=observed_y[:50],
        target_y=observed_y[50:],
        sampling_time=SAMPLE_TIME_SECONDS,
    )


def _lazy_validation_series(tmp_path: Path) -> SilverboxControlledValidation:
    source = tmp_path / "synthetic-validation-source.zip"
    source.write_bytes(b"synthetic validation source")
    return SilverboxControlledValidation(
        source_start=VALIDATION_SOURCE_START,
        source_stop=VALIDATION_SOURCE_STOP,
        input_u=np.full(VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START, 0.02),
        initialization_y=np.full(50, 0.10),
        sampling_time=SAMPLE_TIME_SECONDS,
        _raw_archive=source,
        _source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        _source_bytes=source.stat().st_size,
    )


def _write_source_bound_development_archive(path: Path) -> str:
    rows = ["V1,V2,\n"]
    for index in range(SAMPLE_COUNT):
        output = (
            0.47
            if VALIDATION_SOURCE_START + 50 <= index < VALIDATION_SOURCE_STOP
            else 0.10
        )
        rows.append(f"0.02000000,{output:.8f},\n")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in REQUIRED_ARCHIVE_MEMBERS:
            archive.writestr(
                member,
                "".join(rows) if member == SNLS_CSV_MEMBER else "synthetic placeholder",
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_fake_abc(prior_sampler, simulator, discrepancy, **kwargs):
    """Return two complete populations and exercise the callbacks/simulator."""
    attempt_cap = kwargs["max_attempts_per_population"]
    expected = {
        "target_samples": 64,
        "max_attempts_per_population": attempt_cap,
        "covariance_scale": 2.0,
        "lambda_noise": 0.01,
        "nugget": 1e-9,
    }
    for key, value in expected.items():
        assert kwargs[key] == value
    dimension = len(kwargs["bounds"])
    expected_bounds = first_fit.LINEAR_BOUNDS if dimension == 4 else first_fit.NONLINEAR_BOUNDS
    np.testing.assert_array_equal(kwargs["bounds"], expected_bounds)
    assert kwargs["seed"] == (26092511 if dimension == 4 else 26092512)
    assert len(kwargs["epsilon_schedule"]) == 2
    assert kwargs["epsilon_schedule"][0] >= kwargs["epsilon_schedule"][1]
    assert kwargs["prior_logpdf"](np.mean(np.asarray(expected_bounds), axis=1)) > float("-inf")
    assert kwargs["prior_logpdf"](np.full(dimension, 100.0)) == float("-inf")

    probe_rng = np.random.default_rng(42)
    probe = np.asarray(prior_sampler(probe_rng), dtype=float)
    discrepancy(simulator(probe, probe_rng))

    populations = []
    bounds = np.asarray(kwargs["bounds"], dtype=float)
    accepted = np.tile(np.mean(bounds, axis=1), (64, 1))
    weights = np.full(64, 1.0 / 64.0)
    for generation, epsilon in enumerate(kwargs["epsilon_schedule"]):
        kwargs["population_event"]("start", generation, None)
        if generation == 0:
            covariance = None
            log_prior = np.full(64, np.nan)
            log_proposal = np.full(64, np.nan)
            log_weights = np.log(weights)
        else:
            covariance = first_fit.make_gaussian_kernel_covariance(
                accepted,
                weights,
                covariance_scale=kwargs["covariance_scale"],
                lambda_noise=kwargs["lambda_noise"],
                nugget=kwargs["nugget"],
            )
            log_prior = np.asarray([kwargs["prior_logpdf"](point) for point in accepted])
            log_proposal = np.asarray(
                [
                    first_fit.gaussian_mixture_logpdf(point, accepted, weights, covariance)
                    for point in accepted
                ]
            )
            log_weights = log_prior - log_proposal
        population = {
            "generation": generation,
            "epsilon": float(epsilon),
            "accepted_params": accepted.copy(),
            "distances": np.full(64, float(epsilon)),
            "weights": weights.copy(),
            "log_weights": log_weights,
            "log_prior_density": log_prior,
            "log_proposal_mixture_density": log_proposal,
            "proposal_covariance": covariance,
            "effective_sample_size": 64.0,
            "diagnostics": {
                # One simulated, finite draw per generation is omitted from
                # the accepted ledger because it exceeded epsilon. The
                # reference diagnostics expose it only as the nonnegative
                # residual `simulated - accepted - failed_discrepancies`.
                "proposed": 65,
                "simulated": 65,
                "accepted": 64,
                "out_of_support": 0,
                "failed_prior_draws": 0,
                "failed_proposals": 0,
                "failed_simulations": 0,
                "failed_discrepancies": 0,
                "weight_failures": 0,
                "max_attempts": attempt_cap,
                "complete": True,
                "termination_reason": "target_reached",
                "ancestor_indices": [None] * 65 if generation == 0 else [0] * 65,
            },
        }
        kwargs["population_event"]("end", generation, population)
        populations.append(population)
    return {
        "status": "complete",
        "complete": True,
        "termination_reason": "completed",
        "reference_path": "gaussian_abc_smc_reference_opt_in",
        "canonical_runner_integrated": False,
        "target_samples": 64,
        "epsilon_schedule": [float(value) for value in kwargs["epsilon_schedule"]],
        "max_attempts_per_population": [attempt_cap, attempt_cap],
        "seed": kwargs["seed"],
        "accepted_params": populations[-1]["accepted_params"],
        "distances": populations[-1]["distances"],
        "accepted_distances": populations[-1]["distances"],
        "weights": populations[-1]["weights"],
        "accepted_weights": populations[-1]["weights"],
        "effective_sample_size": populations[-1]["effective_sample_size"],
        "diagnostics": populations[-1]["diagnostics"],
        "populations": populations,
    }


def test_train_only_scalers_use_fixed_target_and_observed_lag_indices():
    windows = _train_windows()
    scales = first_fit.train_only_scalers(windows, "y_cubed")
    all_targets = np.concatenate([window.target_y for window in windows])
    expected_sy = np.sqrt(np.mean(all_targets**2))
    observed_cubes = []
    for window in windows:
        observed = np.concatenate((window.initialization_y, window.target_y))
        observed_cubes.append(observed[49:255] ** 3)
    expected_sf = np.sqrt(np.mean(np.concatenate(observed_cubes) ** 2))
    assert scales["training_target_count"] == 824
    assert scales["s_y"] == pytest.approx(expected_sy)
    assert scales["s_feature"] == pytest.approx(expected_sf)
    assert scales["divergence_bound"] == pytest.approx(1.0)


def test_recurrence_uses_k_minus_one_input_and_free_run_output_history():
    u = np.full(52, 0.25)
    initializer = np.zeros(50)
    initializer[48:50] = [3.0, 2.0]
    linear = first_fit.simulate_controlled_ar2(
        u,
        initializer,
        [1.0, 1.0, 2.0, 0.5],
        hypothesis="linear",
        term_id="u_y_product",
        s_y=2.0,
        s_feature=4.0,
        divergence_bound=100.0,
    )
    # k=50 uses y[49]=2, y[48]=3 and u[49]=0.25.
    assert linear[0] == pytest.approx(2.0 + 3.0 + 0.5 + 0.5)
    # k=51 uses the generated prediction at k=50 and the recorded u[50].
    assert linear[1] == pytest.approx(linear[0] + 2.0 + 0.5 + 0.5)

    nonlinear = first_fit.simulate_controlled_ar2(
        u,
        initializer,
        [1.0, 1.0, 2.0, 0.5, 1.0],
        hypothesis="nonlinear",
        term_id="u_y_product",
        s_y=2.0,
        s_feature=4.0,
        divergence_bound=100.0,
    )
    expected_extra = 1.0 * 2.0 * (0.25 * 2.0) / 4.0
    assert nonlinear[0] == pytest.approx(linear[0] + expected_extra)


def test_simulator_failure_is_explicit_and_never_clips():
    with pytest.raises(first_fit.SimulationFailure, match="wrong_parameter_shape"):
        first_fit.simulate_controlled_ar2(
            np.zeros(51), np.zeros(50), [1.0], hypothesis="linear", term_id="y_cubed",
            s_y=1.0, s_feature=1.0, divergence_bound=1.0,
        )
    with pytest.raises(first_fit.SimulationFailure, match="divergence_bound_exceeded"):
        first_fit.simulate_controlled_ar2(
            np.zeros(51), np.ones(50), [1.0, 1.0, 0.0, 0.0], hypothesis="linear", term_id="y_cubed",
            s_y=1.0, s_feature=1.0, divergence_bound=1.0,
        )


def test_pointwise_weighted_median_uses_stable_first_cumulative_half():
    trajectories = np.asarray([[0.0, 3.0], [2.0, 1.0], [4.0, 2.0]])
    weights = np.asarray([0.25, 0.50, 0.25])
    # At the first time, the middle particle lands exactly on cumulative 0.5;
    # at the second time, the first sorted particle carries exactly 0.5.
    np.testing.assert_array_equal(first_fit.pointwise_weighted_median(trajectories, weights), [2.0, 1.0])
    tie_paths = np.asarray([[1.0], [1.0], [9.0]])
    np.testing.assert_array_equal(first_fit.pointwise_weighted_median(tie_paths, [0.25, 0.25, 0.5]), [1.0])


def test_fit_api_is_explicitly_train_only_and_requires_upstream_term_id():
    parameters = inspect.signature(first_fit.fit_silverbox_development).parameters
    assert "development" not in parameters
    assert "validation_series" not in parameters
    assert parameters["term_id"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="no fallback"):
        first_fit.fit_silverbox_development(
            _train_windows(), "", source_sha256="a" * 64, proposal_receipt_sha256="b" * 64,
            run_id="invalid", receipt_dir="unused"
        )
    with pytest.raises(TypeError, match="proposal_receipt_sha256"):
        first_fit.fit_silverbox_development(
            _train_windows(), "y_cubed", source_sha256="a" * 64, run_id="missing-provenance", receipt_dir="unused"
        )
    with pytest.raises(ValueError, match="lowercase"):
        first_fit.fit_silverbox_development(
            _train_windows(), "y_cubed", source_sha256="a" * 64, proposal_receipt_sha256="B" * 64,
            run_id="invalid-provenance", receipt_dir="unused"
        )


def test_selection_rejects_preloaded_validation_series():
    with pytest.raises(ValueError, match="deferred, source-bound"):
        first_fit._validate_validation_series(_validation_series())


def test_caller_monotonic_start_is_included_in_fit_wall_budget(monkeypatch, tmp_path):
    def forbidden_abc(*args, **kwargs):
        raise AssertionError("an already expired global pilot must not start ABC")

    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", forbidden_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "y_cubed",
        source_sha256=hashlib.sha256(b"synthetic-deadline-source").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-deadline-proposal").hexdigest(),
        run_id="synthetic-expired-global-deadline",
        receipt_dir=tmp_path,
        pilot_started_monotonic=0.0,
    )

    assert fit.status == "incomplete"
    receipt = json.loads(fit.receipt_path.read_text())
    assert receipt["fit_started_monotonic"] == 0.0
    assert receipt["budget"]["wall_started_monotonic"] == 0.0
    assert receipt["budget"]["stop_reason"] == "wall_time_limit"
    assert receipt["selection"]["validation_accessed"] is False


def test_calibration_and_reference_call_contract_and_separate_validation_selection(monkeypatch, tmp_path):
    source_archive = tmp_path / "source-bound-development.zip"
    source_sha256 = _write_source_bound_development_archive(source_archive)
    development = load_silverbox_controlled_development(source_archive)
    windows = development.train_windows
    validation = development.validation
    simulated_values = []

    def fake_simulator(input_u, initialization_y, parameters, *, hypothesis, term_id, s_y, s_feature, divergence_bound):
        value = 0.50 if hypothesis == "linear" else 0.47
        simulated_values.append(hypothesis)
        return np.full(len(input_u) - 50, value)

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", fake_simulator)
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", _complete_fake_abc)
    fit = first_fit.fit_silverbox_development(
        windows,
        "y_cubed",
        source_sha256=source_sha256,
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-proposal-receipt").hexdigest(),
        run_id="synthetic-complete",
        receipt_dir=tmp_path,
    )
    assert fit.status == "complete"
    fit_payload = json.loads(fit.receipt_path.read_text())
    assert fit_payload["proposal_receipt_sha256"] == fit.proposal_receipt_sha256
    assert fit_payload["selection"]["validation_accessed"] is False
    assert fit_payload["selection"]["validation_target_loaded"] is False
    assert "validation_target_sha256" not in fit_payload
    for hypothesis in ("linear", "nonlinear"):
        calibration = fit_payload["hypotheses"][hypothesis]["calibration"]
        assert calibration["draws_completed"] == 256
        assert calibration["finite_count"] == 256
        assert calibration["quantile_method"] == "linear"
        finite = np.asarray([row["discrepancy"] for row in calibration["calibration_draws"]])
        expected = np.quantile(finite, [0.5, 0.2], method="linear")
        np.testing.assert_allclose(calibration["epsilon_schedule"], expected)
        assert len(fit_payload["hypotheses"][hypothesis]["abc_result"]["populations"]) == 2
        for generation in range(2):
            assert (fit.receipt_path.parent / f"{hypothesis}_population_{generation:02d}.json").exists()
            diagnostics = fit_payload["hypotheses"][hypothesis]["abc_result"]["populations"][generation]["diagnostics"]
            assert diagnostics["simulated"] - diagnostics["accepted"] - diagnostics["failed_discrepancies"] == 1

    simulated_values.clear()
    assert validation.target_loaded is False
    target_load_calls = []

    original_target_loader = silverbox_controlled._read_fixed_validation_target_suffix
    def counted_target_loader(path, *, expected_source_sha256, expected_source_bytes):
        target_load_calls.append((path, expected_source_sha256, expected_source_bytes))
        assert len(simulated_values) == 2 * first_fit.ABC_PARTICLES
        return original_target_loader(
            path,
            expected_source_sha256=expected_source_sha256,
            expected_source_bytes=expected_source_bytes,
        )

    monkeypatch.setattr(
        silverbox_controlled,
        "_read_fixed_validation_target_suffix",
        counted_target_loader,
    )
    selection = first_fit.select_silverbox_development(fit, validation)
    assert selection.status == "selected"
    assert selection.selected_hypothesis == "nonlinear"
    assert validation.target_loaded is True
    assert len(target_load_calls) == 1


    target = validation.load_target_y(fit)
    assert len(target_load_calls) == 1
    assert not target.flags.writeable
    np.testing.assert_array_equal(
        target,
        np.full(VALIDATION_SOURCE_STOP - VALIDATION_SOURCE_START - 50, 0.47),
    )
    selection_payload = json.loads(selection.receipt_path.read_text())
    assert selection_payload["validation_accessed"] is True
    assert selection_payload["validation_target_loaded"] is True
    assert set(selection_payload["forecasts"]) == {"linear", "nonlinear"}
    assert selection_payload["scores"]["rmse"]["nonlinear"] == pytest.approx(0.0)
    assert selection_payload["scores"]["rmse"]["linear"] > 0.0
    assert selection_payload["scores"]["persistence_rmse"] > 0.0

    mutations = (
        ("source_indices", lambda body: body.update({"source_indices": [[1, 2]]})),
        ("sampling_time", lambda body: body.update({"sampling_time": 2.0})),
        (
            "protocol_constants",
            lambda body: body["protocol_constants"].update({"abc_particles": 63}),
        ),
        ("implementation_sha256", lambda body: body.update({"implementation_sha256": "0" * 64})),
        ("controlled_contract_sha256", lambda body: body.update({"controlled_contract_sha256": "0" * 64})),
        ("abc_reference_sha256", lambda body: body.update({"abc_reference_sha256": "0" * 64})),
        (
            "calibration_seed",
            lambda body: body["hypotheses"]["linear"]["calibration"].update({"seed": 0}),
        ),
        (
            "calibration_draw",
            lambda body: body["hypotheses"]["linear"]["calibration"]["calibration_draws"][0]["parameters"].__setitem__(0, 1000.0),
        ),
        (
            "quantile_method",
            lambda body: body["hypotheses"]["nonlinear"]["calibration"].update({"quantile_method": "higher"}),
        ),
        (
            "abc_seed",
            lambda body: body["hypotheses"]["nonlinear"]["abc_result"].update({"seed": 0}),
        ),
        (
            "particle_outside_support",
            lambda body: body["hypotheses"]["linear"]["abc_result"]["populations"][0]["accepted_params"][0].__setitem__(0, 1000.0),
        ),
        (
            "particle_ess",
            lambda body: body["hypotheses"]["linear"]["abc_result"]["populations"][1].update({"effective_sample_size": 1.0}),
        ),
        (
            "population_attempt_counter",
            lambda body: body["hypotheses"]["linear"]["abc_result"]["populations"][0]["diagnostics"].update({"proposed": 63}),
        ),
        (
            "negative_epsilon_rejection_residual",
            lambda body: body["hypotheses"]["linear"]["abc_result"]["populations"][0]["diagnostics"].update({"simulated": 63, "failed_discrepancies": 1}),
        ),
        (
            "accepted_distance_over_epsilon",
            lambda body: body["hypotheses"]["linear"]["abc_result"]["populations"][0]["distances"].__setitem__(0, 1000.0),
        ),
    )
    for index, (name, mutate) in enumerate(mutations):
        body = json.loads(fit.receipt_path.read_text())
        body.pop("receipt_sha256")
        mutate(body)
        path, digest = first_fit._write_signed_json(
            tmp_path / f"tampered-fit-{name}-{index}.json", body
        )
        tampered_fit = replace(fit, receipt_path=path, receipt_sha256=digest)
        with pytest.raises(ValueError):
            validation.load_target_y(tampered_fit)
    with pytest.raises(ValueError):
        validation.load_target_y(replace(fit, run_id="different-cached-fit"))
    assert len(target_load_calls) == 1


def test_v2_fit_receipts_record_2048_cap_and_v1_receipts_cannot_be_relabelled(monkeypatch, tmp_path):
    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", lambda input_u, initialization_y, parameters, **kwargs: np.full(len(input_u) - 50, 0.25))
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", _complete_fake_abc)
    source_archive = tmp_path / "v2-source-bound.zip"
    source_sha = _write_source_bound_development_archive(source_archive)
    development = load_silverbox_controlled_development(source_archive)
    proposal_sha = hashlib.sha256(b"synthetic-v2-proposal").hexdigest()
    manifest_sha = hashlib.sha256(b"caller-frozen-v2-manifest").hexdigest()

    v2_fit = first_fit.fit_silverbox_development(
        development.train_windows,
        "u_cubed",
        source_sha256=source_sha,
        proposal_receipt_sha256=proposal_sha,
        run_id="synthetic-v2-complete",
        receipt_dir=tmp_path,
        protocol_id=first_fit.V2_PROTOCOL_ID,
        preflight_manifest_sha256=manifest_sha,
    )
    assert v2_fit.status == "complete"
    v2_body = json.loads(v2_fit.receipt_path.read_text())
    assert v2_body["protocol_id"] == first_fit.V2_PROTOCOL_ID
    assert v2_body["abc_attempts_per_population"] == 2_048
    assert v2_body["preflight_manifest_sha256"] == manifest_sha
    assert v2_body["protocol_constants"]["abc_attempts_per_population"] == 2_048
    for family in v2_body["hypotheses"].values():
        assert family["abc_result"]["max_attempts_per_population"] == [2_048, 2_048]
        assert [row["diagnostics"]["max_attempts"] for row in family["abc_result"]["populations"]] == [2_048, 2_048]
    first_fit._verify_complete_fit_receipt(v2_fit, v2_body)
    v2_selection = first_fit.select_silverbox_development(v2_fit, development.validation)
    assert v2_selection.protocol_id == first_fit.V2_PROTOCOL_ID
    assert v2_selection.preflight_manifest_sha256 == manifest_sha
    selection_body = json.loads(v2_selection.receipt_path.read_text())
    assert selection_body["protocol_id"] == first_fit.V2_PROTOCOL_ID
    assert selection_body["abc_attempts_per_population"] == 2_048
    assert selection_body["preflight_manifest_sha256"] == manifest_sha

    v1_fit = first_fit.fit_silverbox_development(
        development.train_windows,
        "u_cubed",
        source_sha256=source_sha,
        proposal_receipt_sha256=proposal_sha,
        run_id="synthetic-v1-stays-v1",
        receipt_dir=tmp_path,
    )
    v1_body = json.loads(v1_fit.receipt_path.read_text())
    assert v1_body["protocol_id"] == first_fit.PROTOCOL_ID
    assert "abc_attempts_per_population" not in v1_body
    with pytest.raises(ValueError, match="protocol"):
        first_fit._verify_complete_fit_receipt(
            replace(v1_fit, protocol_id=first_fit.V2_PROTOCOL_ID, preflight_manifest_sha256=manifest_sha),
            v1_body,
        )


def test_public_target_loader_rejects_minimal_content_hashed_fit_handle(monkeypatch, tmp_path):
    validation = _lazy_validation_series(tmp_path)
    proposal_digest = hashlib.sha256(b"synthetic-proposal").hexdigest()
    minimal_receipt = {
        "protocol_id": first_fit.PROTOCOL_ID,
        "run_id": "fabricated-minimal-fit",
        "source_sha256": validation._source_sha256,
        "proposal_receipt_sha256": proposal_digest,
        "term_id": "y_cubed",
        "status": "complete",
        "reason": None,
        "selection": {"validation_accessed": False, "validation_target_loaded": False},
    }
    receipt_path, receipt_digest = first_fit._write_signed_json(
        tmp_path / "fabricated-minimal-fit" / "fit.json", minimal_receipt
    )
    fabricated_fit = first_fit.FrozenSilverboxFit(
        run_id="fabricated-minimal-fit",
        receipt_path=receipt_path,
        receipt_sha256=receipt_digest,
        status="complete",
        term_id="y_cubed",
        source_sha256=validation._source_sha256,
        proposal_receipt_sha256=proposal_digest,
    )

    def forbidden_target_read(*args, **kwargs):
        raise AssertionError("a minimally fabricated receipt must not open validation targets")

    monkeypatch.setattr(
        silverbox_controlled,
        "_read_fixed_validation_target_suffix",
        forbidden_target_read,
    )
    with pytest.raises(ValueError, match="frozen source split or protocol constants"):
        validation.load_target_y(fabricated_fit)
    assert validation.target_loaded is False


def test_incomplete_fit_writes_unresolved_receipt_without_reading_validation(monkeypatch, tmp_path):
    def fake_incomplete_abc(prior_sampler, simulator, discrepancy, **kwargs):
        return {
            "status": "incomplete",
            "complete": False,
            "termination_reason": "attempt_budget_exhausted",
            "populations": [],
        }

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", lambda input_u, initialization_y, parameters, **kwargs: np.zeros(len(input_u) - 50))
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", fake_incomplete_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "u_cubed",
        source_sha256=hashlib.sha256(b"synthetic-incomplete").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-incomplete-proposal").hexdigest(),
        run_id="synthetic-incomplete",
        receipt_dir=tmp_path,
    )
    assert fit.status == "incomplete"
    validation = _lazy_validation_series(tmp_path)

    def forbidden_target_load(*args, **kwargs):
        raise AssertionError("incomplete fit must not open validation target outputs")

    monkeypatch.setattr(silverbox_controlled, "_read_fixed_validation_target_suffix", forbidden_target_load)
    with pytest.raises(ValueError, match="complete frozen fit"):
        validation.load_target_y(fit)
    assert validation.target_loaded is False
    result = first_fit.select_silverbox_development(fit, validation)
    assert result.status == "unresolved"
    assert validation.target_loaded is False
    payload = json.loads(result.receipt_path.read_text())
    assert payload["reason"] == "fit_incomplete"
    assert payload["validation_accessed"] is False
    assert payload["validation_target_loaded"] is False
    assert payload["scores"] is None


def test_selection_honors_same_global_wall_deadline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        first_fit,
        "simulate_controlled_ar2",
        lambda input_u, initialization_y, parameters, **kwargs: np.zeros(len(input_u) - 50),
    )
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", _complete_fake_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "y_cubed",
        source_sha256=hashlib.sha256(b"synthetic-selection-deadline-source").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-selection-deadline-proposal").hexdigest(),
        run_id="synthetic-selection-deadline",
        receipt_dir=tmp_path,
    )
    assert fit.status == "complete"
    receipt = json.loads(fit.receipt_path.read_text())
    original_start = receipt["budget"]["wall_started_monotonic"]
    calls = 0

    def advancing_clock():
        nonlocal calls
        calls += 1
        return original_start + (1.0 if calls < 3 else first_fit.PILOT_WALL_SECONDS + 1.0)

    monkeypatch.setattr(first_fit.time, "monotonic", advancing_clock)
    selection = first_fit.select_silverbox_development(fit, _lazy_validation_series(tmp_path))

    assert selection.status == "unresolved"
    assert selection.reason == "wall_time_limit"
    selection_receipt = json.loads(selection.receipt_path.read_text())
    assert selection_receipt["validation_accessed"] is True
    assert selection_receipt["validation_target_loaded"] is False
    assert selection_receipt["forecasts"] == {}
    assert selection_receipt["scores"] is None


def test_selection_that_crosses_deadline_during_scoring_is_unresolved(monkeypatch, tmp_path):
    validation = _lazy_validation_series(tmp_path)
    monkeypatch.setattr(
        first_fit,
        "simulate_controlled_ar2",
        lambda input_u, initialization_y, parameters, **kwargs: np.zeros(len(input_u) - 50),
    )
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", _complete_fake_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "y_cubed",
        source_sha256=validation._source_sha256,
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-late-score-proposal").hexdigest(),
        run_id="synthetic-late-score-deadline",
        receipt_dir=tmp_path,
    )
    assert fit.status == "complete"
    fit_receipt = json.loads(fit.receipt_path.read_text())
    started = fit_receipt["budget"]["wall_started_monotonic"]
    now = [started + 1.0]
    original_rmse = first_fit._rmse
    score_calls = 0

    def advance_after_final_score(prediction, target):
        nonlocal score_calls
        result = original_rmse(prediction, target)
        score_calls += 1
        if score_calls == 3:
            now[0] = started + first_fit.PILOT_WALL_SECONDS + 1.0
        return result

    monkeypatch.setattr(first_fit.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(first_fit, "_rmse", advance_after_final_score)
    monkeypatch.setattr(
        silverbox_controlled,
        "_read_fixed_validation_target_suffix",
        lambda *args, **kwargs: np.full(validation.prediction_count, 0.0),
    )
    selection = first_fit.select_silverbox_development(fit, validation)

    assert score_calls == 3
    assert selection.status == "unresolved"
    assert selection.selected_hypothesis is None
    assert selection.reason == "wall_time_limit"
    receipt = json.loads(selection.receipt_path.read_text())
    assert receipt["status"] == "unresolved"
    assert receipt["reason"] == "wall_time_limit"
    assert receipt["selected_hypothesis"] is None
    assert receipt["scores"]["rmse"] is not None


def test_selection_write_that_crosses_deadline_is_replaced_with_unresolved(monkeypatch, tmp_path):
    validation = _lazy_validation_series(tmp_path)
    def fake_simulator(input_u, initialization_y, parameters, *, hypothesis, **kwargs):
        value = 0.50 if hypothesis == "linear" else 0.47
        return np.full(len(input_u) - 50, value)

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", fake_simulator)
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", _complete_fake_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "y_cubed",
        source_sha256=validation._source_sha256,
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-write-deadline-proposal").hexdigest(),
        run_id="synthetic-write-deadline",
        receipt_dir=tmp_path,
    )
    fit_receipt = json.loads(fit.receipt_path.read_text())
    started = fit_receipt["budget"]["wall_started_monotonic"]
    now = [started + 1.0]
    original_write = first_fit._write_signed_json
    selection_write_statuses = []

    def cross_budget_during_selected_write(path, payload):
        path, digest = original_write(path, payload)
        if path.name == "selection.json":
            selection_write_statuses.append(payload["status"])
            if payload["status"] == "selected":
                now[0] = started + first_fit.PILOT_WALL_SECONDS + 1.0
        return path, digest

    monkeypatch.setattr(first_fit.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(first_fit, "_write_signed_json", cross_budget_during_selected_write)
    monkeypatch.setattr(
        silverbox_controlled,
        "_read_fixed_validation_target_suffix",
        lambda *args, **kwargs: np.full(validation.prediction_count, 0.47),
    )
    selection = first_fit.select_silverbox_development(fit, validation)

    assert selection_write_statuses == ["selected", "unresolved"]
    assert selection.status == "unresolved"
    assert selection.selected_hypothesis is None
    assert selection.reason == "wall_time_limit"
    receipt = json.loads(selection.receipt_path.read_text())
    assert receipt["status"] == "unresolved"
    assert receipt["reason"] == "wall_time_limit"
    assert receipt["selected_hypothesis"] is None
    assert receipt["receipt_sha256"] == selection.receipt_sha256
    assert first_fit._load_signed_json(selection.receipt_path, selection.receipt_sha256) == receipt


def test_calibration_failure_count_and_finite_draw_gate_are_retained(monkeypatch, tmp_path):
    def failed_simulator(*args, **kwargs):
        raise first_fit.SimulationFailure("divergence_bound_exceeded", "synthetic divergence")

    def forbidden_abc(*args, **kwargs):
        raise AssertionError("ABC must not run after the finite-calibration gate fails")

    monkeypatch.setattr(first_fit, "simulate_controlled_ar2", failed_simulator)
    monkeypatch.setattr(first_fit, "run_gaussian_abc_smc_reference", forbidden_abc)
    fit = first_fit.fit_silverbox_development(
        _train_windows(),
        "y_cubed",
        source_sha256=hashlib.sha256(b"synthetic-calibration-failure").hexdigest(),
        proposal_receipt_sha256=hashlib.sha256(b"synthetic-calibration-proposal").hexdigest(),
        run_id="synthetic-calibration-failure",
        receipt_dir=tmp_path,
    )
    assert fit.status == "incomplete"
    receipt = json.loads(fit.receipt_path.read_text())
    for family_name in ("linear", "nonlinear"):
        family = receipt["hypotheses"][family_name]
        calibration = family["calibration"]
        assert family["status"] == "incomplete"
        assert calibration["finite_count"] == 0
        assert calibration["failure_count"] == 256
        assert calibration["epsilon_schedule"] is None
        assert calibration["failure_modes"]["divergence_bound_exceeded"] == 256
        assert all(row["discrepancy"] == "Infinity" for row in calibration["calibration_draws"])


def test_no_sealed_scorer_or_qwen_client_is_exported():
    assert "load_silverbox_for_scoring" not in first_fit.__dict__
    assert not any("score" in name.lower() and "selection" not in name.lower() for name in first_fit.__all__)
    assert "request_silverbox_term_id" not in first_fit.__dict__

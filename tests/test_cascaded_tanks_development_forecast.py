from __future__ import annotations

from pathlib import Path

import pytest

from core.real_data import cascaded_tanks_controlled as source
from core.real_data.cascaded_tanks_development_forecast import (
    CascadedTanksForecastEnsemble,
    CascadedTanksForecastError,
    CascadedTanksForecastParticle,
    _forecast_synthetic_fixture,
    run_cascaded_tanks_development_forecast,
)
from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)


def _synthetic_view(*, boundary_kick: float = 100.0):
    training_inputs = [0.0] * 768
    training_inputs[766] = boundary_kick
    training_inputs[767] = 2.0
    training_outputs = [0.0] * 768
    development_inputs = [0.0] * 256
    visible_sha256 = "a" * 64
    receipts = source._make_stage_receipts(
        training_inputs,
        training_outputs,
        visible_sha256,
        contract_sha256=source.SYNTHETIC_FIXTURE_CONTRACT_SHA256,
        archive_sha256=None,
    )
    return source.SyntheticCascadedTanksDevelopmentData(
        track_id=source.SYNTHETIC_FIXTURE_TRACK_ID,
        contract_sha256=source.SYNTHETIC_FIXTURE_CONTRACT_SHA256,
        source_visible_sha256=visible_sha256,
        sample_interval_seconds=4.0,
        training_indices=tuple(range(768)),
        training_u_est=tuple(training_inputs),
        training_y_est=tuple(training_outputs),
        development_input_indices=tuple(range(768, 1024)),
        development_u_est=tuple(development_inputs),
        stage_receipts=receipts,
    )


def _particle(*, p: float = 1.0, weight: float = 1.0):
    return CascadedTanksForecastParticle(
        model=TankModel.S0,
        parameters=TankParameters(a=0.5, c=0.5, p=p),
        initial_state=TankState(x1=0.0, x2=0.0),
        weight=weight,
    )


def _limits(*, max_magnitude: float = 1.0e12):
    return TankSimulationLimits(max_steps=768, max_magnitude=max_magnitude)


def test_forecast_carries_state_after_training_u767_into_dev_y768() -> None:
    data = _synthetic_view()
    particle = CascadedTanksForecastParticle(
        model=TankModel.S0,
        parameters=TankParameters(a=0.5, c=0.5, p=1.0),
        initial_state=TankState(x1=4.0, x2=4.0),
        weight=1.0,
    )
    outcome = _forecast_synthetic_fixture(
        data,
        CascadedTanksForecastEnsemble((particle,)),
        limits=_limits(),
    )

    training = simulate_cascaded_tanks(
        data.training_u_est,
        particle.parameters,
        particle.initial_state,
        model=particle.model,
        limits=_limits(),
    )
    assert isinstance(training, TankSimulationSuccess)
    truncated_training = simulate_cascaded_tanks(
        data.training_u_est[:767],
        particle.parameters,
        particle.initial_state,
        model=particle.model,
        limits=_limits(),
    )
    assert isinstance(truncated_training, TankSimulationSuccess)
    forecast_without_last_training_transition = simulate_cascaded_tanks(
        data.development_u_est,
        particle.parameters,
        truncated_training.terminal_state,
        model=particle.model,
        limits=_limits(),
    )
    assert isinstance(forecast_without_last_training_transition, TankSimulationSuccess)
    zero_last_input = simulate_cascaded_tanks(
        data.training_u_est[:767] + (0.0,),
        particle.parameters,
        particle.initial_state,
        model=particle.model,
        limits=_limits(),
    )
    assert isinstance(zero_last_input, TankSimulationSuccess)
    forecast_with_zero_last_input = simulate_cascaded_tanks(
        data.development_u_est,
        particle.parameters,
        zero_last_input.terminal_state,
        model=particle.model,
        limits=_limits(),
    )
    assert isinstance(forecast_with_zero_last_input, TankSimulationSuccess)

    assert outcome.status == "complete"
    assert outcome.forecast is not None and len(outcome.forecast) == 256
    assert data.training_u_est[767] != 0.0
    assert outcome.forecast[0] == training.terminal_state.x2
    assert (
        outcome.forecast[0] != forecast_without_last_training_transition.observations[0]
    )
    assert training.terminal_state.x1 != truncated_training.terminal_state.x1
    assert outcome.forecast[0] == forecast_with_zero_last_input.observations[0]
    assert outcome.forecast[1] != forecast_with_zero_last_input.observations[1]
    x1_767 = truncated_training.terminal_state.x1
    x2_767 = truncated_training.terminal_state.x2
    expected_y768 = max(
        0.0,
        x2_767 + x1_767**0.5 - particle.parameters.c * x2_767**0.5,
    )
    expected_x1_768 = max(
        0.0,
        x1_767
        + particle.parameters.p * data.training_u_est[767]
        - particle.parameters.a * x1_767**0.5,
    )
    expected_y769 = max(
        0.0,
        expected_y768
        + expected_x1_768**0.5
        - particle.parameters.c * expected_y768**0.5,
    )
    assert outcome.forecast[0] == pytest.approx(expected_y768)
    assert outcome.forecast[1] == pytest.approx(expected_y769)
    # The final input changes x1[768], so its delayed effect appears in y[769].
    # y[768] reflects prior x1[767]. Omitting the final transition itself also
    # omits the x1[767]-to-x2[768] update and therefore misaligns y[768].
    assert outcome.receipt.training_indices == (0, 768)
    assert outcome.receipt.boundary_state_index == 768
    assert outcome.receipt.forecast_indices == (768, 1024)
    assert outcome.receipt.particle_receipts[0].boundary_state_sha256 is not None
    assert outcome.receipt.particle_receipts[0].forecast_sha256 is not None
    assert outcome.receipt.weighted_median_forecast_sha256 is not None
    assert outcome.receipt.target_sha256 is None
    assert outcome.receipt.source_kind == "synthetic-fixture-only"
    assert outcome.receipt.track_id == source.SYNTHETIC_FIXTURE_TRACK_ID


def test_one_failed_particle_fails_whole_ensemble_without_target_materialization() -> (
    None
):
    data = _synthetic_view()
    ensemble = CascadedTanksForecastEnsemble(
        (_particle(p=100.0, weight=0.2), _particle(p=0.01, weight=0.8))
    )
    assert not hasattr(data, "development_y_est")
    assert not hasattr(data, "u_val")
    assert not hasattr(data, "y_val")

    outcome = _forecast_synthetic_fixture(
        data,
        ensemble,
        limits=_limits(max_magnitude=100.0),
    )

    assert outcome.status == "terminal_failure"
    assert outcome.forecast is None
    assert outcome.receipt.weighted_median_forecast_sha256 is None
    assert outcome.receipt.target_sha256 is None
    assert len(outcome.receipt.failures) == 1
    assert outcome.receipt.failures[0].particle_index == 0
    assert outcome.receipt.failures[0].phase == "training"
    assert outcome.receipt.failures[0].step_index == 766
    assert outcome.receipt.particle_receipts[0].forecast_sha256 is None
    assert outcome.receipt.particle_receipts[1].forecast_sha256 is not None
    assert len(outcome.receipt.particle_receipts) == 2


def test_synthetic_forecast_replays_exactly_and_binds_hashes() -> None:
    data = _synthetic_view()
    ensemble = CascadedTanksForecastEnsemble((_particle(),))

    first = _forecast_synthetic_fixture(data, ensemble, limits=_limits())
    second = _forecast_synthetic_fixture(data, ensemble, limits=_limits())

    assert first == second
    assert first.forecast is not None
    assert first.receipt.source_receipt_sha256
    assert first.receipt.ensemble_parameter_sha256
    assert first.receipt.code_sha256
    assert first.receipt.particle_receipts[0].parameter_sha256
    assert first.receipt.weighted_median_forecast_sha256
    assert first.receipt.particle_receipts[0].model == "S0"
    assert first.receipt.model_alignment == ((0, "S0"),)


def test_production_entrypoint_loads_by_path_and_rejects_fixture_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path = tmp_path / "never-opened.zip"
    calls: list[Path] = []
    fixture = _synthetic_view()

    def fixture_loader(path: str | Path):
        calls.append(Path(path))
        return fixture

    monkeypatch.setattr(source, "load_development_data", fixture_loader)

    with pytest.raises(CascadedTanksForecastError, match="official loader"):
        run_cascaded_tanks_development_forecast(
            archive_path,
            CascadedTanksForecastEnsemble((_particle(),)),
            limits=_limits(),
        )

    assert calls == [archive_path]


def test_ensemble_requires_immutable_normalized_nonnegative_weights() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        CascadedTanksForecastEnsemble((_particle(weight=0.9),))
    with pytest.raises(ValueError, match="nonnegative"):
        _particle(weight=-0.1)
    with pytest.raises(TypeError, match="immutable tuple"):
        CascadedTanksForecastEnsemble([_particle()])  # type: ignore[arg-type]

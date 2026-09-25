from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from core.real_data.silverbox import SAMPLE_COUNT, SCHROEDER_CSV_MEMBER, SNLS_CSV_MEMBER
from core.real_data.silverbox_controlled import (
    BLOCK_SOURCE_START,
    BLOCK_SOURCE_STOP,
    STATE_INITIALIZATION_LENGTH,
    TRAIN_SOURCE_START,
    TRAIN_SOURCE_STOP,
    TRAIN_WINDOW_LENGTH,
    TRAIN_WINDOW_PREDICTION_LENGTH,
    TRAIN_WINDOW_RELATIVE_STARTS,
    UNUSED_GAP_SOURCE_START,
    UNUSED_GAP_SOURCE_STOP,
    VALIDATION_SOURCE_START,
    VALIDATION_SOURCE_STOP,
    SilverboxControlledDevelopment,
    load_silverbox_controlled_development,
    simulate_controlled_series,
)


def _write_index_coded_archive(path: Path) -> None:
    rows = ["V1,V2,\n"]
    rows.extend(f"{index / 10:.8f},{index / 20:.8f},\n" for index in range(SAMPLE_COUNT))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SNLS_CSV_MEMBER, "".join(rows) + "\n")
        archive.writestr(SCHROEDER_CSV_MEMBER, "Ovld2,Ovld1,V1,V2,\n")
        for name in (
            "SilverboxFiles/SNLS80mV.mat",
            "SilverboxFiles/Schroeder80mV.mat",
            "SilverboxFiles/README.txt",
            "SilverboxFiles/README.m",
        ):
            archive.writestr(name, "placeholder")


@pytest.fixture(scope="module")
def development(tmp_path_factory: pytest.TempPathFactory) -> SilverboxControlledDevelopment:
    archive = tmp_path_factory.mktemp("silverbox-controlled") / "SilverboxFiles.zip"
    _write_index_coded_archive(archive)
    return load_silverbox_controlled_development(archive)


def test_fixed_source_indices_roles_gap_and_50_sample_initializers(development: SilverboxControlledDevelopment) -> None:
    assert development.block_source_range == (BLOCK_SOURCE_START, BLOCK_SOURCE_STOP) == (40_650, 48_842)
    assert development.train_source_range == (TRAIN_SOURCE_START, TRAIN_SOURCE_STOP) == (40_650, 46_794)
    assert development.unused_gap_source_range == (UNUSED_GAP_SOURCE_START, UNUSED_GAP_SOURCE_STOP) == (46_794, 47_050)
    assert development.validation_source_range == (VALIDATION_SOURCE_START, VALIDATION_SOURCE_STOP) == (47_050, 48_842)
    assert development.sampling_time == 1.0 / 610.35

    expected_starts = tuple(TRAIN_SOURCE_START + offset for offset in TRAIN_WINDOW_RELATIVE_STARTS)
    assert tuple(window.source_start for window in development.train_windows) == expected_starts
    for start, window in zip(expected_starts, development.train_windows):
        assert window.role == "train"
        assert window.source_stop - window.source_start == TRAIN_WINDOW_LENGTH == 256
        assert window.initialization_y.size == STATE_INITIALIZATION_LENGTH == 50
        assert window.target_y.size == TRAIN_WINDOW_PREDICTION_LENGTH == 206
        np.testing.assert_array_equal(window.input_u, np.arange(start, start + 256) / 10.0)
        np.testing.assert_array_equal(window.initialization_y, np.arange(start, start + 50) / 20.0)
        np.testing.assert_array_equal(window.target_y, np.arange(start + 50, start + 256) / 20.0)
        assert window.source_start >= TRAIN_SOURCE_START
        assert window.source_stop <= TRAIN_SOURCE_STOP

    validation = development.validation
    assert validation.role == "validation"
    assert validation.source_stop - validation.source_start == 1_792
    assert validation.initialization_y.size == 50
    assert validation.target_y.size == 1_742
    np.testing.assert_array_equal(validation.input_u, np.arange(VALIDATION_SOURCE_START, VALIDATION_SOURCE_STOP) / 10.0)
    np.testing.assert_array_equal(
        validation.initialization_y,
        np.arange(VALIDATION_SOURCE_START, VALIDATION_SOURCE_START + 50) / 20.0,
    )
    np.testing.assert_array_equal(
        validation.target_y,
        np.arange(VALIDATION_SOURCE_START + 50, VALIDATION_SOURCE_STOP) / 20.0,
    )
    assert all(window.source_stop <= UNUSED_GAP_SOURCE_START for window in development.train_windows)
    assert validation.source_start == UNUSED_GAP_SOURCE_STOP


def test_simulator_receives_measured_input_initializer_and_native_interval_without_targets(
    development: SilverboxControlledDevelopment,
) -> None:
    series = development.train_windows[0]
    parameters = object()
    received: dict[str, object] = {}

    def simulator(**kwargs):
        received.update(kwargs)
        return np.full(series.prediction_count, 1.0e12)

    result = simulate_controlled_series(series, simulator, parameters)
    assert result.status == "success"
    assert result.failure_mode is None
    assert set(received) == {"input_u", "initialization_y", "parameters", "sampling_time"}
    np.testing.assert_array_equal(received["input_u"], series.input_u)
    np.testing.assert_array_equal(received["initialization_y"], series.initialization_y)
    assert received["parameters"] is parameters
    assert received["sampling_time"] == 1.0 / 610.35
    assert not received["input_u"].flags.writeable
    assert not received["initialization_y"].flags.writeable
    assert np.all(result.trajectory == 1.0e12)
    assert not result.trajectory.flags.writeable


def test_validation_simulator_gets_only_the_50_sample_initializer_and_forcing(
    development: SilverboxControlledDevelopment,
) -> None:
    validation = development.validation
    received: dict[str, object] = {}

    def simulator(**kwargs):
        received.update(kwargs)
        return np.zeros(validation.prediction_count)

    result = simulate_controlled_series(validation, simulator, parameters=None)
    assert result.status == "success"
    assert len(received["input_u"]) == 1_792
    assert len(received["initialization_y"]) == 50
    assert result.trajectory.shape == (1_742,)
    assert "target_y" not in received


def test_simulator_exceptions_bad_shapes_and_nonfinite_outputs_are_failures(
    development: SilverboxControlledDevelopment,
) -> None:
    series = development.train_windows[0]

    def raises(**kwargs):
        raise RuntimeError("integrator diverged")

    exception_result = simulate_controlled_series(series, raises, parameters=None)
    assert exception_result.status == "failed"
    assert exception_result.trajectory is None
    assert exception_result.failure_mode == "simulator_exception"
    assert "integrator diverged" in exception_result.failure_detail

    nonfinite_result = simulate_controlled_series(
        series,
        lambda **kwargs: np.full(series.prediction_count, np.inf),
        parameters=None,
    )
    assert nonfinite_result.status == "failed"
    assert nonfinite_result.trajectory is None
    assert nonfinite_result.failure_mode == "nonfinite_output"

    wrong_shape_result = simulate_controlled_series(
        series,
        lambda **kwargs: np.zeros(series.sample_count),
        parameters=None,
    )
    assert wrong_shape_result.status == "failed"
    assert wrong_shape_result.trajectory is None
    assert wrong_shape_result.failure_mode == "wrong_output_shape"


def test_development_contract_does_not_expose_sealed_test_outputs(development: SilverboxControlledDevelopment) -> None:
    assert not hasattr(development, "test_candidates")
    assert not hasattr(development, "test_records")
    assert not hasattr(development, "final")
    assert not hasattr(development, "gap")
    assert all(not hasattr(series, "output_y") for series in (*development.train_windows, development.validation))

"""Artificial-only candidate surface for Phase 68 TensorFlow Speech Commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SAMPLES = 3_200
CLASS_COUNT = 35
PCM_BOUNDS = (-32_768, 32_767)


@dataclass(frozen=True)
class SpeechCommandsWaveform:
    samples: np.ndarray

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=float)
        if (
            samples.shape != (SAMPLES,)
            or not np.all(np.isfinite(samples))
            or np.any(samples < PCM_BOUNDS[0])
            or np.any(samples > PCM_BOUNDS[1])
            or np.any(samples != np.rint(samples))
        ):
            raise ValueError("Speech Commands candidate must be finite integral 3200-sample PCM")
        object.__setattr__(self, "samples", samples.astype(np.int16, copy=True))


def waveform_from_payload(payload: Mapping[str, object]) -> SpeechCommandsWaveform:
    if set(payload) != {"samples"}:
        raise ValueError("Speech Commands payload may contain only samples")
    return SpeechCommandsWaveform(np.asarray(payload["samples"], dtype=float))


def predict_independent(
    waveforms: list[SpeechCommandsWaveform],
    predictor: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    probabilities = []
    for waveform in waveforms:
        value = np.asarray(predictor(0.0, waveform.samples.copy()), dtype=float)
        if (
            value.shape != (CLASS_COUNT,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0)
            or np.any(value > 1)
            or not np.isclose(value.sum(), 1.0)
        ):
            raise ValueError("invalid Speech Commands probabilities")
        probabilities.append(value)
    return np.asarray(probabilities)

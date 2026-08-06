#!/usr/bin/env python3
"""Run the injected-only Phase 68 Speech Commands adapter mechanics smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.speech_commands import (
    CLASS_COUNT,
    PCM_BOUNDS,
    SAMPLES,
    SpeechCommandsWaveform,
    predict_independent,
    waveform_from_payload,
)


OUT = Path("artifacts/evaluations/phase68_tensorflow_speech_commands_v0_02_raw_waveform_adapter_mechanics_smoke_20260803")


def rejected(payload: dict[str, object]) -> bool:
    try:
        waveform_from_payload(payload)
    except ValueError:
        return True
    return False


def run(output_dir: Path = OUT) -> dict[str, object]:
    generator = np.random.default_rng(2_068_001)
    waveforms = [
        SpeechCommandsWaveform(generator.integers(PCM_BOUNDS[0], PCM_BOUNDS[1] + 1, SAMPLES))
        for _ in range(3)
    ]
    states: list[float] = []
    uniform = np.full(CLASS_COUNT, 1.0 / CLASS_COUNT)
    probabilities = predict_independent(waveforms, lambda state, samples: states.append(state) or uniform)
    reordered = predict_independent(waveforms[::-1], lambda state, samples: uniform)
    samples = waveforms[0].samples
    controls = {
        "zero_reset": states == [0.0, 0.0, 0.0],
        "reorder_invariant": bool(np.array_equal(probabilities, reordered[::-1])),
        "target_rejected": rejected({"samples": samples, "label": 0}),
        "file_rejected": rejected({"samples": samples, "file": "testing_list.txt"}),
        "split_rejected": rejected({"samples": samples, "split": "external"}),
        "row_rejected": rejected({"samples": samples, "row": 0}),
        "path_rejected": rejected({"samples": samples, "path": "yes/0123abcd_nohash_0.wav"}),
        "speaker_rejected": rejected({"samples": samples, "speaker_hash": "0123abcd"}),
        "utterance_rejected": rejected({"samples": samples, "utterance_index": 0}),
        "suffix_rejected": rejected({"samples": samples, "suffix": samples}),
        "duplicate_group_rejected": rejected({"samples": samples, "duplicate_group": 0}),
        "finite_probabilities": bool(
            np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)
        ),
    }
    result = {
        "phase": 68,
        "status": "passed_adapter_mechanics_only" if all(controls.values()) else "closed_negative_adapter_mechanics_failure",
        "uses_observed_values": False,
        "uses_observed_labels": False,
        "uses_source_statistics": False,
        "controls": controls,
        "abc_smc_calls": 0,
        "llm_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

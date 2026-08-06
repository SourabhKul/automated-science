from __future__ import annotations

import numpy as np

from core.real_data.uci_spoken_arabic_digit import DIGIT_LABELS, MFCC_COEFFICIENTS, PREFIX_FRAMES, ArabicDigitInjectedRecord, frozen_speakers, partition_injected_records, predict_independent_prefixes, prefix_from_payload


SPLIT = "data/real/uci_spoken_arabic_digit/source_speaker_split.json"


def main() -> None:
    groups = frozen_speakers(SPLIT)
    records = [ArabicDigitInjectedRecord(source, speaker, split, f"a-{source}-{speaker}", np.ones((PREFIX_FRAMES, MFCC_COEFFICIENTS)), np.zeros((1, MFCC_COEFFICIENTS)), 0) for split in groups for source, speaker in groups[split]]
    assert {name: len(items) for name, items in partition_injected_records(records, SPLIT).items()} == {"train": 44, "selection": 22, "external": 22}
    candidate = records[0].candidate_input()
    assert candidate.frames.shape == (PREFIX_FRAMES, MFCC_COEFFICIENTS)
    assert not hasattr(candidate, "artificial_label") and not hasattr(candidate, "speaker_id")
    assert np.array_equal(prefix_from_payload({"frames": candidate.frames}).frames, candidate.frames)
    for field in ("label", "speaker_id", "source_file", "block_index", "suffix"):
        try: prefix_from_payload({"frames": candidate.frames, field: 0})
        except ValueError: pass
        else: raise AssertionError(f"{field} must be rejected")
    probabilities = predict_independent_prefixes([candidate], lambda state, frames: np.full(len(DIGIT_LABELS), 1 / len(DIGIT_LABELS)))
    assert np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1)
    print("SUCCESS: Spoken Arabic Digit injected adapter is frame-only, reset-safe, isolated, and split-locked")


if __name__ == "__main__": main()

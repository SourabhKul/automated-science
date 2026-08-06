from __future__ import annotations

import numpy as np

from scripts.run_uci_spoken_arabic_digit_source_gate import _source_speaker, parse_blocks


def main() -> None:
    raw = ((" ".join(str(value) for value in range(13)) + "\n") * 4 + "\n").encode()
    blocks = parse_blocks(raw, "artificial")
    assert len(blocks) == 1 and blocks[0].shape == (4, 13) and np.all(np.isfinite(blocks[0]))
    assert _source_speaker(0) == 1 and _source_speaker(329) == 33 and _source_speaker(330) == 34 and _source_speaker(659) == 66
    print("SUCCESS: Spoken Arabic Digit source block grammar and speaker-order mapping are fixed")


if __name__ == "__main__":
    main()

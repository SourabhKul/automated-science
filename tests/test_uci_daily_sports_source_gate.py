from __future__ import annotations

import numpy as np

from scripts.run_uci_daily_sports_source_gate import expected_members, parse_segment


def main() -> None:
    assert len(expected_members()) == 9120
    raw = ("1,2," + ",".join("3" for _ in range(43)) + "\n") * 125
    values = parse_segment(raw.encode(), "data/a01/p1/s01.txt")
    assert values.shape == (125, 45) and np.all(np.isfinite(values))
    try:
        parse_segment(b"1,2,3\n", "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid segment was accepted")
    print("SUCCESS: Daily Sports source-gate grammar and raw shape checks are fixed")


if __name__ == "__main__":
    main()

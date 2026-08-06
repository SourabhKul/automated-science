from __future__ import annotations

import numpy as np

from core.real_data.parallel_wiener_hammerstein import ParallelWHInputRecord, build_estimation_input_split
from scripts.run_parallel_wiener_hammerstein_synthetic_controls_smoke import run


def main() -> None:
    assert "output_y" not in ParallelWHInputRecord.__dataclass_fields__
    # The actual archive run verifies native input parsing; this fixture verifies
    # the deterministic 16/4 split is available without measured outputs.
    records = []
    for level, amplitude in enumerate((0.1, 0.325, 0.55, 0.775, 1.0), start=1):
        for index in range(1, 21):
            records.append(ParallelWHInputRecord(f"est_{level}_{index}", "estimation", "fixed_amplitude_multisine", amplitude, index, np.linspace(-1, 1, 128), "synthetic"))
    records.extend(ParallelWHInputRecord(f"val_{index}", "official_validation", "fixed_amplitude_multisine", None, None, np.linspace(-1, 1, 128), "synthetic") for index in range(6))
    split = build_estimation_input_split(records)
    assert {key: len(value) for key, value in split.items()} == {"train": 80, "selection": 20, "official_validation": 6}
    result = run()
    assert result["input_only_records"] and result["passed"]
    assert result["gates"]["pairing_pass"] and result["gates"]["time_order_pass"]
    assert result["gates"]["growing_amplitude_warmup_pass"]
    print("SUCCESS: Parallel WH synthetic controls are output-isolated and input-specific")


if __name__ == "__main__":
    main()

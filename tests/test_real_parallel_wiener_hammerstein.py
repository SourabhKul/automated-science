from __future__ import annotations

import numpy as np

from core.real_data.parallel_wiener_hammerstein import (
    ParallelWHRecord,
    build_estimation_split,
    evaluate_parallel_wiener_hammerstein_baselines,
    load_parallel_wiener_hammerstein,
)
from scripts.legacy.real_parallel_wiener_hammerstein_data_loader import DEFAULT_RAW_ARCHIVE


def _synthetic_record(record_id: str, split: str, realization: int | None, *, amplitude: float | None = 0.1) -> ParallelWHRecord:
    u = np.sin(np.linspace(0.0, 4.0, 128))
    y = np.empty_like(u)
    y[0] = 0.0
    for index in range(1, len(y)):
        y[index] = 0.9 * y[index - 1] + 0.1 * u[index]
    return ParallelWHRecord(record_id, split, "fixed_amplitude_multisine", amplitude, realization, u, y, "synthetic")


def main() -> None:
    records = load_parallel_wiener_hammerstein(DEFAULT_RAW_ARCHIVE)
    assert len(records) == 106
    assert len([record for record in records if record.source_split == "estimation"]) == 100
    assert len([record for record in records if record.source_split == "official_validation"]) == 6
    assert all(record.input_u.shape == (32768,) and np.all(np.isfinite(record.output_y)) for record in records)
    split = build_estimation_split(records)
    assert {key: len(value) for key, value in split.items()} == {"train": 80, "selection": 20, "official_validation": 6}

    small_records = []
    for level, amplitude in enumerate((0.1, 0.325, 0.55, 0.775, 1.0), start=1):
        small_records.extend(_synthetic_record(f"estimation_{level}_{index}", "estimation", index, amplitude=amplitude) for index in range(1, 21))
    small_records.extend(_synthetic_record(f"validation_{level}", "official_validation", None, amplitude=amplitude) for level, amplitude in enumerate((0.1, 0.325, 0.55, 0.775, 1.0), start=1))
    small_records.append(_synthetic_record("validation_arrow", "official_validation", None, amplitude=None))
    baselines = evaluate_parallel_wiener_hammerstein_baselines(build_estimation_split(small_records))
    assert set(baselines) == {"persistence", "regularized_arx", "stable_first_order", "fixed_parallel_wiener_hammerstein"}
    for baseline in baselines.values():
        assert np.isfinite(baseline["selection"]["aggregate"]["nrmse"])
        assert np.isfinite(baseline["official_validation"]["aggregate"]["nrmse"])
    print("SUCCESS: Parallel Wiener-Hammerstein loader, split, and causal baseline contract are stable")


if __name__ == "__main__":
    main()

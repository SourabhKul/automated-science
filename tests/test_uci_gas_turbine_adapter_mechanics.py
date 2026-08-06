from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.real_data.uci_gas_turbine import (
    CANDIDATE_FIELDS,
    UCIGasTurbineInjectedRecord,
    candidate_input_from_payload,
    partition_injected_records,
    predict_independent_rows,
)


SPLIT_PATH = Path("data/real/uci_gas_turbine/source_year_split.json")


def _records() -> list[UCIGasTurbineInjectedRecord]:
    split = json.loads(SPLIT_PATH.read_text())
    year_split = [(int(year), "train") for year in split["train_years"]]
    year_split.extend([(int(split["selection_year"]), "selection"), (int(split["external_year"]), "external")])
    return [
        UCIGasTurbineInjectedRecord(
            year=year,
            split=split_name,
            artificial_row_key=f"artificial-{year}",
            raw_source_units=np.arange(len(CANDIDATE_FIELDS), dtype=float) + float(index),
            artificial_nox_mg_m3=30.0 + float(index),
            artificial_co_mg_m3=2.0 + float(index),
        )
        for index, (year, split_name) in enumerate(year_split)
    ]


def main() -> None:
    records = _records()
    partitioned = partition_injected_records(records, SPLIT_PATH)
    assert {name: len(items) for name, items in partitioned.items()} == {"train": 3, "selection": 1, "external": 1}
    candidate = records[0].candidate_input()
    assert candidate.raw_source_units.shape == (len(CANDIDATE_FIELDS),)
    assert not hasattr(candidate, "artificial_nox_mg_m3")
    assert not hasattr(candidate, "artificial_co_mg_m3")
    assert not hasattr(candidate, "year")
    assert np.array_equal(candidate.raw_source_units, records[0].raw_source_units)

    assert np.array_equal(candidate_input_from_payload({"raw_source_units": candidate.raw_source_units}).raw_source_units, candidate.raw_source_units)
    for forbidden in ("NOX", "CO", "year", "row_index", "duplicate_fingerprint"):
        try:
            candidate_input_from_payload({"raw_source_units": candidate.raw_source_units, forbidden: 1})
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError(f"{forbidden} must be rejected from the candidate payload")

    states: list[float] = []
    predictions = predict_independent_rows(
        [record.candidate_input() for record in records],
        lambda state, values: states.append(state) or 25.0 + float(np.sum(values)) * 0.01,
    )
    assert states == [0.0] * len(records)
    assert np.all(np.isfinite(predictions)) and np.all((predictions >= 0.0) & (predictions <= 150.0))
    try:
        predict_independent_rows([candidate], lambda state, values: float("nan"))
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite UCI gas-turbine prediction must fail")

    wrong = list(records)
    first = wrong[0]
    wrong[0] = UCIGasTurbineInjectedRecord(
        first.year, "external", first.artificial_row_key, first.raw_source_units, first.artificial_nox_mg_m3, first.artificial_co_mg_m3
    )
    try:
        partition_injected_records(wrong, SPLIT_PATH)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("cross-split injected row must fail")
    print("SUCCESS: UCI gas-turbine injected adapter is raw-unit-only, reset-safe, target-isolated, and year-split-locked")


if __name__ == "__main__":
    main()

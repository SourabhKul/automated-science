from __future__ import annotations

from pathlib import Path

from core.real_data.camels_us import deterministic_camels_split, eligible_camels_basins, inspect_camels_us_archive, validate_camels_split


ARCHIVE = Path("data/real/camels_us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip")


def main() -> None:
    inspection = inspect_camels_us_archive(ARCHIVE)
    assert inspection["gauge_count"] == 671
    assert not inspection["gauge_missing_flow"]
    assert not inspection["gauge_missing_daymet"]
    assert inspection["raw_schema"]["forcing"]["precipitation"] == "prcp(mm/day)"
    assert inspection["raw_schema"]["streamflow"]["missing_value"] == -999.0
    eligible = eligible_camels_basins(inspection)
    assert len(eligible) >= 64
    assert all(row["aligned_daily_grid"] and row["complete_hydrologic_years"] >= 8 for row in eligible)
    split = deterministic_camels_split(eligible)
    validate_camels_split(split, eligible)
    assert deterministic_camels_split(eligible) == split
    malformed = {name: list(values) for name, values in split.items()}
    malformed["selection"][0] = malformed["train"][0]
    try:
        validate_camels_split(malformed, eligible)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping CAMELS split must fail")
    print("SUCCESS: CAMELS archive source-quality and static basin-split contracts are stable")


if __name__ == "__main__":
    main()

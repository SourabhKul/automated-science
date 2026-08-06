from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np


CAMELS_ROOT = "basin_dataset_public_v1p2"
CAMELS_FORCING_HEADER = (
    "Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) "
    "tmax(C) tmin(C) vp(Pa)"
)
CAMELS_SPLIT_SEED = "phase28_camels_us_20260721"
HYDROLOGIC_YEAR_START_MONTH = 10
MIN_COMPLETE_HYDROLOGIC_YEARS = 8
RUNOFF_MM_DAY_PER_CFS_KM2 = 2.4465755455488005
CAMELS_SPLIT_COUNTS = {"train": 36, "selection": 12, "external": 16}


@dataclass(frozen=True)
class CAMELSBasinQuality:
    basin_id: str
    huc_02: str
    drainage_area_km2: float
    forcing_days: int
    flow_days: int
    aligned_daily_grid: bool
    complete_hydrologic_years: int
    missing_flow_days: int
    estimated_flow_days: int
    censored_low_flow_days: int


@dataclass(frozen=True)
class CAMELSInputEpisode:
    """One complete hydrologic-year, causal forcing-only episode."""

    dates: tuple[date, ...]
    input_u: np.ndarray

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        values = np.asarray(self.input_u, dtype=float)
        if len(dates) not in {365, 366} or values.shape != (len(dates), 2):
            raise ValueError("CAMELS input episode must have daily two-channel forcing for one hydrologic year")
        if dates[0].month != HYDROLOGIC_YEAR_START_MONTH or dates[0].day != 1:
            raise ValueError("CAMELS input episode must start on October 1")
        if any(right - left != timedelta(days=1) for left, right in zip(dates, dates[1:])):
            raise ValueError("CAMELS input episode dates must be contiguous")
        if dates[-1] != date(dates[0].year + 1, HYDROLOGIC_YEAR_START_MONTH, 1) - timedelta(days=1):
            raise ValueError("CAMELS input episode must end on September 30")
        if not np.all(np.isfinite(values)):
            raise ValueError("CAMELS input episode must be finite")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "input_u", values.copy())


@dataclass(frozen=True)
class CAMELSOutcomeEpisode:
    """Outcome-bearing diagnostic wrapper that never reaches a causal candidate."""

    basin_id: str
    split: str
    input_episode: CAMELSInputEpisode
    raw_flow_cfs: np.ndarray
    drainage_area_km2: float

    def __post_init__(self) -> None:
        flow = np.asarray(self.raw_flow_cfs, dtype=float)
        if self.split not in CAMELS_SPLIT_COUNTS or len(self.basin_id) != 8:
            raise ValueError("CAMELS outcome episode has an invalid basin ID or split")
        if flow.shape != (len(self.input_episode.dates),) or not np.all(np.isfinite(flow)) or np.any(flow < 0):
            raise ValueError("CAMELS outcome episode needs finite nonnegative raw flow")
        if not np.isfinite(self.drainage_area_km2) or self.drainage_area_km2 <= 0:
            raise ValueError("CAMELS outcome episode needs positive drainage area")
        object.__setattr__(self, "raw_flow_cfs", flow.copy())

    @property
    def target_mm_day(self) -> np.ndarray:
        return convert_cfs_to_mm_day(self.raw_flow_cfs, self.drainage_area_km2)

    def causal_input(self) -> CAMELSInputEpisode:
        return self.input_episode


def convert_cfs_to_mm_day(flow_cfs: np.ndarray, drainage_area_km2: float) -> np.ndarray:
    """Convert daily mean cfs to basin-depth runoff without modifying the input."""
    flow = np.asarray(flow_cfs, dtype=float)
    if flow.ndim != 1 or not np.all(np.isfinite(flow)) or np.any(flow < 0):
        raise ValueError("CAMELS cfs conversion requires finite nonnegative one-dimensional flow")
    if not np.isfinite(drainage_area_km2) or drainage_area_km2 <= 0:
        raise ValueError("CAMELS cfs conversion requires positive drainage area")
    return flow * (RUNOFF_MM_DAY_PER_CFS_KM2 / float(drainage_area_km2))


def partition_camels_outcome_episodes(
    episodes: list[CAMELSOutcomeEpisode], split: dict[str, list[str]]
) -> dict[str, list[CAMELSOutcomeEpisode]]:
    """Verify the frozen basin partition while keeping targets outside candidate inputs."""
    partition_ids = {name: [str(basin_id) for basin_id in split[name]] for name in CAMELS_SPLIT_COUNTS}
    if {name: len(values) for name, values in partition_ids.items()} != CAMELS_SPLIT_COUNTS:
        raise ValueError("CAMELS frozen split must be 36/12/16")
    expected = {basin_id: name for name, basin_ids in partition_ids.items() for basin_id in basin_ids}
    if len(expected) != sum(CAMELS_SPLIT_COUNTS.values()):
        raise ValueError("CAMELS frozen split has duplicate basin IDs")
    partitioned = {name: [] for name in CAMELS_SPLIT_COUNTS}
    for episode in episodes:
        if expected.get(episode.basin_id) != episode.split:
            raise ValueError("CAMELS episode crosses the frozen basin split")
        partitioned[episode.split].append(episode)
    if {episode.basin_id for episode in episodes} != set(expected):
        raise ValueError("CAMELS episode set omits or adds a locked basin")
    return partitioned


def rollout_camels_causal_input(
    episode: CAMELSInputEpisode,
    step: Any,
) -> np.ndarray:
    """Run one reset, causal-only forcing episode with no target or identity surface."""
    state = 0.0
    output = np.empty(len(episode.dates), dtype=float)
    for index, current_u in enumerate(episode.input_u):
        next_state = float(step(float(state), np.asarray(current_u, dtype=float).copy()))
        if not np.isfinite(next_state) or next_state < 0:
            raise ValueError("CAMELS causal rollout emitted a non-finite or negative state")
        output[index] = next_state
        state = next_state
    return output


def load_camels_forcing_only_episodes(
    raw_archive: str | Path,
    split: dict[str, list[str]],
    *,
    hydrologic_years: tuple[int, ...] = tuple(range(1980, 1988)),
) -> dict[str, list[CAMELSInputEpisode]]:
    """Load fixed Daymet inputs only; this path never opens a streamflow member."""
    partition_ids = {name: [str(basin_id) for basin_id in split[name]] for name in CAMELS_SPLIT_COUNTS}
    if {name: len(values) for name, values in partition_ids.items()} != CAMELS_SPLIT_COUNTS:
        raise ValueError("CAMELS forcing-only loader needs the frozen 36/12/16 split")
    if not hydrologic_years or len(set(hydrologic_years)) != len(hydrologic_years):
        raise ValueError("CAMELS forcing-only loader needs distinct hydrologic years")
    result = {name: [] for name in CAMELS_SPLIT_COUNTS}
    with ZipFile(Path(raw_archive)) as archive:
        forcing_members = _member_by_basin(archive, "/basin_mean_forcing/daymet/", "_lump_cida_forcing_leap.txt")
        for split_name, basin_ids in partition_ids.items():
            for basin_id in basin_ids:
                member = forcing_members.get(basin_id)
                if member is None:
                    raise ValueError("CAMELS locked basin lacks a Daymet forcing member")
                rows: dict[date, tuple[float, float]] = {}
                with archive.open(member) as raw:
                    [next(raw) for _ in range(3)]
                    if next(raw).decode("utf-8").strip() != CAMELS_FORCING_HEADER:
                        raise ValueError("CAMELS forcing-only header contract failed")
                    for line in raw:
                        fields = line.decode("utf-8").split()
                        if len(fields) != 11:
                            raise ValueError("CAMELS forcing-only row column-count contract failed")
                        current = date(int(fields[0]), int(fields[1]), int(fields[2]))
                        precipitation = float(fields[5])
                        mean_temperature = (float(fields[8]) + float(fields[9])) / 2.0
                        if not np.isfinite(precipitation) or not np.isfinite(mean_temperature) or current in rows:
                            raise ValueError("CAMELS forcing-only value/date contract failed")
                        rows[current] = (precipitation, mean_temperature)
                for start_year in hydrologic_years:
                    start = date(start_year, HYDROLOGIC_YEAR_START_MONTH, 1)
                    end = date(start_year + 1, HYDROLOGIC_YEAR_START_MONTH, 1) - timedelta(days=1)
                    expected = tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))
                    if any(current not in rows for current in expected):
                        raise ValueError("CAMELS forcing-only episode is incomplete")
                    result[split_name].append(CAMELSInputEpisode(expected, np.asarray([rows[current] for current in expected], dtype=float)))
    expected_counts = {name: len(partition_ids[name]) * len(hydrologic_years) for name in CAMELS_SPLIT_COUNTS}
    if {name: len(values) for name, values in result.items()} != expected_counts:
        raise ValueError("CAMELS forcing-only loader episode count contract failed")
    return result


def load_camels_outcome_episodes(
    raw_archive: str | Path,
    basin_ids: list[str],
    split: str,
    *,
    episode_count: int = 8,
) -> list[CAMELSOutcomeEpisode]:
    """Load earliest complete source episodes for one allowed split only."""
    if split not in CAMELS_SPLIT_COUNTS or episode_count <= 0 or len(set(basin_ids)) != len(basin_ids):
        raise ValueError("CAMELS outcome loader split or basin contract failed")
    episodes: list[CAMELSOutcomeEpisode] = []
    with ZipFile(Path(raw_archive)) as archive:
        gauges = _gauge_metadata(archive)
        forcing_members = _member_by_basin(archive, "/basin_mean_forcing/daymet/", "_lump_cida_forcing_leap.txt")
        flow_members = _member_by_basin(archive, "/usgs_streamflow/", "_streamflow_qc.txt")
        for basin_id in basin_ids:
            if basin_id not in gauges or basin_id not in forcing_members or basin_id not in flow_members:
                raise ValueError("CAMELS locked basin source join failed")
            forcing: dict[date, tuple[float, float]] = {}
            with archive.open(forcing_members[basin_id]) as raw:
                [next(raw) for _ in range(3)]
                if next(raw).decode("utf-8").strip() != CAMELS_FORCING_HEADER:
                    raise ValueError("CAMELS outcome loader forcing header contract failed")
                for line in raw:
                    fields = line.decode("utf-8").split()
                    if len(fields) != 11:
                        raise ValueError("CAMELS outcome loader forcing row contract failed")
                    current = date(int(fields[0]), int(fields[1]), int(fields[2]))
                    values = (float(fields[5]), (float(fields[8]) + float(fields[9])) / 2.0)
                    if current in forcing or not all(np.isfinite(values)):
                        raise ValueError("CAMELS outcome loader forcing value contract failed")
                    forcing[current] = values
            flow: dict[date, float] = {}
            with archive.open(flow_members[basin_id]) as raw:
                for line in raw:
                    fields = line.decode("utf-8").split()
                    if len(fields) != 6 or fields[0] != basin_id:
                        raise ValueError("CAMELS outcome loader flow row contract failed")
                    current = date(int(fields[1]), int(fields[2]), int(fields[3]))
                    value, flag = float(fields[4]), fields[5]
                    if current in flow or flag not in {"A", "A:e", "M", "A:<"} or (flag == "M") != (value == -999.0):
                        raise ValueError("CAMELS outcome loader flow QC contract failed")
                    flow[current] = value
            years: list[tuple[tuple[date, ...], np.ndarray, np.ndarray]] = []
            for start_year in range(1980, 2014):
                start = date(start_year, HYDROLOGIC_YEAR_START_MONTH, 1)
                end = date(start_year + 1, HYDROLOGIC_YEAR_START_MONTH, 1) - timedelta(days=1)
                dates = tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))
                if any(current not in forcing or current not in flow or flow[current] < 0 for current in dates):
                    continue
                years.append((dates, np.asarray([forcing[current] for current in dates], dtype=float), np.asarray([flow[current] for current in dates], dtype=float)))
            if len(years) < episode_count:
                raise ValueError("CAMELS locked basin lacks enough complete source episodes")
            area = gauges[basin_id][1]
            episodes.extend(
                CAMELSOutcomeEpisode(basin_id, split, CAMELSInputEpisode(dates, inputs), raw_flow, area)
                for dates, inputs, raw_flow in years[:episode_count]
            )
    expected_count = len(basin_ids) * episode_count
    if len(episodes) != expected_count:
        raise ValueError("CAMELS outcome loader episode count contract failed")
    return episodes


def _member_by_basin(archive: ZipFile, fragment: str, suffix: str) -> dict[str, str]:
    members = {
        Path(member).name.split("_")[0]: member
        for member in archive.namelist()
        if member.startswith(f"{CAMELS_ROOT}/") and fragment in member and member.endswith(suffix)
    }
    if len(members) != len(
        [member for member in archive.namelist() if member.startswith(f"{CAMELS_ROOT}/") and fragment in member and member.endswith(suffix)]
    ):
        raise ValueError(f"duplicate CAMELS basin IDs in {fragment}")
    return members


def _gauge_metadata(archive: ZipFile) -> dict[str, tuple[str, float]]:
    member = f"{CAMELS_ROOT}/basin_metadata/gauge_information.txt"
    gauges: dict[str, tuple[str, float]] = {}
    with archive.open(member) as raw:
        header = next(raw).decode("utf-8").strip()
        if "GAGE_ID" not in header or "DRAINAGE AREA (KM^2)" not in header:
            raise ValueError("CAMELS gauge metadata header contract failed")
        for line in raw:
            fields = line.decode("utf-8").split()
            if len(fields) < 6:
                raise ValueError("CAMELS gauge metadata row contract failed")
            basin_id, huc_02, area = fields[1], fields[0], float(fields[-1])
            if len(basin_id) != 8 or basin_id in gauges or area <= 0:
                raise ValueError("CAMELS gauge metadata identity or drainage-area contract failed")
            gauges[basin_id] = (huc_02, area)
    return gauges


def _forcing_dates(archive: ZipFile, member: str) -> set[date]:
    dates: set[date] = set()
    with archive.open(member) as raw:
        preamble = [next(raw).decode("utf-8").strip() for _ in range(3)]
        if not all(preamble):
            raise ValueError("CAMELS forcing preamble contract failed")
        header = next(raw).decode("utf-8").strip()
        if header != CAMELS_FORCING_HEADER:
            raise ValueError("CAMELS forcing header contract failed")
        for line in raw:
            fields = line.decode("utf-8").split()
            if len(fields) != 11:
                raise ValueError("CAMELS forcing row column-count contract failed")
            year, month, day = map(int, fields[:3])
            values = [float(value) for value in fields[4:]]
            if not all(value == value and abs(value) != float("inf") for value in values):
                raise ValueError("CAMELS forcing contains a non-finite value")
            current = date(year, month, day)
            if current in dates:
                raise ValueError("CAMELS forcing has duplicate dates")
            dates.add(current)
    return dates


def _flow_dates(archive: ZipFile, member: str) -> tuple[set[date], set[date], int, int]:
    dates: set[date] = set()
    missing: set[date] = set()
    estimated = 0
    censored_low_flow = 0
    with archive.open(member) as raw:
        for line in raw:
            fields = line.decode("utf-8").split()
            if len(fields) != 6:
                raise ValueError("CAMELS streamflow row column-count contract failed")
            basin_id, year, month, day, flow, flag = fields
            if len(basin_id) != 8 or flag not in {"A", "A:e", "M", "A:<"}:
                raise ValueError("CAMELS streamflow ID or QC-flag contract failed")
            current = date(int(year), int(month), int(day))
            if current in dates:
                raise ValueError("CAMELS streamflow has duplicate dates")
            value = float(flow)
            if value != value or abs(value) == float("inf"):
                raise ValueError("CAMELS streamflow contains a non-finite value")
            if (flag == "M") != (value == -999.0):
                raise ValueError("CAMELS missing-streamflow sentinel/QC contract failed")
            if flag == "M":
                missing.add(current)
            elif flag == "A:e":
                estimated += 1
            elif flag == "A:<":
                censored_low_flow += 1
            dates.add(current)
    return dates, missing, estimated, censored_low_flow


def _complete_hydrologic_years(forcing_dates: set[date], flow_dates: set[date], missing_flow_dates: set[date]) -> int:
    complete = 0
    for start_year in range(1980, 2015):
        start = date(start_year, HYDROLOGIC_YEAR_START_MONTH, 1)
        end = date(start_year + 1, HYDROLOGIC_YEAR_START_MONTH, 1) - timedelta(days=1)
        current = start
        valid = True
        while current <= end:
            if current not in forcing_dates or current not in flow_dates or current in missing_flow_dates:
                valid = False
                break
            current += timedelta(days=1)
        complete += int(valid)
    return complete


def inspect_camels_us_archive(raw_archive: str | Path) -> dict[str, Any]:
    """Audit source streams only; no normalized forcing or observed-flow model surface."""
    archive_path = Path(raw_archive)
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        required = {
            f"{CAMELS_ROOT}/basin_metadata/gauge_information.txt",
            f"{CAMELS_ROOT}/readme_FIRST.txt",
            f"{CAMELS_ROOT}/readme_streamflow.txt",
            f"{CAMELS_ROOT}/readme_basin_mean_forcing.txt",
            f"{CAMELS_ROOT}/dataset_summary.txt",
        }
        if not required.issubset(members):
            raise ValueError("CAMELS archive lacks required source metadata")
        gauges = _gauge_metadata(archive)
        flow_members = _member_by_basin(archive, "/usgs_streamflow/", "_streamflow_qc.txt")
        forcing_members = _member_by_basin(archive, "/basin_mean_forcing/daymet/", "_lump_cida_forcing_leap.txt")
        shared = sorted(set(gauges) & set(flow_members) & set(forcing_members))
        quality: list[CAMELSBasinQuality] = []
        for basin_id in shared:
            forcing_dates = _forcing_dates(archive, forcing_members[basin_id])
            flow_dates, missing, estimated, censored_low_flow = _flow_dates(archive, flow_members[basin_id])
            huc_02, area = gauges[basin_id]
            quality.append(
                CAMELSBasinQuality(
                    basin_id=basin_id,
                    huc_02=huc_02,
                    drainage_area_km2=area,
                    forcing_days=len(forcing_dates),
                    flow_days=len(flow_dates),
                    aligned_daily_grid=forcing_dates == flow_dates,
                    complete_hydrologic_years=_complete_hydrologic_years(forcing_dates, flow_dates, missing),
                    missing_flow_days=len(missing),
                    estimated_flow_days=estimated,
                    censored_low_flow_days=censored_low_flow,
                )
            )
    return {
        "member_count": len(members),
        "gauge_count": len(gauges),
        "flow_member_count": len(flow_members),
        "daymet_member_count": len(forcing_members),
        "flow_not_in_gauge_metadata": sorted(set(flow_members) - set(gauges)),
        "daymet_not_in_gauge_metadata": sorted(set(forcing_members) - set(gauges)),
        "gauge_missing_flow": sorted(set(gauges) - set(flow_members)),
        "gauge_missing_daymet": sorted(set(gauges) - set(forcing_members)),
        "raw_schema": {
            "forcing": {"header": CAMELS_FORCING_HEADER, "date_columns": ["Year", "Mnth", "Day"], "precipitation": "prcp(mm/day)", "temperature": ["tmax(C)", "tmin(C)"]},
            "streamflow": {"columns": ["GAGEID", "Year", "Month", "Day", "Streamflow(cubic feet per second)", "QC_flag"], "missing_value": -999.0, "qc_flags": ["A", "A:e", "M", "A:<"], "readme_qc_flags": ["A", "A:e", "M"], "source_schema_note": "135 finite observations across four basin files use A:<, an undocumented additional source QC token retained in the audit ledger."},
            "drainage_area": {"source_member": f"{CAMELS_ROOT}/basin_metadata/gauge_information.txt", "units": "km^2"},
            "calendar": "daily Gregorian dates; forcing is 1980-01-01 through 2014-12-31 where source flow coverage permits alignment",
        },
        "basin_quality": [quality_row.__dict__ for quality_row in quality],
    }


def eligible_camels_basins(inspection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in inspection["basin_quality"]
        if row["aligned_daily_grid"] and row["complete_hydrologic_years"] >= MIN_COMPLETE_HYDROLOGIC_YEARS
    ]


def deterministic_camels_split(eligible: list[dict[str, Any]], seed: str = CAMELS_SPLIT_SEED) -> dict[str, list[str]]:
    """Choose the fixed cohort using only basin IDs after source-quality eligibility."""
    ids = sorted({str(row["basin_id"]) for row in eligible})
    if len(ids) != len(eligible) or len(ids) < 64:
        raise ValueError("CAMELS needs at least 64 unique eligible basins for a 36/12/16 split")
    ordered = sorted(ids, key=lambda basin_id: hashlib.sha256(f"{seed}:{basin_id}".encode()).hexdigest())
    return {
        "train": sorted(ordered[:36]),
        "selection": sorted(ordered[36:48]),
        "external": sorted(ordered[48:64]),
    }


def validate_camels_split(split: dict[str, list[str]], eligible: list[dict[str, Any]]) -> None:
    if {name: len(values) for name, values in split.items()} != {"train": 36, "selection": 12, "external": 16}:
        raise ValueError("CAMELS basin split must be 36/12/16")
    ids = [basin_id for values in split.values() for basin_id in values]
    eligible_ids = {str(row["basin_id"]) for row in eligible}
    if len(ids) != len(set(ids)) or not set(ids).issubset(eligible_ids):
        raise ValueError("CAMELS basin split leaks, duplicates, or includes an ineligible basin")

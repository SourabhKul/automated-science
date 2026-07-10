from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


CANONICAL_COLUMNS = [
    "cell_id",
    "cycle_index",
    "elapsed_time_h",
    "ambient_temperature_c",
    "capacity_ah",
    "soh",
    "rul_cycles",
    "mean_discharge_current_a",
    "mean_temperature_c",
    "mean_voltage_v",
    "re_ohm",
    "rct_ohm",
]


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        return obj[name]
    except Exception:
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    try:
        scalar = float(np.ravel(arr)[0])
    except (TypeError, ValueError):
        return default
    return scalar if np.isfinite(scalar) else default


def _nanmean(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return None
    return float(np.nanmean(arr))


def _iter_cycles(mat_payload: dict[str, Any]) -> list[Any]:
    if "cycle" in mat_payload:
        cycles = mat_payload["cycle"]
    else:
        candidates = [value for key, value in mat_payload.items() if not key.startswith("__")]
        if not candidates:
            raise ValueError("MATLAB file does not contain a cycle structure")
        root = candidates[0]
        cycles = _get_field(root, "cycle", root)
    return list(np.ravel(cycles))


def read_nasa_mat_cycles(path: str | Path, *, cell_id: str | None = None) -> pd.DataFrame:
    """Read NASA PCoE MATLAB cycle structures into discharge-cycle rows."""
    path = Path(path)
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    inferred_cell = cell_id or path.stem
    rows: list[dict[str, Any]] = []
    last_re = None
    last_rct = None
    discharge_index = 0
    elapsed_time_h = 0.0

    for cycle in _iter_cycles(payload):
        cycle_type = str(_get_field(cycle, "type", "")).lower()
        data = _get_field(cycle, "data")
        if cycle_type == "impedance":
            last_re = _as_float(_get_field(data, "Re"), last_re)
            last_rct = _as_float(_get_field(data, "Rct"), last_rct)
            continue
        if cycle_type != "discharge":
            continue

        capacity = _as_float(_get_field(data, "Capacity"))
        if capacity is None:
            continue
        time_s = np.asarray(_get_field(data, "Time", []), dtype=float)
        duration_h = float(np.nanmax(time_s) / 3600.0) if time_s.size else 0.0
        rows.append(
            {
                "cell_id": inferred_cell,
                "cycle_index": discharge_index,
                "elapsed_time_h": elapsed_time_h,
                "ambient_temperature_c": _as_float(_get_field(cycle, "ambient_temperature")),
                "capacity_ah": capacity,
                "mean_discharge_current_a": _nanmean(_get_field(data, "Current_measured")),
                "mean_temperature_c": _nanmean(_get_field(data, "Temperature_measured")),
                "mean_voltage_v": _nanmean(_get_field(data, "Voltage_measured")),
                "re_ohm": last_re,
                "rct_ohm": last_rct,
            }
        )
        discharge_index += 1
        elapsed_time_h += duration_h

    if not rows:
        raise ValueError(f"no discharge capacity rows found in {path}")
    return pd.DataFrame(rows)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return read_nasa_mat_cycles(path)
    return pd.read_csv(path, sep=None, engine="python", comment="#")


def normalize_battery_table(raw_path: str | Path, *, eol_soh: float = 0.7) -> pd.DataFrame:
    """Normalize NASA battery cycle-level data for capacity-fade modeling."""
    raw = _read_table(raw_path)
    required = {"cell_id", "cycle_index", "capacity_ah"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"missing required NASA battery columns: {missing}")

    normalized = pd.DataFrame()
    normalized["cell_id"] = raw["cell_id"].astype(str)
    normalized["cycle_index"] = pd.to_numeric(raw["cycle_index"], errors="coerce").astype("Int64")
    if "elapsed_time_h" in raw:
        normalized["elapsed_time_h"] = pd.to_numeric(raw["elapsed_time_h"], errors="coerce")
    else:
        normalized["elapsed_time_h"] = normalized["cycle_index"].astype(float)
    normalized["ambient_temperature_c"] = pd.to_numeric(raw.get("ambient_temperature_c"), errors="coerce")
    normalized["capacity_ah"] = pd.to_numeric(raw["capacity_ah"], errors="coerce")
    normalized["mean_discharge_current_a"] = pd.to_numeric(raw.get("mean_discharge_current_a"), errors="coerce")
    normalized["mean_temperature_c"] = pd.to_numeric(raw.get("mean_temperature_c"), errors="coerce")
    normalized["mean_voltage_v"] = pd.to_numeric(raw.get("mean_voltage_v"), errors="coerce")
    normalized["re_ohm"] = pd.to_numeric(raw.get("re_ohm"), errors="coerce")
    normalized["rct_ohm"] = pd.to_numeric(raw.get("rct_ohm"), errors="coerce")
    normalized = normalized.dropna(subset=["cell_id", "cycle_index", "capacity_ah"])
    if normalized.empty:
        raise ValueError("NASA battery table contains no finite capacity observations")

    normalized = normalized.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    soh_parts = []
    rul_parts = []
    for _, cell_frame in normalized.groupby("cell_id", sort=False):
        initial_capacity = float(cell_frame.iloc[0]["capacity_ah"])
        if initial_capacity <= 0:
            raise ValueError("initial capacity must be positive")
        soh = cell_frame["capacity_ah"].astype(float) / initial_capacity
        below_eol = cell_frame.loc[soh <= eol_soh, "cycle_index"].astype(int)
        eol_cycle = int(below_eol.iloc[0]) if not below_eol.empty else int(cell_frame["cycle_index"].max())
        soh_parts.append(soh)
        rul_parts.append((eol_cycle - cell_frame["cycle_index"].astype(int)).clip(lower=0))
    normalized["soh"] = pd.concat(soh_parts).sort_index()
    normalized["rul_cycles"] = pd.concat(rul_parts).sort_index().astype(int)
    return normalized[CANONICAL_COLUMNS]


def build_capacity_target(
    normalized: pd.DataFrame,
    *,
    cell_id: str | None = None,
    target_column: str = "soh",
) -> tuple[np.ndarray, np.ndarray]:
    if target_column not in {"soh", "capacity_ah"}:
        raise ValueError("target_column must be `soh` or `capacity_ah`")
    selected_cell = str(cell_id) if cell_id is not None else sorted(normalized["cell_id"].astype(str).unique())[0]
    cell = normalized[normalized["cell_id"].astype(str) == selected_cell].sort_values("cycle_index")
    if cell.empty:
        raise ValueError(f"unknown NASA battery cell_id: {selected_cell}")
    if len(cell) < 3:
        raise ValueError("capacity target requires at least three discharge cycles")
    time_points = cell["cycle_index"].to_numpy(dtype=float)
    target = cell[target_column].to_numpy(dtype=float).reshape(-1, 1)
    return time_points, target


def cell_summary(normalized: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    for cell_id, cell_frame in normalized.groupby("cell_id", sort=True):
        ordered = cell_frame.sort_values("cycle_index")
        summaries.append(
            {
                "cell_id": str(cell_id),
                "cycle_count": int(len(ordered)),
                "min_cycle_index": int(ordered["cycle_index"].min()),
                "max_cycle_index": int(ordered["cycle_index"].max()),
                "initial_capacity_ah": float(ordered.iloc[0]["capacity_ah"]),
                "final_capacity_ah": float(ordered.iloc[-1]["capacity_ah"]),
                "initial_soh": float(ordered.iloc[0]["soh"]),
                "final_soh": float(ordered.iloc[-1]["soh"]),
                "eol_reached": bool(np.any(ordered["soh"].to_numpy(dtype=float) <= 0.7)),
            }
        )
    return summaries


def build_cell_holdout_split(
    normalized: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    cells = np.array(sorted(normalized["cell_id"].astype(str).unique().tolist()), dtype=object)
    if len(cells) < 2:
        raise ValueError("cell holdout requires at least two cells")
    rng = np.random.default_rng(seed)
    shuffled = cells.copy()
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_fraction))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_cells = sorted(str(value) for value in shuffled[:train_count])
    test_cells = sorted(str(value) for value in shuffled[train_count:])
    return {
        "schema_version": 1,
        "status": "ok",
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "train_cells": train_cells,
        "test_cells": test_cells,
        "train_cycle_count": int(normalized[normalized["cell_id"].astype(str).isin(train_cells)].shape[0]),
        "test_cycle_count": int(normalized[normalized["cell_id"].astype(str).isin(test_cells)].shape[0]),
    }


def write_cell_holdout_split(
    normalized: pd.DataFrame,
    *,
    path: str | Path,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_cell_holdout_split(normalized, train_fraction=train_fraction, seed=seed)
    except ValueError as exc:
        payload = {
            "schema_version": 1,
            "status": "skipped",
            "reason": str(exc),
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "cell_count": int(normalized["cell_id"].astype(str).nunique()),
        }
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_within_cell_time_splits(
    normalized: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    cells = []
    for cell_id, cell_frame in normalized.groupby("cell_id", sort=True):
        ordered = cell_frame.sort_values("cycle_index")
        if len(ordered) < 3:
            cells.append(
                {
                    "cell_id": str(cell_id),
                    "status": "skipped",
                    "reason": "within-cell split requires at least three cycles",
                    "cycle_count": int(len(ordered)),
                }
            )
            continue
        split_index = int(len(ordered) * train_fraction)
        split_index = min(max(split_index, 1), len(ordered) - 1)
        train_cycles = ordered.iloc[:split_index]["cycle_index"].astype(int).tolist()
        test_cycles = ordered.iloc[split_index:]["cycle_index"].astype(int).tolist()
        cells.append(
            {
                "cell_id": str(cell_id),
                "status": "ok",
                "train_fraction": float(train_fraction),
                "split_index": int(split_index),
                "train_cycles": train_cycles,
                "test_cycles": test_cycles,
                "train_cycle_count": int(len(train_cycles)),
                "test_cycle_count": int(len(test_cycles)),
            }
        )
    return {
        "schema_version": 1,
        "status": "ok" if any(cell["status"] == "ok" for cell in cells) else "skipped",
        "train_fraction": float(train_fraction),
        "cells": cells,
    }


def write_within_cell_time_splits(
    normalized: pd.DataFrame,
    *,
    path: str | Path,
    train_fraction: float = 0.8,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_within_cell_time_splits(normalized, train_fraction=train_fraction)
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_normalized_artifacts(
    raw_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    normalized_csv: str | Path,
    provenance_json: str | Path,
    source_url: str,
) -> pd.DataFrame:
    if isinstance(raw_path, (list, tuple)):
        if not raw_path:
            raise ValueError("at least one NASA battery raw path is required")
        normalized_parts = [normalize_battery_table(path) for path in raw_path]
        normalized = pd.concat(normalized_parts, ignore_index=True).sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
        raw_path_value: str | list[str] = [str(path) for path in raw_path]
    else:
        normalized = normalize_battery_table(raw_path)
        raw_path_value = str(raw_path)
    normalized_path = Path(normalized_csv)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(normalized_path, index=False)

    provenance = {
        "source_url": source_url,
        "raw_path": raw_path_value,
        "normalized_csv": str(normalized_path),
        "schema_version": 1,
        "row_count": int(len(normalized)),
        "cell_count": int(normalized["cell_id"].nunique()),
        "cells": sorted(normalized["cell_id"].astype(str).unique().tolist()),
        "cell_summary": cell_summary(normalized),
        "target": "cycle-level discharge capacity and state of health",
        "notes": [
            "Fixture rows are schema/smoke data only, not a scientific NASA battery result.",
            "First runnable target uses one-cell SOH capacity fade; promotion should require cell-level holdout.",
        ],
    }
    Path(provenance_json).write_text(__import__("json").dumps(provenance, indent=2, sort_keys=True) + "\n")
    return normalized


def write_capacity_target(
    normalized: pd.DataFrame,
    *,
    data_dir: str | Path = "data",
    domain: str = "real_battery_nasa_capacity",
    cell_id: str | None = None,
    target_column: str = "soh",
) -> dict[str, Any]:
    time_points, target = build_capacity_target(normalized, cell_id=cell_id, target_column=target_column)
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = out_dir / f"{domain}_ground_truth.npy"
    time_points_path = out_dir / f"{domain}_time_points.npy"
    np.save(ground_truth_path, target)
    np.save(time_points_path, time_points)
    return {
        "domain": domain,
        "cell_id": cell_id or "first_sorted_cell",
        "ground_truth_path": str(ground_truth_path),
        "time_points_path": str(time_points_path),
        "target_column": target_column,
        "time_points": int(len(time_points)),
        "state_count": int(target.shape[1]),
    }

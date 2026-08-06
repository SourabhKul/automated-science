#!/usr/bin/env python3
"""Fixed official UCI gas-turbine provenance and source-quality gate only."""
from __future__ import annotations

import csv
import hashlib
from io import TextIOWrapper
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARCHIVE = Path("data/real/uci_gas_turbine/raw/uci_gas_turbine_551.zip")
OUT_DIR = Path("artifacts/evaluations/phase38_uci_gas_turbine_nox_year_holdout_archive_gate_20260724")
MANIFEST = Path("data/real/uci_gas_turbine/manifest.json")
SPLIT_PATH = Path("data/real/uci_gas_turbine/source_year_split.json")
MEMBERS = tuple(f"gt_{year}.csv" for year in range(2011, 2016))
HEADERS = ("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX")
TOTAL_ROWS = 36733
SOURCE_UNITS_AND_RANGES = {
    "AT": {"unit": "C", "min": -6.23, "max": 37.10},
    "AP": {"unit": "mbar", "min": 985.85, "max": 1036.56},
    "AH": {"unit": "%", "min": 24.08, "max": 100.20},
    "AFDP": {"unit": "mbar", "min": 2.09, "max": 7.61},
    "GTEP": {"unit": "mbar", "min": 17.70, "max": 40.72},
    "TIT": {"unit": "C", "min": 1000.85, "max": 1100.89},
    "TAT": {"unit": "C", "min": 511.04, "max": 550.61},
    "TEY": {"unit": "MWH", "min": 100.02, "max": 179.50},
    "CDP": {"unit": "mbar", "min": 9.85, "max": 15.16},
    "CO": {"unit": "mg/m3", "min": 0.00, "max": 44.10},
    "NOX": {"unit": "mg/m3", "min": 25.90, "max": 119.91},
}
SOURCE_SPLIT = {"train_years": [2011, 2012, 2013], "selection_year": 2014, "external_year": 2015}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_optional(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


def _inspect_member(archive: ZipFile, member: str) -> dict[str, object]:
    row_count = 0
    nonfinite_cells = 0
    duplicate_rows = 0
    malformed_rows = 0
    values = {name: [] for name in HEADERS}
    fingerprints: set[str] = set()
    with archive.open(member) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8", newline=""))
        headers = tuple(reader.fieldnames or ())
        if headers != HEADERS:
            raise ValueError(f"{member} header {headers!r} does not match the fixed source schema")
        for row in reader:
            row_count += 1
            if set(row) != set(HEADERS) or any(row[name] in (None, "") for name in HEADERS):
                malformed_rows += 1
                continue
            try:
                numeric = {name: float(row[name]) for name in HEADERS}
            except ValueError:
                malformed_rows += 1
                continue
            nonfinite_cells += int(sum(not np.isfinite(value) for value in numeric.values()))
            for name, value in numeric.items():
                values[name].append(value)
            fingerprint = hashlib.sha256(",".join(row[name] for name in HEADERS).encode("utf-8")).hexdigest()
            duplicate_rows += int(fingerprint in fingerprints)
            fingerprints.add(fingerprint)
    ledger = {
        name: {"min": float(np.min(column)) if column else None, "max": float(np.max(column)) if column else None}
        for name, column in values.items()
    }
    return {
        "rows": row_count,
        "headers": list(headers),
        "malformed_rows": malformed_rows,
        "nonfinite_cells": nonfinite_cells,
        "duplicate_complete_rows": duplicate_rows,
        "unique_complete_rows": len(fingerprints),
        "observed_range_ledger": ledger,
    }


def run(archive_path: Path = ARCHIVE, output_dir: Path = OUT_DIR) -> dict[str, object]:
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        if set(members) != set(MEMBERS) or len(members) != len(MEMBERS):
            raise ValueError("official UCI archive does not contain exactly the fixed five annual raw CSV members")
        inspection = {member: _inspect_member(archive, member) for member in MEMBERS}
    row_total = sum(int(item["rows"]) for item in inspection.values())
    checks = {
        "official_member_inventory": members == list(MEMBERS),
        "total_rows": row_total == TOTAL_ROWS,
        "nonempty_annual_members": all(item["rows"] > 0 for item in inspection.values()),
        "fixed_headers": all(item["headers"] == list(HEADERS) for item in inspection.values()),
        "finite_numeric_schema": all(item["malformed_rows"] == 0 and item["nonfinite_cells"] == 0 for item in inspection.values()),
        "frozen_year_split": SOURCE_SPLIT == {"train_years": [2011, 2012, 2013], "selection_year": 2014, "external_year": 2015},
    }
    passed = all(checks.values())
    provenance = Path("data/real/uci_gas_turbine/source_provenance")
    result = {
        "phase": 38,
        "status": "passed_official_source_gate_only" if passed else "blocked_official_source_gate",
        "source": {
            "dataset_page": "https://archive.ics.uci.edu/dataset/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset",
            "archive_url": "https://archive.ics.uci.edu/static/public/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset.zip",
            "doi": "10.24432/C5WC95",
            "license": "CC-BY-4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256_file(archive_path),
            "published_checksum": None,
            "response_headers_sha256": _sha256_optional(provenance / "archive_headers.txt"),
            "page_sha256": _sha256_optional(provenance / "dataset_page.html"),
            "members": members,
        },
        "checks": checks,
        "annual_members": inspection,
        "source_units_and_documented_ranges": SOURCE_UNITS_AND_RANGES,
        "source_limitations": {
            "aggregation": "hourly averages or sums documented by UCI",
            "row_timestamps": "not provided in source rows",
            "source_order": "UCI documents rows as chronologically sorted",
            "claim": "annual member holdout only; no short-horizon forecast or causal inference",
        },
        "source_year_split": SOURCE_SPLIT if passed else None,
        "isolation": {"measurement_transformation_or_fitting": False, "abc_smc_calls": 0, "llm_calls": 0},
        "next_gate": "Write a fixed source-normalized adapter, train-only baseline, and output-isolated control plan." if passed else "Return to source-backed application selection; do not repair or reassign source rows or years.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if passed:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        SPLIT_PATH.write_text(json.dumps(SOURCE_SPLIT, indent=2) + "\n")
    return result


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "checks": result["checks"], "split": result["source_year_split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

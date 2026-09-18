#!/usr/bin/env python3
"""Predeclared driver for the observation-mismatch simulator controls.

The driver only executes the six-cell pilot when both ``--run-pilot`` and the
literal ``--independent-review-approved`` acknowledgement are supplied.  This
keeps implementation tests and plan inspection separate from the pilot that
the control specification reserves for an independent review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observation_mismatch_controls import (
    BASELINE_KIND,
    PROTOCOL_VERSION,
    canonical_json,
    generate_nested_mechanism_dataset,
    generate_observation_mismatch_dataset,
    predeclared_control_cells,
    run_nested_mechanism_abstention_control,
    run_observation_mismatch_control,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    if cell["experiment"] == "observation_mismatch":
        dataset = generate_observation_mismatch_dataset(int(cell["data_seed"]))
        return run_observation_mismatch_control(
            dataset,
            fit_seed_base=int(cell["fit_seed_base"]),
            predictive_seed_base=int(cell["predictive_seed_base"]),
        )
    if cell["experiment"] == "nested_mechanism":
        dataset = generate_nested_mechanism_dataset(int(cell["data_seed"]))
        return run_nested_mechanism_abstention_control(
            dataset,
            fit_seed_base=int(cell["fit_seed_base"]),
            predictive_seed_base=int(cell["predictive_seed_base"]),
        )
    raise ValueError(f"unknown predeclared experiment {cell['experiment']!r}")


def run_pilot(output_dir: Path) -> dict[str, Any]:
    """Run cells serially with atomic receipts after independent review."""

    cells = [dict(cell) for cell in predeclared_control_cells()]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "run_id": run_id,
        "status": "running",
        "protocol_version": PROTOCOL_VERSION,
        "baseline_kind": BASELINE_KIND,
        "pilot_gate": "independent_review_approved_by_cli_acknowledgement",
        "cells": cells,
    }
    _atomic_json(output_dir / "pilot_plan.json", plan)
    records: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        cell_id = str(cell["cell_id"])
        try:
            result = _run_cell(cell)
            status = str(result.get("status", "unknown"))
            record = {"cell": cell, "status": status, "result": result}
        except Exception as exc:  # pragma: no cover - bounded receipt path
            record = {
                "cell": cell,
                "status": "failed",
                "error_category": type(exc).__name__,
                "error": str(exc),
            }
        cell_path = output_dir / f"cell_{index:02d}_{cell_id}.json"
        _atomic_json(cell_path, record)
        records.append(
            {
                "cell_id": cell_id,
                "status": record["status"],
                "path": cell_path.name,
                "bytes": cell_path.stat().st_size,
                "sha256": _file_sha256(cell_path),
            }
        )
        _atomic_json(
            output_dir / "partial_result.json",
            {
                "run_id": run_id,
                "status": "partial",
                "completed_cells": index + 1,
                "total_cells": len(cells),
                "cells": records,
            },
        )
    completed = sum(item["status"] == "complete" for item in records)
    aggregate = {
        "run_id": run_id,
        "status": "completed" if completed == len(cells) else "completed_with_failures",
        "completed_cells": completed,
        "total_cells": len(cells),
        "cells": records,
        "protocol_version": PROTOCOL_VERSION,
        "baseline_kind": BASELINE_KIND,
        "pilot_claim_limit": "known simulator control only; no real-mechanism, coverage, novelty, or model-probability claim",
    }
    _atomic_json(output_dir / "aggregate_result.json", aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-pilot",
        action="store_true",
        help="execute all six predeclared cells (requires --independent-review-approved)",
    )
    parser.add_argument(
        "--independent-review-approved",
        action="store_true",
        help="acknowledge the independent-review gate required before pilot execution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/observation_mismatch_control"),
        help="ignored artifact directory used only by an approved pilot",
    )
    args = parser.parse_args()
    if not args.run_pilot:
        print(canonical_json({"status": "predeclared_not_run", "cells": list(predeclared_control_cells())}))
        return 0
    if not args.independent_review_approved:
        parser.error("pilot execution is gated; pass --independent-review-approved after independent review")
    result = run_pilot(args.output_dir)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

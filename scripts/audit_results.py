#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


DOMAINS = [
    "ecology", "oncology", "econ", "battery", "pkpd", "synbio", "climate", "cardio", "bz_chem", "epidemiology",
    "neuro", "fluid", "astro", "chem_kinetics", "immune", "pop_genetics", "thermo", "materials", "agriculture", "social",
    "real_sunspots", "real_nile", "real_macro", "real_theophylline", "real_co2",
]


def read_domain(root: Path, domain: str) -> dict:
    starts = []
    finals = []
    accepted = 0
    errors = 0
    completed = 0
    for run_idx in range(1, 11):
        path = root / f"run_{run_idx}" / f"qwen36_27b_{domain}_run_history.csv"
        if not path.exists():
            continue
        completed += 1
        start = None
        final = None
        with path.open() as f:
            for row in csv.DictReader(f):
                proposed = float(row["Proposed_Loss"])
                baseline = float(row["Baseline_Loss"])
                if start is None:
                    start = baseline
                final = baseline
                if row["Action"] == "ACCEPTED":
                    accepted += 1
                if math.isinf(proposed) or row["Action"].startswith("ERROR"):
                    errors += 1
        if start is not None:
            starts.append(start)
        if final is not None:
            finals.append(final)

    valid_start = [x for x in starts if not math.isinf(x)]
    valid_final = [x for x in finals if not math.isinf(x)]
    mean_start = mean(valid_start) if valid_start else math.inf
    mean_final = mean(valid_final) if valid_final else math.inf
    relative_improvement = (
        (mean_start - mean_final) / mean_start
        if valid_start and valid_final and mean_start != 0
        else math.nan
    )
    return {
        "domain": domain,
        "completed_runs": completed,
        "accepted_mutations": accepted,
        "error_or_inf_proposals": errors,
        "mean_start_distance": mean_start,
        "mean_final_distance": mean_final,
        "best_final_distance": min(finals) if finals else math.inf,
        "relative_improvement": relative_improvement,
        "weak_result": accepted == 0 or (not math.isnan(relative_improvement) and relative_improvement < 0.05),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize preserved Qwen run histories.")
    parser.add_argument("--root", default="scheduler_logs", help="Run-history root.")
    parser.add_argument("--out", default="artifacts/evaluations/qwen36_result_audit.json", help="Output JSON path.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [read_domain(root, domain) for domain in DOMAINS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    weak = [r["domain"] for r in rows if r["weak_result"]]
    print(f"Wrote {out}")
    print(f"Weak domains (<5% mean improvement or zero accepted): {', '.join(weak)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the fixed Phase 26 VitalDB source-quality adapter assessment only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import vitaldb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.vitaldb import REQUIRED_TRACKS, assess_vitaldb_adapter_contract


SPLIT_PATH = ROOT / "data/real/vitaldb/candidate_split.json"
OUTPUT_DIR = ROOT / "artifacts/evaluations/phase26_vitaldb_adapter_assessment_20260715"


def _reader(case_id: int, tracks: tuple[str, ...], interval: int):
    if tracks != REQUIRED_TRACKS or interval != 10:
        raise ValueError("Phase 26 assessment requires the locked tracks on the 10-second grid")
    return vitaldb.load_case(case_id, list(tracks), interval=interval)


def _markdown(result: dict[str, object]) -> str:
    rows = result["rows"]
    rejected = [row for row in rows if not row["eligible"]]
    lines = [
        "# Phase 26 VitalDB Locked-Cohort Adapter Assessment",
        "",
        "**Scope:** source-quality accounting only; no fitted model, ABC-SMC, LLM, or campaign.",
        "",
        f"**Decision:** {'passed' if result['passed'] else 'failed'}",
        "",
        "## Fixed Contract",
        "",
        f"- tracks: `{', '.join(result['required_tracks'])}`",
        f"- grid: `{result['grid_seconds']}` seconds",
        f"- required contiguous jointly finite samples: `{result['minimum_contiguous_samples']}`",
        f"- locked cases assessed: `{len(rows)}`",
        f"- eligible cases: `{len(rows) - len(rejected)}`",
        f"- rejected cases: `{len(rejected)}`",
        "",
        "## Case Ledger",
        "",
        "| Case | Subject | Split | Department | Native shape | Joint finite | Longest contiguous | Eligible | Reason |",
        "|---:|---:|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['subject_id']} | {row['split']} | {row['department']} | "
            f"{row['native_shape']} | {row['joint_finite_samples']} | {row['longest_contiguous_samples']} | "
            f"{row['eligible']} | {row['rejection_reason'] or ''} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This ledger records data-quality eligibility only. No MAP prediction was fit or scored. "
        "Any rejected locked case closes the cohort without pruning, replacement, interpolation, or split revision.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    result = assess_vitaldb_adapter_contract(SPLIT_PATH, _reader)
    result.update({
        "phase": 26,
        "assessment": "locked_cohort_source_quality_only",
        "real_model_fitting_executed": False,
        "abc_smc_executed": False,
        "llm_called": False,
        "campaign_launched": False,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "assessment.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "report.md").write_text(_markdown(result))
    print(f"VitalDB adapter assessment {'PASSED' if result['passed'] else 'FAILED'}; report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

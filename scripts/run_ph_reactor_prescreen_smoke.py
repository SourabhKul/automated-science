#!/usr/bin/env python3
"""Exercise the pH-reactor static prescreen on all native recorded U grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.ph_reactor import load_ph_reactor, prescreen_ph_reactor_rollouts
from scripts.legacy.real_ph_reactor_data_loader import DEFAULT_RAW_ROOT


def bounded_u_only_rollout(input_u: np.ndarray) -> tuple[np.ndarray, dict[str, bool]]:
    """A deliberately bounded causal surrogate for prescreen plumbing only."""
    output = np.zeros_like(input_u, dtype=float)
    reference = float(np.mean(input_u))
    for index in range(1, len(output)):
        output[index] = 0.90 * output[index - 1] + 0.10 * np.tanh(input_u[index - 1] - reference)
    return output, {"state_clipped": False, "derivative_clipped": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/phase22_ph_reactor_runner_controls_20260712/prescreen_summary.json"),
    )
    args = parser.parse_args()
    result = prescreen_ph_reactor_rollouts(load_ph_reactor(args.raw_root), bounded_u_only_rollout)
    result.update(
        {
            "phase": 22,
            "run_kind": "non_llm_ph_reactor_static_prescreen_smoke",
            "candidate_contract": "bounded causal U-only rollout; synthetic plumbing check, not a fitted simulator candidate",
            "abc_smc_calls": 0,
            "llm_calls": 0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "native_input_grid_count", "failure_count")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the isolated pH input-driven evaluator on planted data over native U grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.input_driven_eval import compile_input_driven_candidate, evaluate_input_driven_candidate, rollout_input_driven
from core.real_data.ph_reactor import PHExperiment, load_ph_reactor
from scripts.legacy.real_ph_reactor_data_loader import DEFAULT_RAW_ROOT


CODE = """def dynamics(t, y, args, u):
    alpha, beta = args
    return jnp.array([-(1.0 - alpha) * y[0] + beta * jnp.tanh(u - 14.4634995)])

metadata = [
    {'name': 'alpha', 'range': (0.70, 0.99)},
    {'name': 'beta', 'range': (0.01, 0.50)}
]"""
PARAMS = np.array([0.92, 0.30])


def planted_data(raw_root: Path = DEFAULT_RAW_ROOT) -> tuple[dict[str, list[PHExperiment]], object]:
    candidate = compile_input_driven_candidate(CODE)
    observed = load_ph_reactor(raw_root)
    synthetic = {}
    for split, experiments in observed.items():
        synthetic[split] = []
        for item in experiments:
            output, _ = rollout_input_driven(candidate, item.input_u, PARAMS)
            synthetic[split].append(PHExperiment(item.experiment_id, item.split, item.input_u, output))
    return synthetic, candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/phase22_ph_reactor_input_driven_synthetic_smoke_20260712/summary.json"))
    args = parser.parse_args()
    data, candidate = planted_data(args.raw_root)
    result = evaluate_input_driven_candidate(data, candidate, PARAMS)
    result.update({"phase": 22, "run_kind": "tiny_non_llm_input_driven_synthetic_contract_smoke", "abc_smc_calls": 0, "llm_calls": 0})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "test": result.get("untouched_test_metric")}, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

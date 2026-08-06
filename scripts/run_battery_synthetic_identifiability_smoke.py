#!/usr/bin/env python3
"""Tiny non-LLM identifiability smoke for the planned NASA covariate family."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.battery_nasa import build_leave_one_cell_out_splits
from core.sbi_engine import resolve_abc_smc_strategy


@dataclass(frozen=True)
class SyntheticBundle:
    cell_id: str
    cycle: np.ndarray
    z_current: np.ndarray
    z_temperature: np.ndarray
    observed: np.ndarray
    true_u: float


PRIORS = [
    {"name": "log_k_pop", "range": [float(np.log(1e-4)), float(np.log(1e-2))]},
    {"name": "u_1", "range": [-0.5, 0.5]},
    {"name": "u_2", "range": [-0.5, 0.5]},
    {"name": "beta_current", "range": [-1.0, 1.0]},
    {"name": "beta_temperature", "range": [-1.0, 1.0]},
]
TRUE = {"k_pop": 0.0024, "u": [-0.12, 0.0, 0.12], "beta_current": 0.20, "beta_temperature": 0.15, "sigma_obs": 0.01}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def decode(params: np.ndarray, cell_count: int) -> tuple[float, np.ndarray, float, float]:
    if cell_count != 3 or len(params) != 5:
        raise ValueError("smoke expects three train cells and five parameters")
    free_u = np.asarray(params[1:3], dtype=float)
    u = np.array([free_u[0], free_u[1], -float(np.sum(free_u))])
    return float(np.exp(params[0])), u, float(params[3]), float(params[4])


def simulate(bundle: SyntheticBundle, k_pop: float, u: float, beta_current: float, beta_temperature: float) -> np.ndarray:
    rate = k_pop * np.exp(u + beta_current * bundle.z_current + beta_temperature * bundle.z_temperature)
    q = np.empty(len(bundle.cycle), dtype=float)
    q[0] = 1.0
    deltas = np.diff(bundle.cycle, prepend=bundle.cycle[0])
    for index in range(1, len(q)):
        q[index] = q[index - 1] * np.exp(-rate[index - 1] * max(float(deltas[index]), 0.0))
    return q


def build_bundles(frame: pd.DataFrame, fold: dict[str, Any], scenario: str, rng: np.random.Generator) -> tuple[list[SyntheticBundle], dict[str, float]]:
    train = frame[frame["cell_id"].astype(str).isin(fold["train_cells"])].copy()
    train["abs_current"] = train["mean_discharge_current_a"].abs()
    current_mean, current_std = float(train["abs_current"].mean()), float(train["abs_current"].std())
    temperature_mean, temperature_std = float(train["mean_temperature_c"].mean()), float(train["mean_temperature_c"].std())
    if current_std <= 0.0 or temperature_std <= 0.0:
        raise ValueError("train-only covariate standard deviation must be positive")
    bundles = []
    for index, cell_id in enumerate(sorted(fold["train_cells"])):
        cell = train[train["cell_id"].astype(str) == cell_id].sort_values("cycle_index")
        z_current = (cell["abs_current"].to_numpy(float) - current_mean) / current_std
        z_temperature = (cell["mean_temperature_c"].to_numpy(float) - temperature_mean) / temperature_std
        template = SyntheticBundle(cell_id, cell["cycle_index"].to_numpy(float), z_current, z_temperature, np.empty(len(cell)), TRUE["u"][index])
        if scenario == "signal":
            beta_current, beta_temperature = TRUE["beta_current"], TRUE["beta_temperature"]
        elif scenario == "null":
            beta_current, beta_temperature = 0.0, 0.0
        elif scenario == "shuffled":
            beta_current, beta_temperature = TRUE["beta_current"], TRUE["beta_temperature"]
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        observed = simulate(template, TRUE["k_pop"], TRUE["u"][index], beta_current, beta_temperature)
        observed *= rng.lognormal(mean=0.0, sigma=TRUE["sigma_obs"], size=len(observed))
        if scenario == "shuffled":
            z_current = rng.permutation(z_current)
            z_temperature = rng.permutation(z_temperature)
        bundles.append(SyntheticBundle(cell_id, template.cycle, z_current, z_temperature, observed, TRUE["u"][index]))
    return bundles, {"current_mean": current_mean, "current_std": current_std, "temperature_mean": temperature_mean, "temperature_std": temperature_std}


def distances(params_batch: np.ndarray, bundles: list[SyntheticBundle]) -> np.ndarray:
    values = []
    for params in np.asarray(params_batch, dtype=float):
        k_pop, u, beta_current, beta_temperature = decode(params, len(bundles))
        residuals = []
        for index, bundle in enumerate(bundles):
            predicted = simulate(bundle, k_pop, u[index], beta_current, beta_temperature)
            residuals.append(predicted - bundle.observed)
        values.append(float(np.sqrt(np.mean(np.square(np.concatenate(residuals))))))
    return np.asarray(values, dtype=float)


def run_abc(bundles: list[SyntheticBundle], *, seed: int, initial_particles: int, generations: int, target_samples: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    strategy = resolve_abc_smc_strategy("gaussian_weighted")
    params = np.column_stack([rng.uniform(prior["range"][0], prior["range"][1], initial_particles) for prior in PRIORS])
    accepted = None
    weights = None
    history = []
    for generation in range(generations):
        if generation:
            params, _ = strategy.transition(
                rng=rng,
                accepted_params=accepted,
                accepted_weights=weights,
                priors=PRIORS,
                initial_particles=initial_particles,
                lambda_noise=0.01,
                nugget=1e-9,
            )
            params = np.asarray(params, dtype=float)
        d = distances(params, bundles)
        finite = np.where(np.isfinite(d))[0]
        if len(finite) < target_samples:
            return {"status": "failed", "reason": "insufficient_finite_particles", "history": history}
        keep = finite[np.argsort(d[finite])[:target_samples]]
        accepted = params[keep]
        selected = d[keep]
        weights = np.full(target_samples, 1.0 / target_samples)
        history.append({"generation": generation, "finite_particle_count": int(len(finite)), "median_distance": float(np.median(selected)), "min_distance": float(np.min(selected))})
    median = np.median(accepted, axis=0)
    k_pop, u, beta_current, beta_temperature = decode(median, len(bundles))
    return {
        "status": "success",
        "history": history,
        "posterior_median": {"k_pop": k_pop, "u": u.tolist(), "beta_current": beta_current, "beta_temperature": beta_temperature},
        "parameter_vector_sum_u": float(np.sum(u)),
        "all_accepted_finite": bool(np.all(np.isfinite(accepted))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-data", type=Path, default=Path("data/real/battery_nasa/cycle_level.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--initial-particles", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--target-samples", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.normalized_data)
    folds = build_leave_one_cell_out_splits(frame)
    results = []
    for fold_index, fold in enumerate(folds):
        for scenario_index, scenario in enumerate(["signal", "null", "shuffled"]):
            rng = np.random.default_rng(args.seed + fold_index * 100 + scenario_index)
            bundles, scaling = build_bundles(frame, fold, scenario, rng)
            result = run_abc(
                bundles,
                seed=args.seed + fold_index * 100 + scenario_index,
                initial_particles=args.initial_particles,
                generations=args.generations,
                target_samples=args.target_samples,
            )
            results.append({"fold_id": fold["fold_id"], "scenario": scenario, "train_cells": fold["train_cells"], "scaling": scaling, **result})
    passed = all(item["status"] == "success" and item.get("all_accepted_finite") and abs(item.get("parameter_vector_sum_u", 1.0)) < 1e-12 for item in results)
    payload = {
        "phase": 18,
        "mode": "synthetic_identifiability_smoke",
        "created_at": utc_now(),
        "run_spec": {"seed": args.seed, "initial_particles": args.initial_particles, "generations": args.generations, "target_samples": args.target_samples},
        "results": results,
        "smoke_passed": passed,
        "note": "Smoke validates finite trajectories, train-only scaling, and parameter-vector bookkeeping only. It does not satisfy the 50k recovery or negative-control calibration gates.",
    }
    write_json(args.output, payload)
    print(json.dumps({"smoke_passed": passed, "result_count": len(results)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

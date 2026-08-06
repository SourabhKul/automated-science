#!/usr/bin/env python3
"""Run the pre-registered current-only synthetic NASA smoke."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_data.battery_nasa import build_leave_one_cell_out_splits
from core.sbi_engine import resolve_abc_smc_strategy

PRIORS = [{"name": "log_k_pop", "range": [float(np.log(1e-4)), float(np.log(1e-2))]}, {"name": "beta_current", "range": [-1.0, 1.0]}]
OFFSETS = np.array([-0.12, 0.0, 0.12])


def simulate(cycle, z_current, log_k, beta):
    q = np.empty(len(cycle)); q[0] = 1.0
    rate = np.exp(log_k + beta * z_current)
    for i in range(1, len(q)):
        q[i] = q[i - 1] * np.exp(-rate[i - 1] * max(float(cycle[i] - cycle[i - 1]), 0.0))
    return q


def bundles(frame, fold, scenario, rng):
    train = frame[frame.cell_id.isin(fold["train_cells"])].copy(); train["abs_i"] = train.mean_discharge_current_a.abs()
    mean, std = float(train.abs_i.mean()), float(train.abs_i.std())
    if std <= 0: raise ValueError("nonpositive train-only current scale")
    out = []
    for idx, cell_id in enumerate(sorted(fold["train_cells"])):
        cell = train[train.cell_id == cell_id].sort_values("cycle_index")
        cycle = cell.cycle_index.to_numpy(float); z = (cell.abs_i.to_numpy(float) - mean) / std
        true_beta = 0.2 if scenario != "null" else 0.0
        observed = simulate(cycle, z, np.log(0.0024) + OFFSETS[idx], true_beta) * rng.lognormal(0.0, 0.01, len(cycle))
        if scenario == "shuffled": z = rng.permutation(z)
        out.append((cycle, z, observed))
    return out, {"current_mean": mean, "current_std": std}


def distance(params, data):
    residuals = []
    for idx, (cycle, z, observed) in enumerate(data): residuals.append(simulate(cycle, z, params[0] + OFFSETS[idx], params[1]) - observed)
    return float(np.sqrt(np.mean(np.square(np.concatenate(residuals)))))


def abc(data, seed, n, generations, target):
    rng = np.random.default_rng(seed); strategy = resolve_abc_smc_strategy("gaussian_weighted")
    params = np.column_stack([rng.uniform(p["range"][0], p["range"][1], n) for p in PRIORS]); history=[]
    for g in range(generations):
        if g:
            params, _ = strategy.transition(rng=rng, accepted_params=accepted, accepted_weights=weights, priors=PRIORS, initial_particles=n, lambda_noise=0.01, nugget=1e-9)
            params = np.asarray(params)
        d = np.array([distance(p, data) for p in params]); keep = np.where(np.isfinite(d))[0]
        if len(keep) < target: return {"status":"failed","reason":"insufficient_finite_particles","history":history}
        keep = keep[np.argsort(d[keep])[:target]]; accepted=params[keep]; weights=np.full(target,1/target)
        history.append({"generation":g,"finite_particle_count":int(np.isfinite(d).sum()),"median_distance":float(np.median(d[keep]))})
    summary=[]
    for i, name in enumerate(["log_k_pop","beta_current"]):
        p10,p50,p90=np.percentile(accepted[:,i],[10,50,90]); lo,hi=PRIORS[i]["range"]
        summary.append({"name":name,"p10":float(p10),"median":float(p50),"p90":float(p90),"lower_fraction":float(np.mean(accepted[:,i] <= lo+0.01*(hi-lo))),"upper_fraction":float(np.mean(accepted[:,i] >= hi-0.01*(hi-lo)))})
    return {"status":"success","history":history,"posterior":summary}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--normalized-data",type=Path,default=Path("data/real/battery_nasa/cycle_level.csv")); p.add_argument("--seed",type=int,default=20261920); p.add_argument("--initial-particles",type=int,default=1000); p.add_argument("--generations",type=int,default=3); p.add_argument("--target-samples",type=int,default=50); a=p.parse_args()
    frame=pd.read_csv(a.normalized_data); results=[]
    for fidx, fold in enumerate(build_leave_one_cell_out_splits(frame)):
        for sidx, scenario in enumerate(["signal","null","shuffled"]):
            rng=np.random.default_rng(a.seed+fidx*10+sidx); data, scaling=bundles(frame,fold,scenario,rng); result=abc(data,a.seed+fidx*10+sidx,a.initial_particles,a.generations,a.target_samples)
            results.append({"fold_id":fold["fold_id"],"scenario":scenario,"scaling":scaling,**result})
    def beta(item): return next(x for x in item.get("posterior",[]) if x["name"]=="beta_current")
    gates=[]
    for item in results:
        if item["status"] != "success": gates.append(False); continue
        b=beta(item); off_bound=max(b["lower_fraction"],b["upper_fraction"]) <= .05
        if item["scenario"]=="signal": gates.append(off_bound and b["median"]>0 and b["p10"] <= .2 <= b["p90"])
        else: gates.append(off_bound and b["p10"] <= 0 <= b["p90"])
    payload={"phase":18,"mode":"simplified_current_synthetic_smoke","run_spec":{"seed":a.seed,"initial_particles":a.initial_particles,"generations":a.generations,"target_samples":a.target_samples},"results":results,"smoke_passed":all(gates),"gates":gates}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n"); print(json.dumps({"smoke_passed":payload["smoke_passed"],"result_count":len(results)})); return 0 if payload["smoke_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())

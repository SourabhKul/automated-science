from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.generated_code import GeneratedCodeError, extract_and_validate
from core.sbi_engine import SBIEngine


def build_candidate_module(code: str) -> str:
    return f'''import jax.numpy as jnp
from core.base_model import BaseModel
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

{code}

class CandidateModel(BaseModel):
    def simulate(self, params, time_points, y0):
        saveat = SaveAt(ts=time_points)
        sol = diffeqsolve(
            ODETerm(dynamics),
            Tsit5(),
            t0=float(time_points[0]),
            t1=float(time_points[-1]),
            dt0=0.05,
            y0=y0,
            args=params,
            saveat=saveat,
            max_steps=1000,
        )
        return sol.ys

    def get_parameter_metadata(self):
        return metadata

    def get_initial_conditions(self):
        return jnp.array([1.0])

    def get_latex(self):
        return "Qwen LM Studio smoke-test exponential decay"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Small end-to-end Qwen/LM Studio smoke test.")
    parser.add_argument("--endpoint", default="http://localhost:1234/v1/chat/completions")
    parser.add_argument("--model", default="qwen/qwen3-coder-next")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    prompt = """Return ONLY a Python fenced code block containing a JAX-compatible `dynamics(t, y, args)` function and `metadata` list.
Model a one-state exponential decay process dx/dt = -k*x. IMPORTANT: args is a positional JAX vector, so access k as args[0], never args['k'].
Use one parameter named k with range (0.1, 1.0). Do not include imports, classes, prose, or markdown outside the code block.
"""
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You output only small JAX mathematical model code blocks with dynamics and metadata.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }

    print(f"Calling {args.model} at {args.endpoint}...")
    response = requests.post(args.endpoint, json=payload, timeout=240)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    content = choice["message"].get("content") or ""
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(f"content_length: {len(content)}")
    if not content:
        reasoning = choice["message"].get("reasoning_content") or ""
        print(f"reasoning_preview: {reasoning[:500]}")
        raise RuntimeError("LM Studio returned empty assistant content")

    try:
        extracted = extract_and_validate(content)
    except GeneratedCodeError as exc:
        repair_payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Repair generated JAX math code. Return only a Python fenced code block.",
                },
                {
                    "role": "user",
                    "content": (
                        f"The previous code failed validation: {exc}. "
                        "Return the same exponential-decay dynamics using positional parameter access k = args[0].\n\n"
                        f"{content}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        print(f"Validation failed once: {exc}; requesting repair...")
        repair_response = requests.post(args.endpoint, json=repair_payload, timeout=240)
        repair_response.raise_for_status()
        content = repair_response.json()["choices"][0]["message"].get("content") or ""
        extracted = extract_and_validate(content)
    if extracted is None:
        print(content)
        raise RuntimeError("No code extracted")
    print("Validated code:")
    print(extracted.code)

    spec = importlib.util.spec_from_loader("qwen_smoke_candidate", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(build_candidate_module(extracted.code), module.__dict__)
    model = module.CandidateModel()

    time_points = np.linspace(0.0, 1.0, 12)
    gt_data = np.exp(-0.4 * time_points)[:, None]
    engine = SBIEngine(gt_data, time_points, seed=123)
    result = engine.run_abc_smc(model, target_samples=20, generations=2, initial_particles=80)
    summary = {
        "accepted_samples": int(len(result["accepted_params"])),
        "median_distance": float(result["median_distance"]),
        "param_count": int(len(model.get_parameter_metadata())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

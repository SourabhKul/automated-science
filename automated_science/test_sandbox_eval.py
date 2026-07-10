import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.sandbox_eval import build_diagnostic_packet
from core.domain_configs import DOMAIN_CONFIGS


BROKEN_CODE = """
def dynamics(t, y, args):
    return jnp.array([undefined_symbol, y[1]])

metadata = [
    {'name': 'r', 'range': (0.1, 2.0)}
]
"""


def main():
    packet = build_diagnostic_packet(
        res={
            "median_distance": 1.25,
            "min_distance": 0.5,
            "accepted_params": np.array(
                [
                    [0.3, 0.5],
                    [0.4, 0.55],
                    [0.35, 0.6],
                ]
            ),
        },
        metadata=[
            {"name": "alpha", "range": (0.0, 1.0)},
            {"name": "beta", "range": (0.0, 1.0)},
        ],
        status="success",
        observed=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ]
        ),
        predicted=np.array(
            [
                [1.1, 9.9],
                [2.1, 19.9],
                [3.1, 29.9],
                [4.1, 39.9],
                [5.1, 49.9],
                [6.1, 59.9],
            ]
        ),
        split_idx=4,
        train_metrics={"mse": 0.1, "rmse": 0.316},
        test_metrics={"mse": 0.11, "rmse": 0.332},
    )
    assert packet["schema_version"] == 1
    assert packet["fit"]["accepted_particle_count"] == 3
    assert packet["best_parameters"]["alpha"] == 0.35
    assert len(packet["posterior_summary"]["parameters"]) == 2
    assert len(packet["parameter_bound_pressure"]) == 2
    assert packet["residual_summary"]["segments"][0]["label"] == "early"
    assert packet["residual_summary"]["segments"][0]["states"][0]["state"] == "state_0"
    assert packet["warnings"] == []
    assert packet["failure_modes"] == []

    warning_packet = build_diagnostic_packet(
        res={
            "median_distance": 2.0,
            "min_distance": 0.75,
            "accepted_params": np.array(
                [
                    [0.3, 0.5],
                    [0.4, 0.55],
                    [0.35, 0.6],
                ]
            ),
        },
        metadata=[
            {"name": "alpha", "range": (0.0, 1.0)},
            {"name": "beta", "range": (0.0, 1.0)},
        ],
        status="success",
        observed=np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        predicted=np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.2, 0.3],
                [1.5, 1.6],
                [1.8, 1.9],
                [2.0, 2.1],
            ]
        ),
        split_idx=4,
        train_metrics={"mse": 0.25, "rmse": 0.5},
        test_metrics={"mse": 0.64, "rmse": 0.8},
    )
    assert any("Held-out RMSE is 1.60x train RMSE" in warning for warning in warning_packet["warnings"])
    assert any("candidate may drift over longer horizons" in warning for warning in warning_packet["warnings"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "broken_candidate.py"
        out_path = tmp / "result.json"
        code_path.write_text(BROKEN_CODE)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(code_path),
                "--domain",
                "ecology",
                "--output",
                str(out_path),
                "--target-samples",
                "2",
                "--generations",
                "1",
                "--initial-particles",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode != 0, "broken candidate should fail sandbox evaluation"
        data = json.loads(out_path.read_text())
        assert data["status"] == "error"
        assert data["diagnostics"]["schema_version"] == 1
        assert "exception" in {mode["category"] for mode in data["diagnostics"]["failure_modes"]}
        assert "undefined_symbol" in data.get("traceback", "")
        print("SUCCESS: sandbox rejects runtime-broken generated dynamics")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "ecology_seed.py"
        out_path = tmp / "bdss_result.json"
        code_path.write_text(DOMAIN_CONFIGS["ecology"]["seed_logic"])

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(code_path),
                "--domain",
                "ecology",
                "--output",
                str(out_path),
                "--target-samples",
                "2",
                "--generations",
                "1",
                "--initial-particles",
                "8",
                "--seed",
                "17",
                "--inference-strategy",
                "bdss",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr or result.stdout
        data = json.loads(out_path.read_text())
        assert data["status"] == "success"
        assert data["requested_strategy"] == "bdss"
        assert data["effective_strategy"] == "bdss"
        assert data["diagnostics"]["fit"]["effective_strategy"] == "bdss"
        print("SUCCESS: sandbox threads explicit BDSS strategy into evaluation")


if __name__ == "__main__":
    main()

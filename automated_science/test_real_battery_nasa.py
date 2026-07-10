import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.domain_configs import DOMAIN_CONFIGS
from core.generated_code import validate_generated_math_code
from core.real_data.battery_nasa import (
    build_cell_holdout_split,
    build_capacity_target,
    build_within_cell_time_splits,
    cell_summary,
    normalize_battery_table,
)
from real_battery_nasa_data_loader import DEFAULT_FIXTURE, generate_data, resolve_raw_path


def main():
    normalized = normalize_battery_table(DEFAULT_FIXTURE)
    assert normalized["cell_id"].nunique() == 1
    assert normalized["soh"].iloc[0] == 1.0
    assert normalized["soh"].iloc[-1] < 0.75
    assert normalized["rul_cycles"].iloc[-1] == 0
    fixture_summary = cell_summary(normalized)
    assert fixture_summary[0]["cell_id"] == "BTEST"
    assert fixture_summary[0]["eol_reached"] is False

    time_points, target = build_capacity_target(normalized, cell_id="BTEST")
    assert target.shape == (6, 1)
    assert time_points.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
    assert np.all(np.diff(target[:, 0]) <= 0.0)

    assert "real_battery_nasa_capacity" in DOMAIN_CONFIGS
    config = DOMAIN_CONFIGS["real_battery_nasa_capacity"]
    assert len(config["y0"]) == 1
    validate_generated_math_code(config["seed_logic"])

    manifest = generate_data(raw_path=DEFAULT_FIXTURE)
    assert manifest["fixture_mode"] is True
    assert manifest["raw_path_source"] == "explicit"
    assert Path(manifest["target"]["ground_truth_path"]).exists()
    assert Path(manifest["target"]["time_points_path"]).exists()
    assert manifest["cell_holdout"]["status"] == "skipped"
    assert "cell holdout requires at least two cells" in manifest["cell_holdout"]["reason"]
    assert manifest["within_cell_split"]["status"] == "ok"
    fixture_cell_split = manifest["within_cell_split"]["cells"][0]
    assert fixture_cell_split["cell_id"] == "BTEST"
    assert fixture_cell_split["train_cycles"] == [0, 25, 50, 75]
    assert fixture_cell_split["test_cycles"] == [100, 125]

    with tempfile.TemporaryDirectory() as multi_tmpdir:
        multi = Path(multi_tmpdir)
        raw_dir = multi / "data/real/battery_nasa/raw"
        raw_dir.mkdir(parents=True)
        auto_raw = raw_dir / "cycle_level.csv"
        auto_raw.write_text(
            "\n".join(
                [
                    "cell_id,cycle_index,elapsed_time_h,ambient_temperature_c,capacity_ah",
                    "B1,0,0,24,2.00",
                    "B1,10,1,24,1.95",
                    "B1,20,2,24,1.88",
                    "B2,0,0,30,2.10",
                    "B2,10,1,30,2.00",
                    "B2,20,2,30,1.86",
                ]
            )
            + "\n"
        )
        resolved, fixture_mode = resolve_raw_path(auto_raw)
        assert resolved == auto_raw
        assert fixture_mode is False
        multi_normalized = normalize_battery_table(auto_raw)
        assert sorted(multi_normalized["cell_id"].unique().tolist()) == ["B1", "B2"]
        split = build_cell_holdout_split(multi_normalized, train_fraction=0.5, seed=3)
        assert split["status"] == "ok"
        assert set(split["train_cells"]).isdisjoint(set(split["test_cells"]))
        assert len(split["train_cells"]) == 1
        assert len(split["test_cells"]) == 1
        assert split["train_cycle_count"] == 3
        assert split["test_cycle_count"] == 3
        within_cell = build_within_cell_time_splits(multi_normalized, train_fraction=2 / 3)
        assert within_cell["status"] == "ok"
        assert len(within_cell["cells"]) == 2
        assert within_cell["cells"][0]["train_cycles"] == [0, 10]
        assert within_cell["cells"][0]["test_cycles"] == [20]
        _, b2_target = build_capacity_target(multi_normalized, cell_id="B2")
        assert b2_target.shape == (3, 1)
        multi_manifest = generate_data(
            raw_path=auto_raw,
            cell_id="B2",
            output_root=multi / "out",
            data_dir=multi / "dense",
            holdout_seed=3,
            holdout_train_fraction=0.5,
            within_cell_train_fraction=2 / 3,
        )
        assert multi_manifest["fixture_mode"] is False
        assert multi_manifest["target"]["cell_id"] == "B2"
        assert multi_manifest["cell_holdout"]["status"] == "ok"
        assert multi_manifest["cell_holdout"]["train_cells"] == split["train_cells"]
        assert multi_manifest["cell_holdout"]["test_cells"] == split["test_cells"]
        assert Path(multi_manifest["cell_holdout_json"]).exists()
        assert multi_manifest["within_cell_split"]["cells"][0]["test_cycles"] == [20]
        assert Path(multi_manifest["within_cell_split_json"]).exists()
        assert Path(multi_manifest["normalized_csv"]).exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "real_battery_seed.py"
        out_path = tmp / "result.json"
        code_path.write_text(config["seed_logic"])
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.sandbox_eval",
                "--code-file",
                str(code_path),
                "--domain",
                "real_battery_nasa_capacity",
                "--output",
                str(out_path),
                "--target-samples",
                "2",
                "--generations",
                "1",
                "--initial-particles",
                "8",
                "--held-out",
                "--seed",
                "23",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        payload = json.loads(out_path.read_text())
        assert payload["status"] == "success"
        assert payload["train_metrics"] is not None
        assert payload["test_metrics"] is not None

    print("SUCCESS: real_battery_nasa_capacity adapter smoke is stable")


if __name__ == "__main__":
    main()

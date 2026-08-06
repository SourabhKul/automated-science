from __future__ import annotations

from scripts.run_camels_us_synthetic_controls import run


def main() -> None:
    result = run()
    assert result["uses_measured_camels_streamflow"] is False
    assert result["uses_archive_streamflow_members"] is False
    assert result["episode_counts"] == {"train": 288, "selection": 96, "external": 128}
    assert result["checks"]["target_isolation"]
    assert result["checks"]["all_grid_finite_nonnegative"]
    assert result["selection_nrmse"] <= 0.05
    assert result["pairing_degradation"] >= 2.0
    assert result["time_order_degradation"] >= 2.0
    assert result["precipitation_ablation_degradation"] >= 1.5
    print("SUCCESS: CAMELS native-forcing synthetic recovery and specificity controls are stable")


if __name__ == "__main__":
    main()

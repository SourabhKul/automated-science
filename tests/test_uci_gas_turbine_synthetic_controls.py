from __future__ import annotations

from pathlib import Path

from scripts.run_uci_gas_turbine_synthetic_controls import run


def main() -> None:
    result = run(output_dir=Path("artifacts/evaluations/phase38_uci_gas_turbine_nox_synthetic_controls_20260725"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert result["selected_model"] == "linear_ridge_0.1"
    assert result["selection_rmse_mg_m3"] <= 0.25
    assert result["control_degradation_ratio"]["input_nox_pairing"] >= 2.0
    assert result["control_degradation_ratio"]["ambient_ablation"] >= 1.25
    assert result["control_degradation_ratio"]["process_ablation"] >= 1.25
    assert all(result["checks"].values())
    assert result["uses_measured_source_numeric_cells"] is False
    assert result["uses_measured_nox_values"] is False
    assert result["uses_measured_co_values"] is False
    assert result["external_artificial_records_scored"] is False
    print("SUCCESS: UCI gas-turbine synthetic controls are output-isolated and passed deterministically")


if __name__ == "__main__":
    main()

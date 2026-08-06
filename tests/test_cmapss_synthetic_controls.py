from __future__ import annotations

import numpy as np

from scripts.run_cmapss_fd002_synthetic_controls import run_controls


def main() -> None:
    result = run_controls()
    assert result["decision"] == "passed"
    assert result["budget"] == {"proposal_samples": 0, "generations": 0, "accepted_samples": 0, "replicates": 1, "abc_smc_calls": 0, "llm_calls": 0}
    aggregate = result["aggregate"]
    assert aggregate["all_finite"] and aggregate["output_isolated"] and aggregate["unit_leakage_detected"] and aggregate["prefix_leakage_safe"]
    assert aggregate["pairing_degradation_ratio"] >= 2.0
    assert aggregate["time_order_degradation_ratio"] >= 2.0
    assert aggregate["ablation_degradation_ratio"] >= 2.0
    assert not np.isnan(aggregate["signal_selection_nrmse"])
    print("SUCCESS: C-MAPSS synthetic controls are output-isolated, reset-safe, and input-specific")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from scripts.run_parallel_wiener_hammerstein_non_llm_inference import run


def main() -> None:
    result = run()
    assert result["candidate_count"] == 27
    assert result["gates"]["all_metrics_finite"]
    assert result["gates"]["no_prescreen_warning"]
    assert result["gates"]["pairing_pass"] and result["gates"]["time_order_pass"]
    assert np.isfinite(result["selected"]["selection"]["median_nrmse"])
    assert np.isfinite(result["selected"]["official_validation"]["median_nrmse"])
    assert not result["passed"]
    assert not result["gates"]["parameter_boundary_free"]
    print("SUCCESS: Parallel WH real-output comparison preserves its bounded negative decision")


if __name__ == "__main__":
    main()

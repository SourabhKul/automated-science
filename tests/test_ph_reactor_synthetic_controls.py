import numpy as np

from scripts.run_ph_reactor_synthetic_controls_smoke import run_smoke


def main():
    summary = run_smoke()
    assert summary["spec"]["abc_smc_calls"] == 0
    assert summary["spec"]["llm_calls"] == 0
    assert summary["aggregate"]["all_metrics_finite"]
    assert summary["output_isolation"]["sentinel_passed"]
    assert np.isfinite(summary["aggregate"]["signal_test_nrmse"])
    assert summary["decision"] in {"passed", "failed"}
    print("SUCCESS: pH-reactor synthetic control harness is finite and output-isolated")


if __name__ == "__main__":
    main()

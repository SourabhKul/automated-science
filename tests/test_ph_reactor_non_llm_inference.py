import numpy as np

from core.real_data.ph_reactor import load_ph_reactor
from scripts.legacy.real_ph_reactor_data_loader import DEFAULT_RAW_ROOT
from scripts.run_ph_reactor_non_llm_inference_smoke import fit_fixed_family, run_smoke


def main():
    data = load_ph_reactor(DEFAULT_RAW_ROOT)
    u_reference = float(np.mean(np.concatenate([item.input_u for item in data["train"]])))
    model = fit_fixed_family(data["train"], u_reference)
    assert 0.70 <= model["alpha"] <= 0.99
    assert -0.50 <= model["beta"] <= 0.50
    assert -0.20 <= model["offset"] <= 0.20
    result = run_smoke()
    assert result["spec"]["abc_smc_calls"] == 0
    assert result["spec"]["llm_calls"] == 0
    assert result["aggregate"]["all_metrics_finite"]
    assert result["decision"] in {"passed", "failed"}
    print("SUCCESS: pH non-LLM smoke is bounded, finite, and control-aware")


if __name__ == "__main__":
    main()

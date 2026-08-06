import numpy as np

from core.generated_code import GeneratedCodeError
from core.input_driven_eval import compile_input_driven_candidate, evaluate_input_driven_candidate, rollout_input_driven
from scripts.run_ph_reactor_input_driven_synthetic_smoke import CODE, PARAMS, planted_data


def main():
    candidate = compile_input_driven_candidate(CODE)
    data, candidate = planted_data()
    result = evaluate_input_driven_candidate(data, candidate, PARAMS)
    assert result["status"] == "success"
    assert result["prescreen"]["native_input_grid_count"] == 20
    assert result["untouched_test_metric"]["test_median_nrmse"] < 1e-12

    first, _ = rollout_input_driven(candidate, data["train"][0].input_u, PARAMS)
    second, _ = rollout_input_driven(candidate, data["train"][0].input_u, PARAMS)
    assert np.allclose(first, second), "state must reset for every experiment"

    paired = {split: list(experiments) for split, experiments in data.items()}
    validation = paired["validation"]
    paired["validation"] = [
        type(item)(item.experiment_id, item.split, validation[(index + 1) % len(validation)].input_u, item.output_y)
        for index, item in enumerate(validation)
    ]
    pairing_result = evaluate_input_driven_candidate(paired, candidate, PARAMS)
    assert pairing_result["selection_metric"]["validation_median_nrmse"] > 0.05

    exploding = compile_input_driven_candidate("""def dynamics(t, y, args, u):\n    return jnp.array([1e12])\n\nmetadata = [{'name': 'p', 'range': (0.0, 1.0)}]""")
    failed = evaluate_input_driven_candidate(data, exploding, np.array([0.5]))
    assert failed["status"] == "failed"
    assert "prescreen_bound_exceeded" in failed["failure_modes"]

    try:
        compile_input_driven_candidate("""def dynamics(t, y, args):\n    return jnp.array([0.0])\n\nmetadata = [{'name': 'p', 'range': (0.0, 1.0)}]""")
    except GeneratedCodeError:
        pass
    else:
        raise AssertionError("three-argument autonomous code must not pass the input-driven validator")

    try:
        compile_input_driven_candidate("""def dynamics(t, y, args, u):\n    return jnp.array([u[0]])\n\nmetadata = [{'name': 'p', 'range': (0.0, 1.0)}]""")
    except GeneratedCodeError as exc:
        assert "indexing u" in str(exc)
    else:
        raise AssertionError("input-driven code must not index scalar causal u")
    print("SUCCESS: isolated input-driven evaluator is causal, reset-safe, and prescreened")


if __name__ == "__main__":
    main()

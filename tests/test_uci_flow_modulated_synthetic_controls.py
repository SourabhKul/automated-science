from __future__ import annotations

from scripts.run_uci_flow_modulated_synthetic_controls import run


def main() -> None:
    result = run()
    assert result["status"] == "passed"
    assert result["source_inputs_only"]
    assert not result["raw_dR_opened"] and not result["features_opened"]
    assert result["input_counts"] == {"train": 39, "selection": 11, "external": 8}
    assert all(result["checks"].values())
    print("SUCCESS: UCI flow output-isolated synthetic controls passed")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from scripts.run_uci_statlog_landsat_adapter_mechanics_smoke import run


def main() -> None:
    result = run(output_dir=Path("/tmp/phase47_landsat_adapter_test"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: Statlog Landsat injected adapter is raw-only, reset-safe, isolated, and source-file-locked")


if __name__ == "__main__":
    main()

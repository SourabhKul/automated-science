from __future__ import annotations

from scripts.run_uci_ujiindoorloc_adapter_mechanics_smoke import run


def main() -> None:
    result = run()
    assert result["status"] == "passed_adapter_mechanics_only"
    assert result["target_isolation_sentinel_rejected"]
    assert result["metadata_isolation_sentinel_rejected"]
    print("SUCCESS: UJIIndoorLoc injected adapter is raw-WLAN-only, reset-safe, metadata-isolated, and file-split-locked")


if __name__ == "__main__":
    main()

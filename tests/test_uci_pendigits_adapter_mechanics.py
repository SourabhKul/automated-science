from __future__ import annotations
from scripts.run_uci_pendigits_adapter_mechanics_smoke import run
def main() -> None:
    result=run(); assert result["status"] == "passed_adapter_mechanics_only"; assert result["target_isolation_sentinel_rejected"] and result["writer_isolation_sentinel_rejected"]
    print("SUCCESS: PenDigits injected adapter is raw-coordinate-only, reset-safe, target-isolated, and file-split-locked")
if __name__=="__main__": main()

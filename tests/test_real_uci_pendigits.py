from __future__ import annotations
from scripts.run_uci_pendigits_source_gate import run
def main() -> None:
    result=run(); assert result["status"] == "passed_official_source_gate_only"; assert all(result["checks"]["fixed_rows"].values()); assert result["isolation"]["observed_coordinate_or_label_fitting"] is False
    print("SUCCESS: UCI PenDigits official source gate, writer-file split, and isolation contract passed")
if __name__ == "__main__": main()

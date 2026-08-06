from __future__ import annotations
from scripts.run_uci_pendigits_synthetic_controls import run
def main() -> None:
    result=run(); assert result["status"] == "passed_output_isolated_synthetic_controls"; assert result["label_pairing_drop"] >= .5 and result["coordinate_pair_permutation_drop"] >= .2
    print("SUCCESS: PenDigits output-isolated synthetic controls passed")
if __name__=="__main__": main()

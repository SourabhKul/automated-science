from __future__ import annotations
from scripts.run_uci_isolet_synthetic_controls import run
def main() -> None:
    r=run(); assert r["status"]=="passed_output_isolated_synthetic_controls"; assert r["label_pairing_drop"]>=.4; assert r["block_pairing_drop"]>=.15; assert max(r["quarter_ablation_drops"])>=.05; print("SUCCESS: ISOLET output-isolated synthetic controls passed")
if __name__=="__main__": main()

from __future__ import annotations
from pathlib import Path
from scripts.run_uci_spoken_arabic_digit_synthetic_controls import run
def main():
 r=run(output_dir=Path("/tmp/phase46_arabic_synthetic_test"));assert r["status"]=="passed_output_isolated_synthetic_controls";assert r["planted"]["macro_f1"]>=.95;assert r["controls"]["four_frame_reversal_drop"]>=.05;print("SUCCESS: Spoken Arabic Digit output-isolated synthetic controls passed")
if __name__=="__main__":main()

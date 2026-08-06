from pathlib import Path
from scripts.run_uci_image_segmentation_synthetic_controls import run
def main():
 r=run(Path("/tmp/phase54_segmentation_synthetic_test")); assert r["status"]=="passed_output_isolated_synthetic_controls"; assert r["controls"]["feature_block_pairing_drop"]>=.2; print("SUCCESS: Image Segmentation output-isolated synthetic controls passed")
if __name__=="__main__": main()

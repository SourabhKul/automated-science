from pathlib import Path
from scripts.run_uci_image_segmentation_adapter_mechanics_smoke import run
def main() -> None:
    result=run(Path("/tmp/phase54_segmentation_adapter_test")); assert result["status"]=="passed_adapter_mechanics_only"; assert all(result["controls"].values()); print("SUCCESS: Image Segmentation injected adapter is isolated and reset-safe")
if __name__=="__main__": main()

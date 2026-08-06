from pathlib import Path
from scripts.run_fashion_mnist_synthetic_controls import run
def main():
 r=run(Path("/tmp/phase57_synth"));assert r["status"]=="passed_output_isolated_synthetic_controls";print("SUCCESS: Fashion-MNIST artificial controls passed")
if __name__=="__main__":main()

from pathlib import Path
from scripts.run_fashion_mnist_adapter_mechanics_smoke import run
def main():
 r=run(Path("/tmp/phase57_test"));assert r["status"]=="passed_adapter_mechanics_only";assert all(r["controls"].values());print("SUCCESS: Fashion-MNIST artificial adapter is isolated and reset-safe")
if __name__=="__main__":main()

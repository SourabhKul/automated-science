from pathlib import Path
from scripts.run_kmnist49_adapter_mechanics_smoke import run
def main():
 r=run(Path("/tmp/phase71_k49_adapter_test"));assert r["status"]=="passed_adapter_mechanics_only";assert all(r["controls"].values());print("SUCCESS: K49 artificial adapter is reset-safe and isolated")
if __name__=="__main__":main()

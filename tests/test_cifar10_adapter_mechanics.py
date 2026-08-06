from pathlib import Path
from scripts.run_cifar10_adapter_mechanics_smoke import run
def main():
 r=run(Path('/tmp/phase58_cifar_test'));assert r['status']=='passed_adapter_mechanics_only';assert all(r['controls'].values());print('SUCCESS: CIFAR-10 artificial adapter is isolated and reset-safe')
if __name__=='__main__':main()

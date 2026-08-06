from pathlib import Path
from scripts.run_cifar10_synthetic_controls import run
def main():
 r=run(Path('/tmp/phase58_cifar_synth'));assert r['status']=='passed_output_isolated_synthetic_controls';print('SUCCESS: CIFAR-10 artificial controls passed')
if __name__=='__main__':main()

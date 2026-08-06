from scripts.run_uci_optdigits_synthetic_controls import run
def main():
 r=run();assert r['status']=='passed_output_isolated_synthetic_controls';print('SUCCESS: Optical Digits output-isolated synthetic controls passed')
if __name__=='__main__':main()

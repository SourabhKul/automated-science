from scripts.run_uci_optdigits_adapter_mechanics_smoke import run
def main():
 r=run(); assert r["status"]=="passed_adapter_mechanics_only"; print("SUCCESS: Optical Digits injected grid adapter is reset-safe and target-isolated")
if __name__=="__main__":main()

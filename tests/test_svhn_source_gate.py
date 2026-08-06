from scripts.run_svhn_source_gate import audit


def main() -> None:
    result = audit()
    assert result["status"] == "passed_official_source_gate_only"
    assert result["schema"]["shape"] == [32, 32, 3]
    assert result["frozen_file_split"]["external"] == "test_32x32.mat"
    print("SUCCESS: SVHN official source schema and train/test split are fixed")


if __name__ == "__main__":
    main()

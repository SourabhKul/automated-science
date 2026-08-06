from scripts.run_cifar100_source_gate import audit


def main() -> None:
    result = audit()
    assert result["status"] == "passed_official_source_gate_only"
    assert result["schema"]["features"] == 3072
    assert result["frozen_file_split"]["external"] == "test"
    print("SUCCESS: CIFAR-100 official source schema and creator train/test split are fixed")


if __name__ == "__main__":
    main()

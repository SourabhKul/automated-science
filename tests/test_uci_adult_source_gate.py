from scripts.run_uci_adult_source_gate import parse_rows


def main() -> None:
    rows = (
        b"39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"
        b"50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K\n"
    )
    result = parse_rows(rows, {"<=50K", ">50K"}, False)
    assert result["rows"] == 2
    assert result["field_count"] == 15
    print("SUCCESS: Adult source parser enforces the fixed 15-field contract")


if __name__ == "__main__":
    main()

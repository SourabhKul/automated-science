from scripts.run_stl10_source_gate import ROWS, parse_folds


def main() -> None:
    raw = b"\n".join(
        b" ".join(str(index).encode("ascii") for index in range(1_000))
        for _ in range(10)
    ) + b"\n"
    result = parse_folds(raw)
    assert result["line_count"] == 10
    assert result["indices_per_line"] == 1_000
    assert ROWS["train"] == 5_000
    print("SUCCESS: STL-10 fold parser enforces the fixed ten-fold source contract")


if __name__ == "__main__":
    main()

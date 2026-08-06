import numpy as np

from scripts.run_uci_higgs_source_gate import FIELDS, parse_line


def main() -> None:
    source = b"1," + b",".join(str(float(index)).encode() for index in range(FIELDS - 1)) + b"\n"
    parsed = parse_line(source, 1)
    assert parsed.shape == (FIELDS,)
    assert np.isfinite(parsed).all()
    for bad in (b"1,0\n", b"2," + b",".join(b"0" for _ in range(FIELDS - 1)) + b"\n"):
        try:
            parse_line(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid HIGGS source row was accepted")
    print("SUCCESS: HIGGS source parser enforces the fixed raw row contract")


if __name__ == "__main__":
    main()

import numpy as np

from scripts.run_uci_year_prediction_msd_source_gate import FIELDS, parse_line


def main() -> None:
    source = b"2000," + b",".join(str(float(index)).encode() for index in range(FIELDS - 1)) + b"\n"
    parsed = parse_line(source, 1)
    assert parsed.shape == (FIELDS,)
    assert np.isfinite(parsed).all()
    try:
        parse_line(b"2000,1\n", 2)
    except ValueError:
        pass
    else:
        raise AssertionError("short YearPredictionMSD row was accepted")
    print("SUCCESS: YearPredictionMSD source parser enforces the fixed raw row contract")


if __name__ == "__main__":
    main()

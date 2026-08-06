from __future__ import annotations

from scripts.run_uci_emg_gestures_source_gate import audit


def main() -> None:
    try:
        audit()
    except ValueError as exc:
        assert "wrong field count" in str(exc)
    else:
        raise AssertionError("expected the fixed EMG raw-schema contract to reject the official archive")
    print("SUCCESS: UCI EMG Data for Gestures source gate blocks malformed raw rows")


if __name__ == "__main__":
    main()

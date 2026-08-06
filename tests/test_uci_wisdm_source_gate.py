from __future__ import annotations

from scripts.run_uci_wisdm_source_gate import audit


def main() -> None:
    result = audit()
    assert result["status"] == "passed_official_source_gate_only"
    assert result["schema"]["phone_accelerometer_member_count"] == 51
    assert result["frozen_subject_split"]["train_subjects"] == list(range(1600, 1631))
    print("SUCCESS: WISDM official raw source, participant split, and phone-accelerometer contract passed")


if __name__ == "__main__":
    main()

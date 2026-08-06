from pathlib import Path

from scripts.run_kmnist_source_gate import run


def main() -> None:
    result = run(
        Path("data/real/kmnist/raw"),
        Path("/tmp/phase70_kmnist_source_gate_test"),
        Path("/tmp/phase70_kmnist_manifest.json"),
        Path("/tmp/phase70_kmnist_split.json"),
    )
    assert result["status"] == "passed_source_gate_only"
    assert all(result["contracts"].values())
    print("SUCCESS: KMNIST creator IDX source gate passed")


if __name__ == "__main__":
    main()

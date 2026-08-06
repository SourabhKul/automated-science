from pathlib import Path

from scripts.run_speech_commands_synthetic_controls import run


def main() -> None:
    result = run(Path("/tmp/phase68_speech_commands_synthetic"))
    assert result["status"] == "passed_output_isolated_synthetic_controls"
    assert all(result["controls"].values())
    print("SUCCESS: Speech Commands output-isolated artificial controls passed")


if __name__ == "__main__":
    main()

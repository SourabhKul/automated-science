from pathlib import Path

from scripts.run_speech_commands_adapter_mechanics_smoke import run


def main() -> None:
    result = run(Path("/tmp/phase68_speech_commands_adapter_test"))
    assert result["status"] == "passed_adapter_mechanics_only"
    assert all(result["controls"].values())
    print("SUCCESS: Speech Commands artificial adapter is isolated and reset-safe")


if __name__ == "__main__":
    main()

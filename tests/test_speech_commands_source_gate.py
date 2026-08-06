from scripts.run_speech_commands_source_gate import parse_list, parse_path


def main() -> None:
    paths = parse_list(b"yes/0123abcd_nohash_7.wav\nno/fedcba98_nohash_0.wav\n", "validation_list.txt")
    assert len(paths) == 2
    assert parse_path(paths[0], {"yes", "no"}) == ("yes", "0123abcd", 7)
    try:
        parse_list(b"yes/0123abcd_nohash_7.wav\nyes/0123abcd_nohash_7.wav\n", "validation_list.txt")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate creator list paths must fail")
    try:
        parse_path("_background_noise_/white_noise.wav", {"yes", "no"})
    except ValueError:
        pass
    else:
        raise AssertionError("background-noise paths must be excluded")
    print("SUCCESS: Speech Commands list and filename parsers enforce the fixed source contract")


if __name__ == "__main__":
    main()

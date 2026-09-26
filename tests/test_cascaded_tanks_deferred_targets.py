from __future__ import annotations

from collections.abc import Callable

import pytest

from core.real_data import cascaded_tanks_deferred_targets as deferred_targets

_HEADER = b'"uEst","uVal","yEst","yVal","Ts",'
_NAMES = ("uEst", "uVal", "yEst", "yVal")


def _numeric_field_width(row_index: int, slot: int) -> int:
    return 6 if row_index * 4 + slot < 3_352 else 5


def _csv_bytes(
    *,
    header: bytes = _HEADER,
    mutate: dict[tuple[int, str], bytes] | None = None,
) -> bytes:
    changes = mutate or {}
    result = bytearray(header + b"\n")
    for row_index in range(1_024):
        row = {
            name: (
                f"{(row_index + slot) % 10}.1234"
                if _numeric_field_width(row_index, slot) == 6
                else f"{(row_index + slot) % 10}.123"
            ).encode("ascii")
            for slot, name in enumerate(_NAMES)
        }
        row["Ts"] = b"4.0" if row_index == 0 else b""
        for name in (*_NAMES, "Ts"):
            row[name] = changes.get((row_index, name), row[name])
        result.extend(b",".join(row[name] for name in (*_NAMES, "Ts")))
        result.extend(b",\n")
    result.extend(b"\n")
    return bytes(result)


def _development_target_parse_gate_for_test(
    forecast_succeeded: bool,
    parse_targets: Callable[[], tuple[float, ...]],
) -> tuple[float, ...] | None:
    """Test-only placeholder; it is not a production forecast receipt gate."""

    if not forecast_succeeded:
        return None
    return parse_targets()


def test_synthetic_fixture_has_exact_pinned_wire_shape_and_only_returns_targets() -> (
    None
):
    wire = _csv_bytes()

    assert len(wire) == 30_014
    result = deferred_targets._parse_synthetic_development_targets(wire)

    assert isinstance(result, tuple)
    assert len(result) == 256
    assert all(isinstance(value, float) for value in result)


def test_changes_to_excluded_input_and_test_columns_do_not_change_targets() -> None:
    baseline = _csv_bytes()
    changes = {
        (row_index, name): b"x" * _numeric_field_width(row_index, slot)
        for row_index in range(1_024)
        for slot, name in ((1, "uVal"), (3, "yVal"))
    }
    changed = _csv_bytes(mutate=changes)

    assert len(changed) == len(baseline) == 30_014
    assert deferred_targets._parse_synthetic_development_targets(changed) == (
        deferred_targets._parse_synthetic_development_targets(baseline)
    )


def test_only_changed_development_suffix_targets_change_the_returned_values() -> None:
    baseline = _csv_bytes()
    changed = _csv_bytes(
        mutate={
            (0, "yEst"): b"xxxxxx",  # the training prefix is skipped
            (768, "yEst"): b"9.8765",
            (900, "yEst"): b"8.765",
        }
    )

    original = deferred_targets._parse_synthetic_development_targets(baseline)
    updated = deferred_targets._parse_synthetic_development_targets(changed)

    changed_positions = {
        index for index, pair in enumerate(zip(original, updated)) if pair[0] != pair[1]
    }
    assert changed_positions == {0, 900 - 768}
    assert updated[0] == 9.8765
    assert updated[900 - 768] == 8.765


def test_wrong_pinned_header_fails_closed() -> None:
    invalid = _csv_bytes(header=b'"xEst","uVal","yEst","yVal","Ts",')

    with pytest.raises(ValueError, match="header"):
        deferred_targets._parse_synthetic_development_targets(invalid)


def test_ts_is_required_on_first_row_and_forbidden_afterward() -> None:
    wrong_first = _csv_bytes(mutate={(0, "Ts"): b"3.0"})
    wrong_later = _csv_bytes(mutate={(1, "yVal"): b"xxx", (1, "Ts"): b"4.0"})

    assert len(wrong_first) == len(wrong_later) == 30_014
    with pytest.raises(ValueError, match="first-row Ts"):
        deferred_targets._parse_synthetic_development_targets(wrong_first)
    with pytest.raises(ValueError, match="Ts must be empty"):
        deferred_targets._parse_synthetic_development_targets(wrong_later)


def test_member_size_limit_fails_closed() -> None:
    invalid = _csv_bytes() + b"x"

    with pytest.raises(ValueError, match="member byte count"):
        deferred_targets._parse_synthetic_development_targets(invalid)


def test_field_length_limit_fails_closed_even_when_member_size_is_unchanged() -> None:
    changes: dict[tuple[int, str], bytes] = {(768, "yEst"): b"x" * 257}
    changes.update({(row_index, "uVal"): b"x" for row_index in range(50)})
    changes[(50, "uVal")] = b"xxxxx"
    invalid = _csv_bytes(mutate=changes)

    assert len(invalid) == 30_014
    with pytest.raises(ValueError, match="field exceeds"):
        deferred_targets._parse_synthetic_development_targets(invalid)


def test_nan_target_fails_closed_without_materializing_excluded_fields() -> None:
    invalid = _csv_bytes(mutate={(768, "yEst"): b"NaN", (0, "uVal"): b"abcdefghi"})

    assert len(invalid) == 30_014
    with pytest.raises(ValueError, match="numeric field"):
        deferred_targets._parse_synthetic_development_targets(invalid)


def test_test_only_gate_does_not_call_target_parser_after_failed_forecast() -> None:
    calls: list[str] = []

    def instrumented_target_parse() -> tuple[float, ...]:
        calls.append("called")
        return deferred_targets._parse_synthetic_development_targets(_csv_bytes())

    result = _development_target_parse_gate_for_test(False, instrumented_target_parse)

    assert result is None
    assert calls == []

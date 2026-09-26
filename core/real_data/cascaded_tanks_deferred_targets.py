"""Synthetic-only validation seam for deferred Cascaded Tanks targets.

This module deliberately has no archive access, source identity, production
receipt checks, or public reader.  Its sole parser accepts a synthetic CSV
wire byte stream so the byte-scanning and target-deferral mechanics can be
tested independently.  A source-backed scorer must not call it until a
forecast-success receipt has been verified.
"""

from __future__ import annotations

import math
import re

_HEADER_WIRE = b'"uEst","uVal","yEst","yVal","Ts",\n'
_EXPECTED_MEMBER_BYTES = 30_014
_EXPECTED_ROWS = 1_024
_TARGET_START = 768
_TARGET_COUNT = 256
_SAMPLE_INTERVAL = 4.0
_MAX_HEADER_BYTES = 128
_MAX_FIELD_BYTES = 256
_MAX_ROW_BYTES = 5 * _MAX_FIELD_BYTES + 6
_NUMBER_PATTERN = re.compile(
    rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)


class _SyntheticTargetParseError(ValueError):
    """A synthetic wire stream violated the pinned deferred-target grammar."""


def _parse_synthetic_development_targets(wire: bytes) -> tuple[float, ...]:
    """Return only synthetic ``yEst[768:1024]`` values from exact wire bytes.

    This is a parser-mechanics test seam, not an authorized real-source reader.
    It requires the pinned 30,014-byte CSV member shape.  Excluded columns and
    the ``yEst`` prefix are scanned byte by byte, counted, and discarded; they
    are never decoded, copied into field buffers, or converted to numbers.
    Only suffix ``yEst`` fields and the first-row ``Ts`` token are retained.
    """

    if not isinstance(wire, bytes):
        raise _SyntheticTargetParseError("synthetic CSV input must be bytes")
    if len(wire) != _EXPECTED_MEMBER_BYTES:
        raise _SyntheticTargetParseError(
            "synthetic CSV member byte count is not the pinned size"
        )
    if (
        len(_HEADER_WIRE) > _MAX_HEADER_BYTES
        or wire[: len(_HEADER_WIRE)] != _HEADER_WIRE
    ):
        raise _SyntheticTargetParseError(
            "synthetic CSV header is not the exact pinned wire header"
        )

    offset = len(_HEADER_WIRE)
    targets: list[float] = []

    for row_index in range(_EXPECTED_ROWS):
        row_start = offset
        column = 0
        field_length = 0
        field_buffer: bytearray | None = None
        while True:
            if offset >= len(wire):
                raise _SyntheticTargetParseError(
                    "synthetic CSV ended before all rows were complete"
                )
            value = wire[offset]
            offset += 1
            if offset - row_start > _MAX_ROW_BYTES:
                raise _SyntheticTargetParseError(
                    "synthetic CSV row exceeds the strict byte-length limit"
                )

            if value == 0x0A:
                if column != 5 or field_length != 0:
                    raise _SyntheticTargetParseError(
                        "synthetic CSV row lacks its exact trailing empty field"
                    )
                break

            if value == 0x2C:
                if column >= 5:
                    raise _SyntheticTargetParseError(
                        "synthetic CSV row contains too many fields"
                    )
                _finish_field(column, row_index, field_length, field_buffer, targets)
                column += 1
                field_length = 0
                field_buffer = _selected_buffer(column, row_index)
                continue

            if value == 0x22:
                raise _SyntheticTargetParseError(
                    "quoted data fields are outside the pinned numeric grammar"
                )
            if value < 0x20 or value == 0x7F:
                raise _SyntheticTargetParseError(
                    "synthetic CSV contains a control byte inside a field"
                )

            field_length += 1
            if field_length > _MAX_FIELD_BYTES:
                raise _SyntheticTargetParseError(
                    "synthetic CSV field exceeds the strict byte-length limit"
                )
            if field_buffer is not None:
                field_buffer.append(value)

    if wire[offset:] != b"\n":
        raise _SyntheticTargetParseError(
            "synthetic CSV must end with exactly one blank LF line"
        )
    if len(targets) != _TARGET_COUNT:
        raise _SyntheticTargetParseError(
            "synthetic CSV did not yield exactly 256 deferred targets"
        )
    return tuple(targets)


def _selected_buffer(column: int, row_index: int) -> bytearray | None:
    if column == 2 and row_index >= _TARGET_START:
        return bytearray()
    if column == 4 and row_index == 0:
        return bytearray()
    return None


def _finish_field(
    column: int,
    row_index: int,
    field_length: int,
    field_buffer: bytearray | None,
    targets: list[float],
) -> None:
    if column == 4 and row_index > 0:
        if field_length != 0:
            raise _SyntheticTargetParseError(
                "Ts must be empty after the first logical row"
            )
        return
    if field_length == 0:
        raise _SyntheticTargetParseError(
            "synthetic CSV contains an unexpected empty data field"
        )
    if column == 2 and row_index >= _TARGET_START:
        if field_buffer is None:
            raise _SyntheticTargetParseError("deferred target buffer is unavailable")
        targets.append(_parse_finite_number(bytes(field_buffer)))
    elif column == 4 and row_index == 0:
        if (
            field_buffer is None
            or _parse_finite_number(bytes(field_buffer)) != _SAMPLE_INTERVAL
        ):
            raise _SyntheticTargetParseError(
                "first-row Ts does not match the pinned sample interval"
            )


def _parse_finite_number(raw: bytes) -> float:
    if not _NUMBER_PATTERN.fullmatch(raw):
        raise _SyntheticTargetParseError(
            "selected numeric field is outside the finite-number grammar"
        )
    try:
        value = float(raw)
    except (OverflowError, ValueError) as error:
        raise _SyntheticTargetParseError("selected numeric field is invalid") from error
    if not math.isfinite(value):
        raise _SyntheticTargetParseError("selected numeric field is not finite")
    return value


__all__: tuple[str, ...] = ()

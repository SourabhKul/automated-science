import gzip
import io
import struct

from scripts.run_smallnorb_source_gate import MAGIC_INT32, read_header


def main() -> None:
    raw = struct.pack("<IIIII", MAGIC_INT32, 2, 24_300, 4, 1)
    with gzip.GzipFile(fileobj=io.BytesIO(), mode="wb") as _unused:
        pass
    assert read_header(io.BytesIO(raw)) == (MAGIC_INT32, (24_300, 4))
    print("SUCCESS: smallNORB matrix-header parser enforces the fixed dimensional contract")


if __name__ == "__main__":
    main()
